#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Low-level Vaisala Sigmet/IRIS RAW structure walking for virtualization.

Indexes an IRIS RAW volume (6144-byte records) into per-sweep contiguous
byte spans and walks the Sigmet 16-bit-word RLE ray compression without
decoding whole files. Shared by the parser, which builds the per-sweep
byte-range index and coordinate arrays at parse time, and the
``xradar-iris-sweep`` zarr codec, which demultiplexes one moment out of a
whole sweep's byte span at read time.

Headers come from the eager reader itself: ``read_volume`` runs
``IrisRawFile(buf, loaddata=False)`` (6-21 ms per file; the whole file is
needed) and adds only what it does not check (whole records, the
structure identifiers, the sweep-number runs that make spans contiguous).
Range gates, names, root attrs and the no-data table are the eager
helpers too. Not shared on purpose: the RLE walk (``sweep_words``,
``walk_sweep``, cached per span by ``walk_span``), which the codec runs on
a byte span with no file object;
a shared flat-stream decoder for both readers is a possible follow-up.

Only numpy and xradar's IRIS backend are needed at read time — this
module must stay importable without ``virtualizarr`` installed.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from xradar.io.backends.iris import (
    _NO_DATA_ZERO_TYPES,
    LEN_INGEST_DATA_HEADER,
    LEN_RAW_PROD_BHDR,
    LEN_RAY_HEADER,
    RECORD_BYTES,
    SIGMET_DATA_TYPES,
    STRUCTURE_HEADER_IDENTIFIERS,
    IrisRawFile,
    _data_type_dict,
    _range_centers,
    _root_attrs,
    decode_bin_angle,
)
from xradar.io.virtual._cache import SingleFlightCache
from xradar.io.virtual._checks import MAX_SPAN_CELLS
from xradar.io.virtual._sort import azimuth_sort_order

__all__ = [
    "NO_DATA_ZERO_TYPES",
    "RECORD_SIZE",
    "BHDR_SIZE",
    "IDH_SIZE",
    "RAY_HEADER_WORDS",
    "SIGMET_TYPE_NAMES",
    "IngestHeader",
    "DataTypeHeader",
    "SweepIndex",
    "RayHeader",
    "SweepRays",
    "parse_ingest_header",
    "IrisVolume",
    "read_volume",
    "range_centers",
    "index_sweeps",
    "azimuth_sort_order",
    "azimuth_midpoints",
    "sweep_words",
    "walk_sweep",
    "walk_span",
    "decode_sweep_moment",
]


#: Sizes of the framing structures, from the iris.py LEN_* constants.
RECORD_SIZE = RECORD_BYTES
BHDR_SIZE = LEN_RAW_PROD_BHDR  # raw_prod_bhdr, start of EVERY record
IDH_SIZE = LEN_INGEST_DATA_HEADER  # one per data type per sweep
RAY_HEADER_WORDS = LEN_RAY_HEADER // 2  # 6 int16 words per ray

#: structure_header.structure_identifier values, from the identifiers table.
_ID_BY_NAME = {v["name"]: k for k, v in STRUCTURE_HEADER_IDENTIFIERS.items()}
PRODUCT_HDR_ID = _ID_BY_NAME["PRODUCT_HDR"]
INGEST_HEADER_ID = _ID_BY_NAME["INGEST_HEADER"]
INGEST_DATA_HEADER_ID = _ID_BY_NAME["INGEST_DATA_HEADER"]

#: Data types whose raw word 0 means "no data", from the ``mask`` entries of
#: the eager ``SIGMET_DATA_TYPES`` table (what the parser's ``_FillValue``
#: and the eager decode both follow).
NO_DATA_ZERO_TYPES = _NO_DATA_ZERO_TYPES

#: Sigmet data-type code -> name, from the SIGMET_DATA_TYPES master table.
SIGMET_TYPE_NAMES = {code: entry["name"] for code, entry in SIGMET_DATA_TYPES.items()}


def _structure_id(buf, offset: int) -> int:
    """``structure_header.structure_identifier`` (int16 LE) at ``offset``."""
    return int.from_bytes(bytes(buf[offset : offset + 2]), "little", signed=True)


def _check_framing(buf) -> None:
    """Refuse anything that is not whole IRIS records starting with the
    product and ingest headers, before ``IrisRawFile`` parses it (which
    does not check the structure identifiers and fails with unrelated
    errors on foreign input)."""
    if len(buf) < 2 * RECORD_SIZE or len(buf) % RECORD_SIZE:
        raise ValueError(
            f"file size {len(buf)} is not a whole number (>= 2) of "
            f"{RECORD_SIZE}-byte records — truncated or not an IRIS RAW file"
        )
    sid0 = _structure_id(buf, 0)
    if sid0 != PRODUCT_HDR_ID:
        raise ValueError(
            f"record 0 structure_identifier={sid0}, expected {PRODUCT_HDR_ID} "
            "(PRODUCT_HDR) — not an IRIS RAW file"
        )
    sid1 = _structure_id(buf, RECORD_SIZE)
    if sid1 != INGEST_HEADER_ID:
        raise ValueError(
            f"record 1 structure_identifier={sid1}, expected {INGEST_HEADER_ID} "
            "(INGEST_HEADER) — not an IRIS RAW file"
        )


def bin2deg(word: int) -> float:
    """BIN2 coded angle -> degrees (unsigned view of a possibly signed word)."""
    return decode_bin_angle(word & 0xFFFF, mode=2)


def _text(value) -> str:
    """A header string as ``str``: the eager reader leaves fields that are
    not valid UTF-8 (e.g. a Latin-1 site name) as ``bytes``; Latin-1 decodes
    any byte, so nothing is lost and the attrs stay JSON-serializable."""
    if isinstance(value, bytes):
        value = value.decode("latin-1")
    return str(value).strip("\x00 ")


def _naive_ms(t) -> np.datetime64:
    if t is None:
        raise ValueError("invalid YMDS_TIME in IRIS header")
    return np.datetime64(t.replace(tzinfo=None), "ms")


@dataclass(frozen=True)
class IngestHeader:
    """The subset of the product and ingest headers the virtualization
    needs. ``prf``, ``wavelength`` and ``number_bins`` come from the
    product header's ``product_end``, like the eager ``IrisRawFile``
    decode (they can differ from the task configuration)."""

    site_name: str
    latitude: float
    longitude: float
    height_site: int  # meters MSL (ground)
    height_radar: int  # meters above ground
    altitude_cm: int  # radar altitude, centimeters MSL
    task_name: str
    prf: int  # Hz (product_end)
    multi_prf_mode_flag: int
    wavelength_cm: float  # product_end
    antenna_scan_mode: int
    range_first_bin_cm: int
    range_last_bin_cm: int
    number_output_bins: int
    step_output_bins_cm: int
    variable_range_spacing: bool
    volume_start: np.datetime64
    wavelength: int  # 1/100 cm, as stored (the eager nyquist input)
    number_bins: int  # gates per ray in the data (product_end)


def range_centers(hdr: IngestHeader) -> np.ndarray:
    """Range gate centers in meters (the eager reader's ``_range_centers``)."""
    return _range_centers(
        {
            "range_first_bin": hdr.range_first_bin_cm,
            "range_last_bin": hdr.range_last_bin_cm,
            "step_output_bins": hdr.step_output_bins_cm,
            "number_output_bins": hdr.number_output_bins,
        }
    )


@dataclass(frozen=True)
class DataTypeHeader:
    """One ingest_data_header: a sweep's entry for one data type."""

    sweep_number: int
    sweep_start: np.datetime64
    nrays_expected: int
    nrays_written: int
    fixed_angle: float
    bits_per_bin: int
    type_code: int
    type_name: str


@dataclass(frozen=True)
class SweepIndex:
    """One sweep's record range and its data-type directory."""

    sweep_number: int
    first_record: int
    last_record: int  # inclusive
    headers: tuple[DataTypeHeader, ...] = field(default_factory=tuple)

    @property
    def byte_offset(self) -> int:
        return self.first_record * RECORD_SIZE

    @property
    def byte_length(self) -> int:
        return (self.last_record - self.first_record + 1) * RECORD_SIZE

    @property
    def ndatatypes(self) -> int:
        return len(self.headers)


class IrisVolume(NamedTuple):
    """Everything the parser needs from one IRIS RAW file's headers."""

    header: IngestHeader
    sweeps: list[SweepIndex]
    root_attrs: dict  # the eager reader's volume attrs (``_root_attrs``)
    #: task_end_info ``echo_class_identifiers`` (HydroClass flag attrs)
    echo_class_identifiers: bytes = b""


def read_volume(buf) -> IrisVolume:
    """Ingest header, per-sweep index and root attrs via ``IrisRawFile``.

    ``IrisRawFile(buf, loaddata=False)`` unpacks the product/ingest headers
    and every record's ``raw_prod_bhdr``; record ``i + 2`` is
    ``raw_product_bhdrs[i]`` (only while no data has been loaded). The whole
    file is needed (it walks every record): 6-21 ms per file, no
    header-only probing.

    Raises ``ValueError`` for anything that is not a well-formed IRIS RAW
    file (wrong size, wrong structure identifiers, truncated records).
    """
    _check_framing(buf)
    try:
        raw = IrisRawFile(buf, loaddata=False)
    except (EOFError, OSError, KeyError, IndexError, struct.error) as exc:
        raise ValueError(
            f"cannot read IRIS headers ({type(exc).__name__}: {exc}) — "
            "truncated or not an IRIS RAW file"
        ) from exc
    ih = raw.ingest_header
    ic, tc = ih["ingest_configuration"], ih["task_configuration"]
    tri = tc["task_range_info"]
    lon, lat, _ = raw.site_coords
    # the eager decode takes prf, wavelength and the gate count from here
    product_end = raw.product_hdr["product_end"]
    hdr = IngestHeader(
        site_name=_text(ic["site_name"]),
        latitude=lat,
        longitude=lon,
        height_site=ic["height_site"],
        height_radar=ic["height_radar"],
        altitude_cm=ic["altitude_radar"],
        task_name=_text(tc["task_end_info"]["task_configuration_file_name"]),
        prf=product_end["prf"],
        multi_prf_mode_flag=tc["task_dsp_info"]["multi_prf_mode_flag"],
        wavelength_cm=product_end["wavelength"] / 100.0,
        antenna_scan_mode=raw.scan_mode,
        range_first_bin_cm=tri["range_first_bin"],
        range_last_bin_cm=tri["range_last_bin"],
        number_output_bins=tri["number_output_bins"],
        step_output_bins_cm=tri["step_output_bins"],
        variable_range_spacing=bool(tri["variable_range_bin_spacing_flag"]),
        volume_start=_naive_ms(ic["volume_scan_start_time"]),
        wavelength=product_end["wavelength"],
        number_bins=product_end["number_bins"],
    )

    if hdr.step_output_bins_cm <= 0 or hdr.number_output_bins <= 0:
        raise ValueError(
            f"task_range_info declares step {hdr.step_output_bins_cm} cm and "
            f"{hdr.number_output_bins} bins — corrupt IRIS header"
        )
    # (with variable spacing the gate positions are unknown; the parser
    # refuses those tasks)
    nrange = range_centers(hdr).size
    if not hdr.variable_range_spacing and nrange != hdr.number_bins:
        # the eager reader cannot align these either
        raise ValueError(
            f"product_end declares {hdr.number_bins} gates but "
            f"task_range_info gives {nrange} range bins"
        )

    runs: list[list[int]] = []  # [sweep_number, first_rec, last_rec]
    for r, bhdr in enumerate(raw.raw_product_bhdrs, start=2):
        sweep_number = bhdr["sweep_number"]
        if not runs or runs[-1][0] != sweep_number:
            runs.append([sweep_number, r, r])
        else:
            runs[-1][2] = r
    seen = [run[0] for run in runs]
    if any(number < 1 for number in seen):
        raise ValueError(f"sweep_number sequence {seen} has numbers below 1")
    if len(set(seen)) != len(seen):
        raise ValueError(
            f"sweep_number sequence {seen} revisits a sweep — records are "
            "not consecutive runs; cannot form contiguous sweep spans"
        )
    sweeps = []
    for sweep_number, r0, r1 in runs:
        sweep = raw.data.get(sweep_number)
        if sweep is None or sweep["record_number"] != r0:
            raise ValueError(f"sweep {sweep_number}: no ingest_data_headers at {r0}")
        idhs = list(sweep["ingest_data_hdrs"].values())
        bad = [
            d["structure_header"]["structure_identifier"]
            for d in idhs
            if d["structure_header"]["structure_identifier"] != INGEST_DATA_HEADER_ID
        ]
        if bad or not idhs:
            # IrisRawFile reads one IDH per DSP-mask bit without checking
            raise ValueError(
                f"sweep {sweep_number}: ingest_data_header identifiers {bad} "
                f"at record {r0} (expected {INGEST_DATA_HEADER_ID})"
            )
        headers = tuple(
            DataTypeHeader(
                sweep_number=d["sweep_number"],
                sweep_start=_naive_ms(d["sweep_start_time"]),
                nrays_expected=d["number_rays_file_expected"],
                nrays_written=d["number_rays_file_written"],
                fixed_angle=d["fixed_angle"],
                bits_per_bin=d["bits_per_bin"],
                type_code=d["data_type"],
                type_name=_data_type_dict(d["data_type"])["name"],
            )
            for d in idhs
        )
        sweeps.append(SweepIndex(sweep_number, r0, r1, headers))
    root_attrs = {
        k: _text(v) if isinstance(v, str | bytes) else v
        for k, v in _root_attrs(raw.product_hdr, ih).items()
    }
    identifiers = bytes(tc["task_end_info"]["echo_class_identifiers"])
    return IrisVolume(hdr, sweeps, root_attrs, identifiers)


def parse_ingest_header(buf) -> IngestHeader:
    """Thin wrapper (whole file required: ``IrisRawFile`` walks every record)."""
    return read_volume(buf).header


def index_sweeps(buf) -> list[SweepIndex]:
    """Thin wrapper over :func:`read_volume`."""
    return read_volume(buf).sweeps


def azimuth_midpoints(starts, stops) -> np.ndarray:
    """Wrap-aware per-ray azimuth midpoints on [0, 360).

    This is BOTH the azimuth coordinate value and the ``sort_rays``
    permutation key — using the same quantity for both keeps a sorted
    store's coordinate monotonic even for the ray that straddles north
    (start 359.6, stop 0.5 -> midpoint 0.05, sorted first).
    """
    start = np.asarray(starts, dtype=np.float64)
    stop = np.asarray(stops, dtype=np.float64)
    mid = (start + stop) / 2.0
    wrap = np.abs(stop - start) > 180.0
    mid = np.where(wrap, (mid + 180.0) % 360.0, mid)
    return mid


def sweep_words(span, ndatatypes: int) -> np.ndarray:
    """A sweep span (bytes or uint8 array) -> little-endian int16 words.

    Strips the ``raw_prod_bhdr`` from every record and the
    ``ndatatypes * IDH_SIZE``-byte ingest_data_header prologue from the
    first record, removing every interruption up front so the RLE walk
    below never has to think about record boundaries. One vectorized
    column slice + copy instead of per-record slicing.
    """
    raw = _span_bytes(span)
    if raw.size % RECORD_SIZE:
        raise ValueError(f"sweep span of {raw.size} bytes is not whole records")
    body = raw.reshape(-1, RECORD_SIZE)[:, BHDR_SIZE:].reshape(-1)
    return body[IDH_SIZE * ndatatypes :].view("<i2")


def _span_bytes(span) -> np.ndarray:
    """A sweep span (bytes or uint8 array) as a flat uint8 array (no copy)."""
    if isinstance(span, np.ndarray):
        return span.reshape(-1).view(np.uint8)
    return np.frombuffer(span, dtype=np.uint8)


@dataclass(frozen=True)
class RayHeader:
    """The 6-word header embedded at the start of each decompressed ray."""

    azimuth_start: float
    elevation_start: float
    azimuth_stop: float
    elevation_stop: float
    rbins: int
    dtime_s: int  # seconds since sweep start


@dataclass(frozen=True)
class SweepRays:
    """One data type's written rays of a sweep, in acquisition order.

    ``payload`` holds each ray's decompressed words after its header, cut
    to the width the walk was asked for and zero past the ray's end;
    ``nwords`` is how many of them the ray really decoded. The arrays are
    read-only: one walk is shared by every moment of the sweep.
    """

    headers: np.ndarray  # (rays, 6) int16 ray header words
    payload: np.ndarray  # (rays, width) little-endian int16 words
    nwords: np.ndarray  # (rays,) decoded payload words per ray
    missing: tuple[int, ...]  # ray-group indices of the missing rays

    def ray_headers(self) -> list[RayHeader]:
        """The rays' headers, decoded."""
        return [
            RayHeader(
                azimuth_start=bin2deg(int(words[0])),
                elevation_start=bin2deg(int(words[1])),
                azimuth_stop=bin2deg(int(words[2])),
                elevation_stop=bin2deg(int(words[3])),
                rbins=int(words[4]),
                dtime_s=int(words[5]) & 0xFFFF,
            )
            for words in self.headers.astype(np.int64)
        ]

    def azimuth_key(self) -> np.ndarray:
        """Wrap-aware azimuth midpoints, the ``sort_rays`` key."""
        words = self.headers.astype(np.int64) & 0xFFFF
        return azimuth_midpoints(
            decode_bin_angle(words[:, 0], mode=2),
            decode_bin_angle(words[:, 2], mode=2),
        )

    def rows(self, ngates: int, dtype: np.dtype, fill_value: int) -> np.ndarray:
        """``(rays, ngates)`` raw bins: each ray's first ``rbins`` bins,
        ``fill_value`` past them (or past the decoded words)."""
        dtype = np.dtype(dtype)
        per_word = 2 // dtype.itemsize  # 1-byte types pack two bins a word
        words = self.payload
        bins = words.view(np.uint8) if per_word == 2 else words.view("<u2")
        rbins = np.clip(self.headers[:, 4].astype(np.int64), 0, None)
        valid = np.minimum(np.minimum(rbins, self.nwords * per_word), ngates)
        ncols = min(ngates, bins.shape[1])
        out = np.full((len(words), ngates), fill_value, dtype=dtype)
        keep = np.arange(ncols) < valid[:, None]
        out[:, :ncols] = np.where(keep, bins[:, :ncols], fill_value)
        return out


def walk_sweep(
    words: np.ndarray,
    ndatatypes: int,
    payload_words: int = 0,
    max_rays: int | None = None,
) -> tuple[SweepRays, ...]:
    """One RLE walk of a sweep's word stream, demultiplexed by data type.

    Every ray of every data type is visited in interleave order (ray group
    g's type t is interleave index ``g * ndatatypes + t``) and decompressed
    up to its 6 header words plus ``payload_words`` words; the census
    (``payload_words=0``) reads only the headers. The walk ends when the
    remaining stream is zero padding (records are zero-padded after the
    last ray).

    A missing ray contributes no row, like the eager reader, which drops
    every ray whose header ``rbins`` word is 0: a ray is missing when its
    FIRST code word is the end-of-ray marker (value 1, MSB clear; no header
    at all) or when its header declares 0 bins.

    Returns one :class:`SweepRays` per data type (``ndatatypes`` of them).
    ``max_rays`` bounds the written rays of each type (a store's declared
    ray count); more raise ``ValueError`` before anything else is decoded.
    """
    width = RAY_HEADER_WORDS + payload_words
    rays: list[list[np.ndarray]] = [[] for _ in range(ndatatypes)]
    nwords: list[list[int]] = [[] for _ in range(ndatatypes)]
    missing: list[list[int]] = [[] for _ in range(ndatatypes)]
    n_words = len(words)
    pos = 0
    i = 0

    while pos < n_words and words.item(pos) != 0:
        group, ordinal = divmod(i, ndatatypes)
        ray = np.zeros(width, dtype="<i2")
        got = 0  # decompressed words of this ray so far
        first = True
        while True:
            if pos >= n_words:
                raise ValueError(
                    f"RLE stream exhausted mid-ray at interleave index {i} — "
                    "corrupt sweep span or wrong ndatatypes"
                )
            code = words.item(pos)
            pos += 1
            if code == 1:  # end of ray
                break
            if code >= 0:  # zero run: `code` words, already zero in `ray`
                got += code
            else:  # data run of `run` words
                run = code + 32768
                if got < width:
                    take = min(run, width - got)
                    ray[got : got + take] = words[pos : pos + take]
                got += run
                pos += run
            first = False
        i += 1

        if first:  # end-of-ray as the first code word: no header at all
            missing[ordinal].append(group)
            continue
        if got < RAY_HEADER_WORDS:
            raise ValueError(
                f"ray at interleave index {i - 1} decodes to {got} "
                f"words — shorter than the {RAY_HEADER_WORDS}-word ray header"
            )
        if ray.item(4) == 0:  # a header but no bins: the eager reader drops it
            missing[ordinal].append(group)
            continue
        if max_rays is not None and len(rays[ordinal]) == max_rays:
            raise ValueError(
                f"sweep span holds more than the {max_rays} rays the array "
                "metadata declares — manifest/file mismatch"
            )
        rays[ordinal].append(ray)
        nwords[ordinal].append(min(got, width) - RAY_HEADER_WORDS)

    out = []
    for ordinal in range(ndatatypes):
        block = (
            np.stack(rays[ordinal])
            if rays[ordinal]
            else np.zeros((0, width), dtype="<i2")
        )
        counts = np.asarray(nwords[ordinal], dtype=np.int64)
        for arr in (block, counts):
            arr.setflags(write=False)
        out.append(
            SweepRays(
                headers=block[:, :RAY_HEADER_WORDS],
                payload=block[:, RAY_HEADER_WORDS:],
                nwords=counts,
                missing=tuple(missing[ordinal]),
            )
        )
    return tuple(out)


def _walk_nbytes(walk: tuple[SweepRays, ...]) -> int:
    return sum(r.headers.nbytes + r.payload.nbytes + r.nwords.nbytes for r in walk)


#: Walked spans, shared by every moment of a sweep (12 moments of a
#: 360 x 750 sweep keep ~6.5 MB).
_SPANS: SingleFlightCache[tuple[SweepRays, ...]] = SingleFlightCache(
    max_bytes=128 * 2**20, sizeof=_walk_nbytes
)


def walk_span(
    span, ndatatypes: int, out_shape: tuple[int, int]
) -> tuple[SweepRays, ...]:
    """:func:`walk_sweep` over a whole sweep span, cached.

    The first moment read from a span walks it once for every data type;
    the sweep's other moments (which reference the same bytes) are served
    from a small cache, keyed by the span's content and the walk's
    parameters. Concurrent readers of one span wait for a single walk.
    """
    n_rows, ngates = out_shape
    if ndatatypes * n_rows * (RAY_HEADER_WORDS + ngates) > MAX_SPAN_CELLS:
        raise ValueError(
            f"{ndatatypes} data types x {out_shape} cells exceed "
            f"{MAX_SPAN_CELLS} words per sweep — corrupt or hostile metadata"
        )
    raw = _span_bytes(span)
    digest = hashlib.blake2b(raw, digest_size=16).digest()
    return _SPANS.get(
        (digest, ndatatypes, n_rows, ngates),
        lambda: walk_sweep(
            sweep_words(raw, ndatatypes),
            ndatatypes,
            payload_words=ngates,
            max_rays=n_rows,
        ),
    )


def decode_sweep_moment(
    span: bytes,
    moment_index: int,
    ndatatypes: int,
    out_shape: tuple[int, int],
    dtype: np.dtype,
    fill_value: int = 0,
    sort_rays: bool = False,
    pad_missing_rays: bool = False,
) -> np.ndarray:
    """Demultiplex one data type out of a whole-sweep byte span.

    ``span`` is the contiguous run of complete records of one sweep (bhdr +
    prologue included). Rows are in acquisition order, gates
    padded/truncated to ``out_shape[1]``. The span is walked once for all
    its data types (:func:`walk_span`).

    Missing rays (end-of-ray marker as the FIRST code word, or a ray
    header declaring 0 bins):

    - ``pad_missing_rays=False`` (default, eager-reader parity): missing
      rays contribute no row; ``out_shape[0]`` must equal the WRITTEN ray
      count.
    - ``pad_missing_rays=True``: every ray slot keeps its position —
      missing slots are ``fill_value`` rows; ``out_shape[0]`` must equal
      the EXPECTED slot count.

    With ``sort_rays`` the rows are reordered by ``azimuth_sort_order``
    over the rays' wrap-aware azimuth midpoints; padded missing rows (NaN
    key) sort last.
    """
    n_rows_out, ngates_out = out_shape
    rays = walk_span(span, ndatatypes, out_shape)[moment_index]
    written = rays.rows(ngates_out, dtype, fill_value)
    key = rays.azimuth_key() if sort_rays else None

    if pad_missing_rays:
        missing_slots = set(rays.missing)
        n_slots = len(written) + len(missing_slots)
        if n_slots != n_rows_out:
            raise ValueError(
                f"sweep span holds {n_slots} ray slots ({len(written)} "
                f"written + {len(missing_slots)} missing) but the array "
                f"metadata declares {n_rows_out} — manifest/file mismatch"
            )
        rows = np.full(out_shape, fill_value, dtype=dtype)
        written_slots = [g for g in range(n_rows_out) if g not in missing_slots]
        rows[written_slots] = written
        if sort_rays:
            full_key = np.full(n_rows_out, np.nan)
            full_key[written_slots] = key
            key = full_key
    else:
        if len(written) != n_rows_out:
            raise ValueError(
                f"sweep span decoded {len(written)} written rays but the "
                f"array metadata declares {n_rows_out} — manifest/file "
                "mismatch"
            )
        rows = written

    if sort_rays:
        rows = rows[azimuth_sort_order(key)]
    return np.ascontiguousarray(rows)
