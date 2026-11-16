#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Vaisala Sigmet/IRIS RAW parser: true byte-range virtualization.

One zarr chunk = one whole sweep — the contiguous run of complete records
holding that sweep (rays are RLE-compressed with all data types interleaved
ray-major, so no smaller unit is both contiguous and independently
decodable). Every data-type array of a sweep references the SAME span; the
``xradar-iris-sweep`` codec demultiplexes one interleave ordinal per array at
read time. Coordinate arrays (azimuth/elevation/time/range) are decoded
once at parse time and stored inline.

Moment names, CF attributes, linear scaling, the sweep mode, the nyquist
velocity, range gates and the root attrs come from the eager IRIS backend's
own tables and helpers (``iris_mapping``, ``SIGMET_DATA_TYPES``,
``_moment_names``, ``_sweep_mode``, ``_nyquist``, ``_dual_prf_factor``,
``_range_centers``, ``_root_attrs``), :mod:`xradar.model`, and the eager root builder
(``common._assign_root``), so the virtual view cannot drift from the eager
decode. Not shared on purpose: the header walk of the RLE stream
(``format.walk_sweep``), which the codec runs on a byte span with no file
object.

RHI tasks are refused: the eager reader lays them out as
``(elevation, range)`` and this parser builds ``(azimuth, range)`` sweeps
only. Tasks with variable range bin spacing are refused too: the range
coordinate would need gate positions the header does not give.

Ray times are int64 milliseconds since the epoch, like the eager reader's:
the ``DB_XHDR`` extended header's millisecond times where the sweep has one,
the ray header's whole seconds otherwise.

Requires the ``xradar[virtual]`` extra.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
from virtualizarr.manifests import (
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
    ManifestStore,
)
from virtualizarr.manifests.utils import create_v3_array_metadata

from xradar.io.backends.iris import (
    _dual_prf_factor,
    _elevation_centers,
    _moment_cf_attrs,
    _moment_names,
    _nyquist,
    _sweep_mode,
)
from xradar.io.virtual.iris import codec as _codec  # noqa: F401  (registers)
from xradar.io.virtual.iris.decode import (
    DECODE_CODEC_NAME,
    _cf_scaling,
    decode_codec_config,
)
from xradar.io.virtual.iris.format import (
    XHDR_TIME_WORDS,
    DataTypeHeader,
    IngestHeader,
    RayHeader,
    SweepIndex,
    check_aligned,
    range_attrs,
    range_centers,
    read_volume,
    sweep_words,
    walk_sweep,
)
from xradar.io.virtual.manifest import (
    FM301_STRING_DEFAULTS,
    MOMENT_COORDINATES,
    fetch_bytes,
    inline_root,
    inline_scalar,
    inline_variable,
    to_json_safe,
)
from xradar.model import (
    get_altitude_attrs,
    get_azimuth_attrs,
    get_elevation_attrs,
    get_latitude_attrs,
    get_longitude_attrs,
    get_nyquist_velocity_attrs,
    get_time_attrs,
)

if TYPE_CHECKING:
    from obspec_utils.registry import ObjectStoreRegistry

__all__ = ["IrisParser"]


#: the time of a padded ray (int64 ms), its ``_FillValue``: NaT once decoded
_TIME_FILL = int(np.iinfo(np.int64).min)


def _elevation_midpoints(headers: list[RayHeader]) -> np.ndarray:
    """Per-ray elevation, the eager reader's ``_elevation_centers`` as float32
    (a ray slightly below the horizon is -0.05, not 359.95). Azimuth is
    ``SweepRays.azimuth_key``."""
    start = np.array([h.elevation_start for h in headers], dtype=np.float64)
    stop = np.array([h.elevation_stop for h in headers], dtype=np.float64)
    return _elevation_centers(start, stop).astype(np.float32)


def _sweep_group(
    url: str,
    buf: bytes,
    hdr: IngestHeader,
    sweep: SweepIndex,
    sweep_idx: int,
    rng: np.ndarray,
    rng_attrs: dict,
    nyquist: float,
    nyquist_vel: float,
    pad_missing_rays: bool,
    drop_variables: set[str],
    echo_class_identifiers: bytes = b"",
) -> tuple[ManifestGroup, np.ndarray]:
    """One sweep's group, plus its written rays' epoch-ms times (int64)."""
    headers: tuple[DataTypeHeader, ...] = sweep.headers
    type_names = [dth.type_name for dth in headers]
    # the extended headers' first two payload words are the ray's time in ms
    xhdr = type_names.index("DB_XHDR") if "DB_XHDR" in type_names else None
    ndt = sweep.ndatatypes
    nbins = hdr.number_bins  # product_end, like the eager decode

    # One header walk per sweep gives every per-ray coordinate plus the
    # missing-ray census. Rows stay aligned across the sweep's moment arrays
    # ONLY if missingness is group-consistent, so a sweep where data types
    # disagree on which rays are missing is rejected.
    span = buf[sweep.byte_offset : sweep.byte_offset + sweep.byte_length]
    census = walk_sweep(
        sweep_words(span, ndt),
        ndt,
        payload_words=0 if xhdr is None else XHDR_TIME_WORDS,
    )
    check_aligned(census, f"sweep {sweep.sweep_number}")
    ray_headers = census[0].ray_headers()
    if len(ray_headers) == 0:
        raise ValueError(f"sweep {sweep.sweep_number}: no written rays")

    missing_slots = census[0].missing.tolist()
    azimuth = census[0].azimuth_key()  # the codec's sort_rays key too
    elevation = _elevation_midpoints(ray_headers)
    epoch_ms = headers[0].sweep_start.astype("datetime64[ms]").astype(np.int64)
    if xhdr is None:
        offsets = np.array([h.dtime_s * 1000 for h in ray_headers], dtype=np.int64)
    else:
        try:
            offsets = census[xhdr].dtime_ms()
        except ValueError as err:
            raise ValueError(f"sweep {sweep.sweep_number}: {err}") from None
    ray_ms = time = epoch_ms + offsets

    padded = bool(pad_missing_rays and missing_slots)
    if padded:
        # Every ray SLOT keeps its position; missing slots get NaN angles
        # and the time fill (NaT once decoded); the codec pads the data rows
        # the same way, so row r is the same physical slot in every moment.
        n_slots = len(ray_headers) + len(missing_slots)
        missing_set = set(missing_slots)
        written_slots = [g for g in range(n_slots) if g not in missing_set]

        def _pad(values: np.ndarray, fill: float = np.nan) -> np.ndarray:
            out = np.full(n_slots, fill, dtype=values.dtype)
            out[written_slots] = values
            return out

        azimuth, elevation = _pad(azimuth), _pad(elevation)
        time = _pad(time, _TIME_FILL)
    nrays = len(azimuth)

    # Pure pointers: rows and per-ray coordinates stay in acquisition order.
    # Sorting is a store-builder decision; it sets ``sort_rays: true`` in the
    # codec config and writes the matching sorted coordinates itself.
    # Moment names follow the eager reader: when two data types map to one
    # CfRadial name (DB_DBZ + DB_DBZ2), the later, finer one keeps it and the
    # earlier one keeps its Sigmet name.
    names = _moment_names(type_names)
    arrays: dict[str, ManifestArray] = {}
    for ordinal, (dth, cf_name) in enumerate(zip(headers, names, strict=True)):
        if dth.type_name == "DB_XHDR":
            continue  # extended headers are framing, not a data moment
        if cf_name in drop_variables:
            continue
        if dth.bits_per_bin not in (8, 16):
            raise ValueError(
                f"sweep {sweep.sweep_number} {dth.type_name}: unsupported "
                f"bits_per_bin={dth.bits_per_bin}"
            )
        dtype = np.dtype("uint8") if dth.bits_per_bin == 8 else np.dtype("uint16")
        # numpy flag arrays (HydroClass) become JSON lists for zarr.json
        cf_attrs = to_json_safe(_moment_cf_attrs(dth.type_name, echo_class_identifiers))
        attrs: dict = {
            **cf_attrs,
            "coordinates": MOMENT_COORDINATES,
            "sigmet_data_type": dth.type_name,
        }
        sweep_codec = {
            "name": _codec.CODEC_NAME,
            "configuration": {
                "moment_index": ordinal,
                "ndatatypes": ndt,
                "sort_rays": False,  # frozen config key; see above
                "pad_missing_rays": pad_missing_rays,
            },
        }
        codecs, fill_value = [sweep_codec], 0
        scaling = _cf_scaling(dth.type_code, nyquist, nyquist_vel)
        if scaling is not None:
            attrs["scale_factor"], attrs["add_offset"], fill = scaling
            if fill is not None:
                # Sigmet raw ``mask`` (e.g. 0 for velocity) = no data; with
                # mask_and_scale these bins decode to NaN instead of a
                # plausible-looking physical value.
                attrs["_FillValue"] = int(fill)
        elif (
            decode := decode_codec_config(dth.type_code, hdr.wavelength_cm)
        ) is not None:
            # nonlinear, signed or IRIS-float words: the eager decoder runs in
            # the xradar-iris-decode codec, into float32 like the eager reader
            codecs = [{"name": DECODE_CODEC_NAME, "configuration": decode}, *codecs]
            dtype, fill_value = np.dtype("float32"), np.nan
        elif "flag_meanings" not in attrs:
            # (HydroClass words with CF flag attrs are described by them)
            attrs["comment"] = (
                "raw Sigmet words: the eager backend's SIGMET_DATA_TYPES has "
                "no decoder for this data type — decode per the IRIS "
                "Programmer's Manual"
            )
        metadata = create_v3_array_metadata(
            shape=(nrays, nbins),
            chunk_shape=(nrays, nbins),
            data_type=dtype,
            fill_value=fill_value,
            codecs=codecs,
            attributes=attrs,
            dimension_names=("azimuth", "range"),
        )
        manifest = ChunkManifest(
            entries={
                "0.0": {
                    "path": url,
                    "offset": sweep.byte_offset,
                    "length": sweep.byte_length,
                }
            }
        )
        arrays[cf_name] = ManifestArray(metadata=metadata, chunkmanifest=manifest)

    if not arrays:
        raise ValueError(f"sweep {sweep.sweep_number}: no data moments")

    arrays["azimuth"] = inline_variable(("azimuth",), azimuth, get_azimuth_attrs())
    arrays["elevation"] = inline_variable(
        ("azimuth",), elevation, get_elevation_attrs()
    )
    time_attrs = get_time_attrs(date_unit="milliseconds")
    if padded:
        # xarray masks the padded rays: NaT, or NaN with decode_times=False
        time_attrs["_FillValue"] = _TIME_FILL
    arrays["time"] = inline_variable(("azimuth",), time, time_attrs)
    arrays["range"] = inline_variable(("range",), rng, rng_attrs)
    arrays["sweep_mode"] = inline_scalar(_sweep_mode(hdr.antenna_scan_mode))
    arrays["prt_mode"] = inline_scalar(FM301_STRING_DEFAULTS["prt_mode"])
    arrays["follow_mode"] = inline_scalar(FM301_STRING_DEFAULTS["follow_mode"])
    arrays["sweep_number"] = inline_scalar(np.int64(sweep_idx))
    # rounded like the eager reader (BIN2 0.5 deg decodes as 0.49988)
    fixed_angle = float(np.round(headers[0].fixed_angle, 1))
    arrays["sweep_fixed_angle"] = inline_scalar(np.float64(fixed_angle))
    # Per-file physics behind the VEL/WIDTH CF scaling — stored explicitly
    # so downstream stores can carry a per-volume nyquist even though attrs
    # are written once.
    arrays["nyquist_velocity"] = inline_scalar(
        np.float64(nyquist_vel), get_nyquist_velocity_attrs()
    )
    arrays["latitude"] = inline_scalar(np.float64(hdr.latitude), get_latitude_attrs())
    arrays["longitude"] = inline_scalar(
        np.float64(hdr.longitude), get_longitude_attrs()
    )
    arrays["altitude"] = inline_scalar(
        np.float64(hdr.altitude_cm / 100.0), get_altitude_attrs()
    )

    group_attrs = {
        "fixed_angle": fixed_angle,
        "coordinates": (
            "sweep_mode prt_mode follow_mode sweep_number sweep_fixed_angle "
            "nyquist_velocity latitude longitude altitude"
        ),
    }
    return ManifestGroup(arrays=arrays, attributes=group_attrs), ray_ms


class IrisParser:
    """Parse a Vaisala Sigmet/IRIS RAW file into a ``ManifestStore``.

    The store holds pure byte-range pointers: rays and per-ray coordinates
    are in acquisition order, and every moment's codec config declares
    ``sort_rays: false``. Ordering rays by azimuth is left to whoever builds
    a store from it (the ``xradar-iris-sweep`` codec still honors
    ``sort_rays: true``).

    Parameters
    ----------
    pad_missing_rays : bool
        ``False`` (default): missing rays are compacted away and arrays
        hold only WRITTEN rays, mirroring the eager backend. ``True``:
        every ray slot keeps its position — missing slots read as no data
        (0.0 for the signed types and ``DB_FLIQUID2``, which have no
        no-data word) with NaN per-ray coordinates, and the azimuth dim
        equals the sweep's EXPECTED ray count.
    drop_variables : iterable of str, optional
        CfRadial moment names to exclude when building the store.
    """

    def __init__(
        self,
        pad_missing_rays: bool = False,
        drop_variables: Iterable[str] | None = None,
    ) -> None:
        self.pad_missing_rays = bool(pad_missing_rays)
        self.drop_variables = set(drop_variables or ())

    def __call__(
        self,
        url: str,
        registry: ObjectStoreRegistry,
    ) -> ManifestStore:
        """Parse an IRIS RAW file into a ``ManifestStore``.

        Parameters
        ----------
        url : str
            URL of the IRIS RAW file (``file://`` or any scheme with a
            store registered in ``registry``).
        registry : obspec_utils.registry.ObjectStoreRegistry
            Registry used to read the file at parse time and to resolve
            the virtual references at read time.

        Returns
        -------
        ManifestStore
            One root ``ManifestGroup`` containing one subgroup per sweep.
        """
        buf = memoryview(fetch_bytes(url, registry))  # zero-copy sweep slices
        hdr, sweeps, root_attrs, echo_class_identifiers = read_volume(buf)
        if _sweep_mode(hdr.antenna_scan_mode) == "rhi":
            raise NotImplementedError(
                f"{url}: RHI tasks (antenna_scan_mode=2) are not supported by "
                "the virtual IRIS parser; read them with xradar's eager IRIS "
                "backend (open_iris_datatree)"
            )
        if hdr.variable_range_spacing:
            # gate positions are not a constant step: no range to build
            raise NotImplementedError(
                f"{url}: tasks with variable range bin spacing are not "
                "supported by the virtual IRIS parser (gate positions are "
                "unknown); read them with open_iris_datatree"
            )
        rng, rng_attrs = range_centers(hdr), range_attrs(hdr)

        # the eager reader's nyquist: the multi-PRF factor applies to DB_VEL
        nyquist = _nyquist(hdr.wavelength, hdr.prf)
        nyquist_vel = nyquist * _dual_prf_factor(hdr.multi_prf_mode_flag)

        groups: dict[str, ManifestGroup] = {}
        ray_times: list[np.ndarray] = []
        for position, sweep in enumerate(sweeps):
            # like the eager reader: groups are named by position, while the
            # sweep_number variable is the file's own (0-based) sweep number
            groups[f"sweep_{position}"], times = _sweep_group(
                url,
                buf,
                hdr,
                sweep,
                sweep.sweep_number - 1,
                rng,
                rng_attrs,
                nyquist,
                nyquist_vel,
                self.pad_missing_rays,
                self.drop_variables,
                echo_class_identifiers,
            )
            ray_times.append(times)

        # the eager root: coverage = min/max over every real ray
        root_arrays, attrs = inline_root(
            ray_times,
            np.float64(hdr.latitude),
            np.float64(hdr.longitude),
            np.float64(hdr.altitude_cm / 100.0),
            root_attrs,
        )
        root = ManifestGroup(arrays=root_arrays, groups=groups, attributes=attrs)
        return ManifestStore(group=root, registry=registry)
