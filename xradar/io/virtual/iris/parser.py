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
``_moment_names``, ``_sweep_mode``, ``_nyquist``, ``_range_centers``,
``_root_attrs``), :mod:`xradar.model`, and the eager root builder
(``common._assign_root``), so the virtual view cannot drift from the eager
decode. Not shared on purpose: the header walk of the RLE stream
(``format.walk_sweep``), which the codec runs on a byte span with no file
object.

RHI tasks are refused: the eager reader lays them out as
``(elevation, range)`` and this parser builds ``(azimuth, range)`` sweeps
only.

Known limitation: per-ray times come from the ray header's ``dtime`` field
(1-second resolution). Files carrying DB_XHDR extended headers have
millisecond times inside the XHDR payload which are NOT read yet, so ray
times can differ from the eager decode by up to 999 ms on such files.

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
    SIGMET_DATA_TYPES,
    _moment_names,
    _nyquist,
    _sweep_mode,
    decode_array,
    decode_phidp,
    decode_phidp2,
    decode_vel,
    decode_width,
    iris_mapping,
)
from xradar.io.virtual.iris import codec as _codec  # noqa: F401  (registers)
from xradar.io.virtual.iris.format import (
    DataTypeHeader,
    IngestHeader,
    RayHeader,
    SweepIndex,
    azimuth_midpoints,
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
    moment_attrs,
)
from xradar.model import (
    get_altitude_attrs,
    get_azimuth_attrs,
    get_elevation_attrs,
    get_latitude_attrs,
    get_longitude_attrs,
    get_nyquist_velocity_attrs,
    get_range_attrs,
    get_time_attrs,
)

if TYPE_CHECKING:
    from obspec_utils.registry import ObjectStoreRegistry

__all__ = ["IrisParser"]


def _cf_scaling(
    type_code: int, nyquist: float, nyquist_vel: float
) -> tuple[float, float, float | None] | None:
    """(scale_factor, add_offset, fill_or_None) turning raw words into
    physical values, derived from the eager backend's ``SIGMET_DATA_TYPES``
    entry for this type.

    The eager decoders are ``decode_array = (raw + offset)/scale + offset2``
    plus thin wrappers (``decode_vel`` multiplies by the effective nyquist,
    ``decode_width`` by the unfolded nyquist, ``decode_phidp``/``_phidp2``
    by 180/360 degrees); a ``mask`` value becomes ``_FillValue`` so CF
    mask_and_scale reproduces the eager masking. Types whose table entry
    declares a SIGNED dtype stay raw: the virtual arrays hold the unsigned
    wire words, and a linear transform cannot reproduce the signed
    reinterpretation (words at or above half-range would wrap). Nonlinear
    decoders (sqrt SQI/RHOHV, log KDP), categorical types (HCLASS), and
    float-word types (``tofloat``) stay raw too.
    """
    entry = SIGMET_DATA_TYPES.get(type_code) or {}
    func = entry.get("func")
    fkw = entry.get("fkw") or {}
    scale = fkw.get("scale")
    offset = fkw.get("offset", 0.0)
    offset2 = fkw.get("offset2", 0.0)
    if not scale:
        return None
    if np.dtype(entry.get("dtype", "uint8")).kind != "u":
        return None  # signed wire dtype: linear CF on our unsigned words wraps
    if func is decode_array and "tofloat" not in fkw:
        return 1.0 / scale, offset / scale + offset2, fkw.get("mask")
    if func is decode_vel:
        return nyquist_vel / scale, offset * nyquist_vel / scale, fkw.get("mask")
    if func is decode_width:
        return nyquist / scale, offset * nyquist / scale, None
    if func is decode_phidp:
        return 180.0 / scale, offset * 180.0 / scale, None
    if func is decode_phidp2:
        return 360.0 / scale, offset * 360.0 / scale, None
    return None


def _angle_midpoints(headers: list[RayHeader], which: str) -> np.ndarray:
    """Per-ray angle midpoint (start/stop mean) — the eager backend's
    coordinate convention: azimuth on [0, 360) with wrap-aware midpoints,
    elevation folded to signed degrees (a ray slightly below the horizon is
    -0.05, not 359.95)."""
    start = np.array([getattr(h, f"{which}_start") for h in headers], dtype=np.float64)
    stop = np.array([getattr(h, f"{which}_stop") for h in headers], dtype=np.float64)
    if which == "elevation":
        start = np.where(start > 180.0, start - 360.0, start)
        stop = np.where(stop > 180.0, stop - 360.0, stop)
        return (start + stop) / 2.0
    return azimuth_midpoints(start, stop)


def _sweep_group(
    url: str,
    buf: bytes,
    hdr: IngestHeader,
    sweep: SweepIndex,
    sweep_idx: int,
    rng: np.ndarray,
    nyquist: float,
    nyquist_vel: float,
    pad_missing_rays: bool,
    drop_variables: set[str],
) -> tuple[ManifestGroup, np.ndarray]:
    """One sweep's group, plus its per-ray epoch-ms times (NaN = padded)."""
    headers: tuple[DataTypeHeader, ...] = sweep.headers
    ndt = sweep.ndatatypes
    nbins = hdr.number_output_bins

    # One header walk per sweep gives every per-ray coordinate plus the
    # missing-ray census. Rows stay aligned across the sweep's moment arrays
    # ONLY if missingness is group-consistent, so a sweep where data types
    # disagree on which rays are missing is rejected.
    span = buf[sweep.byte_offset : sweep.byte_offset + sweep.byte_length]
    _, ray_headers, missing = walk_sweep(sweep_words(span, ndt), ndt, None)
    missing_sets = {o: set(groups) for o, groups in missing.items()}
    if missing_sets and len({frozenset(s) for s in missing_sets.values()}) != 1:
        raise ValueError(
            f"sweep {sweep.sweep_number}: missing rays are not group-"
            f"consistent across data types ({missing_sets}) — rows "
            "would misalign between moments"
        )
    if missing_sets and set(missing_sets) != set(range(ndt)):
        raise ValueError(
            f"sweep {sweep.sweep_number}: only ordinals "
            f"{sorted(missing_sets)} of {ndt} report missing rays — rows "
            "would misalign between moments"
        )
    if len(ray_headers) == 0:
        raise ValueError(f"sweep {sweep.sweep_number}: no written rays")

    missing_slots = sorted(missing_sets.get(0, set())) if missing_sets else []
    azimuth = _angle_midpoints(ray_headers, "azimuth")
    elevation = _angle_midpoints(ray_headers, "elevation")
    sweep_start = headers[0].sweep_start
    epoch_ms = sweep_start.astype("datetime64[ms]").astype(np.int64)
    time = np.array(
        [float(epoch_ms + h.dtime_s * 1000) for h in ray_headers],
        dtype=np.float64,
    )

    if pad_missing_rays and missing_slots:
        # Every ray SLOT keeps its position; missing slots become NaN
        # coords (the codec pads the data rows the same way, so row r is
        # the same physical slot in every moment).
        n_slots = len(ray_headers) + len(missing_slots)
        missing_set = set(missing_slots)
        written_slots = [g for g in range(n_slots) if g not in missing_set]

        def _pad(values: np.ndarray) -> np.ndarray:
            padded = np.full(n_slots, np.nan)
            padded[written_slots] = values
            return padded

        azimuth, elevation, time = _pad(azimuth), _pad(elevation), _pad(time)
    nrays = len(azimuth)

    # Pure pointers: rows and per-ray coordinates stay in acquisition order.
    # Sorting is a store-builder decision; it sets ``sort_rays: true`` in the
    # codec config and writes the matching sorted coordinates itself.
    # Moment names follow the eager reader: a second data type mapping to an
    # already used CfRadial name (DB_DBZ + DB_DBZ2) keeps its Sigmet name.
    names = _moment_names([dth.type_name for dth in headers])
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
        attrs: dict = {
            **moment_attrs(iris_mapping.get(dth.type_name, dth.type_name)),
            "coordinates": MOMENT_COORDINATES,
            "sigmet_data_type": dth.type_name,
        }
        scaling = _cf_scaling(dth.type_code, nyquist, nyquist_vel)
        if scaling is not None:
            attrs["scale_factor"], attrs["add_offset"], fill = scaling
            if fill is not None:
                # Sigmet raw ``mask`` (e.g. 0 for velocity) = no data; with
                # mask_and_scale these bins decode to NaN instead of a
                # plausible-looking physical value.
                attrs["_FillValue"] = int(fill)
        else:
            attrs["comment"] = (
                "raw Sigmet words with no CF scaling attached (nonlinear/"
                "categorical encoding per the eager backend's "
                "SIGMET_DATA_TYPES) — decode per the IRIS Programmer's "
                "Manual for this data type"
            )
        metadata = create_v3_array_metadata(
            shape=(nrays, nbins),
            chunk_shape=(nrays, nbins),
            data_type=dtype,
            fill_value=0,
            codecs=[
                {
                    "name": _codec.CODEC_NAME,
                    "configuration": {
                        "moment_index": ordinal,
                        "ndatatypes": ndt,
                        "sort_rays": False,  # frozen config key; see above
                        "pad_missing_rays": pad_missing_rays,
                    },
                }
            ],
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
    arrays["time"] = inline_variable(
        ("azimuth",), time, get_time_attrs(date_unit="milliseconds")
    )
    arrays["range"] = inline_variable(("range",), rng, get_range_attrs(rng))
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
    return ManifestGroup(arrays=arrays, attributes=group_attrs), time


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
        every ray slot keeps its position — missing slots read as fill with
        NaN per-ray coordinates, and the azimuth dim equals the sweep's
        EXPECTED ray count.
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
        hdr, sweeps, root_attrs = read_volume(buf)
        if _sweep_mode(hdr.antenna_scan_mode) == "rhi":
            raise NotImplementedError(
                f"{url}: RHI tasks (antenna_scan_mode=2) are not supported by "
                "the virtual IRIS parser; read them with xradar's eager IRIS "
                "backend (open_iris_datatree)"
            )
        rng = range_centers(hdr)

        # the eager reader's nyquist: the multi-PRF factor applies to DB_VEL
        nyquist = _nyquist(hdr.wavelength, hdr.prf)
        nyquist_vel = _nyquist(hdr.wavelength, hdr.prf, hdr.multi_prf_mode_flag)

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
                nyquist,
                nyquist_vel,
                self.pad_missing_rays,
                self.drop_variables,
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
