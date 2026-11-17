#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Manifest building blocks
^^^^^^^^^^^^^^^^^^^^^^^^

Format-agnostic helpers for building VirtualiZarr manifests from radar
metadata: inline small numpy/xarray values as zarr chunks inside a
``ChunkManifest`` (coordinates, per-sweep scalars, the root group), read
whole source files through an ``obspec`` store registry, and the CF/FM301
conventions the eager backends use. Shared by the IRIS and NEXRAD Level II
parsers, and public for store builders that assemble their own groups.

The CF attributes come from :mod:`xradar.model` and the root group from
:func:`xradar.io.backends.common._assign_root`, the same code the eager
readers run, so the virtual view cannot drift from them.

Requires the ``xradar[virtual]`` extra (``virtualizarr``, which brings
``obstore``/``obspec-utils``); nothing in the read path imports this module.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
import zarr
from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.manifests.utils import create_v3_array_metadata

from xradar.io.backends.common import _assign_root
from xradar.io.virtual._checks import MIN_ZARR_BUILD, plain, require_zarr
from xradar.model import (
    MOMENT_COORDINATES,
    get_altitude_attrs,
    get_latitude_attrs,
    get_longitude_attrs,
)

if TYPE_CHECKING:
    from obspec_utils.registry import ObjectStoreRegistry

require_zarr(
    getattr(zarr, "__version__", "unknown"), MIN_ZARR_BUILD, "xradar's store builders"
)

__all__ = [
    "MOMENT_COORDINATES",
    "FM301_STRING_DEFAULTS",
    "to_json_safe",
    "native_endian_bytes",
    "variable_to_inline_manifest_array",
    "inline_variable",
    "inline_scalar",
    "inline_root",
    "fetch_bytes",
]

__doc__ = __doc__.format("\n   ".join(__all__))


#: FM301 per-sweep scalar string metadata defaults.
FM301_STRING_DEFAULTS = {
    "sweep_mode": "azimuth_surveillance",
    "prt_mode": "not_set",
    "follow_mode": "not_set",
}


def to_json_safe(value):
    """Coerce numpy scalars/arrays (also inside dicts/lists) to native
    Python types for JSON."""
    if isinstance(value, np.generic):
        return plain(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    return value


#: width of the stored strings, wider when a value is longer; it holds every
#: FM301 ``sweep_mode`` value
STRING_WIDTH = 32


def native_endian_bytes(arr: np.ndarray) -> tuple[np.ndarray, str | None]:
    """Contiguous little-endian view of ``arr`` plus the endian tag for the
    zarr ``bytes`` codec (``None`` for 1-byte itemsizes)."""
    arr = np.ascontiguousarray(arr)
    if arr.dtype.itemsize == 1:
        return arr, None
    if arr.dtype.byteorder not in ("=", "<"):
        arr = arr.astype(arr.dtype.newbyteorder("<"), copy=False)
    return arr, "little"


def variable_to_inline_manifest_array(var: xr.Variable) -> ManifestArray:
    """Materialize an ``xr.Variable`` as a single inlined-chunk ManifestArray."""
    values = np.asarray(var.values)
    shape = values.shape
    # zarr's chunk key for a scalar (0-d) array is the literal "c"
    chunk_key = "c" if values.ndim == 0 else ".".join(["0"] * values.ndim)
    attrs = {k: to_json_safe(v) for k, v in var.attrs.items()}

    # Strings are stored fixed-width (zarr's ``fixed_length_utf32``) and read
    # back as ``<U`` like the eager readers' strings: numpy's variable-length
    # StringDType breaks Dataset.where/fillna and the CfRadial2 export.
    # StringDType and object input are converted first, never dumped with
    # ``tobytes()`` (which writes pointers).
    if values.dtype.kind in ("T", "O"):
        items = values.ravel().tolist()
        if not all(isinstance(item, str) for item in items):
            raise TypeError(f"inline object arrays must hold str, got {values!r}")
        values = np.asarray(items, dtype=str).reshape(values.shape)
    if values.dtype.kind == "U":
        # one width for short strings, so volumes whose sweep_mode differs
        # ("sector", "azimuth_surveillance") still concatenate
        width = max(STRING_WIDTH, values.dtype.itemsize // 4)
        values = values.astype(f"<U{width}")
    raw, endian = native_endian_bytes(values)
    data = raw.tobytes()
    data_type = raw.dtype
    if endian is None:
        codecs = [{"name": "bytes"}]
    else:
        codecs = [{"name": "bytes", "configuration": {"endian": endian}}]
    if raw.dtype.kind == "M":
        unit = np.datetime_data(raw.dtype)[0]
        attrs.setdefault("units", f"{unit} since 1970-01-01T00:00:00")
        attrs.setdefault("calendar", "proleptic_gregorian")

    manifest = ChunkManifest(
        entries={
            chunk_key: {
                "path": "",
                "offset": 0,
                "length": len(data),
                "data": data,
            }
        }
    )
    metadata = create_v3_array_metadata(
        shape=shape,
        chunk_shape=shape,
        data_type=data_type,
        codecs=codecs,
        attributes=attrs,
        dimension_names=tuple(str(d) for d in var.dims),
    )
    return ManifestArray(metadata=metadata, chunkmanifest=manifest)


def inline_variable(dims: tuple[str, ...], values, attrs: dict) -> ManifestArray:
    """Inline an n-d variable built from ``dims``/``values``/``attrs``."""
    return variable_to_inline_manifest_array(xr.Variable(dims, values, attrs))


def inline_scalar(value, attrs: dict | None = None) -> ManifestArray:
    """Inline a 0-d variable (numeric or string)."""
    return variable_to_inline_manifest_array(
        xr.Variable((), np.asarray(value), attrs or {})
    )


def inline_root(
    ray_times_ms: list[np.ndarray],
    latitude,
    longitude,
    altitude,
    attrs: dict,
) -> tuple[dict[str, ManifestArray], dict]:
    """Root arrays and attrs through the eager readers' ``_assign_root``.

    ``ray_times_ms`` holds one int64 array of the written rays' epoch
    milliseconds per stored sweep, so the coverage strings are the min/max
    over the real rays. The site
    values keep their dtype and get the :mod:`xradar.model` attrs.
    ``attrs`` (instrument name, scan name, format metadata) is laid over
    the fixed root attrs ``_assign_root`` writes; ``coordinates`` promotes
    the site variables, as the eager root does.

    ``_assign_root`` reads ``Conventions``/``instrument_name``/``comment``
    from its first dataset and merges every attr of the second into the
    root, so both are built attr-free here and the caller's attrs are laid
    on top afterwards.
    """
    site = {
        "latitude": xr.Variable((), latitude, get_latitude_attrs()),
        "longitude": xr.Variable((), longitude, get_longitude_attrs()),
        "altitude": xr.Variable((), altitude, get_altitude_attrs()),
    }
    sweeps = [xr.Dataset()]
    for times in ray_times_ms:
        if len(times) == 0:
            continue
        ms = np.asarray(times, dtype=np.int64).astype("datetime64[ms]")
        sweeps.append(xr.Dataset({"time": ("azimuth", ms), **site}))
    if len(sweeps) == 1:
        raise ValueError("no ray times: cannot build the root group")
    root, _ = _assign_root(sweeps)
    arrays = {
        name: variable_to_inline_manifest_array(root[name].variable)
        for name in root.variables
    }
    root_attrs = {
        **root.attrs,
        **attrs,
        "coordinates": " ".join(str(name) for name in root.coords),
    }
    return arrays, root_attrs


def fetch_bytes(url: str, registry: ObjectStoreRegistry) -> bytes:
    """Read the whole file through the registry.

    ``registry.resolve`` returns the store together with the path inside
    it; raises ``ValueError`` when no store matches the url.
    """
    store, path_in_store = registry.resolve(url)
    return bytes(store.get(path_in_store).bytes())
