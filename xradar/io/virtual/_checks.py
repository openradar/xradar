#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""The output contract every virtual codec enforces before decoding.

Numpy only, so the format walkers (read path) can share it without zarr.
"""

from __future__ import annotations

import math

import numpy as np

# packaging is always installed: a hard dependency of xarray
from packaging.version import InvalidVersion, Version

__all__ = ["MAX_CELLS", "MIN_ZARR", "check_output", "plain", "zarr_supported"]

#: The oldest zarr the virtual codecs run on: zarr v3 with its 3.1 dtype API
#: (3.1.6 is also virtualizarr's floor). zarr v2 has no ``zarr.abc`` codecs.
MIN_ZARR = "3.1.6"


def zarr_supported(version: str) -> bool:
    """Whether a zarr version can run the virtual codecs (same rules as
    ``pytest.importorskip(minversion=...)``: pre-releases of 3.1.6 are
    older than 3.1.6)."""
    try:
        return Version(version) >= Version(MIN_ZARR)
    except InvalidVersion:
        return False


#: Upper bound on one decoded chunk (rays x gates). Real sweeps stay below
#: ~3e6 cells (720 rays x 4000 gates); a store declaring more is corrupt or
#: hostile, and refusing it keeps one chunk from allocating gigabytes.
MAX_CELLS = 2**25


def plain(value):
    """A numpy scalar as the Python value it stands for (JSON-safe);
    anything else unchanged."""
    return value.item() if isinstance(value, np.generic) else value


def _as_int(fill_value) -> int:
    """An integral fill value as ``int``; anything else raises ValueError."""
    if isinstance(fill_value, bool | np.bool_):
        raise ValueError(f"fill_value {fill_value!r} is not an integer")
    if isinstance(fill_value, int | np.integer):
        return int(fill_value)
    if (
        isinstance(fill_value, float | np.floating)
        and math.isfinite(fill_value)
        and float(fill_value).is_integer()
    ):
        # stores written with a JSON float fill (0.0) keep reading
        return int(fill_value)
    raise ValueError(f"fill_value {fill_value!r} is not an integer")


def check_output(dtype, fill_value, out_shape) -> tuple[np.dtype, int]:
    """Validate the array a moment is decoded into; return (dtype, fill).

    Moments are raw ``uint8``/``uint16`` words, the fill an integer inside
    the dtype's range, and the shape a non-empty ``(rays, gates)``.
    Anything else in an array's metadata is out of contract and raises
    ``ValueError`` (what the radish fast path refuses too), rather than
    decoding into wrapped or truncated values.
    """
    dtype = np.dtype(dtype)
    if dtype.kind != "u" or dtype.itemsize not in (1, 2) or not dtype.isnative:
        raise ValueError(
            f"radar moments decode into native uint8/uint16 arrays, not {dtype}"
        )
    fill = _as_int(fill_value)
    info = np.iinfo(dtype)
    if not info.min <= fill <= info.max:
        raise ValueError(f"fill_value {fill_value!r} is not a valid {dtype} value")
    if (
        len(out_shape) != 2
        or not all(isinstance(n, int | np.integer) for n in out_shape)
        or out_shape[0] < 1
        or out_shape[1] < 1
    ):
        raise ValueError(f"expected a non-empty (rays, gates) shape, got {out_shape}")
    if int(out_shape[0]) * int(out_shape[1]) > MAX_CELLS:
        raise ValueError(
            f"chunk shape {tuple(out_shape)} exceeds {MAX_CELLS} cells — "
            "corrupt or hostile array metadata"
        )
    return dtype, fill
