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

__all__ = [
    "MAX_CELLS",
    "MAX_SPAN_CELLS",
    "MIN_ZARR",
    "MIN_ZARR_BUILD",
    "check_output",
    "plain",
    "require_zarr",
    "zarr_supported",
]

#: The oldest zarr the virtual codecs run on: zarr v3 with its 3.1 dtype API
#: (3.1.6 is also virtualizarr's floor). zarr v2 has no ``zarr.abc`` codecs.
MIN_ZARR = "3.1.6"

#: The oldest zarr that builds virtual stores: zarr 3.1.6 to 3.2 warn, each
#: time a store's metadata is written, that the stores' fixed-width strings
#: (``fixed_length_utf32``) have no zarr v3 specification.
MIN_ZARR_BUILD = "3.3"


def zarr_supported(version: str, minimum: str = MIN_ZARR) -> bool:
    """Whether zarr ``version`` is at least ``minimum``; a pre-release sorts
    before its release, as in ``pytest.importorskip(minversion=...)``."""
    try:
        return Version(version) >= Version(minimum)
    except InvalidVersion:
        return False


def require_zarr(found: str, minimum: str, purpose: str) -> None:
    """Raise ImportError (naming zarr) unless zarr ``found`` is at least
    ``minimum``; ``purpose`` says what needs it."""
    if not zarr_supported(found, minimum):
        raise ImportError(
            f"{purpose} need zarr>={minimum}; found zarr {found}", name="zarr"
        )


#: Upper bound on one decoded chunk (rays x gates). Real sweeps stay below
#: ~3e6 cells (720 rays x 4000 gates); a store declaring more is corrupt or
#: hostile, and refusing it keeps one chunk from allocating gigabytes.
MAX_CELLS = 2**25

#: Upper bound on the words one walk of a sweep span may decode (every data
#: type's rays, headers included). A sweep with 20 data types of 720 x 4000
#: gates is ~5.8e7 words; this keeps one walk to 128 MiB.
MAX_SPAN_CELLS = 2**26


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
