#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Virtual (byte-range) access
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Zarr v3 codecs and VirtualiZarr parsers that expose raw radar files
(Sigmet/IRIS RAW) as zarr stores of byte-range references: no data is
copied, the raw bytes are fetched and decoded on read.

This subpackage is intentionally lazy: importing it (or resolving one of the
``zarr.codecs`` entry points it provides) must not require ``virtualizarr``.
Reading a virtual store needs only ``xradar`` + ``zarr``; building one needs
the ``xradar[virtual]`` extra.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

import importlib
from importlib.metadata import PackageNotFoundError, version

from xradar.io.virtual._checks import MIN_ZARR, MIN_ZARR_BUILD, zarr_supported
from xradar.io.virtual._sort import azimuth_sort_order as azimuth_sort_order
from xradar.util import has_import

#: name -> (module, extra that provides its third-party dependencies, oldest
#: zarr it runs on); the parsers build stores, which needs a newer zarr
_LAZY = {
    "IrisSweepCodec": ("xradar.io.virtual.iris.codec", f"zarr>={MIN_ZARR}", MIN_ZARR),
    "IrisDecodeCodec": (
        "xradar.io.virtual.iris.decode",
        f"zarr>={MIN_ZARR}",
        MIN_ZARR,
    ),
    "IrisParser": ("xradar.io.virtual.iris.parser", "xradar[virtual]", MIN_ZARR_BUILD),
}


def _zarr_version() -> str | None:
    """The installed zarr's version, from the package metadata (without
    importing zarr); ``None`` without zarr."""
    try:
        return version("zarr")
    except PackageNotFoundError:
        return None


def _installed(extra: str, min_zarr: str, zarr_version: str | None) -> bool:
    """Whether the dependencies behind a lazy export are installed: zarr at
    least ``min_zarr``, plus virtualizarr for the ``xradar[virtual]``
    exports."""
    if zarr_version is None or not zarr_supported(zarr_version, min_zarr):
        return False
    return extra != "xradar[virtual]" or bool(has_import("virtualizarr"))


# Exports only appear in __all__ (star-imports, docs) where their dependency
# is installed, so a star import never raises MissingDependencyError.
__all__ = sorted(
    [
        "azimuth_sort_order",
        *(
            name
            for name, (_, extra, min_zarr) in _LAZY.items()
            if _installed(extra, min_zarr, _zarr_version())
        ),
    ]
)

__doc__ = __doc__.format("\n   ".join(sorted([*_LAZY, "azimuth_sort_order"])))


class MissingDependencyError(ImportError):
    """A lazy export whose optional dependency is not installed.

    Raised on attribute access or ``from xradar.io.virtual import Name`` and
    says what to install. Star imports and ``dir()`` only list names whose
    dependencies are importable, so they never trigger it.
    """


def lazy_attribute(module_name: str, name: str, lazy: dict):
    """Resolve a lazily exported ``name`` (shared by the subpackages)."""
    if name not in lazy:
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
    target, extra, _ = lazy[name]
    try:
        module = importlib.import_module(target)
    except ImportError as err:
        raise MissingDependencyError(
            f"{module_name}.{name} requires {extra!r} ({err}); reading "
            "existing virtual stores needs only xradar + zarr, building them "
            "needs the xradar[virtual] extra."
        ) from err
    return getattr(module, name)


def __getattr__(name):
    return lazy_attribute(__name__, name, _LAZY)


def __dir__():
    return list(__all__)
