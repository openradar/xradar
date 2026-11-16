#!/usr/bin/env python
# Copyright (c) 2022, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radar Data IO
=============

.. toctree::
    :maxdepth: 4

.. automodule:: xradar.io.backends
.. automodule:: xradar.io.export
.. automodule:: xradar.io.virtual

"""

from functools import partial as _partial

from xarray import open_datatree as _xr_open_datatree

from .backends import *  # noqa
from .backends.cfradial1 import CfRadial1BackendEntrypoint
from .backends.cfradial2 import CfRadial2BackendEntrypoint
from .backends.common import _is_file_list, _open_sweep_files_as_volume
from .backends.datamet import DataMetBackendEntrypoint
from .backends.discover import _discover_common_engine, discover_engine
from .backends.furuno import FurunoBackendEntrypoint
from .backends.gamic import GamicBackendEntrypoint
from .backends.hpl import HPLBackendEntrypoint
from .backends.imd import IMDBackendEntrypoint, open_imd_datatree
from .backends.iris import IrisBackendEntrypoint
from .backends.metek import MRRBackendEntrypoint
from .backends.nexrad_level2 import NexradLevel2BackendEntrypoint
from .backends.odim import OdimBackendEntrypoint
from .backends.rainbow import RainbowBackendEntrypoint
from .backends.uf import UFBackendEntrypoint
from .export import *  # noqa

#: Registry mapping engine names to backend classes that support groups.
_ENGINE_REGISTRY = {
    "odim": OdimBackendEntrypoint,
    "cfradial1": CfRadial1BackendEntrypoint,
    "cfradial2": CfRadial2BackendEntrypoint,
    "nexradlevel2": NexradLevel2BackendEntrypoint,
    "gamic": GamicBackendEntrypoint,
    "iris": IrisBackendEntrypoint,
    "furuno": FurunoBackendEntrypoint,
    "rainbow": RainbowBackendEntrypoint,
    "datamet": DataMetBackendEntrypoint,
    "hpl": HPLBackendEntrypoint,
    "metek": MRRBackendEntrypoint,
    "uf": UFBackendEntrypoint,
    "imd": IMDBackendEntrypoint,
}


def open_datatree(filename_or_obj, *, engine=None, **kwargs):
    """Open a radar file as :py:class:`xarray.DataTree`.

    The engine is the given ``engine`` or, without it, the one found by
    :func:`xradar.io.discover_engine`. This works for all radar formats,
    including the netCDF/HDF5 based ones (ODIM_H5, GAMIC, CfRadial) which
    :py:func:`xarray.open_datatree` without ``engine`` hands to xarray's own
    netCDF/HDF5 backends.

    A list of files is stacked into one volume (see
    :func:`xradar.util.create_volume`): this is for formats which write one
    sweep per file, e.g. IMD or IRIS/Sigmet files of single sweeps. All files
    need the same engine, the sweeps of all files are sorted by time and
    numbered in this order. The keyword arguments ``time_coverage_start``,
    ``time_coverage_end``, ``min_angle``, ``max_angle`` and ``volume_number``
    of :func:`xradar.util.create_volume` select and label the sweeps, all other
    arguments are used for each file.

    Parameters
    ----------
    filename_or_obj : str, Path, file-like or list of str or Path
        Path to the radar file, or the paths of files with one sweep each.
    engine : str, optional
        Backend engine name (e.g., ``"odim"``, ``"cfradial1"``, ``"nexradlevel2"``).
        Default is to discover the engine from the file with
        :func:`xradar.io.discover_engine` (raises ``ValueError`` if no xradar
        engine can open it).
    **kwargs
        Keyword arguments for :py:func:`xarray.open_datatree`: xarray options
        such as ``chunks`` or ``cache``, and the backend options documented on
        the engine's ``open_groups_as_dict`` (``sweep``, ``first_dim``, ...).

    Returns
    -------
    dtree : xarray.DataTree
        DataTree with CfRadial2 group structure.

    Examples
    --------
    >>> import xradar as xd
    >>> dtree = xd.open_datatree("file.h5", engine="odim")
    >>> dtree = xd.open_datatree("file.h5")  # engine discovered
    >>> volume = xd.open_datatree(["sweep_1.nc", "sweep_2.nc"])  # stacked sweeps
    """
    files = list(filename_or_obj) if _is_file_list(filename_or_obj) else None
    if files is not None and not files:
        raise ValueError("open_datatree requires at least one file.")
    if engine is None:
        engine = (
            discover_engine(filename_or_obj)
            if files is None
            else _discover_common_engine(files)
        )
    if engine not in _ENGINE_REGISTRY:
        supported = ", ".join(sorted(_ENGINE_REGISTRY))
        raise ValueError(f"Unknown engine {engine!r}. Supported engines: {supported}")
    open_file = _partial(_xr_open_datatree, engine=_ENGINE_REGISTRY[engine])
    if files is None:
        return open_file(filename_or_obj, **kwargs)
    if engine == "imd":
        return open_imd_datatree(files, **kwargs)
    return _open_sweep_files_as_volume(files, open_file, **kwargs)


def list_engines():
    """Return a sorted list of engine names registered with :func:`open_datatree`.

    Examples
    --------
    >>> import xradar as xd
    >>> "odim" in xd.io.list_engines()
    True
    """
    return sorted(_ENGINE_REGISTRY)


__all__ = [s for s in dir() if not s.startswith("_")]


def __getattr__(name):
    # `virtual` stays lazy: importing it must not pull zarr/virtualizarr
    # into ordinary eager-reader sessions.
    if name == "virtual":
        import importlib

        return importlib.import_module("xradar.io.virtual")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
