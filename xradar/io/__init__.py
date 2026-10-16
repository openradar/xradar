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

"""

from .backends import *  # noqa
from .export import *  # noqa

from .backends.cfradial1 import CfRadial1BackendEntrypoint
from .backends.cfradial2 import CfRadial2BackendEntrypoint
from .backends.datamet import DataMetBackendEntrypoint
from .backends.discover import discover_engine
from .backends.furuno import FurunoBackendEntrypoint
from .backends.gamic import GamicBackendEntrypoint
from .backends.hpl import HPLBackendEntrypoint
from .backends.imd import IMDBackendEntrypoint
from .backends.iris import IrisBackendEntrypoint
from .backends.metek import MRRBackendEntrypoint
from .backends.nexrad_level2 import NexradLevel2BackendEntrypoint
from .backends.odim import OdimBackendEntrypoint
from .backends.rainbow import RainbowBackendEntrypoint
from .backends.uf import UFBackendEntrypoint
from xarray import open_datatree as _xr_open_datatree

from .backends.common import _call_all, _close_all

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


_VOLUME_KWARGS = (
    "time_coverage_start",
    "time_coverage_end",
    "min_angle",
    "max_angle",
    "volume_number",
)


def _is_file_list(filename_or_obj):
    import os

    return isinstance(filename_or_obj, (list, tuple)) and all(
        isinstance(item, (str, os.PathLike)) for item in filename_or_obj
    )


def _open_files_as_volume(files, engine, **kwargs):
    """Open files with one sweep each and stack them into one volume."""
    from functools import partial

    import xarray as xr

    from .. import util

    if not files:
        raise ValueError("open_datatree requires at least one file.")
    if engine is None:
        engines = {discover_engine(file) for file in files}
        if len(engines) > 1:
            raise ValueError(
                f"The files need different engines ({', '.join(sorted(engines))}), "
                "a volume is stacked from files of one format."
            )
        (engine,) = engines
    if engine not in _ENGINE_REGISTRY:
        supported = ", ".join(sorted(_ENGINE_REGISTRY))
        raise ValueError(f"Unknown engine {engine!r}. Supported engines: {supported}")
    if engine == "imd":
        from .backends.imd import open_imd_datatree

        return open_imd_datatree(files, **kwargs)
    volume_kwargs = {k: kwargs.pop(k) for k in _VOLUME_KWARGS if k in kwargs}
    if len(files) == 1:
        if volume_kwargs:
            raise TypeError(
                f"{', '.join(_VOLUME_KWARGS)} are only valid when opening "
                "multiple files."
            )
        return _xr_open_datatree(files[0], engine=_ENGINE_REGISTRY[engine], **kwargs)
    trees = []
    try:
        for file in files:
            trees.append(
                _xr_open_datatree(file, engine=_ENGINE_REGISTRY[engine], **kwargs)
            )
        volume = util.create_volume(trees, **volume_kwargs)
    except Exception:
        _close_all(trees)
        raise
    # create_volume copies sweeps verbatim: number them by their position
    for i, key in enumerate(util.get_sweep_keys(volume)):
        sweep = volume[key].to_dataset()
        if "sweep_number" in sweep:
            sweep["sweep_number"] = xr.DataArray(i)
            volume[key] = xr.DataTree(sweep)
    # closing the volume closes every source file; the source trees and their
    # data are not kept
    closers = [
        node._close
        for tree in trees
        for node in tree.subtree
        if node._close is not None
    ]
    volume.set_close(partial(_call_all, closers))
    return volume


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
    if _is_file_list(filename_or_obj):
        return _open_files_as_volume(list(filename_or_obj), engine, **kwargs)
    if engine is None:
        engine = discover_engine(filename_or_obj)
    if engine not in _ENGINE_REGISTRY:
        supported = ", ".join(sorted(_ENGINE_REGISTRY))
        raise ValueError(f"Unknown engine {engine!r}. Supported engines: {supported}")
    return _xr_open_datatree(filename_or_obj, engine=_ENGINE_REGISTRY[engine], **kwargs)


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
