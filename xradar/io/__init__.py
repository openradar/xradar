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
from .backends.nexrad_level2 import NexradLevel2BackendEntrypoint
from .backends.odim import OdimBackendEntrypoint

#: Registry mapping engine names to backend classes that support groups.
_ENGINE_REGISTRY = {
    "odim": OdimBackendEntrypoint,
    "cfradial1": CfRadial1BackendEntrypoint,
    "nexradlevel2": NexradLevel2BackendEntrypoint,
}


def open_datatree(filename_or_obj, *, engine, **kwargs):
    """Open a radar file as :py:class:`xarray.DataTree` using the specified engine.

    Parameters
    ----------
    filename_or_obj : str, Path, or file-like
        Path to the radar file.
    engine : str
        Backend engine name (e.g., ``"odim"``, ``"cfradial1"``, ``"nexradlevel2"``).
    **kwargs
        Additional keyword arguments passed to the backend's ``open_datatree`` method.

    Returns
    -------
    dtree : xarray.DataTree
        DataTree with CfRadial2 group structure.

    Examples
    --------
    >>> import xradar as xd
    >>> dtree = xd.open_datatree("file.h5", engine="odim")
    """
    if engine not in _ENGINE_REGISTRY:
        supported = ", ".join(sorted(_ENGINE_REGISTRY))
        raise ValueError(f"Unknown engine {engine!r}. Supported engines: {supported}")
    backend = _ENGINE_REGISTRY[engine]()
    return backend.open_datatree(filename_or_obj, **kwargs)


__all__ = [s for s in dir() if not s.startswith("_")]
