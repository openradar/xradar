#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Engine discovery
^^^^^^^^^^^^^^^^

Find the xradar backend engine which can open a given file, based on the
backends' ``guess_can_open``.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = ["discover_engine", "list_xradar_engines"]

__doc__ = __doc__.format("\n   ".join(__all__))

import os
from importlib.metadata import entry_points

from xarray.backends import plugins


def list_xradar_engines():
    """Names of the xarray backend engines provided by xradar.

    Returns
    -------
    engines : list of str
    """
    return sorted(
        ep.name
        for ep in entry_points(group="xarray.backends")
        if ep.value.startswith("xradar.")
    )


def discover_engine(filename_or_obj):
    """Find the xradar engine which can open the given file.

    Each xradar backend checks the file with its ``guess_can_open`` (format
    signatures, e.g. ``AR2V`` for NEXRAD Level II, ``ODIM_H5`` conventions in
    HDF5 files), see :func:`list_xradar_engines`.

    Parameters
    ----------
    filename_or_obj : str, os.PathLike, bytes or file-like
        Radar file.

    Returns
    -------
    engine : str
        Name of the engine, to be used with e.g.
        ``xr.open_dataset(filename, engine=engine, group="sweep_0")``.

    Raises
    ------
    ValueError
        If no xradar engine can open the file.

    Examples
    --------
    >>> engine = xd.io.discover_engine(filename)  # doctest: +SKIP
    >>> ds = xr.open_dataset(filename, engine=engine, group="sweep_0")  # doctest: +SKIP
    """
    if isinstance(filename_or_obj, os.PathLike):
        filename_or_obj = os.fspath(filename_or_obj)
    engines = list_xradar_engines()
    for engine in engines:
        # guess_can_open returns False for inputs it can't read (missing,
        # truncated or corrupt files, see ``_UNREADABLE_ERRORS``); anything
        # else it raises is a bug and is not hidden here
        if plugins.get_backend(engine).guess_can_open(filename_or_obj):
            return engine
    raise ValueError(
        f"None of the xradar engines ({', '.join(engines)}) can open "
        f"{filename_or_obj!r}."
    )


def _discover_common_engine(files):
    """The engine which can open all ``files`` (a volume is stacked from files of
    one format)."""
    engines = {discover_engine(file) for file in files}
    if len(engines) > 1:
        raise ValueError(
            f"The files need different engines ({', '.join(sorted(engines))}), "
            "a volume is stacked from files of one format."
        )
    (engine,) = engines
    return engine
