#!/usr/bin/env python
# Copyright (c) 2022-2025, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""

CfRadial2
=========

This submodule provides an xarray backend for reading CfRadial2-compliant radar data
into Xarray `DataTree` structures. It serves as a wrapper around `xarray.open_datatree`.

Example::

    import xradar as xd
    dtree = xd.io.open_cfradial2_datatree(filename)

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}

"""

__all__ = [
    "open_cfradial2_datatree",
]

__doc__ = __doc__.format("\n   ".join(__all__))

from xarray import open_datatree


def open_cfradial2_datatree(filename, **kwargs):
    """
    Open a CfRadial2 file as an xarray DataTree.

    Parameters
    ----------
    filename : str
        Path to the CfRadial2 file.
    **kwargs : dict
        Additional keyword arguments passed to `xarray.open_datatree`.

    Returns
    -------
    xarray.DataTree
        The opened DataTree containing the radar data.
    """
    return open_datatree(filename, **kwargs)
