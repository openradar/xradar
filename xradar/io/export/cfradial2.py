#!/usr/bin/env python
# Copyright (c) 2022-2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""

CfRadial2 output
================

This sub-module contains the writer for export of CfRadial2-based radar
data.

Code ported from wradlib.

Example::

    import xradar as xd
    dtree = xd.io.to_cfradial2(dtree, filename)

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}

"""

__all__ = [
    "to_cfradial2",
]

__doc__ = __doc__.format("\n   ".join(__all__))

from importlib.metadata import version

import xarray as xr

from ...model import conform_cfradial2_sweep_group
from ...util import has_import


def to_cfradial2(dtree, filename, engine=None, timestep=None, deepcopy=False):
    """Save DataTree to CfRadial2 compliant file.

    The input DataTree is never modified. All sweep rewrites and root
    attribute edits are applied to a copy, which is shallow by default.

    Parameters
    ----------
    dtree : xarray.DataTree
        DataTree with CfRadial2 groups.
    filename : str
        output filename

    Keyword Arguments
    -----------------
    timestep : int
        timestep of wanted volume, currently not used
    engine : str
        Either `netcdf4` or `h5netcdf`.
    deepcopy : bool
        If True, work on a deep copy of ``dtree`` (the data is duplicated in
        memory). If False (default), work on a shallow copy, which shares the
        underlying data arrays with ``dtree`` but not its structure and
        attributes.
    """
    if engine is None:
        if has_import("netCDF4"):
            engine = "netcdf4"
        elif has_import("h5netcdf"):
            engine = "h5netcdf"
        else:
            raise ImportError(
                "xradar: ``netCDF4`` or ``h5netcdf`` needed to perform this operation."
            )

    # Work on a copy so the caller's DataTree is not modified in-place.
    # A shallow copy is sufficient, since only the tree structure and the
    # attributes are changed while the data arrays are only read. Use
    # ``deepcopy=True`` to also decouple the data from the input.
    dtree = dtree.copy(deep=deepcopy)

    # iterate over DataTree and make subgroups cfradial2 compliant
    for grp in dtree.groups:
        if "sweep" in grp:
            dtree[grp] = xr.DataTree(
                conform_cfradial2_sweep_group(
                    dtree[grp].to_dataset(inherit="all_coords"),
                    optional=False,
                )
            )

    root = dtree["/"]
    # fix Conventions
    root.attrs["Conventions"] = "Cf/Radial"
    root.attrs["version"] = "2.0"
    # add xradar version to history
    xradar_version = version("xradar")
    root.attrs["history"] += f": xradar v{xradar_version} CfRadial2 export"

    # write DataTree
    dtree.to_netcdf(filename, engine=engine)
