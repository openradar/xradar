#!/usr/bin/env python
# Copyright (c) 2022, openradar developers.
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

from datatree import DataTree

from ...model import conform_cfradial2_sweep_group
from ...util import has_import


def to_cfradial2(dtree, filename, engine=None, timestep=None):
    """Save DataTree to CfRadial2 compliant file.

    Parameters
    ----------
    dtree : DataTree
        DataTree with CfRadial2 groups.
    filename : str
        output filename

    Keyword Arguments
    ----------------
    timestep : int
        timestep of wanted volume, currently not used
    engine : str
        Either `netcdf4` or `h5netcdf`.
    """
    if engine is None:
        if has_import("netCDF4"):
            engine == "netcdf4"
        elif has_import("h5netcdf"):
            engine == "h5netcdf"
        else:
            raise ImportError(
                "xradar: ``netCDF4`` or ``h5netcdf`` needed to perform this operation."
            )

    # iterate over DataTree and make subgroups cfradial2 compliant
    for grp in dtree.groups:
        if "sweep" in grp:
            dtree[grp] = DataTree(
                conform_cfradial2_sweep_group(
                    dtree[grp].to_dataset(), optional=False, dim0="azimuth"
                )
            )

    root = dtree["/"].to_dataset()
    # fix Conventions
    root.attrs["Conventions"] = "Cf/Radial"
    root.attrs["version"] = "2.0"
    # add xradar version to history
    xradar_version = version("xradar")
    root.attrs["history"] += f": xradar v{xradar_version} CfRadial2 export"

    # write DataTree
    dtree.to_netcdf(filename, engine=engine)
