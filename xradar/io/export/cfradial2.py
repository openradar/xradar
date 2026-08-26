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

from ...model import conform_cfradial2_sweep_group, validate_global_attrs
from ...util import has_import


def to_cfradial2(
    dtree,
    filename,
    engine=None,
    timestep=None,
    global_attrs: dict[str, object] | None = None,
):
    """Save DataTree to CfRadial2 compliant file.

    Parameters
    ----------
    dtree : xarray.DataTree
        DataTree with CfRadial2 groups.
    filename : str
        output filename

    Keyword Arguments
    ----------------
    timestep : int
        timestep of wanted volume, currently not used
    engine : str
        Either `netcdf4` or `h5netcdf`.
    global_attrs : dict
        Dictionary of root-group global attributes to set manually, e.g. those
        which cannot be deduced from the DataTree or should be overwritten.
        These are applied last and take precedence over any attributes
        derived from the DataTree.
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

    if global_attrs is not None:
        validate_global_attrs(global_attrs)

    # all sweep rewrites and root-attribute edits need to happen on a copy instead of the caller’s DataTree object.
    # otherwise, the caller’s DataTree will be modified in-place, which is not expected behavior for a writer function.
    # Using a deep copy makes the code future-proof against any in-place modifications that might be added later on.
    dtree = dtree.copy(deep=True)

    # iterate over DataTree and make subgroups cfradial2 compliant
    for grp in dtree.groups:
        if "sweep" in grp:
            dtree[grp] = xr.DataTree(
                conform_cfradial2_sweep_group(
                    dtree[grp].to_dataset(inherit="all_coords"),
                    optional=False,
                )
            )

    root = dtree["/"].ds
    # fix Conventions
    root.attrs["Conventions"] = "Cf/Radial"
    root.attrs["version"] = "2.0"
    # add xradar version to history
    xradar_version = version("xradar")
    root.attrs["history"] += f": xradar v{xradar_version} CfRadial2 export"

    # apply user-supplied global attributes last, so they take precedence
    if global_attrs is not None:
        root.attrs.update(global_attrs)

    # write DataTree
    dtree.to_netcdf(filename, engine=engine)
