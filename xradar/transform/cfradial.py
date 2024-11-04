#!/usr/bin/env python
# Copyright (c) 2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.
"""
Transform CF-Radial
===================

This module provides the utilities to convert between CfRadial1 and
CfRadial2 formats, following WMO guidelines for radar data processing.

.. autosummary::
   :nosignatures:
   :toctree: generated/

    {}

to_cfradial1
------------
Convert a `xarray.DataTree` object to CfRadial1 format. The output is
an `xarray.Dataset` following the CfRadial1 standard, which can be used
for radar data visualization and further processing.

Parameters:
    - dtree: xarray.DataTree
        Radar xarray object to convert.
    - filename: str (optional)
        Output file name for the CfRadial1 dataset.
    - calibs: bool (default: True)
        Whether to include calibration parameters.

Returns:
    - dataset: xarray.Dataset
        Converted dataset in CfRadial1 format.

to_cfradial2
------------
Convert a `xarray.Dataset` object to CfRadial2 format by mapping the
structure back into a `xarray.DataTree`.

Parameters:
    - ds: xarray.Dataset
        CfRadial1 dataset to convert.
    - kwargs: dict
        Additional keyword arguments for controlling the conversion, such
        as specifying sweeps or metadata inclusion.

Returns:
    - dtree: xarray.DataTree
        Converted data tree in CfRadial2 format.

"""

__all__ = [
    "to_cfradial2",
    "to_cfradial1",
]

__doc__ = __doc__.format("\n   ".join(__all__))

from importlib.metadata import version

from xarray import DataTree

from ..io.backends.cfradial1 import (
    _get_radar_calibration,
    _get_required_root_dataset,
    _get_subgroup,
    _get_sweep_groups,
)
from ..io.backends.common import _attach_sweep_groups
from ..io.export.cfradial1 import (
    _calib_mapper,
    _variable_mapper,
)
from ..model import (
    georeferencing_correction_subgroup,
    radar_parameters_subgroup,
)


# to_cfradial1 function implementation
def to_cfradial1(dtree=None, calibs=True):
    """
    Convert a radar xarray.DataTree to the CFRadial1 format
    and save it to a file. Ensure that the resulting dataset
    is well-formed and does not include specified extraneous variables.

    Parameters
    ----------
    dtree: xarray.DataTree
        Radar xarray.DataTree object.
    calibs: Bool, optional
        Whether to include calibration parameters.
    """
    # Generate the initial ds_cf using the existing mapping functions
    dataset = _variable_mapper(dtree)

    # Handle calibration parameters
    if calibs:
        if "radar_calibration" in dtree:
            calib_params = dtree["radar_calibration"].to_dataset()
            calibs = _calib_mapper(calib_params)
            dataset.update(calibs)

    # Add additional parameters if they exist in dtree
    if "radar_parameters" in dtree:
        radar_params = dtree["radar_parameters"].to_dataset().reset_coords()
        dataset.update(radar_params)

    if "georeferencing_correction" in dtree:
        radar_georef = dtree["georeferencing_correction"].to_dataset().reset_coords()
        dataset.update(radar_georef)

    # Ensure that the data type of sweep_mode and similar variables matches
    if "sweep_mode" in dataset.variables:
        dataset["sweep_mode"] = dataset["sweep_mode"].astype("S")

    # Update global attributes
    dataset.attrs = dtree.attrs
    dataset.attrs["Conventions"] = "Cf/Radial"
    dataset.attrs["version"] = "1.2"
    xradar_version = version("xradar")
    dataset.attrs["history"] += f": xradar v{xradar_version} CfRadial1 export"

    return dataset


# to_cfradial2 function implementation
def to_cfradial2(ds, **kwargs):
    """Convert a CfRadial1 Dataset to CfRadial2 DataTree."""
    first_dim = kwargs.pop("first_dim", "auto")
    optional = kwargs.pop("optional", True)
    kwargs.pop("site_coords", True)
    sweep = kwargs.pop("sweep", None)

    # Create DataTree root node with required data
    root_data = _get_required_root_dataset(ds, optional=optional)
    dtree = DataTree(dataset=root_data, name="root")

    # Attach additional root metadata groups as child nodes
    radar_parameters = _get_subgroup(ds, radar_parameters_subgroup)
    if radar_parameters:
        dtree["radar_parameters"] = DataTree(radar_parameters, name="radar_parameters")

    calib = _get_radar_calibration(ds)
    if calib:
        dtree["radar_calibration"] = DataTree(calib, name="radar_calibration")

    georeferencing = _get_subgroup(ds, georeferencing_correction_subgroup)
    if georeferencing:
        dtree["georeferencing_correction"] = DataTree(
            georeferencing, name="georeferencing_correction"
        )

    # Attach sweep child nodes
    sweep_groups = list(
        _get_sweep_groups(
            ds,
            sweep=sweep,
            first_dim=first_dim,
            optional=optional,
            site_coords=True,
        ).values()
    )
    dtree = _attach_sweep_groups(dtree, sweep_groups)

    return dtree
