#!/usr/bin/env python
# Copyright (c) 2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

import xarray as xr
from open_radar_data import DATASETS

import xradar as xd


def test_to_cfradial1():
    """Test the conversion from DataTree to CfRadial1 format."""
    file = DATASETS.fetch("cfrad.20080604_002217_000_SPOL_v36_SUR.nc")
    dtree = xd.io.open_cfradial1_datatree(file)

    # Call the conversion function
    ds_cf1 = xd.transform.to_cfradial1(dtree)

    # Verify key attributes and data structures in the resulting dataset
    assert isinstance(ds_cf1, xr.Dataset), "Output is not a valid xarray Dataset"
    assert "Conventions" in ds_cf1.attrs and ds_cf1.attrs["Conventions"] == "Cf/Radial"
    assert "sweep_mode" in ds_cf1.variables, "Missing sweep_mode in converted dataset"
    assert ds_cf1.attrs["version"] == "1.2", "Incorrect CfRadial version"


def test_to_cfradial2():
    """Test the conversion from CfRadial1 to CfRadial2 DataTree format."""
    file = DATASETS.fetch("cfrad.20080604_002217_000_SPOL_v36_SUR.nc")
    dtree = xd.io.open_cfradial1_datatree(file)

    # Convert to CfRadial1 dataset first
    ds_cf1 = xd.transform.to_cfradial1(dtree)

    # Call the conversion back to CfRadial2
    dtree_cf2 = xd.transform.to_cfradial2(ds_cf1)

    # Verify key attributes and data structures in the resulting datatree
    assert isinstance(dtree_cf2, xr.DataTree), "Output is not a valid DataTree"
    assert "radar_parameters" in dtree_cf2, "Missing radar_parameters in DataTree"
    assert dtree_cf2.attrs == ds_cf1.attrs, "Attributes mismatch between formats"
