#!/usr/bin/env python
# Copyright (c) 2023-2025, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

import pytest
import xarray as xr
from open_radar_data import DATASETS

import xradar as xd


@pytest.fixture
def cfradial1_file():
    return DATASETS.fetch("cfrad.20080604_002217_000_SPOL_v36_SUR.nc")


@pytest.fixture
def temp_file(tmp_path):
    return tmp_path / "cfradial2.nc"


def test_open_cfradial2(temp_file, cfradial1_file):
    # Open the original file using CfRadial1 reader
    dtree = xd.io.open_cfradial1_datatree(cfradial1_file)

    # Write to CfRadial2 NetCDF format
    dtree.to_netcdf(temp_file)

    # Read back using CfRadial2 reader
    dtree2 = xd.io.open_cfradial2_datatree(temp_file)

    # Check structural correctness
    assert dtree2 is not None, "Failed to open CfRadial2 datatree"
    assert hasattr(dtree2, "children"), "Returned object is not a valid DataTree"
    assert "sweep_0" in dtree2.children, "Missing expected sweep node"
    assert "DBZ" in dtree2["sweep_0"].data_vars, "Missing reflectivity field"

    # Optional data structure comparison
    ds1 = dtree["sweep_0"].ds.copy(deep=False)
    ds2 = dtree2["sweep_0"].ds.copy(deep=False)
    xr.testing.assert_isomorphic(dtree, dtree2)
