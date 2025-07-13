#!/usr/bin/env python
# Copyright (c) 2023-2025, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.


import pytest
from open_radar_data import DATASETS

from xradar.io.cfradial2 import open_cfradial2_datatree


@pytest.fixture
def cfradial2_zarr(tmp_path):
    # Fetch the CfRadial1 NetCDF file
    file = DATASETS.fetch("cfrad.20080604_002217_000_SPOL_v36_SUR.nc")

    # Open as CfRadial1 and write to Zarr (to simulate CfRadial2-style storage)
    import xradar as xd

    dtree = xd.io.open_cfradial1_datatree(file)
    zarr_path = tmp_path / "CFRADIAL_zarr.zarr"
    dtree.to_zarr(zarr_path)

    return zarr_path


def test_open_cfradial2_datatree(cfradial2_zarr):
    dtree = open_cfradial2_datatree(cfradial2_zarr)

    assert dtree is not None, "Failed to open CfRadial2 datatree"
    assert hasattr(dtree, "children"), "Returned object is not a valid DataTree"
    assert "sweep_0" in dtree.children, "Missing expected sweep node"
    assert "DBZ" in dtree["sweep_0"].data_vars, "Missing reflectivity field"

    # Optional: check attributes
    assert "Conventions" in dtree.attrs, "Missing global attribute: Conventions"
