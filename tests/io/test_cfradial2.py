#!/usr/bin/env python
# Copyright (c) 2023-2025, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

import tempfile
from pathlib import Path

import xarray as xr
from open_radar_data import DATASETS

import xradar as xd


def test_open_cfradial2():
    # Fetch a CfRadial1 file to simulate CfRadial2 Zarr format
    filename = DATASETS.fetch("cfrad.20080604_002217_000_SPOL_v36_SUR.nc")

    # Open using CfRadial1 reader
    dtree = xd.io.open_cfradial1_datatree(filename)

    # Write it to a temporary Zarr store (to simulate CfRadial2)
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "cfradial2.nc"
        dtree.to_netcdf(file_path)

        # Read back using CfRadial2-style reader
        dtree2 = xd.io.open_cfradial2_datatree(file_path)

        # # Compare sweeps
        # for sweep_num in range(3):
        #     ds1 = dtree[f"sweep_{sweep_num}"].ds.copy(deep=False)
        #     ds2 = dtree2[f"sweep_{sweep_num}"].ds.copy(deep=False)

        #     xr.testing.assert_isomorphic(ds1, ds2)
        #     xr.testing.assert_allclose(ds1, ds2, rtol=1e-5)

        xr.testing.assert_isomorphic(dtree, dtree2)
        assert dtree is not None, "Failed to open CfRadial2 datatree"
        assert hasattr(dtree, "children"), "Returned object is not a valid DataTree"
        assert "sweep_0" in dtree.children, "Missing expected sweep node"
        assert "DBZ" in dtree["sweep_0"].data_vars, "Missing reflectivity field"
