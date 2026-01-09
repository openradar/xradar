#!/usr/bin/env python
# Copyright (c) 2023-2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.


import xarray as xr
from open_radar_data import DATASETS

import xradar as xd


def test_cfradial1_open_mfdataset_context_manager(cfradial1_file):
    with xr.open_mfdataset(
        [cfradial1_file],
        engine="cfradial1",
        concat_dim="volume_time",
        combine="nested",
        group="sweep_0",
    ) as ds:
        assert ds is not None
        # closer must exist while inside context
        assert callable(getattr(ds, "_close", None))


def test_cfradial1_dataset_has_close(cfradial1_file):
    ds = xr.open_dataset(cfradial1_file, engine="cfradial1", group="sweep_0")
    assert callable(getattr(ds, "_close", None))
    ds.close()


def test_compare_sweeps(temp_file):
    # Fetch the radar data file
    filename = DATASETS.fetch("cfrad.20080604_002217_000_SPOL_v36_SUR.nc")

    # Open the data tree
    # todo: implement a roundtrip function
    dtree = xd.io.open_cfradial1_datatree(filename)
    # Save the modified data tree to the temporary file
    xd.io.to_cfradial1(dtree.copy(), temp_file, calibs=True)

    # Open the modified data tree
    dtree1 = xd.io.open_cfradial1_datatree(temp_file)
    # todo: check, if we can use xarray machinery for
    #  testing tree equality
    # Compare the values of the DataArrays for all sweeps
    for sweep_num in range(9):  # there are 9 sweeps in this file
        xr.testing.assert_equal(
            dtree[f"sweep_{sweep_num}"].ds, dtree1[f"sweep_{sweep_num}"].ds
        )
