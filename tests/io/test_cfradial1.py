#!/usr/bin/env python
# Copyright (c) 2023-2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.


import numpy as np
import xarray as xr
from open_radar_data import DATASETS

import xradar as xd
from xradar.io.export import cfradial1 as cf1_export


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


def test_cfradial1_export_helper_scalar_normalization():
    assert cf1_export._first_valid_scalar(xr.DataArray(np.array([np.nan, 3.0]))) == 3.0

    masked = xr.DataArray(np.ma.array([1.0, 2.0], mask=[True, False]))
    assert cf1_export._first_valid_scalar(masked) == 2.0

    nat = xr.DataArray(np.array(["NaT", "2025-01-01T00:00:00"], dtype="datetime64[ns]"))
    assert np.datetime64(cf1_export._first_valid_scalar(nat), "ns") == np.datetime64(
        "2025-01-01T00:00:00", "ns"
    )

    text = xr.DataArray(np.array(["azimuth_surveillance"], dtype=object))
    assert cf1_export._first_valid_scalar(text) == "azimuth_surveillance"

    missing = xr.DataArray(np.array([np.nan, np.nan]))
    assert np.isnan(cf1_export._first_valid_scalar(missing))


def test_cfradial1_export_helper_metadata_and_indices():
    sweep = xr.Dataset(
        data_vars={
            "sweep_number": (
                ("azimuth", "range"),
                np.array([[0.0, np.nan], [0.0, np.nan]]),
            ),
            "sweep_mode": (
                ("azimuth", "range"),
                np.array(
                    [
                        ["azimuth_surveillance", None],
                        ["azimuth_surveillance", None],
                    ],
                    dtype=object,
                ),
            ),
            "DBZ": (("azimuth", "range"), np.ones((2, 2), dtype="float32")),
        },
        coords={
            "azimuth": ("azimuth", np.array([0.0, 1.0], dtype="float32")),
            "range": ("range", np.array([100.0, 200.0], dtype="float32")),
            "time": (
                "azimuth",
                np.array(["2025-01-01", "2025-01-01"], dtype="datetime64[ns]"),
            ),
            "elevation": ("azimuth", np.array([0.5, 0.5], dtype="float32")),
        },
    )

    normalized = cf1_export._normalize_sweep_metadata(sweep)
    assert normalized["sweep_number"].dims == ()
    assert normalized["sweep_number"].item() == 0.0
    assert normalized["sweep_mode"].dims == ()
    assert normalized["sweep_mode"].item() == "azimuth_surveillance"

    valid = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(),
            "/sweep_0": xr.Dataset(
                coords={"elevation": ("azimuth", np.array([0.5, 0.5], dtype="float32"))}
            ),
        }
    )
    out = cf1_export.calculate_sweep_indices(valid)
    assert out["sweep_start_ray_index"].dims == ("sweep",)
    assert out["sweep_end_ray_index"].dims == ("sweep",)


def test_cfradial1_export_helper_empty_sweep_info_and_time_fallback():
    empty = xr.DataTree.from_dict({"/": xr.Dataset()})
    sweep_info = cf1_export._sweep_info_mapper(empty)
    assert "sweep_number" in sweep_info
    assert np.isnan(sweep_info["sweep_number"].values[0])

    sweep = xr.Dataset(
        data_vars={
            "DBZ": (("time", "range"), np.ones((2, 2), dtype="float32")),
            "sweep_mode": ((), "manual"),
            "sweep_number": ((), 0),
            "sweep_fixed_angle": ((), 0.5),
        },
        coords={
            "time": (
                "time",
                np.array(["2025-01-01", "2025-01-01T00:00:01"], dtype="datetime64[ns]"),
            ),
            "range": ("range", np.array([100.0, 200.0], dtype="float32")),
            "azimuth": ("time", np.array([0.0, 1.0], dtype="float32")),
            "elevation": ("time", np.array([0.5, 0.5], dtype="float32")),
        },
    )
    dtree = xr.DataTree.from_dict({"/": xr.Dataset(), "/sweep_0": sweep})
    mapped = cf1_export._variable_mapper(dtree)
    assert "DBZ" in mapped
    assert mapped["DBZ"].dims == ("time", "range")
