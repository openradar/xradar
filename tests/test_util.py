#!/usr/bin/env python
# Copyright (c) 2022, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for `xradar` util package."""

import numpy as np
import pytest
import xarray as xr
from open_radar_data import DATASETS

from xradar import model, util


@pytest.fixture(
    params=[("PPI", "azimuth", "elevation"), ("RHI", "elevation", "azimuth")]
)
def sweep(request):
    return request.param


def test_get_first_angle(sweep):
    ds = model.create_sweep_dataset(sweep=sweep[0])
    with pytest.raises(ValueError):
        util.get_first_angle(ds)
    ds = ds.swap_dims({"time": sweep[1]})
    assert util.get_first_angle(ds) == sweep[1]


def test_get_second_angle(sweep):
    ds = model.create_sweep_dataset(sweep=sweep[0])
    ds = ds.swap_dims({"time": sweep[1]})
    assert util.get_second_angle(ds) == sweep[2]


def test_remove_duplicate_rays():
    filename = DATASETS.fetch("DWD-Vol-2_99999_20180601054047_00.h5")
    ds = xr.open_dataset(filename, group="sweep_7", engine="gamic", first_dim="auto")
    ds_out = util.remove_duplicate_rays(ds)
    assert ds.dims["azimuth"] == 361
    assert ds_out.dims["azimuth"] == 360


def test_reindex_angle():
    filename = DATASETS.fetch("DWD-Vol-2_99999_20180601054047_00.h5")
    ds = xr.open_dataset(filename, group="sweep_7", engine="gamic", first_dim="auto")
    ds_out = util.remove_duplicate_rays(ds)
    ds_out = util.reindex_angle(
        ds_out, start_angle=0, stop_angle=360, angle_res=1.0, direction=1
    )
    assert ds_out.dims["azimuth"] == 360
    np.testing.assert_array_equal(ds_out.azimuth.values, np.arange(0.5, 360, 1.0))


def test_extract_angle_parameters():
    filename = DATASETS.fetch("DWD-Vol-2_99999_20180601054047_00.h5")
    ds = xr.open_dataset(filename, group="sweep_7", engine="gamic", first_dim="auto")
    angle_dict = util.extract_angle_parameters(ds)
    expected_dict = {
        "a1gate_idx": np.array(243),
        "a1gate_val": np.array(243.52020263671875),
        "angle_res": np.array(1.0),
        "angles_are_unique": False,
        "ascending": np.array(True),
        "direction": 1,
        "excess_rays": np.array(True),
        "expected_angle_span": 360,
        "expected_number_rays": 360,
        "first_angle": "azimuth",
        "max_angle": np.array(359.5111083984375),
        "max_time": np.datetime64("2018-06-01T05:43:08.042000128"),
        "min_angle": np.array(0.52459716796875),
        "min_time": np.datetime64("2018-06-01T05:42:44.042000128"),
        "missing_rays": np.array(False),
        "second_angle": "elevation",
        "start_angle": 0,
        "stop_angle": 360,
        "times_are_unique": True,
        "uniform_angle_spacing": False,
    }
    assert angle_dict == expected_dict


def test_ipol_time():
    filename = DATASETS.fetch("DWD-Vol-2_99999_20180601054047_00.h5")
    ds = xr.open_dataset(filename, group="sweep_7", engine="gamic", first_dim="auto")

    # remove block of rays (100-150)
    ds_in = xr.concat(
        [
            ds.isel(azimuth=slice(None, 100)),
            ds.isel(azimuth=slice(150, None)),
        ],
        "azimuth",
        data_vars="minimal",
    )

    # fix sweep and interpolate times
    ds_out = util.remove_duplicate_rays(ds_in)
    ds_out = util.reindex_angle(
        ds_out, start_angle=0, stop_angle=360, angle_res=1.0, direction=1
    )
    ds_out = util.ipol_time(ds_out)

    # extract times and compare
    time_in = util.remove_duplicate_rays(ds).time.astype(int)
    time_out = ds_out.time.astype(int)
    xr.testing.assert_allclose(
        time_out.drop_vars(time_out.coords), time_in.drop_vars(time_in.coords)
    )
