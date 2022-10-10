#!/usr/bin/env python
# Copyright (c) 2022, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for `xradar` model package."""

import numpy as np

from xradar import model


# todo: possibly use fixtures here
def test_create_sweep_dataset():
    # default setup (360, 1000)
    # azimuth-res 1deg, fixed elevation 1deg, range-res 100m, time-res 0.25s
    ds = model.create_sweep_dataset()
    assert ds.azimuth.shape == (360,)
    assert ds.elevation.shape == (360,)
    assert ds.time.shape == (360,)
    assert ds.range.shape == (1000,)
    assert ds.dims == {"time": 360, "range": 1000}
    assert np.unique(ds.elevation) == [1.0]
    assert ds.azimuth[0] == 0.5
    assert ds.azimuth[-1] == 359.5
    assert ds.range[0] == 50
    assert ds.range[-1] == 99950
    assert ds.time[0].values == np.datetime64("2022-08-27T10:00:00.000000000")
    assert ds.time[-1].values == np.datetime64("2022-08-27T10:01:29.750000000")

    # provide azimuth- and time-resolution and fixed elevation
    ds = model.create_sweep_dataset(azimuth=1.0, elevation=5.0, time=1)
    assert ds.azimuth.shape == (360,)
    assert ds.elevation.shape == (360,)
    assert ds.time.shape == (360,)
    assert ds.range.shape == (1000,)
    assert ds.dims == {"time": 360, "range": 1000}
    assert np.unique(ds.elevation) == [5.0]
    assert ds.azimuth[0] == 0.5
    assert ds.azimuth[-1] == 359.5
    assert ds.time[0].values == np.datetime64("2022-08-27T10:00:00.000000000")
    assert ds.time[-1].values == np.datetime64("2022-08-27T10:05:59.000000000")

    # provide shape and range-res, fixed-elevation
    ds = model.create_sweep_dataset(shape=(180, 100), rng=50.0, elevation=5.0)
    assert ds.dims == {"time": 180, "range": 100}
    assert ds.range[-1] == 4975
    assert ds.time[-1].values == np.datetime64("2022-08-27T10:00:44.750000000")
    assert np.unique(ds.elevation) == [5.0]

    # provide shape and range-res, fixed-elevation, RHI
    ds = model.create_sweep_dataset(
        shape=(90, 100), rng=50.0, azimuth=205.0, sweep="RHI"
    )
    assert ds.dims == {"time": 90, "range": 100}
    assert ds.range[-1] == 4975
    assert ds.time[-1].values == np.datetime64("2022-08-27T10:00:22.250000000")
    assert np.unique(ds.azimuth) == [205.0]


def test_get_range_dataarray():
    rng = model.get_range_dataarray(100, 100)
    assert rng[0] == 50
    assert rng[-1] == 9950
    attrs = rng.attrs
    assert attrs["units"] == "meters"
    assert attrs["standard_name"] == "projection_range_coordinate"
    assert attrs["long_name"] == "range_to_measurement_volume"
    assert attrs["axis"] == "radial_range_coordinate"
    assert attrs["meters_between_gates"] == 100.0
    assert attrs["spacing_is_constant"] == "true"
    assert attrs["meters_to_center_of_first_gate"] == 50.0


def test_get_azimuth_dataarray():
    # provide resolution
    azi = model.get_azimuth_dataarray(1.0)
    assert azi[0] == 0.5
    assert azi[-1] == 359.5
    attrs = azi.attrs
    assert attrs["units"] == "degrees"
    assert attrs["standard_name"] == "ray_azimuth_angle"
    assert attrs["long_name"] == "azimuth_angle_from_true_north"
    assert attrs["axis"] == "radial_azimuth_coordinate"
    assert attrs["a1gate"] == 0

    # provide constant value and number of rays
    azi = model.get_azimuth_dataarray(1.0, nrays=360)
    assert np.unique(azi) == [1.0]


def test_get_elevation_dataarray():
    # provide resolution
    ele = model.get_elevation_dataarray(1.0)
    assert ele[0] == 0.5
    assert ele[-1] == 89.5
    attrs = ele.attrs
    assert attrs["units"] == "degrees"
    assert attrs["standard_name"] == "ray_elevation_angle"
    assert attrs["long_name"] == "elevation_angle_from_horizontal_plane"
    assert attrs["axis"] == "radial_elevation_coordinate"

    # provide constant value and number of rays
    ele = model.get_elevation_dataarray(1.0, nrays=360)
    assert np.unique(ele) == [1.0]


def test_get_time_dataarray():
    time = model.get_time_dataarray(0.25, nrays=360, date_str="2022-08-27T00:00:00")
    assert time[0] == 0
    assert time[-1] == 89.75
    attrs = time.attrs
    assert attrs["units"] == "seconds since 2022-08-27T00:00:00"
    assert attrs["standard_name"] == "time"


def test_get_sweep_dataarray():
    da = model.get_sweep_dataarray((360, 100), "DBZH", fill=42.0)
    assert da.dims == ("time", "range")
    assert np.unique(da) == [42.0]
    attrs = da.attrs
    attrs["standard_name"] == "radar_equivalent_reflectivity_factor_h"
    attrs["long_name"] == "Equivalent reflectivity factor H"
    attrs["short_name"] == "DBZH"
    attrs["units"] == "dBZ"
