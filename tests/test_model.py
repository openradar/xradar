#!/usr/bin/env python
# Copyright (c) 2022-2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for `xradar` model package."""

import numpy as np
import pytest

from xradar import model


def test_get_range_attrs_with_float32_precision():
    rng = np.arange(
        37.500034,
        75.000034 * 8000,
        75.000034,
        dtype="float32",
    )
    range_attrs = model.get_range_attrs(rng)
    assert range_attrs == {
        "units": "meters",
        "standard_name": "projection_range_coordinate",
        "long_name": "range_to_measurement_volume",
        "axis": "radial_range_coordinate",
        "meters_between_gates": np.float32(75.00003),
        "spacing_is_constant": "true",
        "meters_to_center_of_first_gate": np.float32(37.500034),
    }


@pytest.mark.parametrize("dtype", ["int16", "int32", "int64", "uint32"])
def test_get_range_attrs_integer_range(dtype):
    # integer ranges are compared exactly
    rng = np.arange(125, 125 + 250 * 10, 250, dtype=dtype)
    range_attrs = model.get_range_attrs(rng)
    assert range_attrs["spacing_is_constant"] == "true"
    assert range_attrs["meters_between_gates"] == 250
    assert range_attrs["meters_to_center_of_first_gate"] == 125

    rng[-1] += 1
    range_attrs = model.get_range_attrs(rng)
    assert range_attrs["spacing_is_constant"] == "false"
    assert "meters_between_gates" not in range_attrs


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_get_range_attrs_float_not_constant(dtype):
    # deviations larger than the float precision are not constant spacing
    rng = np.arange(37.5, 75.0 * 100, 75.0, dtype=dtype)
    rng[50:] += 1.0
    range_attrs = model.get_range_attrs(rng)
    assert range_attrs["spacing_is_constant"] == "false"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(azimuth=1.0, elevation=1.0), "Either `shape` or"),
        (dict(sweep="PPI"), "elevation need to be specified"),
        (dict(sweep="RHI"), "azimuth need to be specified"),
    ],
)
def test_create_sweep_dataset_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        model.create_sweep_dataset(shape=(360, 100), **kwargs)


def test_create_sweep_dataset_shape():
    ds = model.create_sweep_dataset(shape=(720, 100), elevation=1.0)
    assert ds.azimuth.shape == (720,)
    assert ds.range.shape == (100,)
    np.testing.assert_allclose(ds.azimuth.diff("time"), 0.5)
    ds = model.create_sweep_dataset(shape=(90, 100), azimuth=10.0, sweep="RHI")
    assert ds.elevation.shape == (90,)
    np.testing.assert_allclose(ds.elevation.diff("time"), 1.0)


def test_get_sweep_dataarray_fill():
    da = model.get_sweep_dataarray((10, 20), "DBZH", fill=5.0)
    assert da.dims == ("time", "range")
    assert da.shape == (10, 20)
    np.testing.assert_array_equal(da, 5.0)
    assert da.name == "DBZH"
    assert da.attrs["units"] == "dBZ"
    # without fill: range index along each ray
    da = model.get_sweep_dataarray((3, 4), "DBZH")
    np.testing.assert_array_equal(da, np.tile(np.arange(4), (3, 1)))


# todo: possibly use fixtures here
def test_create_sweep_dataset():
    # default setup (360, 1000)
    # azimuth-res 1deg, fixed elevation 1deg, range-res 100m, time-res 0.25s
    ds = model.create_sweep_dataset()
    assert ds.azimuth.shape == (360,)
    assert ds.elevation.shape == (360,)
    assert ds.time.shape == (360,)
    assert ds.range.shape == (1000,)
    assert ds.sizes == {"time": 360, "range": 1000}
    assert np.unique(ds.elevation) == [1.0]
    assert ds.azimuth[0] == 0.5
    assert ds.azimuth[-1] == 359.5
    assert ds.range[0] == 50
    assert ds.range[-1] == 99950
    assert ds.time[0].values == np.datetime64("2022-08-27T10:00:00.000000000")
    assert ds.time[-1].values == np.datetime64("2022-08-27T10:01:29.750000000")
    assert ds.altitude == 375
    assert ds.longitude == 8.7877271
    assert ds.latitude == 46.172541

    # provide azimuth- and time-resolution and fixed elevation
    ds = model.create_sweep_dataset(azimuth=1.0, elevation=5.0, time=1)
    assert ds.azimuth.shape == (360,)
    assert ds.elevation.shape == (360,)
    assert ds.time.shape == (360,)
    assert ds.range.shape == (1000,)
    assert ds.sizes == {"time": 360, "range": 1000}
    assert np.unique(ds.elevation) == [5.0]
    assert ds.azimuth[0] == 0.5
    assert ds.azimuth[-1] == 359.5
    assert ds.time[0].values == np.datetime64("2022-08-27T10:00:00.000000000")
    assert ds.time[-1].values == np.datetime64("2022-08-27T10:05:59.000000000")

    # provide shape and range-res, fixed-elevation
    ds = model.create_sweep_dataset(shape=(180, 100), rng=50.0, elevation=5.0)
    assert ds.sizes == {"time": 180, "range": 100}
    assert ds.range[-1] == 4975
    assert ds.time[-1].values == np.datetime64("2022-08-27T10:00:44.750000000")
    assert np.unique(ds.elevation) == [5.0]

    # provide shape and range-res, fixed-elevation, RHI
    ds = model.create_sweep_dataset(
        shape=(90, 100), rng=50.0, azimuth=205.0, sweep="RHI"
    )
    assert ds.sizes == {"time": 90, "range": 100}
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
    assert attrs["standard_name"] == "radar_equivalent_reflectivity_factor_h"
    assert attrs["long_name"] == "Equivalent reflectivity factor H"
    assert attrs["short_name"] == "DBZH"
    assert attrs["units"] == "dBZ"


def test_cf_moment_attrs_order_is_fixed():
    """Attr key order must not depend on the process (``model.moment_attrs``
    is a set); the virtual stores write these attrs in this order."""
    assert list(model._cf_moment_attrs("DBZH")) == [
        "units",
        "standard_name",
        "long_name",
    ]
    assert model._cf_moment_attrs("NOT_A_MOMENT") == {}
