#!/usr/bin/env python
# Copyright (c) 2022-2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for `xradar` util package."""

import numpy as np
import pytest
import xarray as xr
from open_radar_data import DATASETS

import xradar as xd
from xradar import io, model, util
from xradar.io.backends.common import (
    _get_radar_calibration,
    _get_required_root_dataset,
    _get_subgroup,
)


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
    assert ds.sizes["azimuth"] == 361
    assert ds_out.sizes["azimuth"] == 360


def test_reindex_angle():
    filename = DATASETS.fetch("DWD-Vol-2_99999_20180601054047_00.h5")
    ds = xr.open_dataset(filename, group="sweep_7", engine="gamic", first_dim="auto")
    ds_out = util.remove_duplicate_rays(ds)
    ds_out = util.reindex_angle(
        ds_out, start_angle=0, stop_angle=360, angle_res=1.0, direction=1
    )
    assert ds_out.sizes["azimuth"] == 360
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
        "max_time": np.datetime64("2018-06-01T05:43:08.042000000"),
        "min_angle": np.array(0.52459716796875),
        "min_time": np.datetime64("2018-06-01T05:42:44.042000000"),
        "missing_rays": np.array(False),
        "second_angle": "elevation",
        "start_angle": 0,
        "stop_angle": 360,
        "times_are_unique": True,
        "uniform_angle_spacing": False,
    }
    assert angle_dict == expected_dict


@pytest.mark.parametrize("low", [-0.25, -0.1, 0])
@pytest.mark.parametrize("high", [0.25, 0.1, 0])
def test_extract_angle_parameters2(low, high):
    seed = 12345
    rng = np.random.default_rng(seed)
    noise = rng.uniform(low=low, high=high, size=360)
    azimuth = np.arange(0.5, 360.0, 1.0) + noise
    ds = model.create_sweep_dataset(a1gate=0, direction=1, azimuth=azimuth)
    ds = ds.swap_dims({"time": "azimuth"})
    ds = ds.sortby("azimuth")
    ds = ds.assign_coords(sweep_mode="azimuth_surveillance")

    # remove block of rays
    ds_in = xr.concat(
        [
            ds.isel(azimuth=slice(None, 100)),
            ds.isel(azimuth=slice(110, None)),
        ],
        "azimuth",
        data_vars="minimal",
    )
    angle_dict = util.extract_angle_parameters(ds_in)
    expected_dict = {
        "first_angle": "azimuth",
        "second_angle": "elevation",
        "min_angle": np.array(0.36366801123358483),
        "max_angle": np.array(359.6511639563352),
        "min_time": np.datetime64("2022-08-27T10:00:00.000000000"),
        "max_time": np.datetime64("2022-08-27T10:01:29.750000000"),
        "angles_are_unique": True,
        "times_are_unique": True,
        "a1gate_idx": np.array(0),
        "a1gate_val": np.array(0.36366801123358483),
        "uniform_angle_spacing": False,
        "ascending": np.array(True),
        "direction": 1,
        "angle_res": np.array(1.0),
        "start_angle": 0,
        "stop_angle": 360,
        "expected_angle_span": 360,
        "missing_rays": True,
        "excess_rays": False,
        "expected_number_rays": 360,
    }
    expected = {
        (0.25, -0.25): {
            "min_angle": np.array(0.36366801123358483),
            "max_angle": np.array(359.6511639563352),
            "a1gate_val": np.array(0.36366801123358483),
        },
        (0.25, -0.1): {
            "min_angle": np.array(0.4795676078635094),
            "max_angle": np.array(359.6808147694347),
            "a1gate_val": np.array(0.4795676078635094),
        },
        (0.25, 0.0): {
            "min_angle": np.array(0.5568340056167924),
            "max_angle": np.array(359.70058197816763),
            "a1gate_val": np.array(0.5568340056167924),
        },
        (0.1, -0.25): {
            "min_angle": np.array(0.32956760786350936),
            "max_angle": np.array(359.53081476943464),
            "a1gate_val": np.array(0.32956760786350936),
        },
        (0.1, -0.1): {
            "min_angle": np.array(0.4454672044934339),
            "max_angle": np.array(359.5604655825341),
            "a1gate_val": np.array(0.4454672044934339),
        },
        (0.1, 0.0): {
            "min_angle": np.array(0.5227336022467169),
            "max_angle": np.array(359.58023279126706),
            "a1gate_val": np.array(0.5227336022467169),
        },
        (0.0, -0.25): {
            "min_angle": np.array(0.3068340056167924),
            "max_angle": np.array(359.45058197816763),
            "a1gate_val": np.array(0.3068340056167924),
        },
        (0.0, -0.1): {
            "min_angle": np.array(0.42273360224671697),
            "max_angle": np.array(359.48023279126704),
            "a1gate_val": np.array(0.42273360224671697),
        },
        (0.0, 0.0): {
            "min_angle": np.array(0.5),
            "max_angle": np.array(359.5),
            "a1gate_val": np.array(0.5),
        },
    }
    for k in angle_dict:
        if k in ["min_angle", "max_angle", "a1gate_val"]:
            assert angle_dict[k] == expected[(high, low)][k]
            continue
        assert angle_dict[k] == expected_dict[k]


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


@pytest.mark.parametrize("missing", [0, 1, 2, 10, 50])
@pytest.mark.parametrize("a1gate", [0, 1, 2, 179, 180, 181, 357, 358, 359])
@pytest.mark.parametrize("rot", ["cw", "ccw"])
def test_ipol_time2(missing, a1gate, rot):
    # prepare Dataset with one NaT in between
    if rot == "cw":
        start_ang, stop_ang = 0, 360
        direction = 1
    else:
        start_ang, stop_ang = 360, 0
        direction = -1

    ds = model.create_sweep_dataset(
        a1gate=a1gate, direction=direction, start_ang=start_ang, stop_ang=stop_ang
    )
    ds = ds.swap_dims({"time": "azimuth"})
    ds = ds.sortby("azimuth")
    ds_in = xr.concat(
        [
            ds.isel(azimuth=slice(missing, 180 - missing)),
            ds.isel(azimuth=slice(180 + missing, 360 - missing)),
        ],
        "azimuth",
        data_vars="minimal",
    )

    ds_out = util.remove_duplicate_rays(ds_in)
    ds_out = util.reindex_angle(
        ds_out,
        start_angle=start_ang,
        stop_angle=stop_ang,
        angle_res=1.0,
        direction=direction,
        method="nearest",
    )
    dsx = ds_out.copy(deep=True)
    dsy = dsx.pipe(util.ipol_time, a1gate_idx=a1gate, direction=direction)
    # see if all time values were reconstructed
    xr.testing.assert_equal(ds, dsy)

    # check emit warning
    if a1gate in [179, 180, 181] and missing and direction == 1:
        with pytest.warns(
            UserWarning, match="Rays might miss on beginning and/or end of sweep."
        ):
            dsx.pipe(util.ipol_time, direction=direction)


def test_get_sweep_keys():
    # Test finding sweep keys
    filename = DATASETS.fetch("sample_sgp_data.nc")
    dtree = io.open_cfradial1_datatree(filename)
    # set a fake group
    dtree["sneep_1"] = dtree["sweep_1"]
    keys = util.get_sweep_keys(dtree)
    assert keys == [
        "sweep_0",
        "sweep_1",
        "sweep_2",
        "sweep_3",
        "sweep_4",
        "sweep_5",
    ]


def test_apply_to_sweeps():
    # Fetch the sample radar file
    filename = DATASETS.fetch("sample_sgp_data.nc")

    # Open the radar file into a DataTree object
    dtree = io.open_cfradial1_datatree(filename)

    # Define a simple function to test with apply_to_sweeps
    def dummy_function(ds):
        """A dummy function that adds a constant field to the dataset."""
        ds["dummy_field"] = (
            ds["reflectivity_horizontal"] * 0
        )  # Adding a field with all zeros
        ds["dummy_field"].attrs = {"units": "dBZ", "long_name": "Dummy Field"}
        return ds

    # Apply the dummy function to all sweeps using apply_to_sweeps
    modified_dtree = util.apply_to_sweeps(dtree, dummy_function)

    # Verify that the dummy field has been added to each sweep
    sweep_keys = util.get_sweep_keys(modified_dtree)
    for key in sweep_keys:
        assert (
            "dummy_field" in modified_dtree[key].data_vars
        ), f"dummy_field not found in {key}"
        assert modified_dtree[key]["dummy_field"].attrs["units"] == "dBZ"
        assert modified_dtree[key]["dummy_field"].attrs["long_name"] == "Dummy Field"

    # Check that the original data has not been modified
    assert (
        "dummy_field" not in dtree["/"].data_vars
    ), "dummy_field should not be in the root node"

    # Test that an exception is raised when a function that causes an error is applied
    with pytest.raises(ValueError, match="This is an intentional error"):

        def error_function(ds):
            raise ValueError("This is an intentional error")

        util.apply_to_sweeps(dtree, error_function)


def test_apply_to_volume():
    # Fetch the sample radar file
    filename = DATASETS.fetch("sample_sgp_data.nc")

    # Open the radar file into a DataTree object
    dtree = io.open_cfradial1_datatree(filename)

    # Define a simple function to test with apply_to_volume
    def dummy_function(ds):
        """A dummy function that adds a constant field to the dataset."""
        ds["dummy_field"] = (
            ds["reflectivity_horizontal"] * 0
        )  # Adding a field with all zeros
        ds["dummy_field"].attrs = {"units": "dBZ", "long_name": "Dummy Field"}
        return ds

    # Apply the dummy function to all sweeps using apply_to_volume
    modified_dtree = util.apply_to_volume(dtree, dummy_function)

    # Verify that the modified_dtree is an instance of DataTree
    assert isinstance(
        modified_dtree, xr.DataTree
    ), "The result should be a DataTree instance."

    # Verify that the dummy field has been added to each sweep
    sweep_keys = util.get_sweep_keys(modified_dtree)
    for key in sweep_keys:
        assert (
            "dummy_field" in modified_dtree[key].data_vars
        ), f"dummy_field not found in {key}"
        assert modified_dtree[key]["dummy_field"].attrs["units"] == "dBZ"
        assert modified_dtree[key]["dummy_field"].attrs["long_name"] == "Dummy Field"

    # Check that the original DataTree (dtree) has not been modified
    original_sweep_keys = util.get_sweep_keys(dtree)
    for key in original_sweep_keys:
        assert (
            "dummy_field" not in dtree[key].data_vars
        ), f"dummy_field should not be in the original DataTree at {key}"

    # Test edge case: Apply a function that modifies only certain sweeps
    def selective_function(ds):
        """Only modifies sweeps with a specific condition."""
        if "reflectivity_horizontal" in ds:
            ds["selective_field"] = ds["reflectivity_horizontal"] * 1
        return ds

    # Apply the selective function to all sweeps using apply_to_volume
    selectively_modified_dtree = util.apply_to_volume(dtree, selective_function)

    # Verify that the selective field was added only where the condition was met
    for key in sweep_keys:
        if "reflectivity_horizontal" in modified_dtree[key].data_vars:
            assert (
                "selective_field" in selectively_modified_dtree[key].data_vars
            ), f"selective_field not found in {key} where it should have been added."
        else:
            assert (
                "selective_field" not in selectively_modified_dtree[key].data_vars
            ), f"selective_field should not be present in {key}"

    # Test that an exception is raised when a function that causes an error is applied
    with pytest.raises(ValueError, match="This is an intentional error"):

        def error_function(ds):
            raise ValueError("This is an intentional error")

        util.apply_to_volume(dtree, error_function)


def test_map_over_sweeps_decorator_dummy_function():
    """
    Test applying a dummy function to all sweep nodes using the map_over_sweeps decorator.
    """
    # Fetch the sample radar file
    filename = DATASETS.fetch("sample_sgp_data.nc")

    # Open the radar file into a DataTree object
    dtree = xd.io.open_cfradial1_datatree(filename)

    # Use the decorator on the dummy function
    @xd.map_over_sweeps
    def dummy_function(ds, refl="none"):
        ds = ds.assign(
            dummy_field=ds["reflectivity_horizontal"] * 0
        )  # Field with zeros
        ds["dummy_field"].attrs = {
            "unit": "dBZ",
            "long_name": "Dummy Field",
            "test": refl,
        }
        return ds

    # Apply using pipe and decorator
    dtree_modified = dtree.pipe(dummy_function, refl="test")

    # Check that the new field exists in sweep_0 and has the correct attributes
    sweep_0 = dtree_modified["sweep_0"]
    assert "dummy_field" in sweep_0.data_vars
    assert sweep_0.dummy_field.attrs["unit"] == "dBZ"
    assert sweep_0.dummy_field.attrs["long_name"] == "Dummy Field"
    assert sweep_0.dummy_field.attrs["test"] == "test"

    # Ensure all non-NaN values are 0 (accounting for -0.0 and NaN values)
    non_nan_values = np.nan_to_num(
        sweep_0.dummy_field.values
    )  # Convert NaNs to zero for comparison
    assert np.all(np.isclose(non_nan_values, 0))


def test_get_required_root_dataset():

    filename = DATASETS.fetch("cor-main131125105503.RAW2049")
    sweeps = [f"sweep_{i}" for i in range(10)]
    ls_ds = [xr.open_dataset(filename, engine="iris", group=sweep) for sweep in sweeps]
    root = _get_required_root_dataset(ls_ds, optional=True)
    elevations = [
        0.5,
        1.0,
        2.0,
        3.0,
        5.0,
        7.0,
        10.0,
        15.0,
        20.0,
        30.0,
    ]
    assert len(root.variables) == 10
    assert root.variables["time_coverage_start"] == "2013-11-25T10:55:04Z"
    assert root.variables["time_coverage_end"] == "2013-11-25T10:59:24Z"
    np.testing.assert_equal(
        root.variables["sweep_fixed_angle"].values, np.array(elevations)
    )
    assert len(list(root.attrs.keys())) == 10
    assert root.attrs["instrument_name"] == "Corozal, Radar"
    assert root.attrs["scan_name"] == "SURV_HV_300 "
    assert root.attrs["comment"] == "AEROCIVIL OPERATIONAL DUAL POLE SCAN"


def test_get_radar_calibration():
    filename = DATASETS.fetch("DWD-Vol-2_99999_20180601054047_00.h5")
    sweeps = [f"sweep_{i}" for i in range(10)]
    ls_ds = [xr.open_dataset(filename, engine="gamic", group=sweep) for sweep in sweeps]
    subgroup = _get_radar_calibration(ls_ds, model.radar_calibration_subgroup)
    assert len(subgroup.variables) == 6
    assert subgroup["noise_power_h"] == "-3.8298"
    assert subgroup["rx_loss_h"] == "3"
    assert subgroup["ant_gain_v"] == "43"
    assert subgroup["ant_gain_h"] == "43"


def test_get_subgroup():
    filename = DATASETS.fetch("71_20181220_060628.pvol.h5")
    sweeps = [f"sweep_{i}" for i in range(10)]
    ls_ds = [xr.open_dataset(filename, engine="odim", group=sweep) for sweep in sweeps]
    subgroup = _get_subgroup(ls_ds, model.radar_parameters_subgroup)
    assert len(subgroup.variables) == 3
    assert list(subgroup.variables) == ["longitude", "latitude", "altitude"]
    np.testing.assert_almost_equal(subgroup.longitude.values.item(), 151.20899963378906)
    np.testing.assert_almost_equal(subgroup.latitude.values.item(), -33.700801849365234)
    assert isinstance(subgroup.altitude.values.item(), float)
