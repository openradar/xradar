#!/usr/bin/env python
# Copyright (c) 2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for `io.backends.gamic` module.

ported from wradlib
"""

import numpy as np
import pytest
from xarray import DataTree

from xradar.io.backends import gamic
from xradar.io.backends.gamic import open_gamic_datatree


def create_ray_header(nrays=360):
    dtype = {
        "names": [
            "azimuth_start",
            "azimuth_stop",
            "elevation_start",
            "elevation_stop",
            "timestamp",
        ],
        "formats": ["<f8", "<f8", "<f8", "<f8", "<i8"],
        "offsets": [0, 8, 16, 24, 32],
        "itemsize": 40,
    }
    ray_header = np.zeros(nrays, dtype=dtype)
    start_az = np.linspace(0, 360, nrays, endpoint=False, dtype=np.float64)
    stop_az = np.linspace(1, 361, nrays, endpoint=False, dtype=np.float64)
    ray_header["azimuth_start"] = start_az
    ray_header["azimuth_stop"] = stop_az

    start_el = np.ones_like(start_az, dtype=np.float64)
    stop_el = np.ones_like(start_az, dtype=np.float64)
    ray_header["elevation_start"] = start_el
    ray_header["elevation_stop"] = stop_el

    time = np.arange(
        1527831788042000, 1527831788042000 + nrays * 100000, 100000, dtype=np.int64
    )
    ray_header["timestamp"] = time
    return ray_header


@pytest.mark.parametrize("nrays", [180, 240, 360, 720])
def test_get_azimuth(nrays):
    ray_header = create_ray_header(nrays)
    actual = gamic._get_azimuth(ray_header)
    udiff = np.unique(np.diff(actual))
    assert len(actual) == nrays
    assert len(udiff) == 1
    assert udiff[0] == 360.0 / nrays


@pytest.mark.parametrize("nrays", [180, 240, 360, 720])
def test_get_elevation(nrays):
    ray_header = create_ray_header(nrays)
    actual = gamic._get_elevation(ray_header)
    unique = np.unique(actual)
    assert len(actual) == nrays
    assert len(unique) == 1
    assert unique[0] == 1.0


@pytest.mark.parametrize("nrays", [180, 240, 360, 720])
def test_get_timestamp(nrays):
    ray_header = create_ray_header(nrays)
    actual = gamic._get_time(ray_header)
    udiff = np.unique(np.diff(actual))
    assert len(actual) == nrays
    assert len(udiff) == 1
    assert udiff[0] == 100000


@pytest.mark.parametrize(
    "ang",
    [("elevation", "azimuth"), ("azimuth", "elevation")],
)
def test_get_fixed_dim_and_angle(ang):
    how = {ang[0]: 1.0}
    dim, angle = gamic._get_fixed_dim_and_angle(how)
    assert dim == ang[1]
    assert angle == 1.0


@pytest.mark.parametrize("range_step", [100, 150, 300, 1000])
@pytest.mark.parametrize("range_samples", [1, 2, 4])
def test_get_range(range_step, range_samples):
    where = dict(bin_count=10, range_step=range_step, range_samples=range_samples)
    rng, cent_first, bin_range = gamic._get_range(where)
    assert len(rng) == 10
    assert np.unique(np.diff(rng))[0] == range_step * range_samples
    assert cent_first == (range_step * range_samples) / 2
    assert bin_range == range_step * range_samples


def test_GamicH5NetCDFMetadata(gamic_file):
    store = gamic.GamicStore.open(gamic_file, group="sweep_0")
    with pytest.warns(DeprecationWarning):
        assert store.root.first_dim == "azimuth"


def test_open_gamic_datatree(gamic_file):
    # Define kwargs to pass into the function
    kwargs = {
        "sweep": [0, 1],  # Test with specific sweeps
        "first_dim": "auto",
        "reindex_angle": {
            "start_angle": 0.0,
            "stop_angle": 360.0,
            "angle_res": 1.0,
            "direction": 1,  # Set a valid direction within reindex_angle
        },
        "fix_second_angle": True,
        "site_coords": True,
    }

    # Call the function with an actual GAMIC HDF5 file
    dtree = open_gamic_datatree(gamic_file, **kwargs)

    # Assertions
    assert isinstance(dtree, DataTree), "Expected a DataTree instance"
    assert "/" in dtree.subtree, "Root group should be present in the DataTree"
    assert (
        "/radar_parameters" in dtree.subtree
    ), "Radar parameters group should be in the DataTree"
    assert (
        "/georeferencing_correction" in dtree.subtree
    ), "Georeferencing correction group should be in the DataTree"
    assert (
        "/radar_calibration" in dtree.subtree
    ), "Radar calibration group should be in the DataTree"

    # Check if at least one sweep group is attached (e.g., "/sweep_0")
    sweep_groups = [key for key in dtree.match("sweep_*")]
    assert len(sweep_groups) == 2, "Expected two sweep groups in the DataTree"

    # Verify a sample variable in one of the sweep groups
    sample_sweep = sweep_groups[0]
    assert (
        len(dtree[sample_sweep].data_vars) == 17
    ), f"Expected data variables in {sample_sweep}"
    assert dtree[sample_sweep]["DBZH"].shape == (360, 360)
    assert (
        "DBZH" in dtree[sample_sweep].data_vars
    ), f"DBZH should be a data variable in {sample_sweep}"
    assert (
        "VRADV" in dtree[sample_sweep].data_vars
    ), f"VRADV should be a data variable in {sample_sweep}"

    # Validate coordinates are attached correctly in the root dataset
    assert (
        "latitude" in dtree[sample_sweep]
    ), "Latitude should be attached to the root dataset"
    assert (
        "longitude" in dtree[sample_sweep]
    ), "Longitude should be attached to the root dataset"
    assert (
        "altitude" in dtree[sample_sweep]
    ), "Altitude should be attached to the root dataset"

    # Validate attributes
    assert len(dtree.attrs) == 9
    assert dtree.attrs["source"] == "gamic", "Source should match expected value"
