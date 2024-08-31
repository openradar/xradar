#!/usr/bin/env python
# Copyright (c) 2022-2023, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

import datatree as dt
import numpy as np
from numpy.testing import assert_almost_equal
from open_radar_data import DATASETS

import xradar as xd


def test_georeference_dataarray():
    radar = xd.model.create_sweep_dataset()
    radar["sample_field"] = radar.azimuth + radar.range

    geo = radar.sample_field.xradar.georeference()
    assert_almost_equal(geo.x.values[:3, 0], np.array([0.436241, 1.3085901, 2.1805407]))
    assert_almost_equal(
        geo.y.values[:3, 0], np.array([49.9882679, 49.973041, 49.9425919])
    )
    assert_almost_equal(
        geo.z.values[:3, 0], np.array([375.8727675, 375.8727675, 375.8727675])
    )


def test_georeference_dataset():
    radar = xd.model.create_sweep_dataset()
    geo = radar.xradar.georeference()
    assert_almost_equal(geo.x.values[:3, 0], np.array([0.436241, 1.3085901, 2.1805407]))
    assert_almost_equal(
        geo.y.values[:3, 0], np.array([49.9882679, 49.973041, 49.9425919])
    )
    assert_almost_equal(
        geo.z.values[:3, 0], np.array([375.8727675, 375.8727675, 375.8727675])
    )


def test_georeference_datatree():
    radar = xd.model.create_sweep_dataset()
    tree = dt.DataTree.from_dict({"sweep_0": radar})
    geo = tree.xradar.georeference()["sweep_0"]
    assert_almost_equal(
        geo["x"].values[:3, 0], np.array([0.436241, 1.3085901, 2.1805407])
    )
    assert_almost_equal(
        geo["y"].values[:3, 0], np.array([49.9882679, 49.973041, 49.9425919])
    )
    assert_almost_equal(
        geo["z"].values[:3, 0], np.array([375.8727675, 375.8727675, 375.8727675])
    )


def test_xradar_datatree_accessor_apply():
    # Fetch the sample radar file
    filename = DATASETS.fetch("sample_sgp_data.nc")

    # Open the radar file into a DataTree object
    dtree = xd.io.open_cfradial1_datatree(filename)

    # Define a simple function to test with the apply method
    def dummy_function(ds, sweep=None):
        """A dummy function that adds a constant field to the dataset."""
        ds["dummy_field"] = (
            ds["reflectivity_horizontal"] * 0
        )  # Adding a field with all zeros
        ds["dummy_field"].attrs = {
            "unit": "dBZ",
            "long_name": "Dummy Field",
            "sweep": sweep,
        }
        return ds

    # Apply the dummy function using the xradar accessor
    dtree = dtree.xradar.apply(dummy_function, pass_sweep_name=True)

    # Verify that the dummy field has been added to each sweep and includes the correct sweep name
    for key in xd.util.get_sweep_keys(dtree):
        assert "dummy_field" in dtree[key].data_vars, f"dummy_field not found in {key}"
        assert dtree[key]["dummy_field"].attrs["unit"] == "dBZ"
        assert dtree[key]["dummy_field"].attrs["long_name"] == "Dummy Field"
        assert (
            dtree[key]["dummy_field"].attrs["sweep"] == key
        ), f"sweep name incorrect in {key}"


def test_xradar_dataset_accessor_apply():
    # Fetch the sample radar file
    filename = DATASETS.fetch("sample_sgp_data.nc")

    # Open the radar file into a DataTree object and extract a Dataset
    dtree = xd.io.open_cfradial1_datatree(filename)
    ds = dtree["sweep_0"].to_dataset()  # Extracting the Dataset from one sweep

    # Define a simple function to test with the apply method
    def dummy_function(ds):
        """A dummy function that adds a constant field to the dataset."""
        ds["dummy_field"] = (
            ds["reflectivity_horizontal"] * 0
        )  # Adding a field with all zeros
        ds["dummy_field"].attrs = {"unit": "dBZ", "long_name": "Dummy Field"}
        return ds

    # Apply the dummy function using the xradar accessor
    ds = ds.xradar.apply(dummy_function)

    # Verify that the dummy field has been added
    assert "dummy_field" in ds.data_vars, "dummy_field not found in dataset"
    assert ds["dummy_field"].attrs["unit"] == "dBZ"
    assert ds["dummy_field"].attrs["long_name"] == "Dummy Field"


def test_xradar_dataarray_accessor_apply():
    # Fetch the sample radar file
    filename = DATASETS.fetch("sample_sgp_data.nc")

    # Open the radar file into a DataTree object, extract a Dataset, and then a DataArray
    dtree = xd.io.open_cfradial1_datatree(filename)
    ds = dtree["sweep_0"].to_dataset()  # Extracting the Dataset from one sweep
    da = ds["reflectivity_horizontal"]  # Extracting a DataArray from the Dataset

    # Define a simple function to test with the apply method
    def dummy_function(da):
        """A dummy function that adds a constant value to the data."""
        return da + 10  # Add 10 to every element in the DataArray

    # Apply the dummy function using the xradar accessor
    da_modified = da.xradar.apply(dummy_function)

    # Verify that the data was modified correctly using numpy's allclose
    assert np.allclose(
        da_modified.values, ds["reflectivity_horizontal"].values + 10, equal_nan=True
    ), "DataArray values not correctly modified"
