#!/usr/bin/env python
# Copyright (c) 2022-2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

import numpy as np
import xarray as xr
from numpy.testing import assert_almost_equal, assert_raises
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
    tree = xr.DataTree.from_dict({"sweep_0": radar})
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


def test_crs_dataarray():
    """Test the add_crs and get_crs methods on a DataArray."""
    # Create a sample DataArray with radar data
    radar = xd.model.create_sweep_dataset()
    radar["sample_field"] = radar.azimuth + radar.range
    dataarray = radar.sample_field

    # Apply add_crs to add coordinate system
    da_with_crs = dataarray.xradar.add_crs()

    # Check if either 'spatial_ref' or 'crs_wkt' was added as coordinate
    assert (
        "spatial_ref" in da_with_crs.coords or "crs_wkt" in da_with_crs.coords
    ), "CRS coordinate is missing in DataArray"

    # Retrieve CRS using get_crs
    crs = da_with_crs.xradar.get_crs()

    # Verify that the CRS is valid and projected
    assert crs is not None, "CRS could not be retrieved from DataArray"
    assert crs.is_projected, "Expected a projected CRS in DataArray"


def test_crs_dataset():
    """Test the add_crs and get_crs methods on a Dataset."""
    # Create a sample Dataset with radar data
    radar = xd.model.create_sweep_dataset()

    # Apply add_crs to add coordinate system
    ds_with_crs = radar.xradar.add_crs()

    # Check if either 'spatial_ref' or 'crs_wkt' was added as coordinate
    assert (
        "spatial_ref" in ds_with_crs.coords or "crs_wkt" in ds_with_crs.coords
    ), "CRS coordinate is missing in Dataset"

    # Retrieve CRS using get_crs
    crs = ds_with_crs.xradar.get_crs()

    # Verify that the CRS is valid and projected
    assert crs is not None, "CRS could not be retrieved from Dataset"
    assert crs.is_projected, "Expected a projected CRS in Dataset"


def test_crs_datatree():
    """Test the add_crs method on a DataTree."""
    # Create a sample DataTree with radar data
    radar = xd.model.create_sweep_dataset()
    tree = xr.DataTree.from_dict({"sweep_0": radar})

    # Apply add_crs to add coordinate system
    tree_with_crs = tree.xradar.add_crs()

    # Check if either 'spatial_ref' or 'crs_wkt' was added as coordinate in each sweep
    assert (
        "spatial_ref" in tree_with_crs["sweep_0"].coords
        or "crs_wkt" in tree_with_crs["sweep_0"].coords
    ), "CRS coordinate is missing in DataTree"


def test_map_over_sweeps_apply_dummy_function():
    """
    Test applying a dummy function to all sweep nodes using map_over_sweeps.
    """
    # Fetch the sample radar file
    filename = DATASETS.fetch("sample_sgp_data.nc")

    # Open the radar file into a DataTree object
    dtree = xd.io.open_cfradial1_datatree(filename)

    # Define a simple dummy function that adds a constant field to the dataset
    def dummy_function(ds):
        ds = ds.assign(
            dummy_field=ds["reflectivity_horizontal"] * 0
        )  # Field with zeros
        ds["dummy_field"].attrs = {"unit": "dBZ", "long_name": "Dummy Field"}
        return ds

    # Apply using map_over_sweeps accessor
    dtree_modified = dtree.xradar.map_over_sweeps(dummy_function)

    # Check that the new field exists in sweep_0 and has the correct attributes
    sweep_0 = dtree_modified["sweep_0"]
    assert "dummy_field" in sweep_0.data_vars
    assert sweep_0.dummy_field.attrs["unit"] == "dBZ"
    assert sweep_0.dummy_field.attrs["long_name"] == "Dummy Field"

    # Ensure all non-NaN values are 0 (accounting for -0.0 and NaN values)
    non_nan_values = np.nan_to_num(
        sweep_0.dummy_field.values
    )  # Convert NaNs to zero for comparison
    assert np.all(np.isclose(non_nan_values, 0))


def test_map_over_sweeps_non_sweep_nodes():
    """
    Test that non-sweep nodes remain unchanged when using map_over_sweeps.
    """
    # Fetch the sample radar file
    filename = DATASETS.fetch("sample_sgp_data.nc")

    # Open the radar file into a DataTree object and add a non-sweep node
    dtree = xd.io.open_cfradial1_datatree(filename)
    non_sweep_data = xr.Dataset({"non_sweep_data": ("dim", np.arange(10))})
    dtree["non_sweep_node"] = non_sweep_data

    # Define a simple function that only modifies sweep nodes
    def dummy_function(ds):
        if "range" in ds.dims:
            ds = ds.assign(
                dummy_field=ds["reflectivity_horizontal"] * 0
            )  # Field with zeros
        return ds

    # Apply using map_over_sweeps
    dtree_modified = dtree.xradar.map_over_sweeps(dummy_function)

    # Check that non-sweep nodes remain unchanged
    assert "non_sweep_data" in dtree_modified["non_sweep_node"].data_vars
    assert "dummy_field" not in dtree_modified["non_sweep_node"].data_vars
    assert "dummy_field" in dtree_modified["sweep_0"].data_vars


def test_map_over_sweeps_invalid_input():
    """
    Test map_over_sweeps with invalid input to ensure appropriate error handling.
    """
    radar = xd.model.create_sweep_dataset()
    tree = xr.DataTree.from_dict({"sweep_0": radar})

    # Define a function that raises an error for invalid input
    def invalid_rain_rate(ds, ref_field="INVALID_FIELD"):
        return ds[ref_field]  # This will raise a KeyError if the field is not present

    # Expect a KeyError when applying the function with an invalid field
    with assert_raises(KeyError):
        tree.xradar.map_over_sweeps(invalid_rain_rate, ref_field="INVALID_FIELD")


def test_accessor_to_cfradial1():
    """Test the accessor function to convert DataTree to CfRadial1 Dataset."""
    file = DATASETS.fetch("cfrad.20080604_002217_000_SPOL_v36_SUR.nc")
    dtree = xd.io.open_cfradial1_datatree(file)

    # Use accessor method to convert to CfRadial1
    ds_cf1 = dtree.xradar.to_cfradial1_dataset()

    # Test alias for conversion to CfRadial1
    ds_cf1_alias = dtree.xradar.to_cf1()

    # Verify key properties of the resulting dataset
    assert isinstance(ds_cf1, xr.Dataset), "Conversion to CfRadial1 failed"
    assert "sweep_mode" in ds_cf1.variables, "Missing sweep_mode in CfRadial1 dataset"

    # Verify alias
    assert isinstance(ds_cf1_alias, xr.Dataset), "Alias conversion to CfRadial1 failed"
    assert (
        "sweep_mode" in ds_cf1_alias.variables
    ), "Missing sweep_mode in CfRadial1 dataset"


def test_accessor_to_cfradial2():
    """Test the accessor function to convert CfRadial1 Dataset back to DataTree."""
    file = DATASETS.fetch("cfrad.20080604_002217_000_SPOL_v36_SUR.nc")
    dtree = xd.io.open_cfradial1_datatree(file)

    # Convert to CfRadial1 dataset
    ds_cf1 = dtree.xradar.to_cfradial1_dataset()

    # Use accessor method to convert back to CfRadial2 DataTree
    dtree_cf2 = ds_cf1.xradar.to_cfradial2_datatree()

    # Test aliases for CfRadial2 conversion
    dtree_cf2_alias1 = ds_cf1.xradar.to_cfradial2()
    dtree_cf2_alias2 = ds_cf1.xradar.to_cf2()

    # Verify the properties of the resulting DataTree
    assert isinstance(dtree_cf2, xr.DataTree), "Conversion to CfRadial2 failed"
    assert (
        "radar_parameters" in dtree_cf2
    ), "Missing radar_parameters in CfRadial2 DataTree"

    # Verify alias1
    assert isinstance(
        dtree_cf2_alias1, xr.DataTree
    ), "Alias conversion to CfRadial2 failed"
    assert (
        "radar_parameters" in dtree_cf2_alias1
    ), "Missing radar_parameters in CfRadial2 DataTree"

    # Verify alias2
    assert isinstance(
        dtree_cf2_alias2, xr.DataTree
    ), "Alias conversion to CfRadial2 failed"
    assert (
        "radar_parameters" in dtree_cf2_alias2
    ), "Missing radar_parameters in CfRadial2 DataTree"
