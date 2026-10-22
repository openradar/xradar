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


def test_georeference_target_crs_dataset():
    radar = xd.model.create_sweep_dataset()
    geo = radar.xradar.georeference(target_crs=4326)
    crs = geo.xradar.get_crs()
    assert crs.is_geographic
    assert geo.x.attrs["standard_name"] == "longitude"
    assert geo.y.attrs["standard_name"] == "latitude"
    assert_almost_equal(geo.x.values[0, 0], radar.longitude.values, decimal=3)
    assert_almost_equal(geo.y.values[0, 0], radar.latitude.values, decimal=3)


def test_georeference_target_crs_datatree():
    radar = xd.model.create_sweep_dataset()
    tree = xr.DataTree.from_dict({"sweep_0": radar})
    geo = tree.xradar.georeference(target_crs=4326)["sweep_0"]
    assert geo["x"].attrs["standard_name"] == "longitude"
    assert geo["y"].attrs["standard_name"] == "latitude"
    assert_almost_equal(geo["x"].values[0, 0], radar.longitude.values, decimal=3)
    assert_almost_equal(geo["y"].values[0, 0], radar.latitude.values, decimal=3)


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


def test_map_over_sweeps_apply_dummy_function(cfradial1_sgp_file, closing):
    """
    Test applying a dummy function to all sweep nodes using map_over_sweeps.
    """
    # Open the radar file into a DataTree object
    dtree = closing(xd.io.open_cfradial1_datatree(cfradial1_sgp_file))

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


def test_map_over_sweeps_non_sweep_nodes(cfradial1_sgp_file, closing):
    """
    Test that non-sweep nodes remain unchanged when using map_over_sweeps.
    """
    # Open the radar file into a DataTree object and add a non-sweep node
    dtree = closing(xd.io.open_cfradial1_datatree(cfradial1_sgp_file))
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


def test_accessor_to_cfradial1(closing):
    """Test the accessor function to convert DataTree to CfRadial1 Dataset."""
    file = DATASETS.fetch("cfrad.20080604_002217_000_SPOL_v36_SUR.nc")
    dtree = closing(xd.io.open_cfradial1_datatree(file))

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


def test_accessor_to_cfradial2(closing):
    """Test the accessor function to convert CfRadial1 Dataset back to DataTree."""
    file = DATASETS.fetch("cfrad.20080604_002217_000_SPOL_v36_SUR.nc")
    dtree = closing(xd.io.open_cfradial1_datatree(file, optional_groups=True))

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


def test_create_xradar_dataarray_accessor():
    da = xr.DataArray(np.arange(4.0), dims="x")
    xd.accessors.create_xradar_dataarray_accessor(
        "xradar_test", {"double": lambda obj: obj * 2, "total": lambda obj: obj.sum()}
    )
    xr.testing.assert_equal(da.xradar_test.double(), da * 2)
    assert float(da.xradar_test.total()) == 6.0


def _jittered_sweep(azimuth, range_start, ngates=6, res=250.0):
    # a small PPI whose angles and ranges differ slightly between sweeps
    rng = (range_start + np.arange(ngates) * res).astype("float32")
    data = np.arange(4 * ngates, dtype="float32").reshape(4, ngates)
    return xr.Dataset(
        {"DBZH": (("azimuth", "range"), data, {"_FillValue": np.float32(np.nan)})},
        coords={
            "azimuth": ("azimuth", np.array(azimuth, dtype="float32")),
            "elevation": ("azimuth", np.full(4, 0.5, dtype="float32")),
            "range": ("range", rng, {"units": "meters"}),
        },
    )


def test_accessor_reindex_dataset():
    ds = _jittered_sweep([44.0, 136.0, 224.0, 316.0], 125.4)
    # the same as the functions
    angle = dict(start_angle=0, stop_angle=360, angle_res=90.0, direction=1)
    rng = dict(start_range=125.0, stop_range=1375.0, range_res=250.0)
    xr.testing.assert_identical(
        ds.xradar.reindex_angle(**angle), xd.util.reindex_angle(ds, **angle)
    )
    xr.testing.assert_identical(
        ds.xradar.reindex_range(**rng), xd.util.reindex_range(ds, **rng)
    )
    out = ds.xradar.reindex_angle(**angle).xradar.reindex_range(**rng)
    np.testing.assert_array_equal(out.azimuth, [45.0, 135.0, 225.0, 315.0])
    np.testing.assert_array_equal(out.range, 125.0 + np.arange(6) * 250.0)


def test_accessor_reindex_datatree():
    sweeps = {
        "/sweep_0": _jittered_sweep([44.0, 136.0, 224.0, 316.0], 125.4),
        "/sweep_1": _jittered_sweep([46.0, 134.0, 226.0, 314.0], 124.6),
    }
    dtree = xr.DataTree.from_dict({"/": xr.Dataset(attrs={"title": "t"}), **sweeps})
    angle = dict(start_angle=0, stop_angle=360, angle_res=90.0, direction=1)
    rng = dict(start_range=125.0, stop_range=1375.0, range_res=250.0)
    out = dtree.xradar.reindex_angle(**angle).xradar.reindex_range(**rng)
    # every sweep is on the same grid, so that the sweeps can be stacked
    for name in ("sweep_0", "sweep_1"):
        np.testing.assert_array_equal(out[name].azimuth, [45.0, 135.0, 225.0, 315.0])
        np.testing.assert_array_equal(out[name].range, 125.0 + np.arange(6) * 250.0)
    stacked = xr.concat(
        [out["sweep_0"].to_dataset(), out["sweep_1"].to_dataset()], dim="volume_time"
    )
    assert stacked.sizes == {"volume_time": 2, "azimuth": 4, "range": 6}
    assert not np.isnan(stacked.DBZH).any()
    # the other nodes are not touched, the input is not changed
    assert out.attrs["title"] == "t"
    assert dtree["sweep_0"].azimuth[0] == 44.0
    # same result as the function on a sweep
    expected = xd.util.reindex_range(sweeps["/sweep_0"], **rng)
    xr.testing.assert_equal(
        out["sweep_0"].to_dataset().xradar.reindex_range(**rng).DBZH,
        expected.pipe(xd.util.reindex_angle, **angle).DBZH,
    )
