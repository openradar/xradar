#!/usr/bin/env python
# Copyright (c) 2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

import pytest
import xarray as xr
from open_radar_data import DATASETS
from xarray import MergeError

import xradar as xd


def test_to_cfradial1(cfradial1_file):
    """Test the conversion from DataTree to CfRadial1 format."""
    with xd.io.open_cfradial1_datatree(cfradial1_file) as dtree:
        # Call the conversion function
        ds_cf1 = xd.transform.to_cfradial1(dtree)

        # Verify key attributes and data structures in the resulting dataset
        assert isinstance(ds_cf1, xr.Dataset), "Output is not a valid xarray Dataset"
        assert (
            "Conventions" in ds_cf1.attrs and ds_cf1.attrs["Conventions"] == "Cf/Radial"
        )
        assert (
            "sweep_mode" in ds_cf1.variables
        ), "Missing sweep_mode in converted dataset"
        assert ds_cf1.attrs["version"] == "1.2", "Incorrect CfRadial version"


def test_to_cfradial2(cfradial1_file):
    """Test the conversion from CfRadial1 to CfRadial2 DataTree format."""
    with xd.io.open_cfradial1_datatree(cfradial1_file, optional_groups=True) as dtree:
        # Convert to CfRadial1 dataset first
        ds_cf1 = xd.transform.to_cfradial1(dtree)

        # Call the conversion back to CfRadial2
        dtree_cf2 = xd.transform.to_cfradial2(ds_cf1)

        # Verify key attributes and data structures in the resulting datatree
        assert isinstance(dtree_cf2, xr.DataTree), "Output is not a valid DataTree"
        assert "radar_parameters" in dtree_cf2, "Missing radar_parameters in DataTree"
        assert (
            "georeferencing_correction" in dtree_cf2
        ), "Missing georeferencing_correction in DataTree"
        # Round-tripped attrs should be a subset of the original — backend-specific
        # attrs (e.g. NEXRAD ICD metadata) are not preserved by the generic reader.
        for key, val in dtree_cf2.attrs.items():
            assert ds_cf1.attrs[key] == val, f"Attr {key!r} mismatch after round-trip"


def test_to_cfradial2_root_only_dataset_reopens_source(cfradial1_file):
    ds = xr.open_dataset(cfradial1_file, engine="cfradial1")

    dtree = xd.transform.to_cfradial2(ds)

    assert isinstance(dtree, xr.DataTree)
    assert "georeferencing_correction" in dtree
    assert "sweep_0" in dtree


def test_to_cfradial2_root_only_dataset_without_source_errors(cfradial1_file):
    ds = xr.open_dataset(cfradial1_file, engine="cfradial1").copy()
    ds.encoding = {}

    with pytest.raises(ValueError, match="does not retain a readable"):
        xd.transform.to_cfradial2(ds)


def test_to_cfradial1_with_different_range_shapes(nexradlevel2_bzfile):
    with xd.io.open_nexradlevel2_datatree(nexradlevel2_bzfile) as dtree:
        ds_cf1 = xd.transform.to_cfradial1(dtree)
        # Verify key attributes and data structures in the resulting dataset
        assert isinstance(ds_cf1, xr.Dataset), "Output is not a valid xarray Dataset"
        assert (
            "Conventions" in ds_cf1.attrs and ds_cf1.attrs["Conventions"] == "Cf/Radial"
        )
        assert (
            "sweep_mode" in ds_cf1.variables
        ), "Missing sweep_mode in converted dataset"
        assert ds_cf1.attrs["version"] == "1.2", "Incorrect CfRadial version"
        assert ds_cf1.sizes.mapping == {"time": 5400, "range": 1832, "sweep": 11}

        # Call the conversion back to CfRadial2
        dtree_cf2 = xd.transform.to_cfradial2(ds_cf1)
        # Verify key attributes and data structures in the resulting datatree
        assert isinstance(dtree_cf2, xr.DataTree), "Output is not a valid DataTree"
        # todo: this needs to be fixed in nexrad level2reader
        # assert "radar_parameters" in dtree_cf2, "Missing radar_parameters in DataTree"
        # Round-tripped attrs should be a subset of the original — backend-specific
        # attrs (e.g. NEXRAD ICD metadata) are not preserved by the generic reader.
        for key, val in dtree_cf2.attrs.items():
            assert ds_cf1.attrs[key] == val, f"Attr {key!r} mismatch after round-trip"


def test_to_cfradial1_error_with_different_range_bin_sizes(gamic_file):
    with xd.io.open_gamic_datatree(gamic_file) as dtree:
        with pytest.raises(MergeError):
            xd.transform.to_cfradial1(dtree)


def test_to_cfradial1_after_map_over_sweeps_where(tmp_path):
    file = DATASETS.fetch("swx_20120520_0641.nc")
    dtree = xd.io.open_cfradial1_datatree(file)

    def filter_radar(
        ds,
        vel_name="mean_doppler_velocity",
        ref_name="corrected_reflectivity_horizontal",
    ):
        vel_texture = ds[vel_name].rolling(range=30, min_periods=2, center=True).std()
        ds = ds.assign(velocity_texture=vel_texture)
        return ds.where(
            (ds.velocity_texture < 10) & ((ds[ref_name] >= -10) & (ds[ref_name] <= 75))
        )

    filtered = dtree.xradar.map_over_sweeps(filter_radar)
    ds_cf1 = xd.transform.to_cfradial1(filtered)

    assert ds_cf1["sweep_number"].dims == ("sweep",)
    assert ds_cf1["fixed_angle"].dims == ("sweep",)
    assert ds_cf1["sweep_mode"].dims == ("sweep",)

    outfile = tmp_path / "filtered_cfradial1.nc"
    xd.io.to_cfradial1(filtered, outfile)
    reopened = xd.io.open_cfradial1_datatree(outfile)
    assert "sweep_0" in reopened
