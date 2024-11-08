#!/usr/bin/env python
# Copyright (c) 2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

import pytest
import xarray as xr
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
    with xd.io.open_cfradial1_datatree(cfradial1_file) as dtree:

        # Convert to CfRadial1 dataset first
        ds_cf1 = xd.transform.to_cfradial1(dtree)

        # Call the conversion back to CfRadial2
        dtree_cf2 = xd.transform.to_cfradial2(ds_cf1)

        # Verify key attributes and data structures in the resulting datatree
        assert isinstance(dtree_cf2, xr.DataTree), "Output is not a valid DataTree"
        assert "radar_parameters" in dtree_cf2, "Missing radar_parameters in DataTree"
        assert dtree_cf2.attrs == ds_cf1.attrs, "Attributes mismatch between formats"


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
        assert dtree_cf2.attrs == ds_cf1.attrs, "Attributes mismatch between formats"


def test_to_cfradial1_error_with_different_range_bin_sizes(gamic_file):
    with xd.io.open_gamic_datatree(gamic_file) as dtree:
        with pytest.raises(MergeError):
            xd.transform.to_cfradial1(dtree)
