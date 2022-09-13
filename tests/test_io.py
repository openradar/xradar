#!/usr/bin/env python
# Copyright (c) 2022, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for `io` module."""

import numpy as np
import xarray as xr

from xradar.io import open_cfradial1_datatree, open_odim_datatree
from xradar.model import (
    non_standard_sweep_dataset_vars,
    required_sweep_metadata_vars,
    sweep_dataset_vars,
)


def test_open_cfradial1_datatree(cfradial1_file):
    dtree = open_cfradial1_datatree(cfradial1_file)
    attrs = dtree.attrs

    # root_attrs
    assert (
        attrs["Conventions"]
        == "CF/Radial instrument_parameters radar_parameters radar_calibration geometry_correction"
    )
    assert attrs["version"] == "1.2"
    assert attrs["title"] == "TIMREX"
    assert attrs["instrument_name"] == "SPOLRVP8"
    assert attrs["platform_is_mobile"] == "false"

    # root vars
    rvars = dtree.data_vars
    assert rvars["volume_number"] == 36
    assert rvars["platform_type"] == b"fixed"
    assert rvars["instrument_type"] == b"radar"
    assert rvars["time_coverage_start"] == b"2008-06-04T00:15:03Z"
    assert rvars["time_coverage_end"] == b"2008-06-04T00:22:17Z"
    np.testing.assert_almost_equal(rvars["latitude"].values, np.array(22.526699))
    np.testing.assert_almost_equal(rvars["longitude"].values, np.array(120.4335022))
    np.testing.assert_almost_equal(rvars["altitude"].values, np.array(45.0000018))

    # iterate over subgroups and check some values
    moments = ["DBZ", "VR"]
    elevations = [0.5, 1.1, 1.8, 2.6, 3.6, 4.7, 6.5, 9.1, 12.8]
    azimuths = [483, 483, 482, 483, 481, 482, 482, 484, 483]
    ranges = [996, 996, 996, 996, 996, 996, 996, 996, 996]
    for i, grp in enumerate(dtree.groups[1:]):
        if grp == "/":
            continue
        # print(grp)
        # print(dtree[grp])
        ds = dtree[grp].ds
        assert dict(ds.dims) == {"time": azimuths[i], "range": ranges[i]}
        assert set(ds.data_vars) & (
            sweep_dataset_vars | non_standard_sweep_dataset_vars
        ) == set(moments)
        assert set(ds.data_vars) & (required_sweep_metadata_vars) == set(
            required_sweep_metadata_vars ^ {"azimuth", "elevation"}
        )
        assert set(ds.coords) == {
            "azimuth",
            "elevation",
            "time",
            "latitude",
            "longitude",
            "altitude",
            "range",
        }
        assert np.round(ds.elevation.mean().values.item(), 1) == elevations[i]


def test_open_cfradial1_dataset(cfradial1_file):
    # open first sweep group
    ds = xr.open_dataset(cfradial1_file, group="sweep_0", engine="cfradial1")
    assert dict(ds.dims) == {"time": 483, "range": 996}
    assert set(ds.data_vars) & (
        sweep_dataset_vars | non_standard_sweep_dataset_vars
    ) == {"DBZ", "VR"}

    # open last sweep group
    ds = xr.open_dataset(cfradial1_file, group="sweep_8", engine="cfradial1")
    assert dict(ds.dims) == {"time": 483, "range": 996}
    assert set(ds.data_vars) & (
        sweep_dataset_vars | non_standard_sweep_dataset_vars
    ) == {"DBZ", "VR"}


def test_open_odim_datatree(odim_file):
    dtree = open_odim_datatree(odim_file)

    # root_attrs
    attrs = dtree.attrs
    assert attrs["Conventions"] == "ODIM_H5/V2_2"

    # root vars
    rvars = dtree.data_vars
    assert rvars["volume_number"] == 0
    assert rvars["platform_type"] == "fixed"
    assert rvars["instrument_type"] == "radar"
    assert rvars["time_coverage_start"] == "2018-12-20T06:06:28Z"
    assert rvars["time_coverage_end"] == "2018-12-20T06:10:42Z"
    np.testing.assert_almost_equal(rvars["latitude"].values, np.array(-33.7008018))
    np.testing.assert_almost_equal(rvars["longitude"].values, np.array(151.2089996))
    np.testing.assert_almost_equal(rvars["altitude"].values, np.array(195.0))

    # iterate over subgroups and check some values
    moments = ["PHIDP", "VRADH", "DBZH", "TH", "ZDR", "RHOHV", "WRADH", "KDP"]
    elevations = [
        0.5,
        10.0,
        13.3,
        17.9,
        23.9,
        32.0,
        0.9,
        1.3,
        1.8,
        2.4,
        3.1,
        4.2,
        5.6,
        7.4,
    ]
    azimuths = [360] * 14
    ranges = [
        1200,
        480,
        360,
        280,
        200,
        200,
        1200,
        1200,
        1200,
        1200,
        1180,
        1100,
        1100,
        590,
    ]
    for i, grp in enumerate(dtree.groups[1:]):
        ds = dtree[grp].ds
        assert dict(ds.dims) == {"time": azimuths[i], "range": ranges[i]}
        assert set(ds.data_vars) & (
            sweep_dataset_vars | non_standard_sweep_dataset_vars
        ) == set(moments)
        assert set(ds.data_vars) & (required_sweep_metadata_vars) == set(
            required_sweep_metadata_vars ^ {"azimuth", "elevation"}
        )
        assert set(ds.coords) == {
            "azimuth",
            "elevation",
            "time",
            "latitude",
            "longitude",
            "altitude",
            "range",
        }
        assert np.round(ds.elevation.mean().values.item(), 1) == elevations[i]


def test_open_odim_dataset(odim_file):
    # open first sweep group
    ds = xr.open_dataset(odim_file, group="dataset1", engine="odim")
    assert dict(ds.dims) == {"time": 360, "range": 1200}
    assert set(ds.data_vars) & (
        sweep_dataset_vars | non_standard_sweep_dataset_vars
    ) == {"WRADH", "VRADH", "PHIDP", "DBZH", "RHOHV", "KDP", "TH", "ZDR"}

    # open last sweep group
    ds = xr.open_dataset(odim_file, group="dataset12", engine="odim")
    assert dict(ds.dims) == {"time": 360, "range": 280}
    assert set(ds.data_vars) & (
        sweep_dataset_vars | non_standard_sweep_dataset_vars
    ) == {"VRADH", "KDP", "WRADH", "TH", "RHOHV", "PHIDP", "ZDR", "DBZH"}

    # open last sweep group, auto
    ds = xr.open_dataset(
        odim_file,
        group="dataset12",
        engine="odim",
        backend_kwargs=dict(first_dim="auto"),
    )
    assert dict(ds.dims) == {"azimuth": 360, "range": 280}
