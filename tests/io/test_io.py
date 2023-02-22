#!/usr/bin/env python
# Copyright (c) 2022, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for `io` module."""

import numpy as np
import xarray as xr

import xradar.io
from xradar.io import (
    open_cfradial1_datatree,
    open_gamic_datatree,
    open_iris_datatree,
    open_odim_datatree,
    open_rainbow_datatree,
)
from xradar.model import (
    non_standard_sweep_dataset_vars,
    required_sweep_metadata_vars,
    sweep_dataset_vars,
)


def test_open_cfradial1_datatree(cfradial1_file):
    dtree = open_cfradial1_datatree(cfradial1_file, first_dim="time", site_coords=False)
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
    for grp in dtree.groups:
        # only iterate sweep groups
        if "sweep" not in grp:
            continue
        ds = dtree[grp].ds
        i = ds.sweep_number.values
        assert i == int(grp[7:])
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
            "range",
        }
        assert np.round(ds.sweep_fixed_angle.values.item(), 1) == elevations[i]


def test_open_cfradial1_dataset(cfradial1_file):
    # open first sweep group
    ds = xr.open_dataset(cfradial1_file, group="sweep_0", engine="cfradial1")
    assert list(ds.dims) == ["azimuth", "range"]
    assert set(ds.data_vars) & (
        sweep_dataset_vars | non_standard_sweep_dataset_vars
    ) == {"DBZ", "VR"}
    assert ds.sweep_number == 0
    # open last sweep group
    ds = xr.open_dataset(cfradial1_file, group="sweep_8", engine="cfradial1")
    assert list(ds.dims) == ["azimuth", "range"]
    assert set(ds.data_vars) & (
        sweep_dataset_vars | non_standard_sweep_dataset_vars
    ) == {"DBZ", "VR"}
    assert ds.sweep_number == 8


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
        0.9,
        1.3,
        1.8,
        2.4,
        3.1,
        4.2,
        5.6,
        7.4,
        10.0,
        13.3,
        17.9,
        23.9,
        32.0,
    ]
    azimuths = [360] * 14
    ranges = [
        1200,
        1200,
        1200,
        1200,
        1200,
        1180,
        1100,
        1100,
        590,
        480,
        360,
        280,
        200,
        200,
    ]
    for i, grp in enumerate(dtree.groups[1:]):
        ds = dtree[grp].ds
        assert dict(ds.dims) == {"azimuth": azimuths[i], "range": ranges[i]}
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
        assert ds.sweep_number.values == int(grp[7:])


def test_open_odim_dataset(odim_file):
    # open first sweep group
    ds = xr.open_dataset(odim_file, group="sweep_0", engine="odim")
    assert dict(ds.dims) == {"azimuth": 360, "range": 1200}
    assert set(ds.data_vars) & (
        sweep_dataset_vars | non_standard_sweep_dataset_vars
    ) == {"WRADH", "VRADH", "PHIDP", "DBZH", "RHOHV", "KDP", "TH", "ZDR"}
    assert ds.sweep_number == 0

    # open last sweep group
    ds = xr.open_dataset(odim_file, group="sweep_11", engine="odim")
    assert dict(ds.dims) == {"azimuth": 360, "range": 280}
    assert set(ds.data_vars) & (
        sweep_dataset_vars | non_standard_sweep_dataset_vars
    ) == {"VRADH", "KDP", "WRADH", "TH", "RHOHV", "PHIDP", "ZDR", "DBZH"}
    assert ds.sweep_number == 11

    # open last sweep group, auto
    ds = xr.open_dataset(
        odim_file,
        group="sweep_11",
        engine="odim",
        backend_kwargs=dict(first_dim="time"),
    )
    assert dict(ds.dims) == {"time": 360, "range": 280}
    assert ds.sweep_number == 11


def test_open_gamic_datatree(gamic_file):
    dtree = open_gamic_datatree(gamic_file)

    # root_attrs
    attrs = dtree.attrs
    assert attrs["Conventions"] == "None"

    # root vars
    rvars = dtree.data_vars
    assert rvars["volume_number"] == 0
    assert rvars["platform_type"] == "fixed"
    assert rvars["instrument_type"] == "radar"
    assert rvars["time_coverage_start"] == "2018-06-01T05:40:47Z"
    assert rvars["time_coverage_end"] == "2018-06-01T05:44:16Z"
    np.testing.assert_almost_equal(rvars["latitude"].values, np.array(50.9287272))
    np.testing.assert_almost_equal(rvars["longitude"].values, np.array(6.4569489))
    np.testing.assert_almost_equal(rvars["altitude"].values, np.array(310.0))

    # iterate over subgroups and check some values
    moments = [
        "WRADH",
        "WRADV",
        "VRADH",
        "VRADV",
        "PHIDP",
        "DBTH",
        "DBTV",
        "DBZH",
        "DBZV",
        "RHOHV",
        "KDP",
        "ZDR",
    ]
    elevations = [
        28.0,
        18.0,
        14.0,
        11.0,
        8.2,
        6.0,
        4.5,
        3.1,
        1.7,
        0.6,
    ]
    azimuths = [361, 361, 361, 360, 361, 360, 360, 361, 360, 360]
    ranges = [
        360,
        500,
        620,
        800,
        1050,
        1400,
        1000,
        1000,
        1000,
        1000,
    ]
    for i, grp in enumerate(dtree.groups[1:]):
        ds = dtree[grp].ds
        assert dict(ds.dims) == {"azimuth": azimuths[i], "range": ranges[i]}
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
        assert ds.sweep_number == i


def test_open_gamic_dataset(gamic_file):
    # open first sweep group
    ds = xr.open_dataset(gamic_file, group="sweep_0", engine="gamic")
    assert dict(ds.dims) == {"azimuth": 361, "range": 360}
    assert set(ds.data_vars) & (
        sweep_dataset_vars | non_standard_sweep_dataset_vars
    ) == {
        "WRADH",
        "WRADV",
        "VRADH",
        "VRADV",
        "PHIDP",
        "DBTH",
        "DBTV",
        "DBZH",
        "DBZV",
        "RHOHV",
        "KDP",
        "ZDR",
    }
    assert ds.sweep_number == 0

    # open last sweep group
    ds = xr.open_dataset(gamic_file, group="sweep_9", engine="gamic")
    assert dict(ds.dims) == {"azimuth": 360, "range": 1000}
    assert set(ds.data_vars) & (
        sweep_dataset_vars | non_standard_sweep_dataset_vars
    ) == {
        "WRADH",
        "WRADV",
        "VRADH",
        "VRADV",
        "PHIDP",
        "DBTH",
        "DBTV",
        "DBZH",
        "DBZV",
        "RHOHV",
        "KDP",
        "ZDR",
    }
    assert ds.sweep_number == 9

    # open last sweep group, auto
    ds = xr.open_dataset(
        gamic_file,
        group="sweep_9",
        engine="gamic",
        backend_kwargs=dict(first_dim="time"),
    )
    assert dict(ds.dims) == {"time": 360, "range": 1000}
    assert ds.sweep_number == 9


def test_open_gamic_dataset_reindex(gamic_file):
    # open first sweep group
    reindex_angle = dict(start_angle=0, stop_angle=360, angle_res=1.0, direction=1)
    ds = xr.open_dataset(
        gamic_file, group="sweep_0", engine="gamic", reindex_angle=reindex_angle
    )
    assert dict(ds.dims) == {"azimuth": 360, "range": 360}


def test_open_furuno_scn_dataset(furuno_scn_file):
    # open sweep group
    ds = xr.open_dataset(furuno_scn_file, first_dim="time", engine="furuno")
    assert dict(ds.dims) == {"time": 1376, "range": 602}
    assert set(ds.data_vars) & (
        sweep_dataset_vars | non_standard_sweep_dataset_vars
    ) == {"KDP", "VRADH", "ZDR", "DBZH", "WRADH", "RHOHV", "PHIDP"}
    for key, value in ds.data_vars.items():
        if key in ["RATE", "KDP", "VRADH", "ZDR", "DBZH", "WRADH", "RHOHV", "PHIDP"]:
            assert value.encoding["_FillValue"] == 0.0
        elif key in ["azimuth", "elevation"]:
            assert value.encoding["_FillValue"] == np.ma.minimum_fill_value(
                value.encoding["dtype"]
            )
        else:
            assert value.encoding.get("_FillValue", None) is None
    assert ds.sweep_number == 0

    # open sweep group, auto
    ds = xr.open_dataset(
        furuno_scn_file,
        engine="furuno",
    )
    assert dict(ds.dims) == {"azimuth": 1376, "range": 602}
    assert ds.sweep_number == 0


def test_open_furuno_scnx_dataset(furuno_scnx_file):
    # open sweep group
    ds = xr.open_dataset(furuno_scnx_file, first_dim="time", engine="furuno")
    assert dict(ds.dims) == {"time": 722, "range": 936}
    assert set(ds.data_vars) & (
        sweep_dataset_vars | non_standard_sweep_dataset_vars
    ) == {"KDP", "VRADH", "ZDR", "DBZH", "WRADH", "RHOHV", "PHIDP"}

    for key, value in ds.data_vars.items():
        if key in ["RATE", "KDP", "VRADH", "ZDR", "DBZH", "WRADH", "RHOHV", "PHIDP"]:
            assert value.encoding["_FillValue"] == 0.0
        elif key in ["azimuth", "elevation"]:
            assert value.encoding["_FillValue"] == np.ma.minimum_fill_value(
                value.encoding["dtype"]
            )
        else:
            assert value.encoding.get("_FillValue", None) is None
    assert ds.sweep_number == 0

    # open sweep group, auto
    ds = xr.open_dataset(
        furuno_scnx_file,
        engine="furuno",
    )
    assert dict(ds.dims) == {"azimuth": 722, "range": 936}
    assert ds.sweep_number == 0


def test_open_rainbow_datatree(rainbow_file):
    dtree = open_rainbow_datatree(rainbow_file)

    # root_attrs
    attrs = dtree.attrs
    assert attrs["Conventions"] == "None"

    # root vars
    rvars = dtree.data_vars
    assert rvars["volume_number"] == 0
    assert rvars["platform_type"] == "fixed"
    assert rvars["instrument_type"] == "radar"
    assert rvars["time_coverage_start"] == "2013-05-10T00:00:06Z"
    assert rvars["time_coverage_end"] == "2013-05-10T00:03:14Z"
    np.testing.assert_almost_equal(rvars["latitude"].values, np.array(50.856633))
    np.testing.assert_almost_equal(rvars["longitude"].values, np.array(6.379967))
    np.testing.assert_almost_equal(rvars["altitude"].values, np.array(116.7))

    # iterate over subgroups and check some values
    moments = [
        "DBZH",
    ]
    elevations = [
        0.6,
        1.4,
        2.4,
        3.5,
        4.8,
        6.3,
        8.0,
        9.9,
        12.2,
        14.8,
        17.9,
        21.3,
        25.4,
        30.0,
    ]
    azimuths = [361] * 14
    ranges = [400] * 14
    for i, grp in enumerate(dtree.groups[1:]):
        ds = dtree[grp].ds
        assert dict(ds.dims) == {"azimuth": azimuths[i], "range": ranges[i]}
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
        assert ds.sweep_number == i


def test_open_rainbow_dataset(rainbow_file):
    # open first sweep group
    ds = xr.open_dataset(rainbow_file, group="sweep_0", engine="rainbow")
    assert dict(ds.dims) == {"azimuth": 361, "range": 400}
    assert set(ds.data_vars) & (
        sweep_dataset_vars | non_standard_sweep_dataset_vars
    ) == {
        "DBZH",
    }
    assert ds.sweep_number == 0

    # open last sweep group
    ds = xr.open_dataset(rainbow_file, group="sweep_13", engine="rainbow")
    assert dict(ds.dims) == {"azimuth": 361, "range": 400}
    assert set(ds.data_vars) & (
        sweep_dataset_vars | non_standard_sweep_dataset_vars
    ) == {
        "DBZH",
    }
    assert ds.sweep_number == 13

    # open last sweep group, auto
    ds = xr.open_dataset(
        rainbow_file,
        group="sweep_13",
        engine="rainbow",
        backend_kwargs=dict(first_dim="time"),
    )
    assert dict(ds.dims) == {"time": 361, "range": 400}
    assert ds.sweep_number == 13


def test_open_iris_datatree(iris0_file):
    dtree = open_iris_datatree(iris0_file)

    # root_attrs
    attrs = dtree.attrs
    assert attrs["Conventions"] == "None"

    # root vars
    rvars = dtree.data_vars
    assert rvars["volume_number"] == 0
    assert rvars["platform_type"] == "fixed"
    assert rvars["instrument_type"] == "radar"
    assert rvars["time_coverage_start"] == "2013-11-25T10:55:04Z"
    assert rvars["time_coverage_end"] == "2013-11-25T10:59:24Z"
    np.testing.assert_almost_equal(rvars["latitude"].values, np.array(9.331))
    np.testing.assert_almost_equal(rvars["longitude"].values, np.array(-75.2829999))
    np.testing.assert_almost_equal(rvars["altitude"].values, np.array(143.0))

    # iterate over subgroups and check some values
    moments = [
        "ZDR",
        "RHOHV",
        "DBZH",
        "PHIDP",
        "KDP",
        "VRADH",
    ]
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
    azimuths = [360] * 10
    ranges = [664] * 10
    for i, grp in enumerate(dtree.groups[1:]):
        ds = dtree[grp].ds
        assert dict(ds.dims) == {"azimuth": azimuths[i], "range": ranges[i]}
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
        assert ds.sweep_number == i


def test_open_iris0_dataset(iris0_file):
    # open first sweep group
    ds = xr.open_dataset(iris0_file, group="sweep_0", engine="iris")
    assert dict(ds.dims) == {"azimuth": 360, "range": 664}
    assert set(ds.data_vars) & (
        sweep_dataset_vars | non_standard_sweep_dataset_vars
    ) == {
        "DBZH",
        "VRADH",
        "KDP",
        "RHOHV",
        "PHIDP",
        "ZDR",
    }
    assert ds.sweep_number == 0

    # open last sweep group
    ds = xr.open_dataset(iris0_file, group="sweep_9", engine="iris")
    assert dict(ds.dims) == {"azimuth": 360, "range": 664}
    assert set(ds.data_vars) & (
        sweep_dataset_vars | non_standard_sweep_dataset_vars
    ) == {
        "DBZH",
        "VRADH",
        "KDP",
        "RHOHV",
        "PHIDP",
        "ZDR",
    }
    assert ds.sweep_number == 9

    # open last sweep group, auto
    ds = xr.open_dataset(
        iris0_file,
        group="sweep_9",
        engine="iris",
        backend_kwargs=dict(first_dim="time"),
    )
    assert dict(ds.dims) == {"time": 360, "range": 664}
    assert ds.sweep_number == 9


def test_open_iris1_dataset(iris1_file):
    # open first and only sweep group
    ds = xr.open_dataset(iris1_file, group="sweep_0", engine="iris")
    assert dict(ds.dims) == {"azimuth": 359, "range": 833}
    assert set(ds.data_vars) & (
        sweep_dataset_vars | non_standard_sweep_dataset_vars
    ) == {
        "DBZH",
        "KDP",
        "DBTH",
        "PHIDP",
        "ZDR",
        "RHOHV",
        "VRADH",
        "WRADH",
    }
    assert ds.sweep_number == 0

    # open first and only sweep group
    ds = xr.open_dataset(
        iris1_file,
        group="sweep_0",
        engine="iris",
        backend_kwargs=dict(first_dim="time"),
    )
    assert dict(ds.dims) == {"time": 359, "range": 833}
    assert ds.sweep_number == 0


def test_odim_roundtrip(odim_file2):
    dtree = open_odim_datatree(odim_file2)
    outfile = "odim_out.h5"
    xradar.io.to_odim(dtree, outfile)
    dtree2 = open_odim_datatree(outfile, reindex_angle=False)
    for d0, d1 in zip(dtree.groups, dtree2.groups):
        print(d0, d1)
        xr.testing.assert_equal(dtree[d0].ds, dtree2[d1].ds)
