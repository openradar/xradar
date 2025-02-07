#!/usr/bin/env python
# Copyright (c) 2022-2025, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for `io` module."""

import io
import tempfile

import fsspec
import h5py
import numpy as np
import pytest
import xarray as xr

import xradar.io
from tests import skip_import
from xradar.io import (
    open_cfradial1_datatree,
    open_datamet_datatree,
    open_gamic_datatree,
    open_iris_datatree,
    open_nexradlevel2_datatree,
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
        assert dict(ds.sizes) == {"time": azimuths[i], "range": ranges[i]}
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
        assert ds.sweep_mode == "azimuth_surveillance"


def test_open_cfradial1_dataset(cfradial1_file):
    # open first sweep group
    with xr.open_dataset(cfradial1_file, group="sweep_0", engine="cfradial1") as ds:
        assert list(ds.dims) == ["azimuth", "range"]
        assert set(ds.data_vars) & (
            sweep_dataset_vars | non_standard_sweep_dataset_vars
        ) == {"DBZ", "VR"}
        assert ds.sweep_number == 0

    # open last sweep group
    with xr.open_dataset(cfradial1_file, group="sweep_8", engine="cfradial1") as ds:
        assert list(ds.dims) == ["azimuth", "range"]
        assert set(ds.data_vars) & (
            sweep_dataset_vars | non_standard_sweep_dataset_vars
        ) == {"DBZ", "VR"}
        assert ds.sweep_number == 8


@pytest.mark.parametrize("sweep", ["sweep_0", 0, [0, 1], ["sweep_0", "sweep_1"]])
def test_open_odim_datatree_sweep(odim_file, sweep):
    dtree = open_odim_datatree(odim_file, sweep=sweep)
    if isinstance(sweep, (str, int)):
        lswp = len([sweep])
    else:
        lswp = len(sweep)
    assert len(dtree.match("sweep_*")) == lswp


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
    for i, grp in enumerate(dtree.match("sweep_*")):
        ds = dtree[grp].ds
        assert dict(ds.sizes) == {"azimuth": azimuths[i], "range": ranges[i]}
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
        assert ds.sweep_number.values == int(grp[6:])


@pytest.mark.parametrize("first_dim", ["auto", "time"])
@pytest.mark.parametrize("fix_second_angle", [False, True])
def test_open_odim_dataset(odim_file, first_dim, fix_second_angle):
    # open first sweep group
    with xr.open_dataset(
        odim_file,
        group="sweep_0",
        engine="odim",
        first_dim=first_dim,
        fix_second_angle=fix_second_angle,
    ) as ds:
        dim0 = "time" if first_dim == "time" else "azimuth"
        assert dict(ds.sizes) == {dim0: 360, "range": 1200}
        assert set(ds.data_vars) & (
            sweep_dataset_vars | non_standard_sweep_dataset_vars
        ) == {"WRADH", "VRADH", "PHIDP", "DBZH", "RHOHV", "KDP", "TH", "ZDR"}
        assert ds.sweep_number == 0

    # open last sweep group
    with xr.open_dataset(
        odim_file,
        group="sweep_11",
        engine="odim",
        first_dim=first_dim,
        fix_second_angle=fix_second_angle,
    ) as ds:
        assert dict(ds.sizes) == {dim0: 360, "range": 280}
        assert set(ds.data_vars) & (
            sweep_dataset_vars | non_standard_sweep_dataset_vars
        ) == {"VRADH", "KDP", "WRADH", "TH", "RHOHV", "PHIDP", "ZDR", "DBZH"}
        assert ds.sweep_number == 11


def test_open_odim_dataset_stream(odim_file):
    with open(odim_file, mode="rb") as fhandle:
        contents = io.BytesIO(fhandle.read())
        with xr.open_dataset(contents, group="sweep_0", engine="odim") as ds:
            assert isinstance(ds, xr.Dataset)


def test_open_odim_dataset_fsspec(odim_file):
    with fsspec.open(odim_file, mode="rb") as fhandle:
        with xr.open_dataset(fhandle, group="sweep_0", engine="odim") as ds:
            assert isinstance(ds, xr.Dataset)


def test_open_odim_store(odim_file):
    store = xradar.io.backends.odim.OdimStore.open(odim_file, group="sweep_0")
    assert store.substore[0].root.a1gate == 86
    assert store.substore[0].root.site_coords == (
        151.20899963378906,
        -33.700801849365234,
        195.0,
    )
    assert store.substore[0].root._get_site_coords() == (
        151.20899963378906,
        -33.700801849365234,
        195.0,
    )
    assert store.substore[0].root.sweep_fixed_angle == 0.5
    assert store.substore[0].root._get_time() == np.datetime64(
        "2018-12-20T06:06:28", "s"
    )


@pytest.mark.parametrize("sweep", ["sweep_0", 0, [0, 1], ["sweep_0", "sweep_1"]])
def test_open_gamic_datatree_sweep(gamic_file, sweep):
    dtree = open_gamic_datatree(gamic_file, sweep=sweep)
    if isinstance(sweep, (str, int)):
        lswp = len([sweep])
    else:
        lswp = len(sweep)
    assert len(dtree.match("sweep_*")) == lswp


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
    for i, grp in enumerate(dtree.match("sweep_*")):
        ds = dtree[grp].ds
        assert dict(ds.sizes) == {"azimuth": azimuths[i], "range": ranges[i]}
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


@pytest.mark.parametrize("first_dim", ["auto", "time"])
@pytest.mark.parametrize("fix_second_angle", [False, True])
def test_open_gamic_dataset(gamic_file, first_dim, fix_second_angle):
    # open first sweep group
    with xr.open_dataset(
        gamic_file,
        group="sweep_0",
        engine="gamic",
        first_dim=first_dim,
        fix_second_angle=fix_second_angle,
    ) as ds:
        dim0 = "time" if first_dim == "time" else "azimuth"
        assert dict(ds.sizes) == {dim0: 361, "range": 360}
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
    with xr.open_dataset(
        gamic_file,
        group="sweep_9",
        engine="gamic",
        first_dim=first_dim,
        fix_second_angle=fix_second_angle,
    ) as ds:
        assert dict(ds.sizes) == {dim0: 360, "range": 1000}
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


def test_open_gamic_dataset_stream(gamic_file):
    with open(gamic_file, mode="rb") as fhandle:
        contents = io.BytesIO(fhandle.read())
        with xr.open_dataset(contents, group="sweep_9", engine="gamic") as ds:
            assert isinstance(ds, xr.Dataset)
            print(ds)


def test_open_gamic_dataset_fsspec(gamic_file):
    with fsspec.open(gamic_file, mode="rb") as fhandle:
        with xr.open_dataset(fhandle, group="sweep_9", engine="gamic") as ds:
            assert isinstance(ds, xr.Dataset)
            print(ds)


def test_open_gamic_store(gamic_file):
    store = xradar.io.backends.gamic.GamicStore.open(gamic_file, group="sweep_0")
    assert store.root.site_coords == (6.4569489, 50.9287272, 310.0)
    assert store.root._get_site_coords() == (6.4569489, 50.9287272, 310.0)
    assert store.root.sweep_fixed_angle == 28.0
    assert store.root._get_time() == np.datetime64("2018-06-01T05:40:47.041000", "us")


def test_open_gamic_dataset_reindex(gamic_file):
    # open first sweep group
    reindex_angle = dict(start_angle=0, stop_angle=360, angle_res=1.0, direction=1)
    with xr.open_dataset(
        gamic_file, group="sweep_0", engine="gamic", reindex_angle=reindex_angle
    ) as ds:
        assert dict(ds.sizes) == {"azimuth": 360, "range": 360}


def test_open_furuno_scn_dataset(furuno_scn_file):
    # open sweep group
    with xr.open_dataset(furuno_scn_file, first_dim="time", engine="furuno") as ds:
        assert dict(ds.sizes) == {"time": 1376, "range": 602}
        assert set(ds.data_vars) & (
            sweep_dataset_vars | non_standard_sweep_dataset_vars
        ) == {"KDP", "VRADH", "ZDR", "DBZH", "WRADH", "RHOHV", "PHIDP"}
        for key, value in ds.data_vars.items():
            if key in [
                "RATE",
                "KDP",
                "VRADH",
                "ZDR",
                "DBZH",
                "WRADH",
                "RHOHV",
                "PHIDP",
            ]:
                assert value.encoding["_FillValue"] == 0.0
            elif key in ["azimuth", "elevation"]:
                assert value.encoding["_FillValue"] == np.ma.minimum_fill_value(
                    value.encoding["dtype"]
                )
            else:
                assert value.encoding.get("_FillValue", None) is None
        assert ds.sweep_number == 0

    # open sweep group, auto
    with xr.open_dataset(
        furuno_scn_file,
        engine="furuno",
    ) as ds:
        assert dict(ds.sizes) == {"azimuth": 1376, "range": 602}
        assert ds.sweep_number == 0


def test_open_furuno_scnx_dataset(furuno_scnx_file):
    # open sweep group
    with xr.open_dataset(furuno_scnx_file, first_dim="time", engine="furuno") as ds:
        assert dict(ds.sizes) == {"time": 722, "range": 936}
        assert set(ds.data_vars) & (
            sweep_dataset_vars | non_standard_sweep_dataset_vars
        ) == {"KDP", "VRADH", "ZDR", "DBZH", "WRADH", "RHOHV", "PHIDP"}

        for key, value in ds.data_vars.items():
            if key in [
                "RATE",
                "KDP",
                "VRADH",
                "ZDR",
                "DBZH",
                "WRADH",
                "RHOHV",
                "PHIDP",
            ]:
                assert value.encoding["_FillValue"] == 0.0
            elif key in ["azimuth", "elevation"]:
                assert value.encoding["_FillValue"] == np.ma.minimum_fill_value(
                    value.encoding["dtype"]
                )
            else:
                assert value.encoding.get("_FillValue", None) is None
        assert ds.sweep_number == 0

    # open sweep group, auto
    with xr.open_dataset(
        furuno_scnx_file,
        engine="furuno",
    ) as ds:
        assert dict(ds.sizes) == {"azimuth": 722, "range": 936}
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
    for i, grp in enumerate(dtree.match("sweep_*")):
        ds = dtree[grp].ds
        assert dict(ds.sizes) == {"azimuth": azimuths[i], "range": ranges[i]}
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
    with xr.open_dataset(rainbow_file, group="sweep_0", engine="rainbow") as ds:
        assert dict(ds.sizes) == {"azimuth": 361, "range": 400}
        assert set(ds.data_vars) & (
            sweep_dataset_vars | non_standard_sweep_dataset_vars
        ) == {
            "DBZH",
        }
        assert ds.sweep_number == 0

    # open last sweep group
    with xr.open_dataset(rainbow_file, group="sweep_13", engine="rainbow") as ds:
        assert dict(ds.sizes) == {"azimuth": 361, "range": 400}
        assert set(ds.data_vars) & (
            sweep_dataset_vars | non_standard_sweep_dataset_vars
        ) == {
            "DBZH",
        }
        assert ds.sweep_number == 13

    # open last sweep group, auto
    with xr.open_dataset(
        rainbow_file,
        group="sweep_13",
        engine="rainbow",
        backend_kwargs=dict(first_dim="time"),
    ) as ds:
        assert dict(ds.sizes) == {"time": 361, "range": 400}
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
    i = 0
    for grp in dtree.match("sweep_*"):
        ds = dtree[grp].ds
        assert dict(ds.sizes) == {"azimuth": azimuths[i], "range": ranges[i]}
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
        i += 1


def test_open_iris0_dataset(iris0_file):
    # open first sweep group
    with xr.open_dataset(iris0_file, group="sweep_0", engine="iris") as ds:
        assert dict(ds.sizes) == {"azimuth": 360, "range": 664}
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
    with xr.open_dataset(iris0_file, group="sweep_9", engine="iris") as ds:
        assert dict(ds.sizes) == {"azimuth": 360, "range": 664}
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
    with xr.open_dataset(
        iris0_file,
        group="sweep_9",
        engine="iris",
        backend_kwargs=dict(first_dim="time"),
    ) as ds:
        assert dict(ds.sizes) == {"time": 360, "range": 664}
        assert ds.sweep_number == 9


def test_open_iris1_dataset(iris1_file):
    # open first and only sweep group
    with xr.open_dataset(iris1_file, group="sweep_0", engine="iris") as ds:
        assert dict(ds.sizes) == {"azimuth": 359, "range": 833}
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
    with xr.open_dataset(
        iris1_file,
        group="sweep_0",
        engine="iris",
        backend_kwargs=dict(first_dim="time"),
    ) as ds:
        assert dict(ds.sizes) == {"time": 359, "range": 833}
        assert ds.sweep_number == 0


@pytest.mark.parametrize(
    "compression, compression_opts", [("gzip", 0), ("gzip", 6), ("gzip", 9)]
)
def test_odim_roundtrip(odim_file2, compression, compression_opts):
    dtree = open_odim_datatree(odim_file2)
    with tempfile.NamedTemporaryFile(mode="w+b") as outfile:
        xradar.io.to_odim(
            dtree,
            outfile.name,
            source="WMO:01104,NOD:norst",
            compression=compression,
            compression_opts=compression_opts,
        )
        dtree2 = open_odim_datatree(outfile.name, reindex_angle=False)
        for d0, d1 in zip(dtree.groups, dtree2.groups):
            xr.testing.assert_equal(dtree[d0].ds, dtree2[d1].ds)


def test_odim_optional_how(odim_file2):
    dtree = open_odim_datatree(odim_file2)
    with tempfile.NamedTemporaryFile(mode="w+b") as outfile:
        xradar.io.to_odim(
            dtree,
            outfile.name,
            source="WMO:01104,NOD:norst",
            optional_how=True,
        )
        ds = h5py.File(outfile.name)

        for i in range(1, 6):
            ds_how = ds[f"dataset{i}"]["how"].attrs
            assert "scan_index" in ds_how
            assert "scan_count" in ds_how
            assert "startazA" in ds_how
            assert "stopazA" in ds_how
            assert "startazT" in ds_how
            assert "startazT" in ds_how
            assert "startelA" in ds_how
            assert "stopelA" in ds_how

    with tempfile.NamedTemporaryFile(mode="w+b") as outfile:
        xradar.io.to_odim(
            dtree,
            outfile.name,
            source="WMO:01104,NOD:norst",
            optional_how=False,
        )
        ds = h5py.File(outfile.name)

        for i in range(1, 6):
            ds_how = ds[f"dataset{i}"]["how"].attrs
            assert "scan_index" not in ds_how
            assert "scan_count" not in ds_how
            assert "startazA" not in ds_how
            assert "stopazA" not in ds_how
            assert "startazT" not in ds_how
            assert "startazT" not in ds_how
            assert "startelA" not in ds_how
            assert "stopelA" not in ds_how


def test_write_odim_source(rainbow_file2):
    dtree = open_rainbow_datatree(rainbow_file2)
    with tempfile.NamedTemporaryFile(mode="w+b") as outfile:
        with pytest.raises(ValueError):
            xradar.io.to_odim(
                dtree,
                outfile.name,
                source="PLC:Wideumont",
            )

        xradar.io.to_odim(
            dtree,
            outfile.name,
            source="NOD:bewid,WMO:06477",
        )
        ds = h5py.File(outfile.name)
        assert ds["what"].attrs["source"].decode("utf-8") == "NOD:bewid,WMO:06477"


def test_open_datamet_dataset(datamet_file):
    # open first sweep group
    with xr.open_dataset(
        datamet_file,
        group="sweep_0",
        engine="datamet",
    ) as ds:
        assert dict(ds.sizes) == {"azimuth": 360, "range": 493}
        assert set(ds.data_vars) & (
            sweep_dataset_vars | non_standard_sweep_dataset_vars
        ) == {"DBTH", "DBZH", "KDP", "PHIDP", "RHOHV", "VRADH", "WRADH", "ZDR"}
        assert ds.sweep_number == 0

    # open last sweep group
    with xr.open_dataset(
        datamet_file,
        group="sweep_10",
        engine="datamet",
    ) as ds:
        assert dict(ds.sizes) == {"azimuth": 360, "range": 1332}
        assert set(ds.data_vars) & (
            sweep_dataset_vars | non_standard_sweep_dataset_vars
        ) == {"DBTH", "DBZH", "KDP", "PHIDP", "RHOHV", "VRADH", "WRADH", "ZDR"}
        assert ds.sweep_number == 10


def test_open_datamet_dataset_reindex(datamet_file):
    # open first sweep group
    reindex_angle = dict(start_angle=0, stop_angle=360, angle_res=2.0, direction=1)
    with xr.open_dataset(
        datamet_file,
        group="sweep_10",
        engine="datamet",
        decode_coords=True,
        reindex_angle=reindex_angle,
    ) as ds:
        assert dict(ds.sizes) == {"azimuth": 180, "range": 1332}


def test_open_datamet_datatree(datamet_file):
    dtree = open_datamet_datatree(datamet_file)

    # root_attrs
    attrs = dtree.attrs
    assert attrs["Conventions"] == "None"

    # root vars
    rvars = dtree.data_vars
    assert rvars["volume_number"] == 0
    assert rvars["platform_type"] == "fixed"
    assert rvars["instrument_type"] == "radar"
    assert rvars["time_coverage_start"] == "2019-07-10T07:00:00Z"
    assert rvars["time_coverage_end"] == "2019-07-10T07:00:00Z"
    np.testing.assert_almost_equal(rvars["latitude"].values, np.array(41.9394))
    np.testing.assert_almost_equal(rvars["longitude"].values, np.array(14.6208))
    np.testing.assert_almost_equal(rvars["altitude"].values, np.array(710))

    # iterate over subgroups and check some values
    moments = ["DBTH", "DBZH", "KDP", "PHIDP", "RHOHV", "VRADH", "WRADH", "ZDR"]
    elevations = [16.1, 13.9, 11.0, 9.0, 7.0, 5.5, 4.5, 3.5, 2.5, 1.5, 0.5]
    azimuths = [360] * 11
    ranges = [493, 493, 493, 664, 832, 832, 1000, 1000, 1332, 1332, 1332]
    i = 0
    for grp in dtree.match("sweep_*"):
        ds = dtree[grp].ds
        assert dict(ds.sizes) == {"azimuth": azimuths[i], "range": ranges[i]}
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
        i += 1

    # Try to reed single sweep
    dtree = open_datamet_datatree(datamet_file, sweep=1)
    assert len(dtree.groups) == 5

    # Try to read list of sweeps
    dtree = open_datamet_datatree(datamet_file, sweep=[1, 2])
    assert len(dtree.groups) == 6


@pytest.mark.parametrize("first_dim", ["time", "auto"])
def test_cfradfial2_roundtrip(cfradial1_file, first_dim):
    dtree0 = open_cfradial1_datatree(cfradial1_file, first_dim=first_dim)
    # first write to cfradial2
    with tempfile.NamedTemporaryFile(mode="w+b") as outfile:
        xradar.io.to_cfradial2(dtree0.copy(), outfile.name)
        # then open cfradial2 file
        dtree1 = xr.open_datatree(outfile.name)
        # and write again
        with tempfile.NamedTemporaryFile(mode="w+b") as outfile1:
            xradar.io.to_cfradial2(dtree1.copy(), outfile1.name)
            # and open second cfradial2
            dtree2 = xr.open_datatree(outfile1.name)
            # check equality
            for d0, d1, d2 in zip(dtree0.groups, dtree1.groups, dtree2.groups):
                if "sweep" in d0:
                    if first_dim == "auto":
                        first_dim = "azimuth"
                    assert first_dim in dtree0[d0].dims
                    assert "time" in dtree1[d1].dims
                    assert "time" in dtree2[d2].dims
                xr.testing.assert_equal(dtree1[d1].ds, dtree2[d2].ds)


def test_cfradial_n_points_file(cfradial1n_file):
    dtree = open_cfradial1_datatree(
        cfradial1n_file, first_dim="auto", site_coords=False
    )
    attrs = dtree.attrs

    # root_attrs
    assert attrs["Conventions"] == "CF-1.7"
    assert attrs["version"] == "CF-Radial-1.4"
    assert attrs["title"] == "VOL_A"
    assert attrs["instrument_name"] == "Desio_Radar"
    assert attrs["platform_is_mobile"] == "false"

    # root vars
    rvars = dtree.data_vars
    assert rvars["volume_number"] == 1
    assert rvars["platform_type"] == b"fixed"
    assert rvars["instrument_type"] == b"radar"
    assert rvars["time_coverage_start"] == b"2024-05-22T16:00:47Z"
    assert rvars["time_coverage_end"] == b"2024-05-22T16:03:20Z"
    np.testing.assert_almost_equal(rvars["latitude"].values, np.array(45.6272661))
    np.testing.assert_almost_equal(rvars["longitude"].values, np.array(9.1963181))
    np.testing.assert_almost_equal(rvars["altitude"].values, np.array(241.0))

    # iterate over subgroups and check some values
    moments = ["ZDR", "RHOHV", "KDP", "DBZ", "VEL", "PHIDP"]
    elevations = [0.7, 1.3, 3.0, 5.0, 7.0, 10.0, 15.0, 25.0]
    azimuths = [360] * 8
    ranges = [416] * 5 + [383, 257, 157]
    for grp in dtree.groups:
        # only iterate sweep groups
        if "sweep" not in grp:
            continue
        ds = dtree[grp].ds
        i = int(ds.sweep_number.values)
        assert i == int(grp[7:])
        assert dict(ds.sizes) == {
            "azimuth": azimuths[i],
            "range": ranges[i],
            "frequency": 1,
        }
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
            "frequency",
        }
        assert np.round(ds.sweep_fixed_angle.values.item(), 1) == elevations[i]
        assert ds.sweep_mode == "azimuth_surveillance"


@pytest.mark.run(order=1)
@pytest.mark.parametrize("sweep", ["sweep_0", 0, [0, 1], ["sweep_0", "sweep_1"]])
@pytest.mark.parametrize(
    "nexradlevel2_files", ["nexradlevel2_gzfile", "nexradlevel2_bzfile"], indirect=True
)
def test_open_nexradlevel2_datatree_sweep(nexradlevel2_files, sweep):
    dtree = open_nexradlevel2_datatree(nexradlevel2_files, sweep=sweep)
    if isinstance(sweep, (str, int)):
        lswp = len([sweep])
    else:
        lswp = len(sweep)
    assert len(dtree.match("sweep*")) == lswp


@pytest.mark.parametrize(
    "nexradlevel2_files", ["nexradlevel2_gzfile", "nexradlevel2_bzfile"], indirect=True
)
def test_open_nexradlevel2_datatree(nexradlevel2_files):
    dtree = open_nexradlevel2_datatree(nexradlevel2_files)
    # root_attrs
    attrs = dtree.attrs
    assert attrs["Conventions"] == "None"
    assert attrs["instrument_name"] == "KLBB"

    # root vars
    rvars = dtree.data_vars
    assert rvars["volume_number"] == 0
    assert rvars["platform_type"] == "fixed"
    assert rvars["instrument_type"] == "radar"
    assert rvars["time_coverage_start"] == "2016-06-01T15:00:25Z"
    assert rvars["time_coverage_end"] == "2016-06-01T15:06:06Z"
    np.testing.assert_almost_equal(rvars["latitude"].values, np.array(33.65414047))
    np.testing.assert_almost_equal(rvars["longitude"].values, np.array(-101.81416321))
    np.testing.assert_almost_equal(rvars["altitude"].values, np.array(1029))

    # iterate over subgroups and check some values
    moments = [
        ["DBZH", "PHIDP", "RHOHV", "ZDR"],
        ["DBZH", "WRADH", "VRADH"],
        ["DBZH", "PHIDP", "RHOHV", "ZDR"],
        ["DBZH", "WRADH", "VRADH"],
        ["DBZH", "PHIDP", "RHOHV", "VRADH", "WRADH", "ZDR"],
        ["DBZH", "PHIDP", "RHOHV", "VRADH", "WRADH", "ZDR"],
        ["DBZH", "PHIDP", "RHOHV", "VRADH", "WRADH", "ZDR"],
        ["DBZH", "PHIDP", "RHOHV", "VRADH", "WRADH", "ZDR"],
        ["DBZH", "PHIDP", "RHOHV", "VRADH", "WRADH", "ZDR"],
        ["DBZH", "PHIDP", "RHOHV", "VRADH", "WRADH", "ZDR"],
        ["DBZH", "PHIDP", "RHOHV", "VRADH", "WRADH", "ZDR"],
    ]
    elevations = [
        0.5,
        0.5,
        1.5,
        1.5,
        2.4,
        3.4,
        4.3,
        6.0,
        9.9,
        14.6,
        19.5,
    ]
    azimuths = [
        720,
        720,
        720,
        720,
        360,
        360,
        360,
        360,
        360,
        360,
        360,
    ]
    ranges = [
        1832,
        1192,
        1632,
        1192,
        1312,
        1076,
        908,
        696,
        448,
        308,
        232,
    ]
    assert len(dtree.groups[1:]) == 14
    for i, grp in enumerate(dtree.match("sweep_*")):
        print(i)
        ds = dtree[grp].ds
        assert dict(ds.sizes) == {"azimuth": azimuths[i], "range": ranges[i]}
        assert set(ds.data_vars) & (
            sweep_dataset_vars | non_standard_sweep_dataset_vars
        ) == set(moments[i])
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
        assert ds.sweep_number.values == int(grp[6:])


@skip_import("dask")
@pytest.mark.parametrize(
    "nexradlevel2_files", ["nexradlevel2_gzfile", "nexradlevel2_bzfile"], indirect=True
)
def test_nexradlevel2_dask_load(nexradlevel2_files):
    ds = xr.open_dataset(nexradlevel2_files, group="sweep_0", engine="nexradlevel2")
    dsc = ds.chunk()
    dsc.load()


@skip_import("dask")
def test_iris_dask_load(iris0_file):
    ds = xr.open_dataset(iris0_file, group="sweep_0", engine="iris")
    dsc = ds.chunk()
    dsc.load()
