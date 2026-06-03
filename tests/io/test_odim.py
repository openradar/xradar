#!/usr/bin/env python
# Copyright (c) 2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for `io.backends.odim` module.

ported from wradlib
"""

from contextlib import nullcontext

import numpy as np
import pytest
from xarray import DataTree, open_dataset, open_mfdataset

from xradar.io.backends import odim, open_odim_datatree


def create_startazA(nrays=360):
    arr = np.linspace(0, 360, 360, endpoint=False, dtype=np.float32)
    if nrays == 361:
        arr = np.insert(arr, 10, (arr[10] + arr[9]) / 2, axis=0)
    return arr


def create_stopazA(nrays=360):
    arr = np.linspace(1, 361, 360, endpoint=False, dtype=np.float32)
    # arr = np.arange(1, 361, 1, dtype=np.float32)
    arr[arr >= 360] -= 360
    if nrays == 361:
        arr = np.insert(arr, 10, (arr[10] + arr[9]) / 2, axis=0)
    return arr


def create_how(nrays=360, stopaz=False):
    how = dict(startazA=create_startazA(nrays))
    if stopaz:
        how.update(stopazA=create_stopazA(nrays))
    return how


@pytest.mark.parametrize("stopaz", [False, True])
def test_get_azimuth_how(stopaz):
    how = create_how(stopaz=stopaz)
    actual = odim._get_azimuth_how(how)
    wanted = np.arange(0.5, 360, 1.0)
    np.testing.assert_equal(actual, wanted)


@pytest.mark.parametrize("nrays", [180, 240, 360, 720])
def test_get_azimuth_where(nrays):
    where = dict(nrays=nrays)
    actual = odim._get_azimuth_where(where)
    udiff = np.unique(np.diff(actual))
    assert len(actual) == nrays
    assert len(udiff) == 1
    assert udiff[0] == 360.0 / nrays


def test_odim_azimuth_nominal(odim_file):
    import xarray as xr

    ds = xr.open_dataset(
        odim_file, engine="odim", angle_spec="nominal", group="sweep_0"
    )

    azimuth = ds.azimuth.values
    udiff = np.unique(np.diff(azimuth))
    assert len(azimuth) == 360
    assert len(udiff) == 1
    assert udiff[0] == 1.0


def test_odim_azimuth_auto(odim_file):
    import xarray as xr

    ds = xr.open_dataset(odim_file, engine="odim", angle_spec="auto", group="sweep_0")

    azimuth = ds.azimuth.values
    udiff = np.unique(np.diff(azimuth))
    assert len(azimuth) == 360
    assert len(udiff) == 1
    assert udiff[0] == 1.0


def test_odim_azimuth_reindex_deprecation(odim_file):
    import xarray as xr

    with pytest.warns(
        DeprecationWarning, match="The 'reindex_angle' kwarg is deprecated"
    ):
        ds = xr.open_dataset(
            odim_file, engine="odim", reindex_angle=False, group="sweep_0"
        )
    azimuth = ds.azimuth.values
    udiff = np.unique(np.diff(azimuth))
    assert len(azimuth) == 360
    assert len(udiff) == 1
    assert udiff[0] == 1.0


def test_odim_azimuth_raw(odim_file):
    import xarray as xr

    ds = xr.open_dataset(odim_file, engine="odim", group="sweep_0", angle_spec="raw")
    azimuth = ds.azimuth.values
    assert len(azimuth) == 360
    # raw should still give uniform spacing for this file
    udiff = np.unique(np.diff(azimuth))
    assert len(udiff) == 1
    assert udiff[0] == 1.0


@pytest.mark.parametrize("angle_spec", ["nominal", "auto", "raw"])
def test_odim_azimuth_nonstandard_nrays(tmp_path, angle_spec):
    import h5netcdf

    nrays = 361
    ascale = 360.0 / nrays

    startazA = np.arange(0, 360, ascale, dtype=np.float32)
    stopazA = np.arange(ascale, 360 + ascale, ascale, dtype=np.float32)

    filepath = tmp_path / "test_nonstandard.h5"
    with h5netcdf.File(filepath, "w") as f:
        f.attrs["Conventions"] = "ODIM_H5/V2_2"
        wg = f.create_group("where")
        wg.attrs["lon"] = 0.0
        wg.attrs["lat"] = 0.0
        wg.attrs["height"] = 0.0
        dg = f.create_group("dataset1")
        dw = dg.create_group("where")
        dw.attrs["nrays"] = nrays
        dw.attrs["nbins"] = 100
        dw.attrs["rstart"] = 0.0
        dw.attrs["rscale"] = 100.0
        dw.attrs["elangle"] = 0.5
        dw.attrs["a1gate"] = 0
        dh = dg.create_group("how")
        dh.attrs["startazA"] = startazA
        dh.attrs["stopazA"] = stopazA
        dh.attrs["startazT"] = np.zeros(nrays, dtype=np.float64)
        dh.attrs["stopazT"] = np.ones(nrays, dtype=np.float64)
        dwhat = dg.create_group("what")
        dwhat.attrs["quantity"] = "DBZH"
        dwhat.attrs["startdate"] = "20000101"
        dwhat.attrs["starttime"] = "000000"
        dwhat.attrs["enddate"] = "20000101"
        dwhat.attrs["endtime"] = "000030"
        dg.create_group("data1")

    import xarray as xr

    if angle_spec == "nominal":
        context = pytest.raises(ValueError, match="Unexpected number of rays")
    elif angle_spec == "auto":
        context = pytest.warns(RuntimeWarning, match="Unexpected number of rays")
        wanted = np.arange(0.5, 360, 1.0)
    else:
        context = nullcontext()
        # should match per-ray midpoint
        wanted = (startazA + np.where(stopazA < startazA, stopazA + 360, stopazA)) / 2
        wanted[wanted >= 360] -= 360

    with context:
        ds = xr.open_dataset(
            filepath, engine="odim", angle_spec=angle_spec, group="sweep_0"
        )

    if angle_spec != "nominal":
        azimuth = ds.azimuth.values
        assert len(azimuth) == 360 if angle_spec == "auto" else nrays
        np.testing.assert_array_almost_equal(azimuth, wanted, decimal=4)


@pytest.mark.parametrize("angle_spec", ["nominal", "auto", "raw"])
def test_odim_azimuth_fallback_nonstandard_nrays_with_duplicate(tmp_path, angle_spec):
    import h5netcdf

    nrays = 361
    startazA = create_startazA(nrays)
    stopazA = create_stopazA(nrays)

    filepath = tmp_path / "test_nonstandard_dup.h5"
    with h5netcdf.File(filepath, "w") as f:
        f.attrs["Conventions"] = "ODIM_H5/V2_2"
        wg = f.create_group("where")
        wg.attrs["lon"] = 0.0
        wg.attrs["lat"] = 0.0
        wg.attrs["height"] = 0.0
        dg = f.create_group("dataset1")
        dw = dg.create_group("where")
        dw.attrs["nrays"] = nrays
        dw.attrs["nbins"] = 100
        dw.attrs["rstart"] = 0.0
        dw.attrs["rscale"] = 100.0
        dw.attrs["elangle"] = 0.5
        dw.attrs["a1gate"] = 0
        dh = dg.create_group("how")
        dh.attrs["startazA"] = startazA
        dh.attrs["stopazA"] = stopazA
        dh.attrs["startazT"] = np.zeros(nrays, dtype=np.float64)
        dh.attrs["stopazT"] = np.ones(nrays, dtype=np.float64)
        dwhat = dg.create_group("what")
        dwhat.attrs["quantity"] = "DBZH"
        dwhat.attrs["startdate"] = "20000101"
        dwhat.attrs["starttime"] = "000000"
        dwhat.attrs["enddate"] = "20000101"
        dwhat.attrs["endtime"] = "000030"
        dg.create_group("data1")

    import xarray as xr

    if angle_spec == "nominal":
        context = pytest.raises(ValueError, match="Unexpected number of rays")
    elif angle_spec == "auto":
        context = pytest.warns(RuntimeWarning, match="Unexpected number of rays")
        wanted = np.arange(0.5, 360, 1.0)
    else:
        context = nullcontext()
        # should match per-ray midpoint
        wanted = (startazA + np.where(stopazA < startazA, stopazA + 360, stopazA)) / 2
        wanted[wanted >= 360] -= 360

    with context:
        ds = xr.open_dataset(
            filepath, engine="odim", angle_spec=angle_spec, group="sweep_0"
        )

    if angle_spec != "nominal":
        azimuth = ds.azimuth.values
        assert len(azimuth) == 360 if angle_spec == "auto" else nrays
        # assert len(azimuth) == nrays
        # wanted = (startazA + np.where(stopazA < startazA, stopazA + 360, stopazA)) / 2
        # wanted[wanted >= 360] -= 360
        print(azimuth)
        np.testing.assert_array_almost_equal(azimuth, wanted, decimal=4)


@pytest.mark.parametrize(
    "ang",
    [("az_angle", "elevation"), ("az_angle", "elevation"), ("elangle", "azimuth")],
)
def test_get_fixed_dim_and_angle(ang):
    where = {ang[0]: 1.0}
    dim, angle = odim._get_fixed_dim_and_angle(where)
    assert dim == ang[1]
    assert angle == 1.0


def create_el_how(rhi):
    if rhi:
        return dict(startelA=1.0, stopelA=2.0)
    else:
        return dict(elangles=1.5)


@pytest.mark.parametrize("rhi", [True, False])
def test_get_elevation_how(rhi):
    how = create_el_how(rhi)
    el = odim._get_elevation_how(how)
    assert el == 1.5


def test_get_elevation_where():
    where = dict(nrays=360, elangle=0.5)
    actual = odim._get_elevation_where(where)
    udiff = np.unique(actual)
    assert len(actual) == 360
    assert len(udiff) == 1
    assert udiff[0] == 0.5
    assert actual.dtype == np.float32


def test_get_time_how():
    how = dict(startazT=np.array([10, 20, 30]), stopazT=np.array([20, 30, 40]))
    time = odim._get_time_how(how)
    np.testing.assert_array_equal(time, np.array([15.0, 25.0, 35.0]))


@pytest.mark.parametrize("a1gate", [(0, 946684800.0416666), (10, 946684829.2083472)])
@pytest.mark.parametrize("enddate", [True, False])
def test_get_time_what(a1gate, enddate):
    what = dict(
        startdate="20000101",
        starttime="000000",
    )
    if enddate:
        what.update(enddate="20000101", endtime="000030")
        a1g = a1gate[1]
    else:
        a1g = 946684800.0
    where = dict(nrays=360, a1gate=a1gate[0])
    if not enddate:
        check = pytest.warns(
            UserWarning, match="Equal ODIM `starttime` and `endtime` values"
        )
    else:
        check = nullcontext()
    with check:
        time = odim._get_time_what(what, where)
    assert time[0] == a1g
    assert len(time) == 360


@pytest.mark.parametrize("rscale", [100, 150, 300, 1000])
def test_get_range(rscale):
    where = dict(nbins=10, rstart=0, rscale=rscale)
    rng, cent_first, bin_range = odim._get_range(where)
    assert np.unique(np.diff(rng))[0] == rscale
    assert cent_first == rscale / 2
    assert bin_range == rscale


@pytest.mark.parametrize(
    "point",
    [
        ("start", np.datetime64("2000-01-01T00:00:00", "s")),
        ("end", np.datetime64("2000-01-01T00:00:30", "s")),
    ],
)
def test_get_time(point):
    what = dict(
        startdate="20000101", starttime="000000", enddate="20000101", endtime="000030"
    )
    time = odim._get_time(what, point=point[0])
    assert time == point[1]


def test_get_a1gate():
    where = dict(a1gate=20)
    assert odim._get_a1gate(where) == 20


def test_OdimH5NetCDFMetadata(odim_file):
    store = odim.OdimStore.open(odim_file, group="sweep_0")
    with pytest.warns(DeprecationWarning):
        assert store.substore[0].root.first_dim == "azimuth"


def test_odim_open_mfdataset_context_manager(odim_file):
    with open_mfdataset(
        [odim_file],
        engine="odim",
        concat_dim="volume_time",
        combine="nested",
        group="sweep_0",
    ) as ds:
        assert ds is not None
        # closer must exist while inside context
        assert callable(getattr(ds, "_close", None))


def test_odim_dataset_has_close(odim_file):
    ds = open_dataset(odim_file, engine="odim", group="sweep_0")
    assert callable(getattr(ds, "_close", None))
    ds.close()


def test_open_odim_datatree(odim_file):
    # Define kwargs to pass into the function
    kwargs = {
        "sweep": [0, 1, 2],  # Specify sweeps to extract
        "first_dim": "auto",
        "reindex_angle": False,
        "fix_second_angle": False,
        "site_as_coords": True,
    }

    # Call the function with an ODIM file
    dtree = open_odim_datatree(odim_file, **kwargs)

    # Assertions to check DataTree structure
    assert isinstance(dtree, DataTree), "Expected a DataTree instance"
    subtree_paths = [n.path for n in dtree.subtree]
    assert "/" in subtree_paths, "Root group should be present in the DataTree"
    # optional_groups=False by default: metadata subgroups should NOT be present
    assert "radar_parameters" not in dtree.children
    assert "georeferencing_correction" not in dtree.children
    assert "radar_calibration" not in dtree.children

    # Check if the correct sweep groups are attached
    sweep_groups = [key for key in dtree.match("sweep_*")]
    assert len(sweep_groups) == 3, "Expected three sweep groups in the DataTree"
    sample_sweep = sweep_groups[0]

    # Check data variables in the sweep group
    assert (
        "DBZH" in dtree[sample_sweep].variables.keys()
    ), "Expected 'DBZH' variable in the sweep group"
    assert (
        "ZDR" in dtree[sample_sweep].variables.keys()
    ), "Expected 'ZDR' variable in the sweep group"
    assert dtree[sample_sweep]["DBZH"].shape == (
        360,
        1200,
    ), "Shape mismatch for 'DBZH' variable"
    # Station coords should be on root as coordinates, NOT on sweeps
    assert "latitude" in dtree.ds.coords
    assert "longitude" in dtree.ds.coords
    assert "altitude" in dtree.ds.coords
    assert "latitude" not in dtree.ds.data_vars

    # Validate attributes
    assert len(dtree.attrs) == 9
    assert (
        dtree.attrs["Conventions"] == "ODIM_H5/V2_2"
    ), "Instrument name should match expected value"


def test_open_odim_datatree_optional_groups(odim_file):
    """Test that optional_groups=True includes metadata subgroups."""
    dtree = open_odim_datatree(odim_file, optional_groups=True)
    assert "radar_parameters" in dtree.children
    assert "georeferencing_correction" in dtree.children
    assert "radar_calibration" in dtree.children
