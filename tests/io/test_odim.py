#!/usr/bin/env python
# Copyright (c) 2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for `io.backends.odim` module.

ported from wradlib
"""
from contextlib import nullcontext

import numpy as np
import pytest

from xradar.io.backends import odim


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


@pytest.mark.parametrize("point", [("start", 946684800.0), ("end", 946684830.0)])
def test_get_time(point):
    what = dict(
        startdate="20000101", starttime="000000", enddate="20000101", endtime="000030"
    )
    time = odim._get_time(what, point=point[0])
    assert time == point[1]


def test_OdimH5NetCDFMetadata(odim_file):
    store = odim.OdimStore.open(odim_file, group="sweep_0")
    with pytest.warns(DeprecationWarning):
        assert store.substore[0].root.first_dim == "azimuth"
