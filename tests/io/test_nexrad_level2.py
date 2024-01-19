#!/usr/bin/env python
# Copyright (c) 2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for `xradar.io.nexrad_archive` module."""

import xarray as xr

from xradar.io.backends import open_nexradlevel2_datatree


def test_open_nexradlevel2_datatree(nexradlevel2_file):
    dtree = open_nexradlevel2_datatree(nexradlevel2_file)
    ds = dtree["sweep_0"]
    assert ds.attrs["instrument_name"] == "KATX"
    assert ds.attrs["nsweeps"] == 16
    assert ds.attrs["Conventions"] == "CF/Radial instrument_parameters"
    assert ds["DBZH"].shape == (719, 1832)
    assert ds["DBZH"].dims == ("azimuth", "range")
    assert int(ds.sweep_number.values) == 0


def test_open_nexrad_level2_backend(nexradlevel2_file):
    ds = xr.open_dataset(nexradlevel2_file, engine="nexradlevel2")
    assert ds.attrs["instrument_name"] == "KATX"
    assert ds.attrs["nsweeps"] == 16
    assert ds.attrs["Conventions"] == "CF/Radial instrument_parameters"
    assert ds["DBZH"].shape == (719, 1832)
    assert ds["DBZH"].dims == ("azimuth", "range")
    assert int(ds.sweep_number.values) == 0
