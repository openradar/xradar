#!/usr/bin/env python
# Copyright (c) 2023, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for `io.backends.furuno` module.

Ported from wradlib.
"""

import datetime

import numpy as np
import pytest

from xradar.io.backends import furuno
from xradar.util import _get_data_file


def test_open_scn(furuno_scn_file, file_or_filelike):
    with _get_data_file(furuno_scn_file, file_or_filelike) as furunofile:
        data = furuno.FurunoFile(furunofile, loaddata=False, obsmode=1)
    assert isinstance(data, furuno.FurunoFile)
    assert isinstance(data.fh, (np.memmap, np.ndarray))
    with _get_data_file(furuno_scn_file, file_or_filelike) as furunofile:
        data = furuno.FurunoFile(furunofile, loaddata=True, obsmode=1)
    assert len(data.data) == 11
    assert data.filename == furunofile
    assert data.version == 3
    assert data.a1gate == 796
    assert data.angle_resolution == 0.26
    assert data.first_dimension == "azimuth"
    assert data.fixed_angle == 7.8
    assert data.site_coords == (15.44729, 47.07734000000001, 407.9)
    assert data.header["scan_start_time"] == datetime.datetime(2021, 7, 30, 16, 0)
    assert list(data.data.keys()) == [
        "RATE",
        "DBZH",
        "VRADH",
        "ZDR",
        "KDP",
        "PHIDP",
        "RHOHV",
        "WRADH",
        "QUAL",
        "azimuth",
        "elevation",
    ]


def test_open_scn_filelike(furuno_scn_file):
    with pytest.raises(
        ValueError, match="Furuno `observation mode` can't be extracted"
    ):
        with _get_data_file(furuno_scn_file, "filelike") as furunofile:
            data = furuno.FurunoFile(furunofile, loaddata=False)
            print(data.first_dimension)


def test_open_scnx(furuno_scnx_file, file_or_filelike):
    with _get_data_file(furuno_scnx_file, file_or_filelike) as furunofile:
        data = furuno.FurunoFile(furunofile, loaddata=False)
    assert isinstance(data, furuno.FurunoFile)
    assert isinstance(data.fh, (np.memmap, np.ndarray))
    with _get_data_file(furuno_scnx_file, file_or_filelike) as furunofile:
        data = furuno.FurunoFile(furunofile, loaddata=True)
    assert data.filename == furunofile
    assert data.version == 10
    assert data.a1gate == 292
    assert data.angle_resolution == 0.5
    assert data.first_dimension == "azimuth"
    assert data.fixed_angle == 0.5
    assert data.site_coords == (13.243970000000001, 53.55478, 38.0)
    assert data.header["scan_start_time"] == datetime.datetime(2022, 3, 24, 0, 0, 1)
    assert len(data.data) == 11
    assert list(data.data.keys()) == [
        "RATE",
        "DBZH",
        "VRADH",
        "ZDR",
        "KDP",
        "PHIDP",
        "RHOHV",
        "WRADH",
        "QUAL",
        "azimuth",
        "elevation",
    ]
