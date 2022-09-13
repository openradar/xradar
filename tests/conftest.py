#!/usr/bin/env python
# Copyright (c) 2022, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.
import os.path
from urllib.parse import urljoin
from urllib.request import urlretrieve

import pytest


@pytest.fixture(scope="session")
def cfradial1_file(tmp_path_factory):
    base_url = "https://raw.githubusercontent.com/wradlib/wradlib-data/main/netcdf/"
    filename = "cfrad.20080604_002217_000_SPOL_v36_SUR.nc"
    url = urljoin(base_url, filename)
    fn = tmp_path_factory.mktemp("cfradial1_data")
    fname = os.path.join(fn, "cfradial1_data.nc")
    urlretrieve(url, filename=fname)
    return fname


@pytest.fixture(scope="session")
def odim_file(tmp_path_factory):
    base_url = "https://raw.githubusercontent.com/wradlib/wradlib-data/main/hdf5/"
    filename = "71_20181220_060628.pvol.h5"
    url = urljoin(base_url, filename)
    fn = tmp_path_factory.mktemp("odim_data")
    fname = os.path.join(fn, "odim_data.h5")
    urlretrieve(url, filename=fname)
    return fname
