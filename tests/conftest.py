#!/usr/bin/env python
# Copyright (c) 2022-2023, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.
import pytest
from open_radar_data import DATASETS


@pytest.fixture(params=["file", "filelike"])
def file_or_filelike(request):
    return request.param


@pytest.fixture(scope="session")
def cfradial1_file(tmp_path_factory):
    return DATASETS.fetch("cfrad.20080604_002217_000_SPOL_v36_SUR.nc")


@pytest.fixture(scope="session")
def odim_file():
    return DATASETS.fetch("71_20181220_060628.pvol.h5")


@pytest.fixture(scope="session")
def odim_file2():
    return DATASETS.fetch("T_PAGZ35_C_ENMI_20170421090837.hdf")


@pytest.fixture(scope="session")
def furuno_scn_file():
    return DATASETS.fetch("0080_20210730_160000_01_02.scn.gz")


@pytest.fixture(scope="session")
def furuno_scnx_file():
    return DATASETS.fetch("2006_20220324_000000_000.scnx.gz")


@pytest.fixture(scope="session")
def gamic_file():
    return DATASETS.fetch("DWD-Vol-2_99999_20180601054047_00.h5")


@pytest.fixture(scope="session")
def rainbow_file():
    return DATASETS.fetch("2013051000000600dBZ.vol")


@pytest.fixture(scope="session")
def iris0_file():
    return DATASETS.fetch("cor-main131125105503.RAW2049")


@pytest.fixture(scope="session")
def iris1_file():
    return DATASETS.fetch("SUR210819000227.RAWKPJV")
