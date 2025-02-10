#!/usr/bin/env python
# Copyright (c) 2022-2025, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.
import os.path

import pytest
from open_radar_data import DATASETS


@pytest.fixture(params=["file", "filelike"])
def file_or_filelike(request):
    return request.param


@pytest.fixture(scope="session")
def cfradial1_file(tmp_path_factory):
    return DATASETS.fetch("cfrad.20080604_002217_000_SPOL_v36_SUR.nc")


@pytest.fixture(scope="session")
def cfradial1n_file(tmp_path_factory):
    return DATASETS.fetch("DES_VOL_RAW_20240522_1600.nc")


@pytest.fixture(scope="session")
def odim_file():
    return DATASETS.fetch("71_20181220_060628.pvol.h5")


@pytest.fixture(scope="session")
def odim_file2():
    return DATASETS.fetch("T_PAGZ35_C_ENMI_20170421090837.hdf")


@pytest.fixture(scope="session")
def datamet_file():
    return DATASETS.fetch("H-000-VOL-ILMONTE-201907100700.tar.gz")


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
def rainbow_file2():
    return DATASETS.fetch("sample_rainbow_5_59.vol")


@pytest.fixture(scope="session")
def iris0_file():
    return DATASETS.fetch("cor-main131125105503.RAW2049")


@pytest.fixture(scope="session")
def iris1_file():
    return DATASETS.fetch("SUR210819000227.RAWKPJV")


@pytest.fixture(scope="session")
def nexradlevel2_file():
    return DATASETS.fetch("KATX20130717_195021_V06")


@pytest.fixture(scope="session")
def nexradlevel2_msg1_file(tmp_path_factory):
    fnamei = DATASETS.fetch("KLIX20050828_180149.gz")
    fnameo = os.path.join(
        tmp_path_factory.mktemp("data"), f"{os.path.basename(fnamei)[:-3]}_gz"
    )
    import gzip
    import shutil

    with gzip.open(fnamei) as fin:
        with open(fnameo, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    return fnameo


@pytest.fixture(scope="session")
def nexradlevel2_gzfile(tmp_path_factory):
    fnamei = DATASETS.fetch("KLBB20160601_150025_V06.gz")
    fnameo = os.path.join(
        tmp_path_factory.mktemp("data"), f"{os.path.basename(fnamei)[:-3]}_gz"
    )
    import gzip
    import shutil

    with gzip.open(fnamei) as fin:
        with open(fnameo, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    return fnameo


@pytest.fixture(scope="session")
def nexradlevel2_bzfile():
    return DATASETS.fetch("KLBB20160601_150025_V06")


@pytest.fixture(scope="session")
def nexradlevel2_files(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session")
def metek_ave_gz_file(tmp_path_factory):
    fnamei = DATASETS.fetch("0308.ave.gz")
    fnameo = os.path.join(
        tmp_path_factory.mktemp("data"), f"{os.path.basename(fnamei)[:-3]}"
    )
    import gzip
    import shutil

    with gzip.open(fnamei) as fin:
        with open(fnameo, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    return fnameo


@pytest.fixture(scope="session")
def metek_pro_gz_file(tmp_path_factory):
    fnamei = DATASETS.fetch("0308.pro.gz")
    fnameo = os.path.join(
        tmp_path_factory.mktemp("data"), f"{os.path.basename(fnamei)[:-3]}"
    )
    import gzip
    import shutil

    with gzip.open(fnamei) as fin:
        with open(fnameo, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    return fnameo


@pytest.fixture(scope="session")
def metek_raw_gz_file(tmp_path_factory):
    fnamei = DATASETS.fetch("0308.raw.gz")
    fnameo = os.path.join(
        tmp_path_factory.mktemp("data"), f"{os.path.basename(fnamei)[:-3]}"
    )
    import gzip
    import shutil

    with gzip.open(fnamei) as fin:
        with open(fnameo, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    return fnameo
