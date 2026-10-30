#!/usr/bin/env python
# Copyright (c) 2022-2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.
import contextlib
import os.path
import shutil
import uuid

import pytest
from filelock import FileLock
from open_radar_data import DATASETS


# silence FutureWarning
def pytest_configure(config):
    config.addinivalue_line(
        "filterwarnings",
        r"ignore:`open_.*_datatree` is deprecated:FutureWarning",
    )
    config.addinivalue_line(
        "filterwarnings",
        r"ignore:.*is incomplete and is padded to a full angle grid:UserWarning",
    )
    config.addinivalue_line(
        "filterwarnings",
        r"ignore:CfRadial2 sweep groups were renumbered:UserWarning",
    )
    config.addinivalue_line(
        "filterwarnings",
        r"ignore:All sweeps are incomplete:UserWarning",
    )
    config.addinivalue_line(
        "filterwarnings",
        r"ignore:Dropped \d+ incomplete sweep:UserWarning",
    )
    config.addinivalue_line(
        "filterwarnings",
        r"ignore:Rays might miss on beginning:UserWarning",
    )
    config.addinivalue_line(
        "filterwarnings",
        r"ignore:.*IRIS multi_prf_mode_flag=\d+ is not defined:UserWarning",
    )
    # prevent adding DataTree.ds by accident, turn warning into error
    config.addinivalue_line(
        "filterwarnings",
        "error:DataTree.ds is deprecated, use DataTree.dataset instead.:FutureWarning",
    )
    # dedicated markers
    config.addinivalue_line("markers", "file: run checks on open-radar-data files")


@pytest.fixture(scope="session")
def file_store():
    """Base fixture for open-radar-data store."""
    return DATASETS


@pytest.fixture(params=["file", "filelike"])
def file_or_filelike(request):
    return request.param


@pytest.fixture
def closing():
    """Close every dataset or tree passed through it when the test ends.

    Leaked files are otherwise closed by the garbage collector at an arbitrary
    point, which can crash HDF5 while another file is being opened (#502).
    Usage: ``dtree = closing(xd.open_datatree(...))``.
    """
    with contextlib.ExitStack() as stack:
        yield stack.enter_context


@pytest.fixture(scope="session")
def cfradial1_sgp_file(file_store):
    return file_store.fetch("sample_sgp_data.nc")


@pytest.fixture(scope="session")
def cfradial1_file(file_store, tmp_path_factory):
    """Safely fetch the dataset across multiple xdist workers using a lock."""
    shared_dir = tmp_path_factory.getbasetemp().parent
    lockfile = shared_dir / "fetch_cfradial1.lock"

    # synchronize the download/cache-check across workers
    with FileLock(str(lockfile)):
        return file_store.fetch("cfrad.20080604_002217_000_SPOL_v36_SUR.nc")


@pytest.fixture(scope="session")
def _cfradial2_source(cfradial1_file, tmp_path_factory):
    """Round-trip a cfradial1 file through to_cfradial2 to get a real CfRadial2 file."""
    import xradar as xd

    # shared tmp directory for file and lockfile
    tmp_dir = tmp_path_factory.mktemp("cfradial2")

    # identical file paths for all workers
    outfile = tmp_dir / "sample_cfradial2.nc"
    lockfile = tmp_dir / "sample_cfradial2.nc.lock"

    # synchronize worker execution
    with FileLock(str(lockfile)):
        if not outfile.is_file():
            # first worker generates the file; close the source right away
            # instead of leaving it open for the whole session (#502)
            with xd.io.open_cfradial1_datatree(
                cfradial1_file, first_dim="time"
            ) as dtree:
                xd.io.to_cfradial2(dtree.copy(), outfile, engine="netcdf4")

    return str(outfile)


@pytest.fixture
def cfradial2_file(_cfradial2_source, tmp_path):
    """A private copy of the CfRadial2 file for every test.

    Tests share no path, so a handle that is still alive from an earlier test
    (or waiting for garbage collection) can't meet the next open of the same
    file, which crashes netCDF4/HDF5 with xarray 2026.9.0 (pydata/xarray#11692).
    """
    path = tmp_path / "sample_cfradial2.nc"
    shutil.copyfile(_cfradial2_source, path)
    return str(path)


@pytest.fixture(scope="session")
def cfradial1n_file(file_store, tmp_path_factory):
    return file_store.fetch("DES_VOL_RAW_20240522_1600.nc")


@pytest.fixture(scope="session")
def odim_file(file_store):
    return file_store.fetch("71_20181220_060628.pvol.h5")


@pytest.fixture(scope="session")
def odim_file2(file_store):
    return file_store.fetch("T_PAGZ35_C_ENMI_20170421090837.hdf")


@pytest.fixture(scope="session")
def odim_file3(file_store):
    return file_store.fetch("T_PAZA63_C_LFPW_20230420065041.h5")


@pytest.fixture(scope="session")
def odim_file4(file_store):
    return file_store.fetch("202506090955_fianj_PVOL.h5")


@pytest.fixture(scope="session")
def odim_file5(file_store):
    return file_store.fetch("SUR.202506091000.VOL.h5")


@pytest.fixture(scope="session")
def datamet_file(file_store):
    return file_store.fetch("H-000-VOL-ILMONTE-201907100700.tar.gz")


@pytest.fixture(scope="session")
def furuno_scn_file(file_store):
    return file_store.fetch("0080_20210730_160000_01_02.scn.gz")


@pytest.fixture(scope="session")
def furuno_scnx_file(file_store):
    return file_store.fetch("2006_20220324_000000_000.scnx.gz")


@pytest.fixture(scope="session")
def gamic_file(file_store):
    return file_store.fetch("DWD-Vol-2_99999_20180601054047_00.h5")


@pytest.fixture(scope="session")
def rainbow_file(file_store):
    return file_store.fetch("2013051000000600dBZ.vol")


@pytest.fixture(scope="session")
def rainbow_file2(file_store):
    return file_store.fetch("sample_rainbow_5_59.vol")


@pytest.fixture(scope="session")
def iris0_file(file_store):
    return file_store.fetch("cor-main131125105503.RAW2049")


@pytest.fixture(scope="session")
def iris1_file(file_store):
    return file_store.fetch("SUR210819000227.RAWKPJV")


@pytest.fixture(scope="session")
def iris_raw_na_file(file_store):
    return file_store.fetch("RAW_NA_000_125_20080411181219")


@pytest.fixture(scope="session")
def iris_arouca_file(file_store):
    return file_store.fetch("PRT231225000005.RAWD2A3")


@pytest.fixture(scope="session")
def nexradlevel2_file(file_store):
    return file_store.fetch("KATX20130717_195021_V06")


@pytest.fixture(scope="session")
def nexradlevel2_ldm_stride_file(file_store):
    """KILX volume with an over-120-message LDM (#376 regression fixture).

    LDM 49 of this file contains 120 MSG_31 + 2 MSG_2 = 122 messages.
    Pre-fix, xradar's mod-120 stride dropped the trailing 2 MSG_31s,
    yielding sweep_10 = 358 (on-wire is 360).
    """
    return file_store.fetch("KILX20230629_154426_V06")


@pytest.fixture(scope="session")
def nexrad_chunks_klot(file_store, tmp_path_factory):
    import tarfile

    archive = file_store.fetch("nexrad_level2_chunks_KLOT.tar.gz")
    extract_dir = tmp_path_factory.mktemp("nexrad_chunks")
    with tarfile.open(archive) as tar:
        tar.extractall(extract_dir, filter="data")
    chunk_dir = extract_dir / "nexrad_chunks_KLOT"
    return sorted(chunk_dir.iterdir())


@pytest.fixture(scope="session")
def nexradlevel2_msg1_file(file_store, tmp_path_factory):
    fnamei = file_store.fetch("KLIX20050828_180149.gz")
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
def nexradlevel2_gzfile(file_store, tmp_path_factory):
    fnamei = file_store.fetch("KLBB20160601_150025_V06.gz")
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
def nexradlevel2_bzfile(file_store):
    return file_store.fetch("KLBB20160601_150025_V06")


@pytest.fixture(scope="session")
def metek_ave_gz_file(file_store, tmp_path_factory):
    fnamei = file_store.fetch("0308.ave.gz")
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
def metek_pro_gz_file(file_store, tmp_path_factory):
    fnamei = file_store.fetch("0308.pro.gz")
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
def metek_raw_gz_file(file_store, tmp_path_factory):
    fnamei = file_store.fetch("0308.raw.gz")
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
def uf_file_1(file_store, tmp_path_factory):
    fnamei = file_store.fetch("20110427_164233_rvp8-rel_v001_SUR.uf.gz")
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
def uf_file_2(file_store, tmp_path_factory):
    fnamei = file_store.fetch("CHL19950617_182848.uf.gz")
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
def uf_file_3(file_store, tmp_path_factory):
    fnamei = file_store.fetch("MC3E_NPOL_2011_0524_2356_hid.uf.gz")
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
def hpl_file1(file_store):
    return file_store.fetch("User1_100_20240714_122137.hpl")


@pytest.fixture(scope="session")
def hpl_file2(file_store):
    return file_store.fetch("User1_184_20240601_013257.hpl")


@pytest.fixture(scope="session")
def imd_file(file_store):
    return file_store.fetch("IMD/JPR220822135253-IMD-B.nc")


@pytest.fixture(scope="session")
def imd_volume_files(file_store):
    return [
        file_store.fetch(f"IMD/JPR220822135253-IMD-B.nc{s}")
        for s in ["", ".1", ".2", ".3", ".4", ".5", ".6", ".7", ".8", ".9"]
    ]


@pytest.fixture
def temp_file(tmp_path):
    rnd = uuid.uuid4().hex[:8]
    return tmp_path / f"testfile_{rnd}"


@pytest.fixture
def make_temp_file(tmp_path):
    def _make_file():
        rnd = uuid.uuid4().hex[:8]
        return tmp_path / f"testfile_{rnd}"

    return _make_file


FILE_FIXTURES = {
    "file_store",
    "engine_and_file",
    "file_fixture",
}


def pytest_collection_modifyitems(items):
    for item in items:
        fixture_names = getattr(item, "fixturenames", ())
        if FILE_FIXTURES.intersection(fixture_names):
            item.add_marker(pytest.mark.file)


@pytest.fixture(autouse=True)
def check_for_missing_file_marker(request, monkeypatch):
    """
    Hooks into every test and intercepts PDATASETS.fetch calls.
    If the test is not marked as 'file', it fails immediately.

    We can run pytest -m "file" and pytest -m "not file" to run
    our file based tests and our general unittests in different runs.
    """
    # keep reference to original fetch method
    original_fetch = DATASETS.fetch

    # cache marker check
    has_marker = request.node.get_closest_marker("file") is not None

    def guarded_fetch(*args, **kwargs):
        # check if the currently running test has the 'file' marker
        if not has_marker:
            # determine which fixtures this test actually uses
            current_fixtures = getattr(request.node, "fixturenames", ())

            pytest.fail(
                f"\nMISSING MARKER DETECTED!\n"
                f"The test '{request.node.name}' loaded data from open-radar-data, "
                f"but does not have the '@pytest.mark.file' marker.\n\n"
                f"Possible causes:\n"
                f"1. One of these active fixtures calls DATASETS.fetch but is missing from FILE_FIXTURES: {list(current_fixtures)}\n"
                f"2. The test calls DATASETS.fetch directly inside the test body and was not manually marked."
            )

        # if the marker is present, proceed normally
        return original_fetch(*args, **kwargs)

    # override the method for the duration of the current test
    monkeypatch.setattr(DATASETS, "fetch", guarded_fetch)
