#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Robustness of backend input handling and file-handle lifetime."""

import gc
import gzip
import io
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import xradar as xd

_PROC_FD = Path("/proc/self/fd")


def _open_handles(path):
    """Return how many file descriptors of this process point at ``path``."""
    target = os.path.realpath(path)
    count = 0
    for fd in _PROC_FD.iterdir():
        try:
            if os.path.realpath(fd.readlink()) == target:
                count += 1
        except OSError:
            continue
    return count


@pytest.mark.skipif(not _PROC_FD.is_dir(), reason="needs /proc/self/fd")
@pytest.mark.parametrize(
    "engine,fixture_name",
    [
        ("datamet", "datamet_file"),
        ("furuno", "furuno_scn_file"),
        ("gamic", "gamic_file"),
        ("hpl", "hpl_file"),
        ("iris", "iris0_file"),
        ("metek", "metek_ave_gz_file"),
        ("nexradlevel2", "nexradlevel2_file"),
        ("odim", "odim_file"),
        ("rainbow", "rainbow_file"),
    ],
)
def test_dataset_close_releases_file(engine, fixture_name, request, tmp_path):
    # private copy: no other test can hold it in xarray's file cache
    source = Path(request.getfixturevalue(fixture_name))
    filename = tmp_path / source.name
    shutil.copyfile(source, filename)
    # str path: this test is about close(), PathLike input is tested below
    ds = xr.open_dataset(str(filename), engine=engine, group="sweep_0")
    # the dataset is still referenced: only close() can release the file
    ds.close()
    assert _open_handles(filename) == 0
    del ds


@pytest.mark.skipif(not _PROC_FD.is_dir(), reason="needs /proc/self/fd")
def test_uf_releases_file_with_its_data(uf_file_1, tmp_path):
    # UF data are views into a memory map, which lives as long as the data
    filename = tmp_path / "uf.uf"
    shutil.copyfile(uf_file_1, filename)
    ds = xr.open_dataset(str(filename), engine="uf", group="sweep_0")
    ds.close()
    del ds
    gc.collect()
    assert _open_handles(filename) == 0


def test_uf_datatree_from_file_like(uf_file_1):
    data = Path(uf_file_1).read_bytes()
    expected = xd.io.open_uf_datatree(uf_file_1, sweep=[0, 1])
    actual = xd.io.open_uf_datatree(io.BytesIO(data), sweep=[0, 1])
    xr.testing.assert_equal(actual, expected)


def test_metek_reads_gzip_path(metek_ave_gz_file, tmp_path):
    gz_path = tmp_path / "0308.ave.gz"
    with open(metek_ave_gz_file, "rb") as fin, gzip.open(gz_path, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    expected = xr.open_dataset(metek_ave_gz_file, engine="metek")
    actual = xr.open_dataset(str(gz_path), engine="metek")
    xr.testing.assert_equal(actual, expected)


@pytest.mark.parametrize(
    "engine,fixture_name",
    [
        ("datamet", "datamet_file"),
        ("furuno", "furuno_scn_file"),
        ("iris", "iris0_file"),
        ("metek", "metek_ave_gz_file"),
    ],
)
def test_open_dataset_accepts_pathlike(engine, fixture_name, request):
    filename = request.getfixturevalue(fixture_name)
    expected = xr.open_dataset(str(filename), engine=engine, group="sweep_0")
    actual = xr.open_dataset(Path(filename), engine=engine, group="sweep_0")
    xr.testing.assert_equal(actual, expected)


def test_iris_datatree_accepts_pathlike(iris0_file):
    expected = xd.io.open_iris_datatree(str(iris0_file), sweep=[0])
    actual = xd.io.open_iris_datatree(Path(iris0_file), sweep=[0])
    xr.testing.assert_equal(actual, expected)


def test_cfradial1_fix_second_angle(cfradial1_file):
    kwargs = dict(engine="cfradial1", group="sweep_0", first_dim="auto")
    raw = xr.open_dataset(cfradial1_file, **kwargs)
    fixed = xr.open_dataset(cfradial1_file, fix_second_angle=True, **kwargs)
    # the secondary angle (elevation on a PPI) collapses onto its median
    assert np.unique(fixed.elevation.values).size == 1
    assert fixed.elevation.values[0] == pytest.approx(float(raw.elevation.median()))
    xr.testing.assert_equal(fixed.DBZ, raw.DBZ.assign_coords(elevation=fixed.elevation))


def test_cfradial1_fix_second_angle_root_group(cfradial1_file):
    # the root group holds no sweep, the fix must not apply there
    expected = xr.open_dataset(cfradial1_file, engine="cfradial1")
    actual = xr.open_dataset(cfradial1_file, engine="cfradial1", fix_second_angle=True)
    xr.testing.assert_equal(actual, expected)
