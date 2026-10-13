#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""``close()`` releases the file for every backend (#471, #502)."""

import io
import os
import shutil
import warnings
from pathlib import Path

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


def _private_copy(path, tmp_path):
    """Copy ``path`` into a folder of its own, so no other test or cache holds
    it (fixtures such as ``cfradial2_file`` may already live in ``tmp_path``)."""
    folder = tmp_path / "private"
    folder.mkdir(exist_ok=True)
    return Path(shutil.copyfile(path, folder / Path(path).name))


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
    # datamet, furuno, hpl and metek read their file completely when opened, so
    # they pass without the fix too; they keep close() from regressing
    # private copy: no other test can hold it in xarray's file cache
    source = Path(request.getfixturevalue(fixture_name))
    filename = _private_copy(source, tmp_path)
    ds = xr.open_dataset(str(filename), engine=engine, group="sweep_0")
    # the dataset is still referenced: only close() can release the file
    ds.close()
    assert _open_handles(filename) == 0


@pytest.mark.skipif(not _PROC_FD.is_dir(), reason="needs /proc/self/fd")
@pytest.mark.parametrize("legacy", [False, True], ids=["engine", "legacy"])
@pytest.mark.parametrize(
    "engine,fixture_name",
    [
        ("cfradial1", "cfradial1_file"),
        ("cfradial2", "cfradial2_file"),
        ("datamet", "datamet_file"),
        ("furuno", "furuno_scn_file"),
        ("gamic", "gamic_file"),
        ("hpl", "hpl_file"),
        ("imd", "imd_file"),
        ("iris", "iris0_file"),
        ("metek", "metek_ave_gz_file"),
        ("nexradlevel2", "nexradlevel2_file"),
        ("odim", "odim_file"),
        ("rainbow", "rainbow_file"),
        ("uf", "uf_file_1"),
    ],
)
def test_datatree_close_releases_file(engine, fixture_name, legacy, request, tmp_path):
    # every group keeps its file's close, so closing the tree releases the
    # file; left to the garbage collector it can crash HDF5 (#471, #502)
    source = Path(request.getfixturevalue(fixture_name))
    filename = _private_copy(source, tmp_path)
    if legacy:
        opener = getattr(xd.io, f"open_{engine}_datatree")
        kwargs = {} if engine == "imd" else {"sweep": 0}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            dtree = opener(str(filename), **kwargs)
    else:
        dtree = xd.open_datatree(str(filename), engine=engine, sweep=0)
    # the tree is still referenced: only close() can release the file
    dtree.close()
    assert _open_handles(filename) == 0


@pytest.mark.skipif(not _PROC_FD.is_dir(), reason="needs /proc/self/fd")
def test_imd_volume_close_releases_all_files(imd_volume_files, tmp_path):
    files = [_private_copy(f, tmp_path) for f in imd_volume_files]
    volume = xd.io.open_imd_datatree(files)
    volume.close()
    assert [_open_handles(f) for f in files] == [0] * len(files)


@pytest.mark.skipif(not _PROC_FD.is_dir(), reason="needs /proc/self/fd")
def test_imd_volumes_close_releases_all_files(imd_volume_files, tmp_path):
    # the vcp_NN tree is flattened with to_dict(), which drops the close
    files = [_private_copy(f, tmp_path) for f in imd_volume_files]
    tree = xd.io.open_imd_volumes(files)
    tree.close()
    assert [_open_handles(f) for f in files] == [0] * len(files)


@pytest.mark.skipif(not _PROC_FD.is_dir(), reason="needs /proc/self/fd")
def test_uf_dataset_close_releases_file_with_time_first(uf_file_1, tmp_path):
    # swap_dims/sortby build new datasets; open_dataset sets the close last
    filename = _private_copy(uf_file_1, tmp_path)
    ds = xr.open_dataset(str(filename), engine="uf", group="sweep_0", first_dim="time")
    ds.close()
    assert _open_handles(filename) == 0


@pytest.mark.skipif(not _PROC_FD.is_dir(), reason="needs /proc/self/fd")
@pytest.mark.parametrize("method", ["georeference", "add_crs"])
def test_georeferenced_tree_still_closes_its_file(method, odim_file, tmp_path):
    # the accessors replace each sweep dataset in place (#471, #502)
    filename = _private_copy(odim_file, tmp_path)
    dtree = xd.open_datatree(str(filename), engine="odim", sweep=[0, 1])
    dtree = getattr(dtree.xradar, method)()
    dtree.close()
    assert _open_handles(filename) == 0


@pytest.mark.parametrize(
    "engine,fixture_name", [("hpl", "hpl_file"), ("metek", "metek_ave_gz_file")]
)
def test_close_keeps_the_callers_file_object_open(engine, fixture_name, request):
    # the backend closes only files it opened itself (#471)
    with open(request.getfixturevalue(fixture_name), "rb") as fobj:
        buffer = io.BytesIO(fobj.read())
    ds = xr.open_dataset(buffer, engine=engine, group="sweep_0")
    ds.close()
    assert not buffer.closed


@pytest.mark.skipif(not _PROC_FD.is_dir(), reason="needs /proc/self/fd")
@pytest.mark.parametrize("engine", ["cfradial2", "odim"])
def test_datatree_close_releases_file_without_site_coords(engine, request, tmp_path):
    # dropping the station variables must keep the file's close (#471, #502)
    fixture = {"cfradial2": "cfradial2_file", "odim": "odim_file"}[engine]
    source = Path(request.getfixturevalue(fixture))
    filename = _private_copy(source, tmp_path)
    dtree = xd.open_datatree(str(filename), engine=engine, sweep=0, site_coords=False)
    dtree.close()
    assert _open_handles(filename) == 0


@pytest.mark.skipif(not _PROC_FD.is_dir(), reason="needs /proc/self/fd")
@pytest.mark.parametrize("sweep", [True, [0, 1.0], [99]])
def test_cfradial2_closes_file_when_opening_fails(cfradial2_file, tmp_path, sweep):
    # an invalid or missing sweep must not leave the file open until garbage
    # collection (#471, #502)
    filename = _private_copy(cfradial2_file, tmp_path)
    with pytest.raises((TypeError, ValueError)):
        xd.open_datatree(filename, engine="cfradial2", sweep=sweep)
    assert _open_handles(filename) == 0
