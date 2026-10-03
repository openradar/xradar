#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for the backends' guess_can_open and engine discovery."""

import io
import pathlib

import pytest
import xarray as xr
from open_radar_data import DATASETS
from xarray.backends import plugins

import xradar as xd

# fixture -> engine
FILES = [
    ("cfradial1_file", "cfradial1"),
    ("cfradial1_sgp_file", "cfradial1"),
    ("cfradial1n_file", "cfradial1"),
    ("odim_file", "odim"),
    ("odim_file2", "odim"),
    ("gamic_file", "gamic"),
    ("datamet_file", "datamet"),
    ("furuno_scn_file", "furuno"),
    ("furuno_scnx_file", "furuno"),
    ("rainbow_file", "rainbow"),
    ("rainbow_file2", "rainbow"),
    ("iris0_file", "iris"),
    ("iris1_file", "iris"),
    ("nexradlevel2_file", "nexradlevel2"),
    ("nexradlevel2_msg1_file", "nexradlevel2"),
    ("hpl_file", "hpl"),
    ("metek_ave_gz_file", "metek"),
    ("metek_pro_gz_file", "metek"),
    ("uf_file_1", "uf"),
    ("uf_file_2", "uf"),
    ("imd_file", "imd"),
]


@pytest.mark.parametrize(("fixture", "engine"), FILES)
def test_guess_can_open(request, fixture, engine):
    filename = request.getfixturevalue(fixture)
    # only the matching xradar engine claims the file
    claims = [
        name
        for name in xd.io.list_xradar_engines()
        if plugins.get_backend(name).guess_can_open(filename)
    ]
    assert claims == [engine]
    assert xd.io.discover_engine(filename) == engine
    assert xd.io.discover_engine(pathlib.Path(filename)) == engine


@pytest.mark.parametrize(
    ("fixture", "engine"),
    [
        ("odim_file", "odim"),
        ("gamic_file", "gamic"),
        ("iris0_file", "iris"),
        ("metek_ave_gz_file", "metek"),
    ],
)
def test_guess_can_open_filelike(request, fixture, engine):
    filename = request.getfixturevalue(fixture)
    with open(filename, "rb") as fh:
        fh.read(3)
        assert plugins.get_backend(engine).guess_can_open(fh)
        # the position is restored
        assert fh.tell() == 3
    with open(filename, "rb") as fh:
        buf = io.BytesIO(fh.read())
    assert xd.io.discover_engine(buf) == engine
    assert buf.tell() == 0


@pytest.mark.parametrize(
    ("fixture", "engine"),
    [
        ("nexradlevel2_file", "nexradlevel2"),
        ("cfradial1_file", "cfradial1"),
        ("iris0_file", "iris"),
        ("uf_file_2", "uf"),
        ("imd_file", "imd"),
    ],
)
def test_guess_can_open_bytes(request, fixture, engine):
    with open(request.getfixturevalue(fixture), "rb") as fh:
        data = fh.read()
    assert xd.io.discover_engine(data) == engine


def test_guess_can_open_input_kinds(nexradlevel2_file, hpl_file, rainbow_file):
    with open(nexradlevel2_file, "rb") as fh:
        data = fh.read()
    nexrad = plugins.get_backend("nexradlevel2")
    # NEXRAD Level II chunks
    assert nexrad.guess_can_open([data[:1000], data[1000:2000]])
    # file objects aren't supported by these backends
    assert not nexrad.guess_can_open(io.BytesIO(data))
    with open(rainbow_file, "rb") as fh:
        assert not plugins.get_backend("rainbow").guess_can_open(fh)
    # HPL is read from text streams
    with open(hpl_file) as fh:
        assert plugins.get_backend("hpl").guess_can_open(fh)
        assert fh.tell() == 0


def test_guess_can_open_compressed_nexrad():
    # gzip-compressed NEXRAD Level II can't be opened directly
    filename = DATASETS.fetch("KLBB20160601_150025_V06.gz")
    assert not plugins.get_backend("nexradlevel2").guess_can_open(filename)


def test_guess_can_open_unknown(tmp_path):
    text = tmp_path / "notes.txt"
    text.write_text("no radar data here\n")
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    nc = tmp_path / "other.nc"
    xr.Dataset({"a": ("x", [1, 2])}).to_netcdf(nc, engine="h5netcdf")
    nc3 = tmp_path / "other3.nc"
    xr.Dataset({"a": ("x", [1, 2])}).to_netcdf(nc3, engine="scipy")
    for filename in [text, empty, nc, nc3, tmp_path / "missing.h5", 12345, None]:
        for name in xd.io.list_xradar_engines():
            assert not plugins.get_backend(name).guess_can_open(filename)
    with pytest.raises(ValueError, match="None of the xradar engines"):
        xd.io.discover_engine(text)


def test_xarray_engine_guessing(nexradlevel2_file, rainbow_file):
    # formats xarray doesn't know are opened without passing the engine
    with xr.open_dataset(nexradlevel2_file, group="sweep_0") as ds:
        assert ds.encoding["engine"] == "nexradlevel2"
    with xr.open_dataset(rainbow_file, group="sweep_0") as ds:
        assert "DBZH" in ds


def test_read_head_helpers(furuno_scn_file, odim_file, tmp_path):
    from xradar.io.backends.common import _read_head, _read_nc_header

    # unsupported input
    assert _read_head(12345) is None
    # gzip-compressed input, decompressed
    with open(furuno_scn_file, "rb") as fh:
        compressed = fh.read()
    expected = _read_head(furuno_scn_file, 4, decompress=True)
    assert expected[2:4] == b"\x03\x00"
    assert _read_head(compressed, 4, decompress=True) == expected
    buf = io.BytesIO(compressed)
    buf.seek(5)
    assert _read_head(buf, 4, decompress=True) == expected
    assert buf.tell() == 5

    # HDF5 signature, but truncated
    broken = tmp_path / "broken.h5"
    with open(odim_file, "rb") as fh:
        broken.write_bytes(fh.read(1000))
    assert _read_nc_header(broken) is None


def test_discover_engine_failing_guess(monkeypatch, odim_file):
    # a failing guess_can_open doesn't stop the discovery
    def _raise(self, filename_or_obj):
        raise RuntimeError("broken")

    monkeypatch.setattr(
        type(plugins.get_backend("cfradial1")), "guess_can_open", _raise
    )
    assert xd.io.discover_engine(odim_file) == "odim"
