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


def test_discover_engine_does_not_hide_bugs(monkeypatch, odim_file):
    # an unexpected error in a guess_can_open is a bug, discovery lets it through
    def _raise(self, filename_or_obj):
        raise RuntimeError("broken")

    monkeypatch.setattr(
        type(plugins.get_backend("cfradial1")), "guess_can_open", _raise
    )
    with pytest.raises(RuntimeError, match="broken"):
        xd.io.discover_engine(odim_file)


@pytest.fixture
def unreadable_inputs(
    tmp_path, odim_file, cfradial1_file, iris1_file, datamet_file, furuno_scn_file
):
    """Inputs which look like a radar file at first but can't be read."""

    def head(path, size):
        with open(path, "rb") as fh:
            return fh.read(size)

    def write(name, data):
        path = tmp_path / name
        path.write_bytes(data)
        return path

    nc3 = tmp_path / "ok3.nc"
    xr.Dataset({"a": ("x", range(50))}).to_netcdf(nc3, engine="scipy")
    corrupt_gz = bytearray(head(furuno_scn_file, 4000))
    corrupt_gz[100:140] = bytes(40)  # raises zlib.error

    def flip(source, offset, value, size=None):
        data = bytearray(head(source, size))
        data[offset] = value
        return bytes(data)

    closed = open(tmp_path / "closed.bin", "wb")
    closed.close()
    (tmp_path / "folder").mkdir()
    return {
        "hdf5_signature_then_garbage": write(
            "garbage.h5", b"\x89HDF\r\n\x1a\n" + bytes(range(256)) * 2
        ),
        "hdf5_truncated": write("t.h5", head(odim_file, 2048)),
        "netcdf4_truncated": write("t.nc", head(cfradial1_file, 3000)),
        "netcdf3_truncated": write("t3.nc", head(nc3, 40)),
        "netcdf3_header_only": write("h3.nc", b"CDF\x01"),
        "iris_truncated_header": write("t.iris", head(iris1_file, 100)),
        # one changed byte makes the parsers raise exceptions other than OSError
        # or ValueError (found by changing bytes of the sample files)
        "netcdf3_bad_byte_IndexError": write("b12.nc", flip(nc3, 12, 0x7F)),
        "netcdf3_bad_byte_KeyError": write("b32.nc", flip(nc3, 32, 0x7F)),
        "hdf5_bad_byte_RuntimeError": write("b17.h5", flip(odim_file, 17, 0xFF)),
        "hdf5_bad_byte_KeyError": write("b112.h5", flip(odim_file, 112, 0xFF)),
        "iris_bad_product_type_KeyError": write(
            "b24.iris", flip(iris1_file, 24, 0xFF, 3000)
        ),
        "datamet_truncated_tar": write("t.tar", head(datamet_file, 300)),
        "gzip_corrupt": write("c.scn.gz", bytes(corrupt_gz)),
        "gzip_truncated": write("t.scn.gz", head(furuno_scn_file, 20)),
        "closed_file_object": closed,
        "directory": tmp_path / "folder",
        "missing_file": tmp_path / "missing.h5",
    }


@pytest.mark.parametrize(
    "name",
    [
        "hdf5_signature_then_garbage",
        "hdf5_truncated",
        "netcdf4_truncated",
        "netcdf3_truncated",
        "netcdf3_header_only",
        "iris_truncated_header",
        "netcdf3_bad_byte_IndexError",
        "netcdf3_bad_byte_KeyError",
        "hdf5_bad_byte_RuntimeError",
        "hdf5_bad_byte_KeyError",
        "iris_bad_product_type_KeyError",
        "datamet_truncated_tar",
        "gzip_corrupt",
        "gzip_truncated",
        "closed_file_object",
        "directory",
        "missing_file",
    ],
)
def test_guess_can_open_unreadable_input(name, unreadable_inputs):
    # every engine says no, and discovery ends with its own error, not with an
    # OSError, struct.error, ... of whatever the engines tried to read
    filename = unreadable_inputs[name]
    for engine in xd.io.list_xradar_engines():
        assert not plugins.get_backend(engine).guess_can_open(filename), engine
    with pytest.raises(ValueError, match="None of the xradar engines"):
        xd.io.discover_engine(filename)


def test_read_head_unexpected_errors_are_not_caught():
    from xradar.io.backends.common import _read_head, _read_nc_header

    class Broken(io.RawIOBase):
        def readable(self):
            return True

        def seekable(self):
            return True

        def tell(self):
            return 0

        def seek(self, *args):
            return 0

        def read(self, size=-1):
            raise TypeError("a bug, not an unreadable file")

    with pytest.raises(TypeError, match="a bug"):
        _read_head(Broken())
    with pytest.raises(TypeError, match="a bug"):
        _read_nc_header(Broken())
