#!/usr/bin/env python
# Copyright (c) 2023, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for `io.backends.rainbow` module.

Ported from wradlib.
"""

import datetime
import zlib
from io import BytesIO

import numpy as np
import pytest
import xmltodict

from xradar.io.backends import rainbow


def test_find_key():
    indict = {
        "A": {
            "AA": {"AAA": 0, "X": 1},
            "AB": {"ABA": 2, "X": 3},
            "AC": {"ACA": 4, "X": 5},
            "AD": [{"ADA": 4, "X": 2}],
        }
    }
    outdict = [
        {"X": 1, "AAA": 0},
        {"ABA": 2, "X": 3},
        {"X": 5, "ACA": 4},
        {"ADA": 4, "X": 2},
    ]

    assert list(rainbow.find_key("X", indict)) == outdict
    assert list(rainbow.find_key("Y", indict)) == []


def test_decompress():
    dstring = b"very special compressed string"
    cstring = zlib.compress(dstring)
    assert rainbow.decompress(cstring) == dstring


def test_get_rb_data_layout():
    assert rainbow.get_rb_data_layout(8) == (1, ">u1")
    assert rainbow.get_rb_data_layout(16) == (2, ">u2")
    assert rainbow.get_rb_data_layout(32) == (4, ">u4")
    with pytest.raises(ValueError):
        rainbow.get_rb_data_layout(128)


def test_get_rb_data_layout_big(monkeypatch):
    monkeypatch.setattr("sys.byteorder", "big")
    assert rainbow.get_rb_data_layout(8) == (1, "<u1")
    assert rainbow.get_rb_data_layout(16) == (2, "<u2")
    assert rainbow.get_rb_data_layout(32) == (4, "<u4")


def test_get_rb_data_attribute():
    data = xmltodict.parse(
        '<slicedata time="13:30:05" date="2013-04-26">'
        '#<rayinfo refid="startangle" blobid="0" '
        'rays="361" depth="16"/> '
        '#<rawdata blobid="1" rays="361" type="dBuZ" '
        'bins="400" min="-31.5" max="95.5" '
        'depth="8"/> #</slicedata>'
    )
    data = list(rainbow.find_key("@blobid", data))
    assert rainbow.get_rb_data_attribute(data[0], "blobid") == 0
    assert rainbow.get_rb_data_attribute(data[1], "blobid") == 1
    assert rainbow.get_rb_data_attribute(data[0], "rays") == 361
    assert rainbow.get_rb_data_attribute(data[1], "rays") == 361
    assert rainbow.get_rb_data_attribute(data[1], "bins") == 400
    with pytest.raises(KeyError):
        rainbow.get_rb_data_attribute(data[0], "Nonsense")
    assert rainbow.get_rb_data_attribute(data[0], "depth") == 16


def test_get_rb_blob_attribute():
    xmldict = xmltodict.parse('<BLOB blobid="0" size="737" compression="qt"></BLOB>')
    assert rainbow.get_rb_blob_attribute(xmldict, "compression") == "qt"
    assert rainbow.get_rb_blob_attribute(xmldict, "size") == "737"
    assert rainbow.get_rb_blob_attribute(xmldict, "blobid") == "0"
    with pytest.raises(KeyError):
        rainbow.get_rb_blob_attribute(xmldict, "Nonsense")


def test_get_rb_data_shape():
    data = xmltodict.parse(
        '<slicedata time="13:30:05" date="2013-04-26">'
        '#<rayinfo refid="startangle" blobid="0" '
        'rays="361" depth="16"/> #<rawdata blobid="1" '
        'rays="361" type="dBuZ" bins="400" '
        'min="-31.5" max="95.5" depth="8"/> #<flagmap '
        'blobid="2" rows="800" type="dBuZ" '
        'columns="400" min="-31.5" max="95.5" '
        'depth="6"/> #<defect blobid="3" type="dBuZ" '
        'columns="400" min="-31.5" max="95.5" '
        'depth="6"/> #<rawdata2 '
        'blobid="4" rows="800" type="dBuZ" '
        'columns="400" min="-31.5" max="95.5" '
        'depth="8"/> #</slicedata>'
    )
    data = list(rainbow.find_key("@blobid", data))
    assert rainbow.get_rb_data_shape(data[0]) == 361
    assert rainbow.get_rb_data_shape(data[1]) == (361, 400)
    assert rainbow.get_rb_data_shape(data[2]) == (800, 400, 6)
    assert rainbow.get_rb_data_shape(data[4]) == (800, 400)
    with pytest.raises(KeyError):
        rainbow.get_rb_data_shape(data[3])


def test_map_rb_data():
    indata = b"0123456789"
    outdata8 = np.array([48, 49, 50, 51, 52, 53, 54, 55, 56, 57], dtype=np.uint8)
    outdata16 = np.array([12337, 12851, 13365, 13879, 14393], dtype=np.uint16)
    outdata32 = np.array([808530483, 875902519], dtype=np.uint32)
    np.testing.assert_allclose(rainbow.map_rb_data(indata, 8), outdata8)
    np.testing.assert_allclose(rainbow.map_rb_data(indata, 16), outdata16)
    np.testing.assert_allclose(rainbow.map_rb_data(indata, 32), outdata32)
    flagdata = b"1"
    np.testing.assert_allclose(
        rainbow.map_rb_data(flagdata, 1, 8), [0, 0, 1, 1, 0, 0, 0, 1]
    )
    # added test for truncation
    np.testing.assert_allclose(rainbow.map_rb_data(flagdata, 1, 6), [0, 0, 1, 1, 0, 0])


def test_get_rb_blob_data():
    datastring = b'<BLOB blobid="0" size="737" compression="qt"></BLOB>'
    with pytest.raises(EOFError):
        rainbow.get_rb_blob_data(datastring, 1)


def test_get_rb_header():
    rb_header = (
        b'<volume version="5.34.16" '
        b'datetime="2013-07-03T08:33:55"'
        b' type="azi" owner="RainAnalyzer"> '
        b'<scan name="analyzer.azi" time="08:34:00" '
        b'date="2013-07-03">'
    )

    buf = BytesIO(rb_header)
    with pytest.raises(IOError):
        rainbow.get_rb_header(buf)


def test_get_rb_header_from_file(rainbow_file):
    with open(rainbow_file, "rb") as rb_fh:
        rb_header = rainbow.get_rb_header(rb_fh)
        assert rb_header["volume"]["@version"] == "5.36.5"


def test_rainbow_file_meta(rainbow_file):
    with rainbow.RainbowFile(rainbow_file, loaddata=False) as rbdict:
        assert rbdict.version == "5.36.5"
        assert rbdict.header
        assert rbdict.site_coords == (6.379967, 50.856633, 116.7)
        assert rbdict.first_dimension == "azimuth"
        assert rbdict.history is None
        assert rbdict.pargroup
        assert rbdict.sensorinfo == dict(
            [
                ("@id", "143DEX"),
                ("@name", "Gematronik"),
                ("@type", "gdrx"),
                ("alt", "116.700000"),
                ("beamwidth", "1.326"),
                ("lat", "50.856633"),
                ("lon", "6.379967"),
                ("wavelen", "0.0319"),
            ]
        )
        assert rbdict.datetime == datetime.datetime(2013, 5, 10, 0, 3, 17)
        assert rbdict.type == "vol"


def test_rainbow_file_data(rainbow_file):
    with rainbow.RainbowFile(rainbow_file, loaddata=True) as rbdict:
        sdata = rbdict.slices[0]["slicedata"]
        print(sdata["rayinfo"]["data"][0])
        np.testing.assert_equal(sdata["rayinfo"]["data"][0], np.array(47.0159912109375))
        np.testing.assert_equal(sdata["rawdata"]["data"][0, 0], np.array(113))
        assert sdata["rawdata"]["data"].shape == (361, 400)
