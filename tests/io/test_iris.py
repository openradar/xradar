#!/usr/bin/env python
# Copyright (c) 2023, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for `io.backends.iris` module.

Ported from wradlib.
"""

import numpy as np

from xradar.io.backends import iris
from xradar.util import _get_data_file


def test_open_iris(iris0_file, file_or_filelike):
    with _get_data_file(iris0_file, file_or_filelike) as sigmetfile:
        data = iris.IrisRawFile(sigmetfile, loaddata=False)
    assert isinstance(data.rh, iris.IrisRecord)
    assert isinstance(data.fh, (np.memmap, np.ndarray))
    with _get_data_file(iris0_file, file_or_filelike) as sigmetfile:
        data = iris.IrisRawFile(sigmetfile, loaddata=True)
    assert data._record_number == 511
    assert data.filepos == 3139584


def test_IrisRecord(iris0_file, file_or_filelike):
    with _get_data_file(iris0_file, file_or_filelike) as sigmetfile:
        data = iris.IrisRecordFile(sigmetfile, loaddata=False)
    # reset record after init
    data.init_record(1)
    assert isinstance(data.rh, iris.IrisRecord)
    assert data.rh.pos == 0
    assert data.rh.recpos == 0
    assert data.rh.recnum == 1
    rlist = [23, 0, 4, 0, 20, 19, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    np.testing.assert_array_equal(data.rh.read(10, 2), rlist)
    assert data.rh.pos == 20
    assert data.rh.recpos == 10
    data.rh.pos -= 20
    np.testing.assert_array_equal(data.rh.read(20, 1), rlist)
    data.rh.recpos -= 10
    np.testing.assert_array_equal(data.rh.read(5, 4), rlist)


def test_decode_bin_angle():
    assert iris.decode_bin_angle(20000, 2) == 109.86328125
    assert iris.decode_bin_angle(2000000000, 4) == 167.63806343078613


def decode_array():
    data = np.arange(0, 11)
    np.testing.assert_array_equal(
        iris.decode_array(data),
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    np.testing.assert_array_equal(
        iris.decode_array(data, offset=1.0),
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
    )
    np.testing.assert_array_equal(
        iris.decode_array(data, scale=0.5),
        [0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
    )
    np.testing.assert_array_equal(
        iris.decode_array(data, offset=1.0, scale=0.5),
        [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0],
    )
    np.testing.assert_array_equal(
        iris.decode_array(data, offset=1.0, scale=0.5, offset2=-2.0),
        [0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
    )
    data = np.array(
        [0, 1, 255, 1000, 9096, 22634, 34922, 50000, 65534], dtype=np.uint16
    )
    np.testing.assert_array_equal(
        iris.decode_array(data, scale=1000, tofloat=True),
        [0.0, 0.001, 0.255, 1.0, 10.0, 100.0, 800.0, 10125.312, 134184.96],
    )


def test_decode_velc():
    data = [0, 1, 2, 128, 129, 254, 255]
    np.testing.assert_array_almost_equal(
        iris.decode_array(data, scale=127 / 75.0, offset=-1, offset2=-75, mask=0.0),
        [np.inf, -75.0, -74.409449, 0.0, 0.590551, 74.409449, 75.0],
    )


def test_decode_kdp():
    np.testing.assert_array_almost_equal(
        iris.decode_kdp(np.arange(-5, 5, dtype="int8"), wavelength=10.0),
        [
            12.243229,
            12.880858,
            13.551695,
            14.257469,
            15.0,
            -0.0,
            -15.0,
            -14.257469,
            -13.551695,
            -12.880858,
        ],
    )


def test_decode_phidp():
    np.testing.assert_array_almost_equal(
        iris.decode_phidp(np.arange(0, 10, dtype="uint8"), scale=254.0, offset=-1),
        [
            -0.70866142,
            0.0,
            0.70866142,
            1.41732283,
            2.12598425,
            2.83464567,
            3.54330709,
            4.2519685,
            4.96062992,
            5.66929134,
        ],
    )


def test_decode_phidp2():
    np.testing.assert_array_almost_equal(
        iris.decode_phidp2(np.arange(0, 10, dtype="uint16"), scale=65534.0, offset=-1),
        [
            -0.00549333,
            0.0,
            0.00549333,
            0.01098666,
            0.01648,
            0.02197333,
            0.02746666,
            0.03295999,
            0.03845332,
            0.04394665,
        ],
    )


def test_decode_sqi():
    np.testing.assert_array_almost_equal(
        iris.decode_sqi(np.arange(0, 10, dtype="uint8"), scale=253.0, offset=-1),
        [
            np.nan,
            0.0,
            0.06286946,
            0.08891084,
            0.1088931,
            0.12573892,
            0.14058039,
            0.1539981,
            0.16633696,
            0.17782169,
        ],
    )


def test_decode_rainrate2():
    vals = np.array(
        [0, 1, 2, 255, 1000, 9096, 22634, 34922, 50000, 65534, 65535],
        dtype="uint16",
    )
    prod = iris.SIGMET_DATA_TYPES[13]
    np.testing.assert_array_almost_equal(
        iris.decode_array(vals.copy(), **prod["fkw"]),
        [
            -1.00000000e-04,
            0.00000000e00,
            1.00000000e-04,
            2.54000000e-02,
            9.99000000e-02,
            9.99900000e-01,
            9.99990000e00,
            7.99999000e01,
            1.01253110e03,
            1.34184959e04,
            1.34201343e04,
        ],
    )


def test_decode_time():
    timestring = b"\xd1\x9a\x00\x000\t\xdd\x07\x0b\x00\x19\x00"
    assert (
        iris.decode_time(timestring).isoformat() == "2013-11-25T11:00:33.304000+00:00"
    )


def test_decode_string():
    assert iris.decode_string(b"EEST\x00\x00\x00\x00") == "EEST"


def test__get_fmt_string():
    fmt = "<12sHHi12s12s12s6s12s12sHiiiiiiiiii2sH12sHB1shhiihh80s16s12s48s"
    assert iris._get_fmt_string(iris.PRODUCT_CONFIGURATION) == fmt
