#!/usr/bin/env python
# Copyright (c) 2023-2025, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for `io.backends.iris` module.

Ported from wradlib.
"""

import numpy as np
from xarray import DataTree

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


def test_read_from_record(iris0_file, file_or_filelike):
    """Test reading a specified number of words from a record."""
    with _get_data_file(iris0_file, file_or_filelike) as sigmetfile:
        data = iris.IrisRecordFile(sigmetfile, loaddata=True)
        data.init_record(0)  # Start from the first record
        record_data = data.read_from_record(10, dtype="int16")
        assert len(record_data) == 10
        assert isinstance(record_data, np.ndarray)


def test_decode_data(iris0_file, file_or_filelike):
    """Test decoding of data with provided product function."""

    # Sample data to decode
    data = np.array([0, 2, 3, 128, 255], dtype="int16")
    # Sample product dict with decoding function and parameters
    prod = {
        "func": iris.decode_vel,
        "dtype": "int16",
        "fkw": {"scale": 0.5, "offset": -1},
    }

    # Open the file as per the testing framework
    with _get_data_file(iris0_file, file_or_filelike) as sigmetfile:
        iris_file = iris.IrisRawFile(sigmetfile, loaddata=False)

        # Decode data using the provided product function
        decoded_data = iris_file.decode_data(data, prod)

    # Check that the decoded data is as expected
    assert isinstance(decoded_data, np.ndarray), "Decoded data should be a numpy array"
    assert decoded_data.dtype in [
        np.float32,
        np.float64,
    ], "Decoded data should have float32 or float64 type"

    # Expected decoded values
    expected_data = [-13.325, 13.325, 26.65, 1692.275, 3384.55]
    np.testing.assert_array_almost_equal(decoded_data, expected_data, decimal=2)


def test_get_sweep(iris0_file, file_or_filelike):
    """Test retrieval of sweep data for specified moments."""

    # Select the sweep number and moments to retrieve
    sweep_number = 1
    moments = ["DB_DBZ", "DB_VEL"]

    # Open the file and load data
    with _get_data_file(iris0_file, file_or_filelike) as sigmetfile:
        iris_file = iris.IrisRawFile(sigmetfile, loaddata=True)

        # Use get_sweep to retrieve data for the selected sweep and moments
        iris_file.get_sweep(sweep_number, moments)
        sweep_data = iris_file.data[sweep_number]["sweep_data"]

    # Verify that sweep_data structure is populated with the selected moments
    for moment in moments:
        assert moment in sweep_data, f"{moment} should be in sweep_data"
        moment_data = sweep_data[moment]
        assert moment_data.shape == (360, 664), f"{moment} data shape mismatch"

        # Check data types for moments, including masked arrays for velocity
        if moment == "DB_VEL":
            assert isinstance(
                moment_data, np.ma.MaskedArray
            ), "DB_VEL should be a masked array"
        else:
            assert isinstance(
                moment_data, np.ndarray
            ), f"{moment} should be a numpy array"

        # Optional: check for expected placeholder/masked values
        if moment == "DB_DBZ":
            assert (
                moment_data == -32
            ).sum() > 0, "DB_DBZ should contain placeholder values (-32)"
        if moment == "DB_VEL":
            assert moment_data.mask.sum() > 0, "DB_VEL should have masked values"


def test_array_from_file(iris0_file, file_or_filelike):
    """Test retrieving an array from a file."""
    with _get_data_file(iris0_file, file_or_filelike) as sigmetfile:
        data = iris.IrisRawFile(sigmetfile, loaddata=True)
        array_data = data.read_from_file(5)  # Adjusted to read_from_file

        # Assertions for the read array
        assert len(array_data) == 5
        assert isinstance(array_data, np.ndarray)


from xradar.io.backends.iris import open_iris_datatree


def test_open_iris_datatree(iris0_file):
    # Define kwargs to pass into the function
    kwargs = {
        "sweep": [0, 1, 2, 4],  # Test with specific sweeps
        "first_dim": "auto",
        "reindex_angle": {
            "start_angle": 0.0,
            "stop_angle": 360.0,
            "angle_res": 1.0,
            "direction": 1,
        },
        "fix_second_angle": True,
        "site_coords": True,
    }

    # Call the function with an actual Iris/Sigmet file
    dtree = open_iris_datatree(iris0_file, **kwargs)

    # Assertions
    assert isinstance(dtree, DataTree), "Expected a DataTree instance"
    assert "/" in dtree.subtree, "Root group should be present in the DataTree"
    assert (
        "/radar_parameters" in dtree.subtree
    ), "Radar parameters group should be in the DataTree"
    assert (
        "/georeferencing_correction" in dtree.subtree
    ), "Georeferencing correction group should be in the DataTree"
    assert (
        "/radar_calibration" in dtree.subtree
    ), "Radar calibration group should be in the DataTree"

    # Check if at least one sweep group is attached (e.g., "/sweep_0")
    sweep_groups = [key for key in dtree.match("sweep_*")]
    assert len(sweep_groups) == 4, "Expected four sweep groups in the DataTree"

    # Verify a sample variable in one of the sweep groups (adjust based on expected variables)
    sample_sweep = sweep_groups[0]
    assert (
        len(dtree[sample_sweep].data_vars) == 12
    ), f"Expected data variables in {sample_sweep}"
    assert dtree[sample_sweep]["DBZH"].shape == (360, 664)
    assert (
        "DBZH" in dtree[sample_sweep].data_vars
    ), f"DBZH should be a data variable in {sample_sweep}"
    assert (
        "VRADH" in dtree[sample_sweep].data_vars
    ), f"VRADH should be a data variable in {sample_sweep}"

    # Validate coordinates are attached correctly in the root dataset
    assert (
        "latitude" in dtree[sample_sweep]
    ), "Latitude should be attached to the root dataset"
    assert (
        "longitude" in dtree[sample_sweep]
    ), "Longitude should be attached to the root dataset"
    assert (
        "altitude" in dtree[sample_sweep]
    ), "Altitude should be attached to the root dataset"

    # Validate attributes
    assert len(dtree.attrs) == 10
    assert (
        dtree.attrs["instrument_name"] == "Corozal, Radar"
    ), "Instrument name should match expected value"
    assert dtree.attrs["source"] == "Sigmet", "Source should match expected value"
