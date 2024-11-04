import logging
import signal
from unittest.mock import MagicMock, patch

import pytest
from open_radar_data import DATASETS

from xradar.io import auto_read


# Mocked functions for testing
def mock_open_success(file):
    mock_dtree = MagicMock(name="DataTree")
    # Mock the behavior of georeference to return the same object without changing attrs
    mock_dtree.xradar.georeference.return_value = mock_dtree
    return mock_dtree


def mock_open_failure(file):
    raise ValueError("Failed to open the file.")


def mock_georeference(dtree):
    return dtree


def mock_timeout_handler(signum, frame):
    raise auto_read.TimeoutException("Radar file reading timed out.")


@pytest.fixture
def sample_file():
    # Fetch a sample radar file for testing
    return DATASETS.fetch("KATX20130717_195021_V06")


# Test case for the timeout handler
def test_timeout_handler():
    # Mock the signal handling and ensure it raises the timeout exception as expected.
    with (
        patch("signal.signal"),
        pytest.raises(
            auto_read.TimeoutException, match="Radar file reading timed out."
        ),
    ):
        auto_read.timeout_handler(signal.SIGALRM, None)


# Test that the alarm is disabled after reading the radar file with timeout
def test_timeout_alarm_disabled(sample_file):
    # Mock the alarm function and check if it gets called with 0 to disable it.
    with (
        patch("xradar.io.auto_read.io.__all__", ["open_nexradlevel2_datatree"]),
        patch(
            "xradar.io.auto_read.io.open_nexradlevel2_datatree",
            side_effect=mock_open_success,
        ),
        patch("signal.signal"),
        patch("signal.alarm") as mock_alarm,
    ):
        auto_read.read(sample_file, timeout=1)
        # Ensure that alarm was called with 0 to disable it after the read.
        mock_alarm.assert_called_with(0)


# Test the nonlocal dtree block
def test_read_nonlocal_dtree(sample_file):
    # Test that it attempts multiple open_ functions and finally succeeds.
    with (
        patch(
            "xradar.io.auto_read.io.__all__",
            ["open_nexradlevel2_datatree", "open_odim_datatree"],
        ),
        patch(
            "xradar.io.auto_read.io.open_nexradlevel2_datatree",
            side_effect=mock_open_failure,
        ),
        patch(
            "xradar.io.auto_read.io.open_odim_datatree", side_effect=mock_open_success
        ),
        patch("xarray.core.dataset.Dataset.pipe", side_effect=mock_georeference),
    ):
        dtree = auto_read.read(sample_file, georeference=True, verbose=True)
        assert dtree is not None


# Test for georeferencing and verbose output
def test_read_with_georeferencing_and_logging(sample_file, caplog):
    # Test successful reading with georeferencing and log output.
    with (
        patch("xradar.io.auto_read.io.__all__", ["open_nexradlevel2_datatree"]),
        patch(
            "xradar.io.auto_read.io.open_nexradlevel2_datatree",
            side_effect=mock_open_success,
        ),
        patch("xarray.core.dataset.Dataset.pipe", side_effect=mock_georeference),
    ):
        with caplog.at_level(logging.DEBUG):
            dtree = auto_read.read(sample_file, georeference=True, verbose=True)
            assert dtree is not None

            # Check that the log messages contain the correct information
            assert "Georeferencing radar data..." in caplog.text
            assert (
                "File opened successfully using open_nexradlevel2_datatree."
                in caplog.text
            )


# Test for exception handling and verbose output during failure
def test_read_failure_with_verbose_output(sample_file, caplog):
    # Test that it handles exceptions and logs the verbose failure message.
    with (
        patch("xradar.io.auto_read.io.__all__", ["open_nexradlevel2_datatree"]),
        patch(
            "xradar.io.auto_read.io.open_nexradlevel2_datatree",
            side_effect=mock_open_failure,
        ),
    ):
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(
                ValueError,
                match="File could not be opened by any supported format in xradar.io.",
            ):
                auto_read.read(sample_file, georeference=True, verbose=True)

            # Check that the failure log messages contain the correct information
            assert (
                "Failed to open with open_nexradlevel2_datatree" in caplog.text
            )  # Capturing log instead of print


# Test for raising ValueError when no format can open the file
def test_read_failure_raises_value_error(sample_file):
    # Test that it raises ValueError when no format can open the file.
    with (
        patch(
            "xradar.io.auto_read.io.__all__",
            ["open_nexradlevel2_datatree", "open_odim_datatree"],
        ),
        patch(
            "xradar.io.auto_read.io.open_nexradlevel2_datatree",
            side_effect=mock_open_failure,
        ),
        patch(
            "xradar.io.auto_read.io.open_odim_datatree", side_effect=mock_open_failure
        ),
    ):
        with pytest.raises(
            ValueError,
            match="File could not be opened by any supported format in xradar.io.",
        ):
            auto_read.read(sample_file)


# Test read success with georeferencing
def test_read_success(sample_file):
    # Test successful reading without timeout and with georeferencing
    with (
        patch("xradar.io.auto_read.io.__all__", ["open_nexradlevel2_datatree"]),
        patch(
            "xradar.io.auto_read.io.open_nexradlevel2_datatree",
            side_effect=mock_open_success,
        ),
        patch("xarray.core.dataset.Dataset.pipe", side_effect=mock_georeference),
    ):
        dtree = auto_read.read(sample_file, georeference=True, verbose=True)
        assert dtree is not None


# Test read success without georeferencing
def test_read_success_without_georeference(sample_file):
    # Test successful reading without georeferencing
    with (
        patch("xradar.io.auto_read.io.__all__", ["open_nexradlevel2_datatree"]),
        patch(
            "xradar.io.auto_read.io.open_nexradlevel2_datatree",
            side_effect=mock_open_success,
        ),
    ):
        dtree = auto_read.read(sample_file, georeference=False)
        assert dtree is not None


# Test read with timeout raising a TimeoutException
def test_read_with_timeout(sample_file):
    # Mock the signal handling and ensure it raises the timeout as expected.
    with (
        patch("xradar.io.auto_read.io.__all__", ["open_nexradlevel2_datatree"]),
        patch(
            "xradar.io.auto_read.io.open_nexradlevel2_datatree",
            side_effect=mock_open_success,
        ),
        patch("signal.signal"),
        patch(
            "signal.alarm",
            side_effect=lambda _: mock_timeout_handler(signal.SIGALRM, None),
        ),
    ):
        with pytest.raises(
            auto_read.TimeoutException, match="Radar file reading timed out."
        ):
            auto_read.read(sample_file, timeout=1)


def test_read_comment_update(sample_file):
    with (
        patch("xradar.io.auto_read.io.__all__", ["open_nexradlevel2_datatree"]),
        patch(
            "xradar.io.auto_read.io.open_nexradlevel2_datatree",
            side_effect=mock_open_success,
        ),
    ):
        dtree = auto_read.read(sample_file)
        assert dtree is not None

        # Print the actual value of the 'comment' attribute for debugging
        print(f"Actual comment: {dtree.attrs['comment']}")

        # The initial comment is "im/exported using xradar" (lowercase 'i')
        expected_comment_start = "im/exported using xradar"

        # Ensure the comment starts with the correct initial value
        assert dtree.attrs["comment"].startswith(expected_comment_start)

        # Ensure the comment has the correct format with 'nexradlevel2'
        expected_comment_end = ",\n'nexradlevel2'"
        assert dtree.attrs["comment"].endswith(expected_comment_end)
