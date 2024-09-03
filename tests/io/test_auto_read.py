import signal
from unittest.mock import MagicMock, patch

import pytest
from open_radar_data import DATASETS

from xradar.io import auto_read


# Mocked functions for testing
def mock_open_success(file):
    mock_dtree = MagicMock(name="DataTree")
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


def test_read_failure(sample_file):
    # Test that it raises ValueError when no format can open the file
    with (
        patch("xradar.io.auto_read.io.__all__", ["open_nexradlevel2_datatree"]),
        patch(
            "xradar.io.auto_read.io.open_nexradlevel2_datatree",
            side_effect=mock_open_failure,
        ),
    ):
        with pytest.raises(
            ValueError,
            match="File could not be opened by any supported format in xradar.io.",
        ):
            auto_read.read(sample_file)


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
