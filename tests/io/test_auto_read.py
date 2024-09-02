#!/usr/bin/env python
# Copyright (c) 2022-2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

from pathlib import Path

import pytest
from open_radar_data import DATASETS

import xradar as xd
from xradar.io import read  # noqa


@pytest.fixture
def test_file():
    """Fixture to provide a valid radar data file for testing."""
    return Path(DATASETS.fetch("KATX20130717_195021_V06"))


def test_read_failed_opening(monkeypatch, test_file):
    """Test that the read function handles failed opening attempts."""

    def mock_open_func(file):
        raise OSError("Simulated failure")

    # Replace all open functions with the mock that raises an error
    monkeypatch.setattr(xd.io, "open_cfradial1_datatree", mock_open_func)
    monkeypatch.setattr(xd.io, "open_datamet_datatree", mock_open_func)
    monkeypatch.setattr(xd.io, "open_furuno_datatree", mock_open_func)
    monkeypatch.setattr(xd.io, "open_gamic_datatree", mock_open_func)
    monkeypatch.setattr(xd.io, "open_hpl_datatree", mock_open_func)
    monkeypatch.setattr(xd.io, "open_iris_datatree", mock_open_func)
    monkeypatch.setattr(xd.io, "open_nexradlevel2_datatree", mock_open_func)

    with pytest.raises(ValueError, match="File could not be opened"):
        read(test_file, verbose=True)
