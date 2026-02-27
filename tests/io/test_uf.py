#!/usr/bin/env python
# Copyright (c) 2024-2025, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for `xradar.io.uf` module."""

import io
import os
from collections import OrderedDict

import numpy as np
import pytest
import xarray
from xarray import DataTree

from xradar.io.backends.uf import (
    UFBackendEntrypoint,
    UFFile,
    open_uf_datatree,
)


def test_open_uf_backend(uf_file_1):
    with UFFile(uf_file_1, loaddata=False) as uff:
        nsweeps = uff.nsweeps

    sweeps = [f"sweep_{i}" for i in range(nsweeps)]
    assert nsweeps == 14
    for i, group in enumerate(sweeps):
        with xarray.open_dataset(
            uf_file_1, engine=UFBackendEntrypoint, group=group
        ) as ds:
            assert ds.attrs["instrument_name"] == "rvp8-rel"
            assert int(ds.sweep_number.values) == i


def test_open_uf_file(uf_file_1):
    with UFFile(uf_file_1) as fh:

        assert fh.endianness == "big"
        assert fh.byteorder == ">"
        assert fh.nsweeps == 14
        assert os.path.basename(fh.filename) == "20110427_164233_rvp8-rel_v001_SUR.uf"
        assert list(fh.ray_indices[0:10]) == [
            0,
            18564,
            37128,
            55692,
            74256,
            92820,
            111384,
            129948,
            148512,
            167076,
        ]
        assert fh.moments[1] == ["DB", "DZ", "VR", "SW", "ZD", "KD", "PH", "SQ", "RH"]

        # ray headers
        sw1h0 = fh.ray_headers[1][0]
        mhead = sw1h0["mhead"]
        assert mhead["ID"] == "UF"
        assert mhead["RecordSize"] == 9278
        assert mhead["OptionalHeaderPosition"] == 46
        assert mhead["LocalUseHeaderPosition"] == 60
        assert mhead["DataHeaderPosition"] == 60
        assert mhead["RecordNumber"] == 0
        assert mhead["VolumeNumber"] == 1
        assert mhead["RayNumber"] == 0
        assert mhead["RecordInRay"] == 1
        assert mhead["SweepNumber"] == 1
        assert mhead["RadarName"] == "rvp8-rel"
        assert mhead["SiteName"] == "MAX"
        assert mhead["LatDegrees"] == 34
        assert mhead["LatMinutes"] == 55
        assert mhead["LatSeconds"] == 54.515625
        assert mhead["LonDegrees"] == -86
        assert mhead["LonMinutes"] == -27
        assert mhead["LonSeconds"] == -56.953125
        assert mhead["Altitude"] == 226
        assert mhead["Year"] == 11
        assert mhead["Month"] == 4
        assert mhead["Day"] == 27
        assert mhead["Hour"] == 16
        assert mhead["Minute"] == 42
        assert mhead["Second"] == 33
        assert mhead["TimeZone"] == "UT"
        assert mhead["Azimuth"] == 211.859375
        assert mhead["Elevation"] == 0.71875
        assert mhead["SweepMode"] == 8
        assert mhead["FixedAngle"] == 0.703125
        assert mhead["SweepRate"] == -512.0
        assert mhead["ConvertYear"] == 12
        assert mhead["ConvertMonth"] == 8
        assert mhead["ConvertDay"] == 2
        assert mhead["ConvertName"] == "Sigmet I"
        assert mhead["NoDataValue"] == -32768

        ohead = sw1h0["ohead"]
        assert ohead["ProjectName"] == "FAY-FULL"
        assert ohead["BaselineAzimuth"] == -6336
        assert ohead["BaselineElevation"] == -6336
        assert ohead["VolumeScanHour"] == 16
        assert ohead["VolumeScanMinute"] == 42
        assert ohead["VolumeScanSecond"] == 33
        assert ohead["FieldTapeName"] == "Conversi"
        assert ohead["Flag"] == 0

        dhead = sw1h0["dhead"]
        assert dhead["FieldsThisRay"] == 9
        assert dhead["RecordsThisRay"] == 1
        assert dhead["FieldsThisRecord"] == 9
        assert dhead["fields"]["DB"] == OrderedDict(
            [
                ("FieldHeaderPosition", 81),
                ("DataPosition", 106),
                ("ScaleFactor", 100),
                ("StartRangeKm", 1),
                ("StartRangeMeters", -499),
                ("BinSpacing", 125),
                ("BinCount", 997),
                ("PulseWidth", -32768),
                ("BeamWidthH", 0.953125),
                ("BeamWidthV", 0.953125),
                ("BandWidth", -156.234375),
                ("Polarization", 0),
                ("WaveLength", 3.171875),
                ("SampleSize", 64),
                ("ThresholdData", ""),
                ("ThresholdValue", -9999),
                ("Scale", 100),
                ("EditCode", "NO"),
                ("PRT", 1000),
                ("BitsPerBin", -32768),
                (
                    "DM",
                    OrderedDict(
                        [
                            ("RadarConstant", -42),
                            ("NoisePower", -10676),
                            ("ReceiverGain", -32768),
                            ("PeakPower", 8342),
                            ("AntennaGain", -32768),
                            ("PulseDuration", 0.796875),
                        ]
                    ),
                ),
            ]
        )


def test_open_uf_file2(uf_file_2):
    with UFFile(uf_file_2) as fh:
        assert fh.endianness == "big"
        assert fh.byteorder == ">"
        assert fh.nsweeps == 7
        assert os.path.basename(fh.filename) == "CHL19950617_182848.uf"
        assert list(fh.ray_indices[0:10]) == [
            0,
            8154,
            16308,
            24462,
            32616,
            40770,
            48924,
            57078,
            65232,
            73386,
        ]
        assert fh.moments[1] == ["DM", "DZ", "VR", "DR", "LD", "LC", "RX", "NC"]
        assert fh.ray_headers[1][0]["mhead"]["SweepMode"] == 1


def test_open_uf_file3(uf_file_3):
    with UFFile(uf_file_3) as fh:
        assert fh.endianness == "big"
        assert fh.byteorder == ">"
        assert fh.nsweeps == 3
        assert os.path.basename(fh.filename) == "MC3E_NPOL_2011_0524_2356_hid.uf"
        assert list(fh.ray_indices[0:10]) == [
            0,
            24616,
            49204,
            73792,
            98380,
            122968,
            147556,
            172144,
            196732,
            221320,
        ]
        assert fh.moments[1] == [
            "ZT",
            "DZ",
            "VR",
            "SW",
            "DR",
            "KD",
            "RH",
            "SQ",
            "PH",
            "CZ",
            "SD",
            "FH",
        ]
        assert fh.ray_headers[1][0]["mhead"]["SweepMode"] == 3


def test_open_uf_datatree(uf_file_1):
    # Define kwargs to pass into the function
    kwargs = {
        "sweep": [0, 1, 2, 5, 7],  # Test with specific sweeps
        "first_dim": "auto",
        "reindex_angle": {
            "start_angle": 0.0,
            "stop_angle": 360.0,
            "angle_res": 1.0,
            "direction": 1,  # Set a valid direction within reindex_angle
        },
        "fix_second_angle": True,
        "site_coords": True,
    }

    # Call the function with an actual UF file
    dtree = open_uf_datatree(uf_file_1, **kwargs)

    # Assertions
    assert isinstance(dtree, DataTree), "Expected a DataTree instance"
    subtree_paths = [n.path for n in dtree.subtree]
    assert "/" in subtree_paths, "Root group should be present in the DataTree"

    # optional_groups=False by default: metadata subgroups should NOT be present
    assert "radar_parameters" not in dtree.children
    assert "georeferencing_correction" not in dtree.children
    assert "radar_calibration" not in dtree.children

    # Check if at least one sweep group is attached (e.g., "/sweep_0")
    sweep_groups = [key for key in dtree.match("sweep_*")]
    assert len(sweep_groups) == 5, "Expected at least one sweep group in the DataTree"

    # Verify a sample variable in one of the sweep groups (adjust as needed based on expected variables)
    sample_sweep = sweep_groups[0]
    assert len(dtree[sample_sweep].data_vars) == 17
    assert (
        "DBTH" in dtree[sample_sweep].data_vars
    ), f"DBTH should be a data variable in {sample_sweep}"
    assert (
        "ZDR" in dtree[sample_sweep].data_vars
    ), f"ZDR should be a data variable in {sample_sweep}"
    assert dtree[sample_sweep]["DBTH"].shape == (360, 997)

    # Station coords should be on root as coordinates, NOT on sweeps
    assert "latitude" in dtree.ds.coords
    assert "longitude" in dtree.ds.coords
    assert "altitude" in dtree.ds.coords
    assert "latitude" not in dtree.ds.data_vars

    assert len(dtree.attrs) == 10
    print(dtree.attrs)
    assert dtree.attrs["instrument_name"] == "rvp8-rel"
    assert dtree.attrs["site_name"] == "MAX"


def test_open_uf_datatree_2(uf_file_2):
    # Define kwargs to pass into the function
    kwargs = {
        "sweep": [0, 1, 2, 5],  # Test with specific sweeps
        "first_dim": "auto",
        "reindex_angle": {
            "start_angle": 0.0,
            "stop_angle": 360.0,
            "angle_res": 1.0,
            "direction": 1,  # Set a valid direction within reindex_angle
        },
        "fix_second_angle": True,
        "site_coords": True,
    }

    # Call the function with an actual NEXRAD Level 2 file
    dtree = open_uf_datatree(uf_file_2, **kwargs)

    # Assertions
    assert isinstance(dtree, DataTree), "Expected a DataTree instance"
    subtree_paths = [n.path for n in dtree.subtree]
    assert "/" in subtree_paths, "Root group should be present in the DataTree"

    # optional_groups=False by default: metadata subgroups should NOT be present
    assert "radar_parameters" not in dtree.children
    assert "georeferencing_correction" not in dtree.children
    assert "radar_calibration" not in dtree.children

    # Check if at least one sweep group is attached (e.g., "/sweep_0")
    sweep_groups = [key for key in dtree.match("sweep_*")]
    assert len(sweep_groups) == 4, "Expected at least one sweep group in the DataTree"

    # Verify a sample variable in one of the sweep groups (adjust as needed based on expected variables)
    sample_sweep = sweep_groups[0]
    assert len(dtree[sample_sweep].data_vars) == 16
    assert (
        "DBTH" in dtree[sample_sweep].data_vars
    ), f"DBTH should be a data variable in {sample_sweep}"
    assert dtree[sample_sweep]["DBTH"].shape == (360, 476)

    # Station coords should be on root as coordinates
    assert "latitude" in dtree.ds.coords
    assert "longitude" in dtree.ds.coords
    assert "altitude" in dtree.ds.coords
    assert "latitude" not in dtree.ds.data_vars

    assert len(dtree.attrs) == 10
    assert dtree.attrs["instrument_name"] == "CHILL"
    assert dtree.attrs["site_name"] == "GREELEY"


@pytest.mark.parametrize(
    "sweeps_input, expected_sweeps, should_raise",
    [
        ("/sweep_0", ["sweep_0"], False),
        (0, ["sweep_0"], False),
        ([0, 1, 2], ["sweep_0", "sweep_1", "sweep_2"], False),
        (["/sweep_0", "/sweep_1"], ["sweep_0", "sweep_1"], False),
        (None, [f"sweep_{i}" for i in range(14)], False),
        (
            [0.1, 1.2],
            None,
            True,
        ),  # This should raise a ValueError due to float types in the list
    ],
)
def test_open_uf_datatree_sweeps_initialization(
    uf_file_1, sweeps_input, expected_sweeps, should_raise
):
    """Test that `open_uf_datatree` correctly initializes sweeps or raises errors."""

    kwargs = {
        "first_dim": "auto",
        "reindex_angle": False,
        "fix_second_angle": False,
        "site_coords": True,
        "optional": True,
    }

    if should_raise:
        with pytest.raises(
            ValueError,
            match="Invalid type in 'sweep' list. Expected integers .* or strings .*",
        ):
            open_uf_datatree(uf_file_1, sweep=sweeps_input, **kwargs)
    else:
        dtree = open_uf_datatree(uf_file_1, sweep=sweeps_input, **kwargs)
        actual_sweeps = list(dtree.match("sweep_*").keys())
        assert (
            actual_sweeps == expected_sweeps
        ), f"Unexpected sweeps for input: {sweeps_input}"


def test_input_types_for_uf(monkeypatch, uf_file_3):
    # Load as bytes
    with open(uf_file_3, "rb") as f:
        file_bytes = f.read()

    # Test with bytes input
    with UFFile(file_bytes) as fh:
        assert isinstance(fh._fh, np.ndarray)
        assert fh._fh.dtype == np.uint8

    # Test with file-like object (BytesIO)
    file_obj = io.BytesIO(file_bytes)
    with UFFile(file_obj) as fh:
        assert isinstance(fh._fh, np.ndarray)
        assert fh._fh.dtype == np.uint8

    # Test with string path
    with UFFile(uf_file_3) as fh:
        assert hasattr(fh, "_fp")
        assert os.path.basename(fh._fp.name) == os.path.basename(uf_file_3)


def test_open_uf_with_bytes(uf_file_3):
    with open(uf_file_3, "rb") as f:
        file_bytes = f.read()

    with UFFile(file_bytes) as fh:
        assert fh.ray_headers[1][0]["mhead"]["SweepMode"] == 3


def test_open_uf_datatree_optional_groups(uf_file_1):
    """Test that optional_groups=True includes metadata subgroups."""
    dtree = open_uf_datatree(uf_file_1, optional_groups=True)
    assert "radar_parameters" in dtree.children
    assert "georeferencing_correction" in dtree.children
    assert "radar_calibration" in dtree.children


def test_uf_unsupported_input_type():
    unsupported_input = 12345  # int is not supported
    with pytest.raises(TypeError, match="Unsupported input type: <class 'int'>"):
        UFFile(unsupported_input)
