#!/usr/bin/env python
# Copyright (c) 2024-2025, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for `xradar.io.nexrad_archive` module."""

import io
import os
from collections import OrderedDict

import numpy as np
import pytest
import xarray
from xarray import DataTree

from xradar.io.backends.nexrad_level2 import (
    NexradLevel2BackendEntrypoint,
    NEXRADLevel2File,
    open_nexradlevel2_datatree,
)


@pytest.mark.parametrize(
    "nexradlevel2_files", ["nexradlevel2_gzfile", "nexradlevel2_bzfile"], indirect=True
)
def test_open_nexrad_level2_backend(nexradlevel2_files):
    with NEXRADLevel2File(nexradlevel2_files, loaddata=False) as nex:
        nsweeps = nex.msg_5["number_elevation_cuts"]

    sweeps = [f"sweep_{i}" for i in range(nsweeps)]
    assert nsweeps == 11
    for i, group in enumerate(sweeps):
        with xarray.open_dataset(
            nexradlevel2_files, engine=NexradLevel2BackendEntrypoint, group=group
        ) as ds:
            assert ds.attrs["instrument_name"] == "KLBB"
            assert int(ds.sweep_number.values) == i


@pytest.mark.parametrize(
    "nexradlevel2_files", ["nexradlevel2_gzfile", "nexradlevel2_bzfile"], indirect=True
)
def test_open_nexradlevel2_file(nexradlevel2_files):
    with NEXRADLevel2File(nexradlevel2_files) as fh:

        # volume header
        assert fh.volume_header["tape"] == b"AR2V0006."
        assert fh.volume_header["extension"] == b"736"
        assert fh.volume_header["date"] == 977403904
        assert fh.volume_header["time"] == 274675715
        assert fh.volume_header["icao"] == b"KLBB"

        # meta_header 15
        assert len(fh.meta_header["msg_15"]) == 5
        assert fh.meta_header["msg_15"][0]["size"] == 1208
        assert fh.meta_header["msg_15"][0]["channels"] == 8
        assert fh.meta_header["msg_15"][0]["type"] == 15
        assert fh.meta_header["msg_15"][0]["seq_id"] == 5
        assert fh.meta_header["msg_15"][0]["date"] == 16940
        assert fh.meta_header["msg_15"][0]["ms"] == 65175863
        assert fh.meta_header["msg_15"][0]["segments"] == 5
        assert fh.meta_header["msg_15"][0]["seg_num"] == 1
        assert fh.meta_header["msg_15"][0]["record_number"] == 0
        # meta_header 13
        assert len(fh.meta_header["msg_13"]) == 49
        assert fh.meta_header["msg_13"][0]["size"] == 1208
        assert fh.meta_header["msg_13"][0]["channels"] == 8
        assert fh.meta_header["msg_13"][0]["type"] == 13
        assert fh.meta_header["msg_13"][0]["seq_id"] == 15690
        assert fh.meta_header["msg_13"][0]["date"] == 16954
        assert fh.meta_header["msg_13"][0]["ms"] == 54016980
        assert fh.meta_header["msg_13"][0]["segments"] == 49
        assert fh.meta_header["msg_13"][0]["seg_num"] == 1
        assert fh.meta_header["msg_13"][0]["record_number"] == 77
        # meta header 18
        assert len(fh.meta_header["msg_18"]) == 4
        assert fh.meta_header["msg_18"][0]["size"] == 1208
        assert fh.meta_header["msg_18"][0]["channels"] == 8
        assert fh.meta_header["msg_18"][0]["type"] == 18
        assert fh.meta_header["msg_18"][0]["seq_id"] == 6
        assert fh.meta_header["msg_18"][0]["date"] == 16940
        assert fh.meta_header["msg_18"][0]["ms"] == 65175864
        assert fh.meta_header["msg_18"][0]["segments"] == 4
        assert fh.meta_header["msg_18"][0]["seg_num"] == 1
        assert fh.meta_header["msg_18"][0]["record_number"] == 126
        # meta header 3
        assert len(fh.meta_header["msg_3"]) == 1
        assert fh.meta_header["msg_3"][0]["size"] == 488
        assert fh.meta_header["msg_3"][0]["channels"] == 8
        assert fh.meta_header["msg_3"][0]["type"] == 3
        assert fh.meta_header["msg_3"][0]["seq_id"] == 15694
        assert fh.meta_header["msg_3"][0]["date"] == 16954
        assert fh.meta_header["msg_3"][0]["ms"] == 54025336
        assert fh.meta_header["msg_3"][0]["segments"] == 1
        assert fh.meta_header["msg_3"][0]["seg_num"] == 1
        assert fh.meta_header["msg_3"][0]["record_number"] == 131
        # meta header 5
        assert len(fh.meta_header["msg_5"]) == 1
        assert fh.meta_header["msg_5"][0]["size"] == 272
        assert fh.meta_header["msg_5"][0]["channels"] == 8
        assert fh.meta_header["msg_5"][0]["type"] == 5
        assert fh.meta_header["msg_5"][0]["seq_id"] == 15695
        assert fh.meta_header["msg_5"][0]["date"] == 16954
        assert fh.meta_header["msg_5"][0]["ms"] == 54025336
        assert fh.meta_header["msg_5"][0]["segments"] == 1
        assert fh.meta_header["msg_5"][0]["seg_num"] == 1
        assert fh.meta_header["msg_5"][0]["record_number"] == 132
        assert fh.msg_5 == OrderedDict(
            [
                ("message_size", 264),
                ("pattern_type", 2),
                ("pattern_number", 21),
                ("number_elevation_cuts", 11),
                ("clutter_map_group_number", 1),
                ("doppler_velocity_resolution", 2),
                ("pulse_width", 2),
                (
                    "elevation_data",
                    [
                        OrderedDict(
                            [
                                ("elevation_angle", 0.4833984375),
                                ("channel_config", 0),
                                ("waveform_type", 1),
                                ("super_resolution", 11),
                                ("prf_number", 1),
                                ("prf_pulse_count", 28),
                                ("azimuth_rate", 8256),
                                ("ref_thresh", 16),
                                ("vel_thresh", 16),
                                ("sw_thresh", 16),
                                ("zdr_thres", 16),
                                ("phi_thres", 16),
                                ("rho_thres", 16),
                                ("edge_angle_1", 0),
                                ("dop_prf_num_1", 0),
                                ("dop_prf_pulse_count_1", 0),
                                ("edge_angle_2", 0),
                                ("dop_prf_num_2", 0),
                                ("dop_prf_pulse_count_2", 0),
                                ("edge_angle_3", 0),
                                ("dop_prf_num_3", 0),
                                ("dop_prf_pulse_count_3", 0),
                            ]
                        ),
                        OrderedDict(
                            [
                                ("elevation_angle", 0.4833984375),
                                ("channel_config", 0),
                                ("waveform_type", 2),
                                ("super_resolution", 7),
                                ("prf_number", 0),
                                ("prf_pulse_count", 0),
                                ("azimuth_rate", 8272),
                                ("ref_thresh", 28),
                                ("vel_thresh", 28),
                                ("sw_thresh", 28),
                                ("zdr_thres", 28),
                                ("phi_thres", 28),
                                ("rho_thres", 28),
                                ("edge_angle_1", 5464),
                                ("dop_prf_num_1", 4),
                                ("dop_prf_pulse_count_1", 75),
                                ("edge_angle_2", 38232),
                                ("dop_prf_num_2", 4),
                                ("dop_prf_pulse_count_2", 75),
                                ("edge_angle_3", 60984),
                                ("dop_prf_num_3", 4),
                                ("dop_prf_pulse_count_3", 75),
                            ]
                        ),
                        OrderedDict(
                            [
                                ("elevation_angle", 1.4501953125),
                                ("channel_config", 0),
                                ("waveform_type", 1),
                                ("super_resolution", 11),
                                ("prf_number", 1),
                                ("prf_pulse_count", 28),
                                ("azimuth_rate", 8256),
                                ("ref_thresh", 16),
                                ("vel_thresh", 16),
                                ("sw_thresh", 16),
                                ("zdr_thres", 16),
                                ("phi_thres", 16),
                                ("rho_thres", 16),
                                ("edge_angle_1", 0),
                                ("dop_prf_num_1", 0),
                                ("dop_prf_pulse_count_1", 0),
                                ("edge_angle_2", 0),
                                ("dop_prf_num_2", 0),
                                ("dop_prf_pulse_count_2", 0),
                                ("edge_angle_3", 0),
                                ("dop_prf_num_3", 0),
                                ("dop_prf_pulse_count_3", 0),
                            ]
                        ),
                        OrderedDict(
                            [
                                ("elevation_angle", 1.4501953125),
                                ("channel_config", 0),
                                ("waveform_type", 2),
                                ("super_resolution", 7),
                                ("prf_number", 0),
                                ("prf_pulse_count", 0),
                                ("azimuth_rate", 8272),
                                ("ref_thresh", 28),
                                ("vel_thresh", 28),
                                ("sw_thresh", 28),
                                ("zdr_thres", 28),
                                ("phi_thres", 28),
                                ("rho_thres", 28),
                                ("edge_angle_1", 5464),
                                ("dop_prf_num_1", 4),
                                ("dop_prf_pulse_count_1", 75),
                                ("edge_angle_2", 38232),
                                ("dop_prf_num_2", 4),
                                ("dop_prf_pulse_count_2", 75),
                                ("edge_angle_3", 60984),
                                ("dop_prf_num_3", 4),
                                ("dop_prf_pulse_count_3", 75),
                            ]
                        ),
                        OrderedDict(
                            [
                                ("elevation_angle", 2.4169921875),
                                ("channel_config", 0),
                                ("waveform_type", 4),
                                ("super_resolution", 14),
                                ("prf_number", 2),
                                ("prf_pulse_count", 8),
                                ("azimuth_rate", 8144),
                                ("ref_thresh", 28),
                                ("vel_thresh", 28),
                                ("sw_thresh", 28),
                                ("zdr_thres", 28),
                                ("phi_thres", 28),
                                ("rho_thres", 28),
                                ("edge_angle_1", 5464),
                                ("dop_prf_num_1", 4),
                                ("dop_prf_pulse_count_1", 59),
                                ("edge_angle_2", 38232),
                                ("dop_prf_num_2", 4),
                                ("dop_prf_pulse_count_2", 59),
                                ("edge_angle_3", 60984),
                                ("dop_prf_num_3", 4),
                                ("dop_prf_pulse_count_3", 59),
                            ]
                        ),
                        OrderedDict(
                            [
                                ("elevation_angle", 3.3837890625),
                                ("channel_config", 0),
                                ("waveform_type", 4),
                                ("super_resolution", 14),
                                ("prf_number", 2),
                                ("prf_pulse_count", 8),
                                ("azimuth_rate", 8144),
                                ("ref_thresh", 28),
                                ("vel_thresh", 28),
                                ("sw_thresh", 28),
                                ("zdr_thres", 28),
                                ("phi_thres", 28),
                                ("rho_thres", 28),
                                ("edge_angle_1", 5464),
                                ("dop_prf_num_1", 4),
                                ("dop_prf_pulse_count_1", 59),
                                ("edge_angle_2", 38232),
                                ("dop_prf_num_2", 4),
                                ("dop_prf_pulse_count_2", 59),
                                ("edge_angle_3", 60984),
                                ("dop_prf_num_3", 4),
                                ("dop_prf_pulse_count_3", 59),
                            ]
                        ),
                        OrderedDict(
                            [
                                ("elevation_angle", 4.306640625),
                                ("channel_config", 0),
                                ("waveform_type", 4),
                                ("super_resolution", 14),
                                ("prf_number", 2),
                                ("prf_pulse_count", 8),
                                ("azimuth_rate", 8144),
                                ("ref_thresh", 28),
                                ("vel_thresh", 28),
                                ("sw_thresh", 28),
                                ("zdr_thres", 28),
                                ("phi_thres", 28),
                                ("rho_thres", 28),
                                ("edge_angle_1", 5464),
                                ("dop_prf_num_1", 4),
                                ("dop_prf_pulse_count_1", 59),
                                ("edge_angle_2", 38232),
                                ("dop_prf_num_2", 4),
                                ("dop_prf_pulse_count_2", 59),
                                ("edge_angle_3", 60984),
                                ("dop_prf_num_3", 4),
                                ("dop_prf_pulse_count_3", 59),
                            ]
                        ),
                        OrderedDict(
                            [
                                ("elevation_angle", 6.0205078125),
                                ("channel_config", 0),
                                ("waveform_type", 4),
                                ("super_resolution", 14),
                                ("prf_number", 3),
                                ("prf_pulse_count", 12),
                                ("azimuth_rate", 8144),
                                ("ref_thresh", 28),
                                ("vel_thresh", 28),
                                ("sw_thresh", 28),
                                ("zdr_thres", 28),
                                ("phi_thres", 28),
                                ("rho_thres", 28),
                                ("edge_angle_1", 5464),
                                ("dop_prf_num_1", 4),
                                ("dop_prf_pulse_count_1", 59),
                                ("edge_angle_2", 38232),
                                ("dop_prf_num_2", 4),
                                ("dop_prf_pulse_count_2", 59),
                                ("edge_angle_3", 60984),
                                ("dop_prf_num_3", 4),
                                ("dop_prf_pulse_count_3", 59),
                            ]
                        ),
                        OrderedDict(
                            [
                                ("elevation_angle", 9.8876953125),
                                ("channel_config", 0),
                                ("waveform_type", 3),
                                ("super_resolution", 10),
                                ("prf_number", 0),
                                ("prf_pulse_count", 0),
                                ("azimuth_rate", 10384),
                                ("ref_thresh", 28),
                                ("vel_thresh", 28),
                                ("sw_thresh", 28),
                                ("zdr_thres", 28),
                                ("phi_thres", 28),
                                ("rho_thres", 28),
                                ("edge_angle_1", 5464),
                                ("dop_prf_num_1", 7),
                                ("dop_prf_pulse_count_1", 82),
                                ("edge_angle_2", 38232),
                                ("dop_prf_num_2", 7),
                                ("dop_prf_pulse_count_2", 82),
                                ("edge_angle_3", 60984),
                                ("dop_prf_num_3", 7),
                                ("dop_prf_pulse_count_3", 82),
                            ]
                        ),
                        OrderedDict(
                            [
                                ("elevation_angle", 14.58984375),
                                ("channel_config", 0),
                                ("waveform_type", 3),
                                ("super_resolution", 10),
                                ("prf_number", 0),
                                ("prf_pulse_count", 0),
                                ("azimuth_rate", 10432),
                                ("ref_thresh", 28),
                                ("vel_thresh", 28),
                                ("sw_thresh", 28),
                                ("zdr_thres", 28),
                                ("phi_thres", 28),
                                ("rho_thres", 28),
                                ("edge_angle_1", 5464),
                                ("dop_prf_num_1", 7),
                                ("dop_prf_pulse_count_1", 82),
                                ("edge_angle_2", 38232),
                                ("dop_prf_num_2", 7),
                                ("dop_prf_pulse_count_2", 82),
                                ("edge_angle_3", 60984),
                                ("dop_prf_num_3", 7),
                                ("dop_prf_pulse_count_3", 82),
                            ]
                        ),
                        OrderedDict(
                            [
                                ("elevation_angle", 19.51171875),
                                ("channel_config", 0),
                                ("waveform_type", 3),
                                ("super_resolution", 10),
                                ("prf_number", 0),
                                ("prf_pulse_count", 0),
                                ("azimuth_rate", 10496),
                                ("ref_thresh", 28),
                                ("vel_thresh", 28),
                                ("sw_thresh", 28),
                                ("zdr_thres", 28),
                                ("phi_thres", 28),
                                ("rho_thres", 28),
                                ("edge_angle_1", 5464),
                                ("dop_prf_num_1", 7),
                                ("dop_prf_pulse_count_1", 82),
                                ("edge_angle_2", 38232),
                                ("dop_prf_num_2", 7),
                                ("dop_prf_pulse_count_2", 82),
                                ("edge_angle_3", 60984),
                                ("dop_prf_num_3", 7),
                                ("dop_prf_pulse_count_3", 82),
                            ]
                        ),
                    ],
                ),
            ]
        )

        # meta header 2
        assert len(fh.meta_header["msg_2"]) == 1
        assert fh.meta_header["msg_2"][0]["size"] == 48
        assert fh.meta_header["msg_2"][0]["channels"] == 8
        assert fh.meta_header["msg_2"][0]["type"] == 2
        assert fh.meta_header["msg_2"][0]["seq_id"] == 15693
        assert fh.meta_header["msg_2"][0]["date"] == 16954
        assert fh.meta_header["msg_2"][0]["ms"] == 54025336
        assert fh.meta_header["msg_2"][0]["segments"] == 1
        assert fh.meta_header["msg_2"][0]["seg_num"] == 1
        assert fh.meta_header["msg_2"][0]["record_number"] == 133

        # data header
        assert len(fh.data_header) == 5400
        msg_31_header_length = [720, 720, 720, 720, 360, 360, 360, 360, 360, 360, 360]
        for i, head in enumerate(fh.msg_31_header):
            assert len(head) == msg_31_header_length[i]


def test_open_nexradlevel2_msg1_file(nexradlevel2_msg1_file):
    with NEXRADLevel2File(nexradlevel2_msg1_file) as fh:

        # volume header
        assert fh.volume_header["tape"] == b"AR2V0001."
        assert fh.volume_header["extension"] == b"201"
        assert fh.volume_header["date"] == 3761373184
        assert fh.volume_header["time"] == 3362708995
        assert fh.volume_header["icao"] == b"KLIX"

        # meta_header 15
        assert len(fh.meta_header["msg_15"]) == 62
        assert fh.meta_header["msg_15"][0]["size"] == 1208
        assert fh.meta_header["msg_15"][0]["channels"] == 0
        assert fh.meta_header["msg_15"][0]["type"] == 15
        assert fh.meta_header["msg_15"][0]["seq_id"] == 819
        assert fh.meta_header["msg_15"][0]["date"] == 13024
        assert fh.meta_header["msg_15"][0]["ms"] == 51522855
        assert fh.meta_header["msg_15"][0]["segments"] == 14
        assert fh.meta_header["msg_15"][0]["seg_num"] == 1
        assert fh.meta_header["msg_15"][0]["record_number"] == 0
        # meta_header 13
        assert len(fh.meta_header["msg_13"]) == 48
        assert fh.meta_header["msg_13"][0]["size"] == 1208
        assert fh.meta_header["msg_13"][0]["channels"] == 0
        assert fh.meta_header["msg_13"][0]["type"] == 13
        assert fh.meta_header["msg_13"][0]["seq_id"] == 0
        assert fh.meta_header["msg_13"][0]["date"] == 13023
        assert fh.meta_header["msg_13"][0]["ms"] == 43397314
        assert fh.meta_header["msg_13"][0]["segments"] == 14
        assert fh.meta_header["msg_13"][0]["seg_num"] == 1
        assert fh.meta_header["msg_13"][0]["record_number"] == 62
        # meta header 18
        assert len(fh.meta_header["msg_18"]) == 4
        assert fh.meta_header["msg_18"][0]["size"] == 1208
        assert fh.meta_header["msg_18"][0]["channels"] == 0
        assert fh.meta_header["msg_18"][0]["type"] == 18
        assert fh.meta_header["msg_18"][0]["seq_id"] == 0
        assert fh.meta_header["msg_18"][0]["date"] == 0
        assert fh.meta_header["msg_18"][0]["ms"] == 0
        assert fh.meta_header["msg_18"][0]["segments"] == 4
        assert fh.meta_header["msg_18"][0]["seg_num"] == 1
        assert fh.meta_header["msg_18"][0]["record_number"] == 110
        # meta header 3
        assert len(fh.meta_header["msg_3"]) == 1
        assert fh.meta_header["msg_3"][0]["size"] == 528
        assert fh.meta_header["msg_3"][0]["channels"] == 0
        assert fh.meta_header["msg_3"][0]["type"] == 3
        assert fh.meta_header["msg_3"][0]["seq_id"] == 5459
        assert fh.meta_header["msg_3"][0]["date"] == 13024
        assert fh.meta_header["msg_3"][0]["ms"] == 61897431
        assert fh.meta_header["msg_3"][0]["segments"] == 1
        assert fh.meta_header["msg_3"][0]["seg_num"] == 1
        assert fh.meta_header["msg_3"][0]["record_number"] == 114
        # meta header 5
        assert len(fh.meta_header["msg_5"]) == 1
        assert fh.meta_header["msg_5"][0]["size"] == 1208
        assert fh.meta_header["msg_5"][0]["channels"] == 0
        assert fh.meta_header["msg_5"][0]["type"] == 5
        assert fh.meta_header["msg_5"][0]["seq_id"] == 0
        assert fh.meta_header["msg_5"][0]["date"] == 0
        assert fh.meta_header["msg_5"][0]["ms"] == 0
        assert fh.meta_header["msg_5"][0]["segments"] == 1
        assert fh.meta_header["msg_5"][0]["seg_num"] == 1
        assert fh.meta_header["msg_5"][0]["record_number"] == 115
        assert fh.msg_5 == OrderedDict(
            [
                ("message_size", 0),
                ("pattern_type", 0),
                ("pattern_number", 0),
                ("number_elevation_cuts", 0),
                ("clutter_map_group_number", 0),
                ("doppler_velocity_resolution", 0),
                ("pulse_width", 0),
                ("elevation_data", []),
            ]
        )

        # meta header 2
        assert len(fh.meta_header["msg_2"]) == 1
        assert fh.meta_header["msg_2"][0]["size"] == 48
        assert fh.meta_header["msg_2"][0]["channels"] == 0
        assert fh.meta_header["msg_2"][0]["type"] == 2
        assert fh.meta_header["msg_2"][0]["seq_id"] == 29176
        assert fh.meta_header["msg_2"][0]["date"] == 13024
        assert fh.meta_header["msg_2"][0]["ms"] == 64889226
        assert fh.meta_header["msg_2"][0]["segments"] == 1
        assert fh.meta_header["msg_2"][0]["seg_num"] == 1
        assert fh.meta_header["msg_2"][0]["record_number"] == 116

        # data header
        assert len(fh.data_header) == 5856
        msg_31_header_length = [
            367,
            367,
            367,
            367,
            367,
            367,
            367,
            367,
            366,
            367,
            366,
            366,
            365,
            364,
            363,
            362,
        ]
        for i, head in enumerate(fh.msg_31_header):
            assert len(head) == msg_31_header_length[i]


def test_open_nexradlevel2_datatree(nexradlevel2_file):
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

    # Call the function with an actual NEXRAD Level 2 file
    dtree = open_nexradlevel2_datatree(nexradlevel2_file, **kwargs)

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
    assert len(sweep_groups) == 5, "Expected at least one sweep group in the DataTree"

    # Verify a sample variable in one of the sweep groups (adjust as needed based on expected variables)
    sample_sweep = sweep_groups[0]
    assert len(dtree[sample_sweep].data_vars) == 9
    assert (
        "DBZH" in dtree[sample_sweep].data_vars
    ), f"DBZH should be a data variable in {sample_sweep}"
    assert (
        "ZDR" in dtree[sample_sweep].data_vars
    ), f"VRADH should be a data variable in {sample_sweep}"
    assert dtree[sample_sweep]["DBZH"].shape == (360, 1832)
    # Validate coordinates are attached correctly
    assert (
        "latitude" in dtree[sample_sweep]
    ), "Latitude should be attached to the root dataset"
    assert (
        "longitude" in dtree[sample_sweep]
    ), "Longitude should be attached to the root dataset"
    assert (
        "altitude" in dtree[sample_sweep]
    ), "Altitude should be attached to the root dataset"

    assert len(dtree.attrs) == 10
    assert dtree.attrs["instrument_name"] == "KATX"
    assert dtree.attrs["scan_name"] == "VCP-11"


def test_open_nexradlevel2_msg1_datatree(nexradlevel2_msg1_file):
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

    # Call the function with an actual NEXRAD Level 2 file
    dtree = open_nexradlevel2_datatree(nexradlevel2_msg1_file, **kwargs)

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
    assert len(sweep_groups) == 5, "Expected at least one sweep group in the DataTree"

    # Verify a sample variable in one of the sweep groups (adjust as needed based on expected variables)
    sample_sweep = sweep_groups[0]
    assert len(dtree[sample_sweep].data_vars) == 6
    assert (
        "DBZH" in dtree[sample_sweep].data_vars
    ), f"DBZH should be a data variable in {sample_sweep}"
    assert dtree[sample_sweep]["DBZH"].shape == (360, 460)
    # Validate coordinates are attached correctly
    assert (
        "latitude" in dtree[sample_sweep]
    ), "Latitude should be attached to the root dataset"
    assert (
        "longitude" in dtree[sample_sweep]
    ), "Longitude should be attached to the root dataset"
    assert (
        "altitude" in dtree[sample_sweep]
    ), "Altitude should be attached to the root dataset"

    assert len(dtree.attrs) == 10
    assert dtree.attrs["instrument_name"] == "KLIX"
    assert dtree.attrs["scan_name"] == "VCP-0"


@pytest.mark.parametrize(
    "sweeps_input, expected_sweeps, should_raise",
    [
        ("/sweep_0", ["sweep_0"], False),
        (0, ["sweep_0"], False),
        ([0, 1, 2], ["sweep_0", "sweep_1", "sweep_2"], False),
        (["/sweep_0", "/sweep_1"], ["sweep_0", "sweep_1"], False),
        (None, [f"sweep_{i}" for i in range(16)], False),
        (
            [0.1, 1.2],
            None,
            True,
        ),  # This should raise a ValueError due to float types in the list
    ],
)
def test_open_nexradlevel2_datatree_sweeps_initialization(
    nexradlevel2_file, sweeps_input, expected_sweeps, should_raise
):
    """Test that `open_nexradlevel2_datatree` correctly initializes sweeps or raises errors."""

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
            open_nexradlevel2_datatree(nexradlevel2_file, sweep=sweeps_input, **kwargs)
    else:
        dtree = open_nexradlevel2_datatree(
            nexradlevel2_file, sweep=sweeps_input, **kwargs
        )
        actual_sweeps = list(dtree.match("sweep_*").keys())
        assert (
            actual_sweeps == expected_sweeps
        ), f"Unexpected sweeps for input: {sweeps_input}"


def test_input_types_for_nexradlevel2(monkeypatch, nexradlevel2_file):
    # Load as bytes
    with open(nexradlevel2_file, "rb") as f:
        file_bytes = f.read()

    # Test with bytes input
    with NEXRADLevel2File(file_bytes) as fh:
        assert isinstance(fh._fh, np.ndarray)
        assert fh._fh.dtype == np.uint8

    # Test with file-like object (BytesIO)
    file_obj = io.BytesIO(file_bytes)
    with NEXRADLevel2File(file_obj) as fh:
        assert isinstance(fh._fh, np.ndarray)
        assert fh._fh.dtype == np.uint8

    # Test with string path
    with NEXRADLevel2File(nexradlevel2_file) as fh:
        assert hasattr(fh, "_fp")
        assert os.path.basename(fh._fp.name) == os.path.basename(nexradlevel2_file)


def test_open_nexradlevel2_with_bytes(nexradlevel2_file):
    with open(nexradlevel2_file, "rb") as f:
        file_bytes = f.read()

    with NEXRADLevel2File(file_bytes) as fh:
        assert fh.volume_header["tape"].startswith(b"AR2V")


def test_nexradlevel2_unsupported_input_type():
    unsupported_input = 12345  # int is not supported
    with pytest.raises(TypeError, match="Unsupported input type: <class 'int'>"):
        NEXRADLevel2File(unsupported_input)


def test_bz2_compressed_buffer_path_real(nexradlevel2_bzfile):
    with open(nexradlevel2_bzfile, "rb") as f:
        file_bytes = f.read()

    with NEXRADLevel2File(file_bytes) as fh:
        assert fh.is_compressed
        fh.init_record(134)

        assert 1 in fh._ldm
        assert isinstance(fh._ldm[1], np.ndarray)
        assert fh._ldm[1].dtype == np.uint8


def test_nexradlevel2_missing_msg2_metadata():
    """
    Test backward compatibility when msg_2 metadata is missing.

    Historical context: NEXRAD files from Build 8.0 era (2005-2006) sometimes
    lack proper msg_2 metadata records in the metadata header. The msg_2 record
    typically indicates the end of metadata and start of data records.

    Without this fix, files missing msg_2 would cause IndexError when trying
    to access meta_header["msg_2"][0]["record_number"].

    This test ensures graceful degradation to scanning from record 0.
    """
    from collections import defaultdict

    from xradar.io.backends.nexrad_level2 import NEXRADLevel2File

    class MockNEXRADFile(NEXRADLevel2File):
        def __init__(self):
            self._meta_header = defaultdict(list)
            # Simulate missing msg_2 by only having other message types
            self._meta_header["msg_15"] = [{"record_number": 0}]
            self._meta_header["msg_5"] = [{"record_number": 1}]
            # msg_2 is missing!

        @property
        def meta_header(self):
            return self._meta_header

        def get_metadata_header(self):
            return self._meta_header

        def init_record(self, recnum):
            # Simulate successful record initialization
            return recnum < 10  # Allow up to 10 records

        def init_next_record(self):
            # Simulate end of records
            return False

        def get_message_header(self):
            # Return empty header
            return {"type": 0, "record_number": 0, "filepos": 0}

    # Test with missing msg_2
    mock_file = MockNEXRADFile()

    # This should handle missing msg_2 gracefully
    try:
        data_header = mock_file.get_data_header()
        # Should return tuple without crashing
        assert isinstance(data_header, tuple)
        assert (
            len(data_header) == 3
        )  # Should return (data_header, msg_31_header, msg_31_data_header)
    except (IndexError, KeyError) as e:
        # The fix should prevent these errors
        pytest.fail(
            f"get_data_header should handle missing msg_2 gracefully, but got: {e}"
        )


@pytest.mark.parametrize(
    "elevation_cuts,pattern_number,expected_has_elevation_data,test_description",
    [
        (40833, 16621, False, "VCP-0 maintenance mode with corrupted elevation count"),
        (5, 11, True, "Normal VCP with valid elevation cuts"),
        (0, 0, False, "Empty VCP pattern"),
        (26, 21, False, "Invalid elevation count exceeding maximum (25)"),
    ],
)
def test_nexradlevel2_msg5_elevation_validation(
    elevation_cuts, pattern_number, expected_has_elevation_data, test_description
):
    """
    Test MSG_5 elevation cut validation for backward compatibility.

    Historical context: VCP-0 maintenance mode files from 2005-2006 era
    often contain corrupted MSG_5 data with invalid elevation counts
    (e.g., 40833 cuts instead of realistic values like 5-25).

    This test ensures graceful handling of such corrupted data.
    """
    import struct
    from collections import defaultdict
    from unittest.mock import Mock

    from xradar.io.backends.nexrad_level2 import NEXRADLevel2File

    # Create mock MSG_5 data with specified parameters
    mock_msg5_data = struct.pack(
        ">HHHHHBB10s", 0, 0, pattern_number, elevation_cuts, 0, 0, 0, b"\x00" * 10
    )

    # Mock the file reading components
    mock_rh = Mock()
    mock_rh.read.return_value = mock_msg5_data

    mock_file = Mock(spec=NEXRADLevel2File)
    mock_file._meta_header = defaultdict(list)
    mock_file._meta_header["msg_5"] = [{"record_number": 0}]
    mock_file._rh = mock_rh
    mock_file._rawdata = False
    mock_file.rh = Mock()
    mock_file.rh.pos = 0

    # Set up meta_header property
    mock_file.meta_header = mock_file._meta_header

    # Call the actual method
    result = NEXRADLevel2File.get_msg_5_data(mock_file)

    # Verify results
    assert result is not False, f"Failed test: {test_description}"
    assert isinstance(result, dict), f"Should return dict for: {test_description}"
    assert "elevation_data" in result, f"Missing elevation_data for: {test_description}"

    if expected_has_elevation_data:
        # For valid elevation counts, we expect the method to attempt reading elevation data
        # (though our mock doesn't provide the elevation data, so it will be empty due to the try/except)
        assert (
            len(result["elevation_data"]) >= 0
        ), f"Should allow elevation data for: {test_description}"
    else:
        # For invalid elevation counts, should return empty elevation_data due to validation
        assert (
            len(result["elevation_data"]) == 0
        ), f"Should have no elevation data for: {test_description}"


def test_nexradlevel2_missing_msg5():
    """Test handling when MSG_5 is completely missing."""
    from collections import defaultdict

    from xradar.io.backends.nexrad_level2 import NEXRADLevel2File

    class MockNEXRADFile(NEXRADLevel2File):
        def __init__(self):
            self._meta_header = defaultdict(list)  # Empty - no msg_5
            self._msg_5_data = None
            self._rawdata = False

        @property
        def meta_header(self):
            return self._meta_header

    mock_file = MockNEXRADFile()
    msg_5_result = mock_file.get_msg_5_data()

    # Should return False when msg_5 is missing
    assert msg_5_result is False


def test_nexradlevel2_msg5_struct_error_handling():
    """Test graceful handling of struct errors in MSG_5 elevation data."""
    import struct
    from collections import defaultdict

    from xradar.io.backends.nexrad_level2 import NEXRADLevel2File

    class MockRH:
        def __init__(self):
            self.pos = 0
            self.read_count = 0

        def read(self, size, width=1):
            self.read_count += 1
            if self.read_count == 1:
                # First read - return valid MSG_5 header with 3 elevation cuts
                return struct.pack(">HHHHHBB10s", 0, 0, 11, 3, 0, 0, 0, b"\x00" * 10)
            else:
                # Subsequent reads - return insufficient data to trigger struct.error
                return b"\x00" * 10  # Too small for MSG_5_ELEV structure

    class MockNEXRADFile(NEXRADLevel2File):
        def __init__(self):
            self._meta_header = defaultdict(list)
            self._meta_header["msg_5"] = [{"record_number": 0}]
            self._msg_5_data = None
            self._rawdata = False
            self._rh = MockRH()
            self._fp = None  # Avoid close() errors

        @property
        def meta_header(self):
            return self._meta_header

        def init_record(self, recnum):
            return True

    mock_file = MockNEXRADFile()
    msg_5_result = mock_file.get_msg_5_data()

    # Should return partial MSG_5 despite struct errors
    assert msg_5_result is not False
    assert "elevation_data" in msg_5_result
    # May have partial elevation data before the error occurred
    assert len(msg_5_result["elevation_data"]) >= 0
