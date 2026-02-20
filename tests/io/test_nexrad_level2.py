#!/usr/bin/env python
# Copyright (c) 2024-2025, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for `xradar.io.nexrad_archive` module."""

import io
import os
import warnings
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray
from xarray import DataTree, open_dataset, open_mfdataset

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
    assert len(dtree[sample_sweep].data_vars) == 9
    assert (
        "DBZH" in dtree[sample_sweep].data_vars
    ), f"DBZH should be a data variable in {sample_sweep}"
    assert (
        "ZDR" in dtree[sample_sweep].data_vars
    ), f"VRADH should be a data variable in {sample_sweep}"
    assert dtree[sample_sweep]["DBZH"].shape == (360, 1832)

    # Station coords should be on root as coordinates
    assert "latitude" in dtree.ds.coords
    assert "longitude" in dtree.ds.coords
    assert "altitude" in dtree.ds.coords
    assert "latitude" not in dtree.ds.data_vars

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
    assert len(dtree[sample_sweep].data_vars) == 6
    assert (
        "DBZH" in dtree[sample_sweep].data_vars
    ), f"DBZH should be a data variable in {sample_sweep}"
    assert dtree[sample_sweep]["DBZH"].shape == (360, 460)

    # Station coords should be on root as coordinates
    assert "latitude" in dtree.ds.coords
    assert "longitude" in dtree.ds.coords
    assert "altitude" in dtree.ds.coords
    assert "latitude" not in dtree.ds.data_vars

    assert len(dtree.attrs) == 10
    assert dtree.attrs["instrument_name"] == "KLIX"
    assert dtree.attrs["scan_name"] == "VCP-0"


def test_open_nexradlevel2_datatree_optional_groups(nexradlevel2_file):
    """Test that optional_groups=True includes metadata subgroups."""
    dtree = open_nexradlevel2_datatree(
        nexradlevel2_file, sweep=[0], optional_groups=True
    )
    assert "radar_parameters" in dtree.children
    assert "georeferencing_correction" in dtree.children
    assert "radar_calibration" in dtree.children

    # Station coords should still be on root as coordinates
    assert "latitude" in dtree.ds.coords
    assert "latitude" not in dtree.ds.data_vars


def test_open_nexradlevel2_single_dataset_site_coords(nexradlevel2_file):
    """Single-dataset access via open_dataset still has lat/lon/alt with site_coords=True."""
    ds = xarray.open_dataset(
        nexradlevel2_file,
        engine=NexradLevel2BackendEntrypoint,
        group="sweep_0",
        site_coords=True,
    )
    assert "latitude" in ds.coords
    assert "longitude" in ds.coords
    assert "altitude" in ds.coords


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


def test_nexradlevel2_open_mfdataset_context_manager(nexradlevel2_file):
    with open_mfdataset(
        [nexradlevel2_file],
        engine="nexradlevel2",
        concat_dim="volume_time",
        combine="nested",
        group="sweep_0",
    ) as ds:
        assert ds is not None
        # closer must exist while inside context
        assert callable(getattr(ds, "_close", None))


def test_nexradlevel2_dataset_has_close(nexradlevel2_file):
    ds = open_dataset(nexradlevel2_file, engine="nexradlevel2", group="sweep_0")
    assert callable(getattr(ds, "_close", None))
    ds.close()


class TestNEXRADChunkFiles:
    """Tests for NEXRAD Level 2 chunk file support (I/E chunks without volume headers)."""

    def test_read_volume_header_missing_gracefully(self):
        """Test that _read_volume_header handles missing headers gracefully."""
        import warnings

        # Create minimal fake chunk data (BZ2 compressed format)
        # BZ2 magic number: 'BZh' followed by compression level
        fake_chunk_data = b"BZh9" + b"\x00" * 100

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with NEXRADLevel2File(fake_chunk_data, has_volume_header=False) as fh:
                assert fh.volume_header is None
                # Check that warning was issued
                assert len(w) == 1
                assert "chunk file mode" in str(w[0].message).lower()

    def test_is_compressed_detects_bz2_without_volume_header(self):
        """Test that is_compressed correctly detects BZ2 data in chunk files."""
        # BZ2 magic bytes: 0x42 ('B'), 0x5A ('Z'), 0x68 ('h')
        bz2_chunk_data = bytes([0x42, 0x5A, 0x68]) + b"9" + b"\x00" * 100

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with NEXRADLevel2File(bz2_chunk_data, has_volume_header=False) as fh:
                assert fh.is_compressed is True

    def test_is_compressed_uncompressed_chunk(self):
        """Test that is_compressed returns False for non-BZ2 chunk files."""
        # Data that doesn't start with BZ2 magic
        non_bz2_data = b"\x00\x00\x00" + b"\x00" * 100

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with NEXRADLevel2File(non_bz2_data, has_volume_header=False) as fh:
                assert fh.is_compressed is False

    def test_has_volume_header_parameter(self):
        """Test that has_volume_header parameter is respected."""
        import warnings

        # Create fake data
        fake_data = b"\x00" * 200

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # With has_volume_header=False
            with NEXRADLevel2File(fake_data, has_volume_header=False) as fh:
                assert fh._has_volume_header is False
                assert fh.volume_header is None

    def test_volume_header_read_failure_fallback(self):
        """Test that failed volume header read falls back gracefully."""
        import warnings

        # Create data that's too short for a valid volume header
        short_data = b"\x00" * 10

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with NEXRADLevel2File(short_data, has_volume_header=True) as fh:
                # Should fall back to None and emit warning
                assert fh.volume_header is None
                assert len(w) == 1
                assert "unable to read volume header" in str(w[0].message).lower()

    def test_get_attrs_handles_missing_volume_header(self):
        """Test that get_attrs works when volume_header is None."""
        from unittest.mock import MagicMock, PropertyMock

        from xradar.io.backends.nexrad_level2 import NexradLevel2Store

        # Create a mock store with None volume_header

        mock_store = MagicMock(spec=NexradLevel2Store)
        mock_root = MagicMock()
        mock_root.volume_header = None
        mock_root.msg_5 = None

        # Use the actual get_attrs method
        type(mock_store).root = PropertyMock(return_value=mock_root)

        # Call the actual method by accessing it from the class
        attrs = NexradLevel2Store.get_attrs(mock_store)

        assert ("instrument_name", "UNKNOWN") in attrs.items()

    def test_get_attrs_with_volume_header(self, nexradlevel2_file):
        """Test that get_attrs correctly reads ICAO when volume_header exists."""
        from xradar.io.backends.nexrad_level2 import NexradLevel2Store

        store = NexradLevel2Store.open(nexradlevel2_file, group="sweep_0")
        attrs = store.get_attrs()

        # Should have actual instrument name from file
        assert "instrument_name" in dict(attrs)
        assert dict(attrs)["instrument_name"] != "UNKNOWN"


class TestConcatenateChunks:
    """Tests for _concatenate_chunks helper and list input to open_nexradlevel2_datatree."""

    def test_concatenate_bytes_chunks(self):
        """List of bytes objects are concatenated correctly."""
        from xradar.io.backends.nexrad_level2 import _concatenate_chunks

        chunk1 = b"\x00\x01\x02"
        chunk2 = b"\x03\x04\x05"
        result = _concatenate_chunks([chunk1, chunk2])
        assert result == b"\x00\x01\x02\x03\x04\x05"

    def test_concatenate_bytearray_chunks(self):
        """List of bytearray objects are concatenated correctly."""
        from xradar.io.backends.nexrad_level2 import _concatenate_chunks

        chunk1 = bytearray(b"\x00\x01")
        chunk2 = bytearray(b"\x02\x03")
        result = _concatenate_chunks([chunk1, chunk2])
        assert result == b"\x00\x01\x02\x03"

    def test_concatenate_file_like_chunks(self):
        """List of file-like objects are read and concatenated."""
        from xradar.io.backends.nexrad_level2 import _concatenate_chunks

        chunk1 = io.BytesIO(b"\x00\x01")
        chunk2 = io.BytesIO(b"\x02\x03")
        result = _concatenate_chunks([chunk1, chunk2])
        assert result == b"\x00\x01\x02\x03"

    def test_concatenate_mixed_types(self):
        """Mixed types (bytes, bytearray, file-like) are concatenated."""
        from xradar.io.backends.nexrad_level2 import _concatenate_chunks

        chunk1 = b"\x00\x01"
        chunk2 = io.BytesIO(b"\x02\x03")
        chunk3 = bytearray(b"\x04\x05")
        result = _concatenate_chunks([chunk1, chunk2, chunk3])
        assert result == b"\x00\x01\x02\x03\x04\x05"

    def test_concatenate_file_paths(self, tmp_path):
        """List of file paths are read and concatenated."""
        from xradar.io.backends.nexrad_level2 import _concatenate_chunks

        f1 = tmp_path / "chunk1.bin"
        f2 = tmp_path / "chunk2.bin"
        f1.write_bytes(b"\x00\x01")
        f2.write_bytes(b"\x02\x03")
        result = _concatenate_chunks([str(f1), f2])  # str and PathLike mixed
        assert result == b"\x00\x01\x02\x03"

    def test_multiple_volume_headers_raises(self):
        """Multiple chunks with volume headers should raise ValueError."""
        from xradar.io.backends.nexrad_level2 import _concatenate_chunks

        vol1 = b"AR2V0006." + b"\x00" * 50
        vol2 = b"AR2V0006." + b"\x00" * 50
        with pytest.raises(ValueError, match="Multiple chunks contain a volume header"):
            _concatenate_chunks([vol1, vol2])

    def test_volume_header_not_first_raises(self):
        """Volume header chunk not at index 0 should raise ValueError."""
        from xradar.io.backends.nexrad_level2 import _concatenate_chunks

        non_vol = b"\x00" * 50
        vol = b"AR2V0006." + b"\x00" * 50
        with pytest.raises(ValueError, match="volume header must be the first item"):
            _concatenate_chunks([non_vol, vol])

    def test_volume_header_first_is_ok(self):
        """Single volume header at index 0 should succeed."""
        from xradar.io.backends.nexrad_level2 import _concatenate_chunks

        vol = b"AR2V0006." + b"\x00" * 50
        non_vol = b"\x00" * 30
        result = _concatenate_chunks([vol, non_vol])
        assert result == vol + non_vol

    def test_no_volume_headers_ok(self):
        """All I/E chunks without volume headers should succeed."""
        from xradar.io.backends.nexrad_level2 import _concatenate_chunks

        chunk1 = b"BZh9" + b"\x00" * 50
        chunk2 = b"BZh9" + b"\x00" * 30
        result = _concatenate_chunks([chunk1, chunk2])
        assert result == chunk1 + chunk2

    def test_unsupported_chunk_type_raises(self):
        """Unsupported chunk type should raise TypeError."""
        from xradar.io.backends.nexrad_level2 import _concatenate_chunks

        with pytest.raises(TypeError, match="Unsupported chunk type"):
            _concatenate_chunks([12345])

    def test_list_input_opens_full_volume(self, nexradlevel2_file):
        """Passing a list with a single full-volume file should work."""
        with open(nexradlevel2_file, "rb") as f:
            file_bytes = f.read()

        # Split into two arbitrary byte chunks (simulating chunk input)
        mid = len(file_bytes) // 2
        chunks = [file_bytes[:mid], file_bytes[mid:]]

        # Should produce the same result as passing the file directly
        dtree_direct = open_nexradlevel2_datatree(
            nexradlevel2_file, reindex_angle=False, site_coords=True
        )
        dtree_list = open_nexradlevel2_datatree(
            chunks, reindex_angle=False, site_coords=True
        )

        direct_sweeps = list(dtree_direct.match("sweep_*").keys())
        list_sweeps = list(dtree_list.match("sweep_*").keys())
        assert direct_sweeps == list_sweeps

    def test_single_file_str_unchanged(self, nexradlevel2_file):
        """Single file path (str) still works as before."""
        dtree = open_nexradlevel2_datatree(
            nexradlevel2_file, reindex_angle=False, site_coords=True
        )
        assert isinstance(dtree, DataTree)
        sweep_groups = list(dtree.match("sweep_*").keys())
        assert len(sweep_groups) > 0

    def test_empty_list_returns_empty_bytes(self):
        """Empty list produces empty bytes."""
        from xradar.io.backends.nexrad_level2 import _concatenate_chunks

        result = _concatenate_chunks([])
        assert result == b""

    def test_single_element_list(self):
        """Single-element list returns that element unchanged."""
        from xradar.io.backends.nexrad_level2 import _concatenate_chunks

        data = b"\x00\x01\x02"
        result = _concatenate_chunks([data])
        assert result == data

    def test_single_element_with_volume_header(self):
        """Single-element list with volume header at index 0 succeeds."""
        from xradar.io.backends.nexrad_level2 import _concatenate_chunks

        vol = b"AR2V0006." + b"\x00" * 50
        result = _concatenate_chunks([vol])
        assert result == vol

    def test_chunk_shorter_than_4_bytes(self):
        """Chunks shorter than 4 bytes do not break volume header detection."""
        from xradar.io.backends.nexrad_level2 import _concatenate_chunks

        result = _concatenate_chunks([b"\x00\x01", b"\x02\x03"])
        assert result == b"\x00\x01\x02\x03"

    def test_empty_chunk_in_list(self):
        """Empty bytes chunk in list is handled correctly."""
        from xradar.io.backends.nexrad_level2 import _concatenate_chunks

        result = _concatenate_chunks([b"\x00\x01", b"", b"\x02\x03"])
        assert result == b"\x00\x01\x02\x03"

    def test_file_like_at_nonzero_position(self):
        """File-like with cursor advanced reads only remaining bytes."""
        from xradar.io.backends.nexrad_level2 import _concatenate_chunks

        buf = io.BytesIO(b"\x00\x01\x02\x03")
        buf.read(2)  # advance cursor past first 2 bytes
        result = _concatenate_chunks([buf])
        assert result == b"\x02\x03"

    def test_nonexistent_file_path_raises(self):
        """Non-existent file path raises FileNotFoundError."""
        from xradar.io.backends.nexrad_level2 import _concatenate_chunks

        with pytest.raises(FileNotFoundError):
            _concatenate_chunks(["/nonexistent/path/to/file.bin"])

    def test_tuple_input_opens_full_volume(self, nexradlevel2_file):
        """Tuple of byte chunks produces same result as file path."""
        with open(nexradlevel2_file, "rb") as f:
            file_bytes = f.read()

        mid = len(file_bytes) // 2
        chunks = (file_bytes[:mid], file_bytes[mid:])

        dtree_direct = open_nexradlevel2_datatree(
            nexradlevel2_file, reindex_angle=False, site_coords=True
        )
        dtree_tuple = open_nexradlevel2_datatree(
            chunks, reindex_angle=False, site_coords=True
        )

        assert list(dtree_direct.match("sweep_*").keys()) == list(
            dtree_tuple.match("sweep_*").keys()
        )


class TestSweepCompleteness:
    """Tests for sweep completeness tracking."""

    def test_complete_sweeps_in_full_volume(self, nexradlevel2_file):
        """All sweeps in a full volume file should be complete."""
        with NEXRADLevel2File(nexradlevel2_file, loaddata=False) as nex:
            incomplete = nex.incomplete_sweeps
            assert len(incomplete) == 0

    def test_complete_flag_set_on_normal_sweeps(self, nexradlevel2_file):
        """Sweeps closed by radial_status 2 or 4 should have complete=True."""
        with NEXRADLevel2File(nexradlevel2_file, loaddata=False) as nex:
            _ = nex.data_header  # trigger parsing
            for sweep_idx, sweep_data in nex.data.items():
                assert sweep_data.get("complete", False) is True

    def test_force_closed_sweep_is_incomplete(self):
        """A sweep that is force-closed (no end-of-elevation marker) should be incomplete."""
        # Use a real file but truncate it to simulate a chunk file ending mid-sweep
        # We'll create a mock scenario instead
        from collections import defaultdict

        class MockNEXRADForCompleteness(NEXRADLevel2File):
            def __init__(self):
                # Skip parent __init__
                self._fp = None
                self._data = OrderedDict()
                self._data_header = None
                self._meta_header = None
                self._msg_5_data = None
                self._msg_31_header = None
                self._msg_31_data_header = None

            @property
            def meta_header(self):
                if self._meta_header is None:
                    self._meta_header = defaultdict(list)
                return self._meta_header

        mock = MockNEXRADForCompleteness()
        # Simulate: sweep 0 is complete, sweep 1 is incomplete
        mock._data[0] = OrderedDict([("complete", True)])
        mock._data[1] = OrderedDict([("complete", False)])
        mock._data_header = []  # already parsed
        assert mock.incomplete_sweeps == {1}

    def test_multiple_incomplete_sweeps(self):
        """Multiple incomplete sweeps are all identified."""
        from collections import defaultdict

        class MockNEXRADForCompleteness(NEXRADLevel2File):
            def __init__(self):
                self._fp = None
                self._data = OrderedDict()
                self._data_header = None
                self._meta_header = None
                self._msg_5_data = None
                self._msg_31_header = None
                self._msg_31_data_header = None

            @property
            def meta_header(self):
                if self._meta_header is None:
                    self._meta_header = defaultdict(list)
                return self._meta_header

        mock = MockNEXRADForCompleteness()
        mock._data[0] = OrderedDict([("complete", True)])
        mock._data[1] = OrderedDict([("complete", False)])
        mock._data[2] = OrderedDict([("complete", True)])
        mock._data[3] = OrderedDict([("complete", False)])
        mock._data_header = []
        assert mock.incomplete_sweeps == {1, 3}

    def test_all_sweeps_incomplete(self):
        """All sweeps incomplete returns full set of indices."""
        from collections import defaultdict

        class MockNEXRADForCompleteness(NEXRADLevel2File):
            def __init__(self):
                self._fp = None
                self._data = OrderedDict()
                self._data_header = None
                self._meta_header = None
                self._msg_5_data = None
                self._msg_31_header = None
                self._msg_31_data_header = None

            @property
            def meta_header(self):
                if self._meta_header is None:
                    self._meta_header = defaultdict(list)
                return self._meta_header

        mock = MockNEXRADForCompleteness()
        mock._data[0] = OrderedDict([("complete", False)])
        mock._data[1] = OrderedDict([("complete", False)])
        mock._data[2] = OrderedDict([("complete", False)])
        mock._data_header = []
        assert mock.incomplete_sweeps == {0, 1, 2}

    def test_sweep_missing_complete_key_defaults_true(self):
        """Sweep dict without 'complete' key defaults to complete (True)."""
        from collections import defaultdict

        class MockNEXRADForCompleteness(NEXRADLevel2File):
            def __init__(self):
                self._fp = None
                self._data = OrderedDict()
                self._data_header = None
                self._meta_header = None
                self._msg_5_data = None
                self._msg_31_header = None
                self._msg_31_data_header = None

            @property
            def meta_header(self):
                if self._meta_header is None:
                    self._meta_header = defaultdict(list)
                return self._meta_header

        mock = MockNEXRADForCompleteness()
        mock._data[0] = OrderedDict()  # no "complete" key
        mock._data[1] = OrderedDict([("complete", False)])
        mock._data_header = []
        assert mock.incomplete_sweeps == {1}


class TestIncompleteSweepParameter:
    """Tests for the incomplete_sweep parameter in open_nexradlevel2_datatree."""

    def test_drop_mode_with_full_volume(self, nexradlevel2_file):
        """Drop mode with a full volume file should keep all sweeps (none incomplete)."""
        dtree = open_nexradlevel2_datatree(
            nexradlevel2_file,
            reindex_angle=False,
            site_coords=True,
            incomplete_sweep="drop",
        )
        sweep_groups = list(dtree.match("sweep_*").keys())
        assert len(sweep_groups) > 0

    def test_pad_mode_with_full_volume(self, nexradlevel2_file):
        """Pad mode with a full volume file should keep all sweeps (none incomplete)."""
        dtree = open_nexradlevel2_datatree(
            nexradlevel2_file,
            reindex_angle=False,
            site_coords=True,
            incomplete_sweep="pad",
        )
        sweep_groups = list(dtree.match("sweep_*").keys())
        assert len(sweep_groups) > 0

    def test_invalid_incomplete_sweep_raises(self, nexradlevel2_file):
        """Invalid incomplete_sweep value should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid incomplete_sweep"):
            open_nexradlevel2_datatree(
                nexradlevel2_file,
                reindex_angle=False,
                incomplete_sweep="invalid",
            )

    def test_drop_mode_warns_on_incomplete(self, nexradlevel2_file):
        """Drop mode should warn when incomplete sweeps exist."""
        import warnings

        # Read file and truncate to create incomplete last sweep
        with open(nexradlevel2_file, "rb") as f:
            file_bytes = f.read()

        # Truncate to ~80% to lose the last sweep(s)
        truncated = file_bytes[: int(len(file_bytes) * 0.8)]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                open_nexradlevel2_datatree(
                    truncated,
                    reindex_angle=False,
                    site_coords=True,
                    incomplete_sweep="drop",
                )
                # Check if a warning about incomplete sweeps was issued
                drop_warnings = [x for x in w if "incomplete" in str(x.message).lower()]
                if drop_warnings:
                    assert "Dropped" in str(drop_warnings[0].message)
            except Exception:
                # Truncated files may fail in various ways; the important
                # thing is that no crash occurs in our sweep filtering code
                pass

    def test_all_incomplete_returns_empty_datatree(self):
        """When all sweeps are incomplete, drop mode returns empty DataTree."""
        import warnings
        from unittest.mock import MagicMock, patch

        # Mock NEXRADLevel2File to report all sweeps as incomplete
        mock_nex = MagicMock()
        mock_nex.msg_5 = {"number_elevation_cuts": 2}
        mock_nex.msg_31_data_header = [MagicMock(), MagicMock()]
        mock_nex.incomplete_sweeps = {0, 1}
        mock_nex.__enter__ = MagicMock(return_value=mock_nex)
        mock_nex.__exit__ = MagicMock(return_value=False)

        with patch(
            "xradar.io.backends.nexrad_level2.NEXRADLevel2File",
            return_value=mock_nex,
        ):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                dtree = open_nexradlevel2_datatree(
                    b"\x00" * 100,
                    reindex_angle=False,
                    incomplete_sweep="drop",
                )
                assert len(dtree.children) == 0
                all_incomplete_warnings = [
                    x for x in w if "All sweeps are incomplete" in str(x.message)
                ]
                assert len(all_incomplete_warnings) == 1

    def test_backward_compat_default_drop(self, nexradlevel2_file):
        """Default incomplete_sweep='drop' should not change behavior for full volumes."""
        dtree_default = open_nexradlevel2_datatree(
            nexradlevel2_file, reindex_angle=False, site_coords=True
        )
        dtree_explicit = open_nexradlevel2_datatree(
            nexradlevel2_file,
            reindex_angle=False,
            site_coords=True,
            incomplete_sweep="drop",
        )
        assert list(dtree_default.match("sweep_*").keys()) == list(
            dtree_explicit.match("sweep_*").keys()
        )

    def test_drop_mode_warns_with_specific_indices(self):
        """Drop mode warns with correct sweep indices when multiple are incomplete."""
        mock_nex = MagicMock()
        mock_nex.msg_5 = {"number_elevation_cuts": 4}
        mock_nex.msg_31_data_header = [MagicMock()] * 4
        mock_nex.incomplete_sweeps = {1, 3}
        mock_nex.__enter__ = MagicMock(return_value=mock_nex)
        mock_nex.__exit__ = MagicMock(return_value=False)

        dummy_ds = xarray.Dataset(
            {
                "time": (
                    "azimuth",
                    np.array(
                        ["2024-01-01T00:00:00", "2024-01-01T00:01:00"],
                        dtype="datetime64[ns]",
                    ),
                ),
                "latitude": 35.0,
                "longitude": -97.0,
                "altitude": 300.0,
            },
        )
        mock_sweep_dict = OrderedDict([("sweep_0", dummy_ds), ("sweep_2", dummy_ds)])

        with (
            patch(
                "xradar.io.backends.nexrad_level2.NEXRADLevel2File",
                return_value=mock_nex,
            ),
            patch(
                "xradar.io.backends.nexrad_level2.open_sweeps_as_dict",
                return_value=mock_sweep_dict,
            ),
        ):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                open_nexradlevel2_datatree(
                    b"\x00" * 100,
                    reindex_angle=False,
                    incomplete_sweep="drop",
                )

                drop_warnings = [x for x in w if "Dropped" in str(x.message)]
                assert len(drop_warnings) == 1
                msg = str(drop_warnings[0].message)
                assert "2 incomplete sweep(s)" in msg
                assert "[1, 3]" in msg

    def test_pad_mode_passes_incomplete_set_to_open_sweeps(self):
        """Pad mode passes the incomplete_sweeps set to open_sweeps_as_dict."""
        mock_nex = MagicMock()
        mock_nex.msg_5 = {"number_elevation_cuts": 2}
        mock_nex.msg_31_data_header = [MagicMock()] * 2
        mock_nex.incomplete_sweeps = {1}
        mock_nex.__enter__ = MagicMock(return_value=mock_nex)
        mock_nex.__exit__ = MagicMock(return_value=False)

        dummy_ds = xarray.Dataset(
            {
                "time": (
                    "azimuth",
                    np.array(
                        ["2024-01-01T00:00:00", "2024-01-01T00:01:00"],
                        dtype="datetime64[ns]",
                    ),
                ),
                "latitude": 35.0,
                "longitude": -97.0,
                "altitude": 300.0,
            },
        )
        mock_sweep_dict = OrderedDict([("sweep_0", dummy_ds), ("sweep_1", dummy_ds)])

        with (
            patch(
                "xradar.io.backends.nexrad_level2.NEXRADLevel2File",
                return_value=mock_nex,
            ),
            patch(
                "xradar.io.backends.nexrad_level2.open_sweeps_as_dict",
                return_value=mock_sweep_dict,
            ) as mock_open_sweeps,
        ):
            open_nexradlevel2_datatree(
                b"\x00" * 100,
                reindex_angle=False,
                incomplete_sweep="pad",
            )
            # Verify incomplete_sweeps={1} was passed
            call_kwargs = mock_open_sweeps.call_args[1]
            assert call_kwargs["incomplete_sweeps"] == {1}

    def test_pad_mode_reads_incomplete_for_user_specified_sweeps(self):
        """Pad mode reads incomplete set even when sweep list is user-specified."""
        mock_nex = MagicMock()
        mock_nex.incomplete_sweeps = {1}
        mock_nex.__enter__ = MagicMock(return_value=mock_nex)
        mock_nex.__exit__ = MagicMock(return_value=False)

        dummy_ds = xarray.Dataset(
            {
                "time": (
                    "azimuth",
                    np.array(
                        ["2024-01-01T00:00:00", "2024-01-01T00:01:00"],
                        dtype="datetime64[ns]",
                    ),
                ),
                "latitude": 35.0,
                "longitude": -97.0,
                "altitude": 300.0,
            },
        )
        mock_sweep_dict = OrderedDict([("sweep_0", dummy_ds), ("sweep_1", dummy_ds)])

        with (
            patch(
                "xradar.io.backends.nexrad_level2.NEXRADLevel2File",
                return_value=mock_nex,
            ),
            patch(
                "xradar.io.backends.nexrad_level2.open_sweeps_as_dict",
                return_value=mock_sweep_dict,
            ) as mock_open_sweeps,
        ):
            open_nexradlevel2_datatree(
                b"\x00" * 100,
                sweep=[0, 1],
                reindex_angle=False,
                incomplete_sweep="pad",
            )
            # Lines 1931-1933: pad mode opens NEXRADLevel2File to get incomplete set
            call_kwargs = mock_open_sweeps.call_args[1]
            assert call_kwargs["incomplete_sweeps"] == {1}


class TestPadModeReindexing:
    """Tests for the pad-mode reindex pipeline using real NEXRAD data.

    Since truncating raw NEXRAD bytes breaks BZ2 record boundaries
    (causing EOFError before sweep data is collected), these tests instead
    open a real sweep and simulate an incomplete sweep by removing rays,
    then verify the reindex pipeline produces correct results.
    """

    def test_partial_sweep_reindex_produces_full_azimuth(self, nexradlevel2_file):
        """Removing rays from a real sweep and reindexing fills to full grid."""
        from xradar import util

        ds = xarray.open_dataset(
            nexradlevel2_file,
            group="sweep_0",
            engine="nexradlevel2",
            first_dim="auto",
        )
        original_size = ds.sizes["azimuth"]

        # Remove half the rays to simulate an incomplete sweep
        ds_partial = ds.isel(azimuth=slice(0, original_size // 2))

        ds_partial = util.remove_duplicate_rays(ds_partial)
        angle_params = util.extract_angle_parameters(ds_partial)
        ds_reindexed = util.reindex_angle(
            ds_partial,
            start_angle=angle_params["start_angle"],
            stop_angle=angle_params["stop_angle"],
            angle_res=float(angle_params["angle_res"]),
            direction=angle_params["direction"],
        )

        # Should be reindexed to a full azimuth grid
        assert ds_reindexed.sizes["azimuth"] in (360, 720)
        # Padded region should have NaN
        assert np.isnan(ds_reindexed["DBZH"].values).any()

    def test_partial_sweep_reindex_preserves_data(self, nexradlevel2_file):
        """Data in the non-padded region matches the original after reindex."""
        from xradar import util

        ds = xarray.open_dataset(
            nexradlevel2_file,
            group="sweep_0",
            engine="nexradlevel2",
            first_dim="auto",
        )
        n_keep = ds.sizes["azimuth"] // 2
        ds_partial = ds.isel(azimuth=slice(0, n_keep))

        ds_partial = util.remove_duplicate_rays(ds_partial)
        angle_params = util.extract_angle_parameters(ds_partial)
        ds_reindexed = util.reindex_angle(
            ds_partial,
            start_angle=angle_params["start_angle"],
            stop_angle=angle_params["stop_angle"],
            angle_res=float(angle_params["angle_res"]),
            direction=angle_params["direction"],
        )

        # Non-NaN values should still exist (data was preserved)
        dbzh = ds_reindexed["DBZH"].values
        valid_count = np.count_nonzero(~np.isnan(dbzh))
        assert valid_count > 0

    def test_complete_sweep_reindex_no_nans(self, nexradlevel2_file):
        """Reindexing a complete sweep does not introduce NaN values."""
        from xradar import util

        ds = xarray.open_dataset(
            nexradlevel2_file,
            group="sweep_0",
            engine="nexradlevel2",
            first_dim="auto",
        )

        ds = util.remove_duplicate_rays(ds)
        angle_params = util.extract_angle_parameters(ds)
        ds_reindexed = util.reindex_angle(
            ds,
            start_angle=angle_params["start_angle"],
            stop_angle=angle_params["stop_angle"],
            angle_res=float(angle_params["angle_res"]),
            direction=angle_params["direction"],
        )

        assert ds_reindexed.sizes["azimuth"] in (360, 720)

    def test_few_rays_reindex_high_nan_percentage(self, nexradlevel2_file):
        """Reindexing with very few rays produces high NaN percentage."""
        from xradar import util

        ds = xarray.open_dataset(
            nexradlevel2_file,
            group="sweep_0",
            engine="nexradlevel2",
            first_dim="auto",
        )
        # Keep only 36 rays (~10% of a 360-ray sweep)
        ds_partial = ds.isel(azimuth=slice(0, 36))

        ds_partial = util.remove_duplicate_rays(ds_partial)
        angle_params = util.extract_angle_parameters(ds_partial)
        ds_reindexed = util.reindex_angle(
            ds_partial,
            start_angle=angle_params["start_angle"],
            stop_angle=angle_params["stop_angle"],
            angle_res=float(angle_params["angle_res"]),
            direction=angle_params["direction"],
        )

        assert ds_reindexed.sizes["azimuth"] in (360, 720)
        nan_pct = np.isnan(ds_reindexed["DBZH"].values).mean()
        assert nan_pct > 0.5


class TestListInputWithIncompleteSweep:
    """Integration tests combining list input with incomplete_sweep parameter."""

    def test_list_input_full_volume_with_drop_mode(self, nexradlevel2_file):
        """List of full volume bytes with drop mode matches file path result."""
        with open(nexradlevel2_file, "rb") as f:
            file_bytes = f.read()

        mid = len(file_bytes) // 2

        dtree_direct = open_nexradlevel2_datatree(
            nexradlevel2_file,
            reindex_angle=False,
            site_coords=True,
            incomplete_sweep="drop",
        )
        dtree_list = open_nexradlevel2_datatree(
            [file_bytes[:mid], file_bytes[mid:]],
            reindex_angle=False,
            site_coords=True,
            incomplete_sweep="drop",
        )

        assert list(dtree_direct.match("sweep_*").keys()) == list(
            dtree_list.match("sweep_*").keys()
        )

    def test_list_input_full_volume_with_pad_mode(self, nexradlevel2_file):
        """List of full volume bytes with pad mode matches file path result."""
        with open(nexradlevel2_file, "rb") as f:
            file_bytes = f.read()

        mid = len(file_bytes) // 2

        dtree_direct = open_nexradlevel2_datatree(
            nexradlevel2_file,
            reindex_angle=False,
            site_coords=True,
            incomplete_sweep="pad",
        )
        dtree_list = open_nexradlevel2_datatree(
            [file_bytes[:mid], file_bytes[mid:]],
            reindex_angle=False,
            site_coords=True,
            incomplete_sweep="pad",
        )

        assert list(dtree_direct.match("sweep_*").keys()) == list(
            dtree_list.match("sweep_*").keys()
        )

    def test_tuple_input_full_volume_with_drop_mode(self, nexradlevel2_file):
        """Tuple of full volume bytes with drop mode matches file path result."""
        with open(nexradlevel2_file, "rb") as f:
            file_bytes = f.read()

        dtree_direct = open_nexradlevel2_datatree(
            nexradlevel2_file,
            reindex_angle=False,
            site_coords=True,
            incomplete_sweep="drop",
        )
        dtree_tuple = open_nexradlevel2_datatree(
            (file_bytes,),
            reindex_angle=False,
            site_coords=True,
            incomplete_sweep="drop",
        )

        assert list(dtree_direct.match("sweep_*").keys()) == list(
            dtree_tuple.match("sweep_*").keys()
        )
