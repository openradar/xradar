#!/usr/bin/env python
# Copyright (c) 2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.


import tarfile

import pytest
from xarray import DataTree, open_dataset, open_mfdataset

from xradar.io.backends import datamet, open_datamet_datatree
from xradar.util import _get_data_file


@pytest.fixture(scope="session")
def data(datamet_file):
    with _get_data_file(datamet_file, "file") as datametfile:
        data = datamet.DataMetFile(datametfile)
        assert data.filename == datametfile
    data.get_sweep(0)
    return data


def test_file_types(data):
    assert isinstance(data, datamet.DataMetFile)
    assert isinstance(data.tarfile, tarfile.TarFile)


def test_basic_content(data):
    assert data.moments == ["UZ", "CZ", "V", "W", "ZDR", "PHIDP", "RHOHV", "KDP"]
    assert len(data.data[0]) == len(data.moments)
    assert data.first_dimension == "azimuth"
    assert data.scan_metadata["origin"] == "ILMONTE"
    assert data.scan_metadata["orig_lat"] == 41.9394
    assert data.scan_metadata["orig_lon"] == 14.6208
    assert data.scan_metadata["orig_alt"] == 710


def test_moment_metadata(data):
    mom_metadata = data.get_mom_metadata("UZ", 0)
    assert mom_metadata["Rangeoff"] == 0.0
    assert mom_metadata["Eloff"] == 16.05
    assert mom_metadata["nlines"] == 360
    assert mom_metadata["ncols"] == 493


@pytest.mark.parametrize(
    "moment, expected_value",
    [
        ("UZ", 56),
        ("CZ", 56),
        ("V", 139),
        ("W", 30),
        ("ZDR", 137),
        ("PHIDP", 16952),
        ("RHOHV", 237),
        ("KDP", 67),
    ],
)
def test_moment_data(data, moment, expected_value):
    assert data.data[0][moment][(4, 107)] == expected_value


def test_datamet_open_mfdataset_context_manager(datamet_file):
    with open_mfdataset(
        [datamet_file],
        engine="datamet",
        concat_dim="volume_time",
        combine="nested",
        group="sweep_0",
    ) as ds:
        assert ds is not None
        # closer must exist while inside context
        assert callable(getattr(ds, "_close", None))


def test_datamet_dataset_has_close(datamet_file):
    ds = open_dataset(datamet_file, engine="datamet", group="sweep_0")
    assert callable(getattr(ds, "_close", None))
    ds.close()


def test_open_datamet_datatree(datamet_file):
    # Define kwargs to pass into the function
    kwargs = {
        "sweep": [0, 1],  # Test with specific sweeps
        "first_dim": "auto",
        "reindex_angle": {
            "start_angle": 0.0,
            "stop_angle": 360.0,
            "angle_res": 1.0,
            "direction": 1,  # Set a valid direction within reindex_angle
        },
        "site_coords": True,
    }

    # Call the function with an actual DataMet file
    dtree = open_datamet_datatree(datamet_file, **kwargs)

    assert isinstance(dtree, DataTree), "Expected a DataTree instance"
    subtree_paths = [n.path for n in dtree.subtree]
    assert "/" in subtree_paths, "Root group should be present in the DataTree"
    # optional_groups=False by default: metadata subgroups should NOT be present
    assert "radar_parameters" not in dtree.children
    assert "georeferencing_correction" not in dtree.children
    assert "radar_calibration" not in dtree.children

    # Check if at least one sweep group is attached (e.g., "/sweep_0")
    sweep_groups = [key for key in dtree.match("sweep_*")]
    assert len(sweep_groups) == 2, "Expected four sweep groups in the DataTree"

    # Verify a sample variable in one of the sweep groups (adjust based on expected variables)
    sample_sweep = sweep_groups[0]
    assert (
        len(dtree[sample_sweep].data_vars) == 13
    ), f"Expected data variables in {sample_sweep}"
    assert dtree[sample_sweep]["DBZH"].shape == (360, 493)
    assert (
        "DBZH" in dtree[sample_sweep].data_vars
    ), f"DBZH should be a data variable in {sample_sweep}"
    assert (
        "VRADH" in dtree[sample_sweep].data_vars
    ), f"VRADH should be a data variable in {sample_sweep}"

    # Station coords should be on root as coordinates, NOT on sweeps
    assert "latitude" in dtree.ds.coords
    assert "longitude" in dtree.ds.coords
    assert "altitude" in dtree.ds.coords
    assert "latitude" not in dtree.ds.data_vars

    # Sweeps should NOT have local station coords
    sweep_ds = dtree[sample_sweep].to_dataset(inherit=False)
    assert "latitude" not in sweep_ds.coords
    assert "latitude" not in sweep_ds.data_vars

    # Validate attributes
    assert len(dtree.attrs) == 10
    assert (
        dtree.attrs["instrument_name"] == "ILMONTE"
    ), "Instrument name should match expected value"
    assert dtree.attrs["source"] == "Datamet", "Source should match expected value"
    assert (
        dtree.attrs["scan_name"] == "VOLUMETRICA"
    ), "Scan name should match expected value"

    # Verify a sample variable in on
