#!/usr/bin/env python
# Copyright (c) 2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.


import pytest
import xarray as xr
from open_radar_data import DATASETS
from xarray import DataTree, open_dataset, open_mfdataset

import xradar as xd


def test_open_datatree_hpl():
    dtree = xd.io.open_hpl_datatree(
        DATASETS.fetch("User1_184_20240601_013257.hpl"),
        sweep=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        backend_kwargs=dict(latitude=41.24276244459537, longitude=-70.1070364814594),
    )
    assert "/sweep_0" in list(dtree.groups)
    assert dtree["sweep_0"]["mean_doppler_velocity"].dims == ("azimuth", "range")
    assert dtree["sweep_0"]["mean_doppler_velocity"].max() == 19.5306

    # regression test for https://github.com/openradar/xradar/issues/296
    assert dtree["sweep_0"]["sweep_mode"].dtype.kind == "S"
    assert dtree["sweep_0"]["sweep_number"].dtype.kind == "i"
    assert "units" not in dtree["sweep_0"]["time"].attrs


def test_open_dataset_hpl():
    with xr.open_dataset(
        DATASETS.fetch("User1_184_20240601_013257.hpl"),
        engine="hpl",
        backend_kwargs=dict(latitude=40, longitude=-70),
    ) as ds:

        assert ds["mean_doppler_velocity"].dims == ("azimuth", "range")
        assert ds["mean_doppler_velocity"].max() == 19.5306


def test_open_dataset_hpl_iobase():
    with open(DATASETS.fetch("User1_184_20240601_013257.hpl")) as fi:  # noqa
        ds = xr.open_dataset(
            fi, engine="hpl", backend_kwargs=dict(latitude=40, longitude=-70)
        )

        assert ds["mean_doppler_velocity"].dims == ("azimuth", "range")
        assert ds["mean_doppler_velocity"].max() == 19.5306


def test_open_rhi():
    with xr.open_dataset(
        DATASETS.fetch("User1_100_20240714_122137.hpl"),
        engine="hpl",
        backend_kwargs=dict(latitude=41.24276244459537, longitude=-70.1070364814594),
    ) as ds:

        assert ds["mean_doppler_velocity"].dims == ("azimuth", "range")
        assert ds["mean_doppler_velocity"].max() == 19.5306


def test_hpl_open_mfdataset_context_manager(hpl_file):
    with open_mfdataset(
        [hpl_file],
        engine="hpl",
        concat_dim="volume_time",
        combine="nested",
        group="sweep_0",
    ) as ds:
        assert ds is not None
        # closer must exist while inside context
        assert callable(getattr(ds, "_close", None))


def test_hpl_dataset_has_close(hpl_file):
    ds = open_dataset(hpl_file, engine="hpl", group="sweep_0")
    assert callable(getattr(ds, "_close", None))
    ds.close()


def test_open_hpl_datatree():
    # Define the kwargs to pass into the function
    kwargs = {
        "sweep": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "first_dim": "auto",
        "site_coords": True,
        "backend_kwargs": {
            "latitude": 41.24276244459537,
            "longitude": -70.1070364814594,
        },
    }

    # Call the function with an actual HPL file
    hpl_file = DATASETS.fetch("User1_184_20240601_013257.hpl")
    dtree = xd.io.open_hpl_datatree(hpl_file, **kwargs)

    # Assertions
    assert isinstance(dtree, DataTree), "Expected a DataTree instance"
    subtree_paths = [n.path for n in dtree.subtree]
    assert "/" in subtree_paths, "Root group should be present in the DataTree"
    # optional_groups=False by default: metadata subgroups should NOT be present
    assert "radar_parameters" not in dtree.children
    assert "georeferencing_correction" not in dtree.children
    assert "radar_calibration" not in dtree.children

    # Verify that each sweep group is attached correctly (e.g., "/sweep_0")
    sweep_groups = [key for key in dtree.match("sweep_*")]
    assert len(sweep_groups) == 9, "Expected nine sweep groups in the DataTree"

    # Verify a sample variable in one of the sweep groups
    sample_sweep = sweep_groups[0]
    assert (
        len(dtree[sample_sweep].data_vars) == 11
    ), f"Expected data variables in {sample_sweep}"
    assert (
        "mean_doppler_velocity" in dtree[sample_sweep].data_vars
    ), f"mean_doppler_velocity should be a data variable in {sample_sweep}"
    assert dtree[sample_sweep]["mean_doppler_velocity"].dims == ("azimuth", "range")
    assert dtree[sample_sweep]["mean_doppler_velocity"].max() == pytest.approx(
        19.5306, rel=1e-3
    )
    assert dtree[sample_sweep]["mean_doppler_velocity"].shape == (34, 400)
    # Station coords should be on root as coordinates, NOT on sweeps
    assert "latitude" in dtree.ds.coords
    assert "longitude" in dtree.ds.coords
    assert "altitude" in dtree.ds.coords
    assert "latitude" not in dtree.ds.data_vars

    # Validate attributes
    assert len(dtree.attrs) == 9
