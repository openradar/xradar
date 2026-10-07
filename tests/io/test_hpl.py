#!/usr/bin/env python
# Copyright (c) 2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.


import io

import numpy as np
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
        "site_as_coords": True,
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
    # all 20 rays of the first sweep (#430)
    assert dtree[sample_sweep]["mean_doppler_velocity"].shape == (20, 400)
    # Station coords should be on root as coordinates, NOT on sweeps
    assert "latitude" in dtree.ds.coords
    assert "longitude" in dtree.ds.coords
    assert "altitude" in dtree.ds.coords
    assert "latitude" not in dtree.ds.data_vars

    # Validate attributes
    assert len(dtree.attrs) == 9


def test_open_hpl_datatree_optional_groups():
    """Test that optional_groups=True includes metadata subgroups."""
    from open_radar_data import DATASETS

    hpl_file = DATASETS.fetch("User1_184_20240601_013257.hpl")
    dtree = xd.io.open_hpl_datatree(hpl_file, optional_groups=True)
    assert "radar_parameters" in dtree.children
    assert "georeferencing_correction" in dtree.children
    assert "radar_calibration" in dtree.children


def test_open_hpl_datatree_sweeps(hpl_file):
    # RHI with 361 rays, the last ray of the sweep is kept (#430)
    dtree = xd.io.open_hpl_datatree(hpl_file)
    ds = dtree["sweep_0"].ds
    assert ds.mean_doppler_velocity.shape == (361, 833)
    assert int(ds.sweep_number) == 0

    dtree = xd.io.open_hpl_datatree(DATASETS.fetch("User1_184_20240601_013257.hpl"))
    sweeps = [k for k in dtree.children if k.startswith("sweep_")]
    assert [int(dtree[k].ds.sweep_number) for k in sweeps] == list(range(len(sweeps)))
    assert list(dtree.ds.sweep_group_name.values) == sweeps


def _modified_hpl(src, dst, extra_header=False, drop_last_line=False):
    with open(src) as f:
        lines = f.readlines()
    if extra_header:
        lines.insert(11, "Some new header line:\tvalue\n")
    if drop_last_line:
        lines = lines[:-1]
    with open(dst, "w") as f:
        f.writelines(lines)
    return dst


def test_hpl_header_length(hpl_file, tmp_path):
    # header length is taken from the "****" separator
    path = _modified_hpl(hpl_file, tmp_path / "extra.hpl", extra_header=True)
    ds = xr.open_dataset(path, engine="hpl")
    assert ds.mean_doppler_velocity.shape == (361, 833)
    xr.testing.assert_equal(
        ds.mean_doppler_velocity,
        xr.open_dataset(hpl_file, engine="hpl").mean_doppler_velocity,
    )


def test_hpl_line_count_mismatch(hpl_file, tmp_path):
    path = _modified_hpl(hpl_file, tmp_path / "broken.hpl", drop_last_line=True)
    with pytest.raises(ValueError, match="does not match the expected format"):
        xr.open_dataset(path, engine="hpl")


def test_open_dataset_hpl_binary_iobase(hpl_file):
    with open(hpl_file, "rb") as fh:
        buf = io.BytesIO(fh.read())
    ds = xr.open_dataset(buf, engine="hpl")
    assert ds.mean_doppler_velocity.shape == (361, 833)


def test_hpl_unsupported_input():
    from xradar.io.backends.hpl import HplFile

    with pytest.raises(TypeError, match="Unsupported input type"):
        HplFile(12345)


def _write_hpl(path, rays, ngates=2):
    """Write a minimal Halo .hpl file, rays as (azimuth, elevation)."""
    header = [
        f"Filename:\t{path.name}",
        "System ID:\t1",
        f"Number of gates:\t{ngates}",
        "Range gate length (m):\t30.0",
        "Gate length (pts):\t10",
        "Pulses/ray:\t1000",
        "No. of waypoints in file:\t3",
        "Scan type:\tUser file 1 - csm",
        "Focus range:\t65535",
        "Start time:\t20231003 14:01:31.40",
        "Resolution (m/s):\t0.0760",
        "Range of measurement (center of gate) = (range gate + 0.5) * Gate length",
        "Data line 1: Decimal time (hours)  Azimuth (degrees)  Elevation (degrees) "
        "Pitch (degrees) Roll (degrees)",
        "f9.6,1x,f6.2,1x,f6.2",
        "Data line 2: Range Gate  Doppler (m/s)  Intensity (SNR + 1)  "
        "Beta (m-1 sr-1)",
        "i3,1x,f6.4,1x,f8.6,1x,e12.6 - repeat for no. gates",
        "****",
    ]
    lines = []
    for i, (az, el) in enumerate(rays):
        lines.append(f"{14.03 + i * 0.001:.8f} {az:6.2f} {el:6.2f} 0.00 0.00")
        for gate in range(ngates):
            lines.append(f"{gate:3d} 1.0000 1.050000  1.000000E-6")
    path.write_text("\n".join(header + lines) + "\n")
    return path


def test_open_hpl_datatree_single_ray_sweeps(tmp_path):
    # custom scan: RHI up, azimuth move at constant elevation, RHI down.
    # It is split into sweeps of a single ray, which were empty and made
    # opening fail (#303)
    up = [(255.0, el) for el in np.arange(0, 40.01, 0.55)]
    move = [(az, 40.0) for az in np.arange(255.5, 265.0, 0.55)]
    down = [(265.0, el) for el in np.arange(40, -0.01, -0.55)]
    path = _write_hpl(tmp_path / "csm.hpl", up + move + down)
    dtree = xd.io.open_hpl_datatree(str(path))
    sweeps = [k for k in dtree.children if k.startswith("sweep_")]
    nrays = [dtree[k].ds.mean_doppler_velocity.shape[0] for k in sweeps]
    assert min(nrays) >= 1
    assert [int(dtree[k].ds.sweep_number) for k in sweeps] == list(range(len(sweeps)))
