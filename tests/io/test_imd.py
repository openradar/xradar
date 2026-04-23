#!/usr/bin/env python
# Copyright (c) 2024-2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for `io.backends.imd` module."""

import os

import numpy as np
import pytest
from xarray import DataTree, open_dataset

from xradar.io.backends import group_imd_files, open_imd_datatree, open_imd_volumes
from xradar.io.backends.imd import _conform_imd_sweep, imd_mapping


def test_imd_mapping_is_cfradial2():
    # Base single-pol moments
    assert imd_mapping["T"] == "DBTH"
    assert imd_mapping["Z"] == "DBZH"
    assert imd_mapping["V"] == "VRADH"
    assert imd_mapping["W"] == "WRADH"
    # Dual-pol moments IMD actually ships (on sites like Jaipur S-band).
    # RHOHV/PHIDP/KDP are *not* distributed by IMD and are intentionally
    # absent from the mapping.
    assert imd_mapping["ZDR"] == "ZDR"
    assert imd_mapping["HCLASS"] == "HCLASS"


def test_open_dataset_imd(imd_file):
    ds = open_dataset(imd_file, engine="imd")
    # CfRadial2 sweep-level scalars are present
    assert "sweep_number" in ds.variables
    assert "sweep_mode" in ds.variables
    assert "scan_type" in ds.variables
    assert str(ds["scan_type"].values) in {
        "ppi",
        "rhi",
        "ppi_sector",
        "rhi_sector",
        "unknown",
    }
    assert "sweep_fixed_angle" in ds.variables
    # at least one moment renamed to CfRadial2
    assert set(ds.data_vars) & {"DBZH", "VRADH", "DBTH", "WRADH"}
    # range built from firstGateRange/gateSize
    assert "range" in ds.coords
    assert ds["range"].attrs.get("units") == "meters"
    # station coords promoted
    assert "latitude" in ds.coords
    assert "longitude" in ds.coords
    assert "altitude" in ds.coords
    # close callable present for resource cleanup
    assert callable(getattr(ds, "_close", None))
    ds.close()


def test_open_dataset_imd_first_dim_time(imd_file):
    ds = open_dataset(imd_file, engine="imd", first_dim="time")
    assert "time" in ds.dims


def test_open_imd_datatree_single_file(imd_file):
    dtree = open_imd_datatree(imd_file)
    assert isinstance(dtree, DataTree)
    assert "/" in [n.path for n in dtree.subtree]
    assert "sweep_0" in dtree.children
    # station coords on root, not on sweep
    assert "latitude" in dtree.ds.coords
    assert "longitude" in dtree.ds.coords
    assert "altitude" in dtree.ds.coords
    # sweep_group_name and sweep_fixed_angle on root
    assert "sweep_group_name" in dtree.ds.variables
    assert "sweep_fixed_angle" in dtree.ds.variables
    # moment present on sweep
    sw0 = dtree["sweep_0"].ds
    assert set(sw0.data_vars) & {"DBZH", "VRADH", "DBTH", "WRADH"}


def test_open_imd_datatree_volume(imd_volume_files):
    dtree = open_imd_datatree(imd_volume_files)
    assert isinstance(dtree, DataTree)
    sweep_groups = [k for k in dtree.match("sweep_*")]
    assert len(sweep_groups) == len(imd_volume_files)
    # sweeps sorted by fixed angle ascending
    angles = dtree.ds["sweep_fixed_angle"].values
    assert np.all(np.diff(angles) >= 0)
    # sweep_number matches position
    for i, sw in enumerate(sweep_groups):
        assert int(dtree[sw].ds["sweep_number"].values) == i


def test_open_imd_datatree_angle_filter(imd_volume_files):
    """min_angle/max_angle forwarded to util.create_volume."""
    # Load first to learn the actual angles, then filter to just the lowest.
    dtree = open_imd_datatree(imd_volume_files)
    angles = sorted(float(a) for a in dtree.ds["sweep_fixed_angle"].values)
    cutoff = (angles[0] + angles[1]) / 2  # keep only the first sweep

    filtered = open_imd_datatree(imd_volume_files, max_angle=cutoff)
    filtered_angles = [float(a) for a in filtered.ds["sweep_fixed_angle"].values]
    assert len(filtered_angles) == 1
    assert filtered_angles[0] <= cutoff


def test_open_imd_datatree_single_file_kwargs_reject_cv_kwargs(imd_file):
    """create_volume kwargs are only valid for multi-file input."""
    with pytest.raises(TypeError):
        open_imd_datatree(imd_file, min_angle=0.0)


def test_open_imd_volumes_single_volume(imd_volume_files):
    """Passing one volume's worth of files yields a single vcp_NN child."""
    tree = open_imd_volumes(imd_volume_files)
    assert list(tree.children) == ["vcp_00"]
    assert "sweep_0" in tree["vcp_00"].children
    assert len(tree["vcp_00"].children) == len(imd_volume_files)
    # parent root has IMD metadata
    assert tree.attrs.get("institution") == "India Meteorological Department"


def test_open_imd_volumes_multi_volume(imd_volume_files, tmp_path):
    """Two volumes with different stems -> vcp_00 and vcp_01."""
    import shutil

    # Fake a second volume by copying the fixture files under a new stem
    vol2 = []
    for src in imd_volume_files:
        base = os.path.basename(src)
        new_base = base.replace("JPR220822135253", "JPR220822145253")
        dst = tmp_path / new_base
        shutil.copyfile(src, dst)
        vol2.append(str(dst))

    tree = open_imd_volumes(imd_volume_files + vol2)
    assert list(tree.children) == ["vcp_00", "vcp_01"]
    for vcp in ("vcp_00", "vcp_01"):
        assert "sweep_0" in tree[vcp].children


def test_open_imd_volumes_empty_raises(tmp_path):
    with pytest.raises(ValueError, match="No IMD files"):
        open_imd_volumes(str(tmp_path))


def test_group_imd_files_from_list(tmp_path):
    # two volumes mixed: A has 3 sweeps, B has 2 sweeps
    names = [
        "GOA210515003646-IMD-C.nc",
        "GOA210515003646-IMD-C.nc.1",
        "GOA210515003646-IMD-C.nc.2",
        "GOA210515012233-IMD-C.nc",
        "GOA210515012233-IMD-C.nc.1",
        "unrelated.txt",  # should be skipped
    ]
    paths = [tmp_path / n for n in names]
    for p in paths:
        p.touch()

    groups = group_imd_files([str(p) for p in paths])
    assert len(groups) == 2
    # lexical stem order: 003646 before 012233
    assert [os.path.basename(p) for p in groups[0]] == [
        "GOA210515003646-IMD-C.nc",
        "GOA210515003646-IMD-C.nc.1",
        "GOA210515003646-IMD-C.nc.2",
    ]
    assert [os.path.basename(p) for p in groups[1]] == [
        "GOA210515012233-IMD-C.nc",
        "GOA210515012233-IMD-C.nc.1",
    ]


def test_group_imd_files_from_directory(tmp_path):
    for name in [
        "GOA210515003646-IMD-C.nc",
        "GOA210515003646-IMD-C.nc.1",
        "GOA210515003646-IMD-C.nc.2",
        "README.md",
    ]:
        (tmp_path / name).touch()

    groups = group_imd_files(str(tmp_path))
    assert len(groups) == 1
    assert len(groups[0]) == 3


def test_group_imd_files_from_glob(tmp_path):
    names = [
        "GOA210515003646-IMD-C.nc",
        "GOA210515003646-IMD-C.nc.1",
        "unrelated.txt",
    ]
    for n in names:
        (tmp_path / n).touch()

    groups = group_imd_files(str(tmp_path / "GOA*"))
    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_open_imd_datatree_optional_groups(imd_file):
    dtree = open_imd_datatree(imd_file, optional_groups=True)
    assert "radar_parameters" in dtree.children
    assert "georeferencing_correction" in dtree.children
    assert "radar_calibration" in dtree.children

    # radar_parameters must carry IMD's mapped beam widths / bandwidth
    rp = dtree["radar_parameters"].ds
    assert "radar_beam_width_h" in rp.variables
    assert "radar_beam_width_v" in rp.variables
    assert "radar_receiver_bandwidth" in rp.variables
    # sanity: beam width is a small positive angle, not NaN
    bw_h = float(rp["radar_beam_width_h"].values)
    assert 0 < bw_h < 10

    # georeferencing_correction is legitimately empty -- IMD has no corrections
    assert not dtree["georeferencing_correction"].ds.data_vars

    # radar_calibration carries IMD-named cal scalars (not canonical CfRadial2).
    # pulse_width now lives on the sweep per FM 301 Table 8a, not in cal.
    rc = dtree["radar_calibration"].ds
    expected_cal = {"calibConst", "radarConst", "calNoise"}
    assert expected_cal.issubset(
        rc.data_vars
    ), f"expected {expected_cal} in /radar_calibration, got {set(rc.data_vars)}"


def test_open_dataset_imd_reindex_angle(imd_file):
    """reindex_angle should produce a uniformly-spaced azimuth grid."""
    ds = open_dataset(
        imd_file,
        engine="imd",
        reindex_angle={
            "start_angle": 0.0,
            "stop_angle": 360.0,
            "angle_res": 1.0,
            "direction": 1,
        },
    )
    assert ds.sizes["azimuth"] == 360
    spacing = np.diff(ds["azimuth"].values)
    np.testing.assert_allclose(spacing, 1.0, atol=1e-6)
    ds.close()


def test_open_imd_datatree_has_cfradial2_required_vars(imd_file):
    """FM 301 Table 7a: prt_mode and follow_mode are required on each sweep."""
    dtree = open_imd_datatree(imd_file)
    sw = dtree["sweep_0"].ds
    assert "prt_mode" in sw.variables
    assert "follow_mode" in sw.variables
    assert str(sw["prt_mode"].values) in {"fixed", "staggered", "not_set"}
    assert str(sw["follow_mode"].values) == "not_set"


def test_open_dataset_imd_fm301_sweep_metadata(imd_file):
    """FM 301 Tables 7b and 8a: PRT, scan_rate, n_samples on (time,) dim."""
    ds = open_dataset(imd_file, engine="imd")
    # azimuth/elevation coord attrs match Table 7b
    assert ds["azimuth"].attrs["standard_name"] == "sensor_to_target_azimuth_angle"
    assert ds["azimuth"].attrs["axis"] == "radial_azimuth_coordinate"
    assert ds["elevation"].attrs["standard_name"] == "sensor_to_target_elevation_angle"
    assert ds["elevation"].attrs["axis"] == "radial_elevation_coordinate"

    # range coord attrs match Table 6b
    assert ds["range"].attrs["standard_name"] == "projection_range_coordinate"
    assert ds["range"].attrs["axis"] == "radial_range_coordinate"

    # Each moment carries the FM 301 Table 9.iv coordinates attribute
    for m in ("DBZH", "VRADH", "WRADH", "DBTH"):
        if m in ds.data_vars:
            assert ds[m].attrs.get("coordinates") == "elevation azimuth range"

    # PRF-derived time-dim metadata (Table 8a)
    n_rays = ds.sizes.get("time") or ds.sizes.get("azimuth")
    for v in ("prt", "scan_rate", "n_samples", "pulse_width"):
        if v in ds.variables:
            assert ds[v].ndim == 1
            assert ds[v].shape[0] == n_rays

    # frequency coord (Table 6a) in Hz
    if "frequency" in ds.coords:
        f = float(ds["frequency"].values[0])
        # C-band ~4-8 GHz, S-band ~2-4 GHz, X-band ~8-12 GHz
        assert 1e9 < f < 2e10
        assert ds["frequency"].attrs.get("units") == "s-1"
    ds.close()


def test_open_imd_datatree_fm301_root_attrs(imd_file):
    """FM 301 Table 2: mandatory global attributes."""
    dtree = open_imd_datatree(imd_file)
    assert dtree.attrs["Conventions"] == "CF-1.8, WMO CF-1.0, ACDD-1.3"
    assert dtree.attrs["wmo__cf_profile"] == "FM 301-XX"
    assert dtree.attrs["platform_is_mobile"] == "false"
    assert dtree.attrs["institution"] == "India Meteorological Department"


def test_open_dataset_imd_moment_attrs(imd_file):
    """Moments should carry canonical CfRadial2 standard_name/long_name/units."""
    ds = open_dataset(imd_file, engine="imd")
    for mname in ("DBTH", "DBZH", "VRADH", "WRADH"):
        if mname not in ds.data_vars:
            continue
        attrs = ds[mname].attrs
        assert "standard_name" in attrs, f"{mname} missing standard_name"
        assert "long_name" in attrs, f"{mname} missing long_name"
        assert "units" in attrs, f"{mname} missing units"
        # stale packed-int8 attrs must not leak through after mask_and_scale
        assert "valid_range" not in attrs
        assert "below_threshold" not in attrs
    if "DBZH" in ds.data_vars:
        assert ds["DBZH"].attrs["units"] == "dBZ"
        assert "reflectivity" in ds["DBZH"].attrs["standard_name"]
    ds.close()


def test_conform_imd_sweep_directly(imd_file):
    """_conform_imd_sweep should accept a raw IMD Dataset and produce CfRadial2 layout."""
    import xarray as xr

    raw = xr.open_dataset(imd_file, engine="netcdf4", decode_timedelta=False)
    ds = _conform_imd_sweep(raw, first_dim="auto", site_as_coords=True)
    assert "sweep_mode" in ds.variables
    assert "sweep_number" in ds.variables
    assert "sweep_fixed_angle" in ds.variables
    raw.close()
