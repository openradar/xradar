#!/usr/bin/env python
# Copyright (c) 2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for `io.backends.gamic` module.

ported from wradlib
"""

import io
import shutil
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest
import xarray as xr
from xarray import DataTree, open_dataset, open_mfdataset

import xradar as xd
from xradar import model
from xradar.io.backends import gamic
from xradar.io.backends.common import _get_radar_calibration, _same_value
from xradar.io.backends.gamic import open_gamic_datatree


def create_ray_header(nrays=360):
    dtype = {
        "names": [
            "azimuth_start",
            "azimuth_stop",
            "elevation_start",
            "elevation_stop",
            "timestamp",
        ],
        "formats": ["<f8", "<f8", "<f8", "<f8", "<i8"],
        "offsets": [0, 8, 16, 24, 32],
        "itemsize": 40,
    }
    ray_header = np.zeros(nrays, dtype=dtype)
    start_az = np.linspace(0, 360, nrays, endpoint=False, dtype=np.float64)
    stop_az = np.linspace(1, 361, nrays, endpoint=False, dtype=np.float64)
    ray_header["azimuth_start"] = start_az
    ray_header["azimuth_stop"] = stop_az

    start_el = np.ones_like(start_az, dtype=np.float64)
    stop_el = np.ones_like(start_az, dtype=np.float64)
    ray_header["elevation_start"] = start_el
    ray_header["elevation_stop"] = stop_el

    time = np.arange(
        1527831788042000, 1527831788042000 + nrays * 100000, 100000, dtype=np.int64
    )
    ray_header["timestamp"] = time
    return ray_header


@pytest.mark.parametrize("nrays", [180, 240, 360, 720])
def test_get_azimuth(nrays):
    ray_header = create_ray_header(nrays)
    actual = gamic._get_azimuth(ray_header)
    udiff = np.unique(np.diff(actual))
    assert len(actual) == nrays
    assert len(udiff) == 1
    assert udiff[0] == 360.0 / nrays


@pytest.mark.parametrize("nrays", [180, 240, 360, 720])
def test_get_elevation(nrays):
    ray_header = create_ray_header(nrays)
    actual = gamic._get_elevation(ray_header)
    unique = np.unique(actual)
    assert len(actual) == nrays
    assert len(unique) == 1
    assert unique[0] == 1.0


@pytest.mark.parametrize("nrays", [180, 240, 360, 720])
def test_get_timestamp(nrays):
    ray_header = create_ray_header(nrays)
    actual = gamic._get_time(ray_header)
    udiff = np.unique(np.diff(actual))
    assert len(actual) == nrays
    assert len(udiff) == 1
    assert udiff[0] == 100000


@pytest.mark.parametrize(
    "ang",
    [("elevation", "azimuth"), ("azimuth", "elevation")],
)
def test_get_fixed_dim_and_angle(ang):
    how = {ang[0]: 1.0}
    dim, angle = gamic._get_fixed_dim_and_angle(how)
    assert dim == ang[1]
    assert angle == 1.0


@pytest.mark.parametrize("range_step", [100, 150, 300, 1000])
@pytest.mark.parametrize("range_samples", [1, 2, 4])
def test_get_range(range_step, range_samples):
    where = dict(bin_count=10, range_step=range_step, range_samples=range_samples)
    rng, cent_first, bin_range = gamic._get_range(where)
    assert len(rng) == 10
    assert np.unique(np.diff(rng))[0] == range_step * range_samples
    assert cent_first == (range_step * range_samples) / 2
    assert bin_range == range_step * range_samples


def test_GamicH5NetCDFMetadata(gamic_file):
    store = gamic.GamicStore.open(gamic_file, group="sweep_0")
    with pytest.warns(DeprecationWarning):
        assert store.root.first_dim == "azimuth"


def test_gamic_open_mfdataset_context_manager(gamic_file):
    with open_mfdataset(
        [gamic_file],
        engine="gamic",
        concat_dim="volume_time",
        combine="nested",
        group="sweep_0",
    ) as ds:
        assert ds is not None
        # closer must exist while inside context
        assert callable(getattr(ds, "_close", None))


def test_gamic_dataset_has_close(gamic_file):
    ds = open_dataset(gamic_file, engine="gamic", group="sweep_0")
    assert callable(getattr(ds, "_close", None))
    ds.close()


def test_open_gamic_datatree(gamic_file, closing):
    # Define kwargs to pass into the function
    kwargs = {
        "sweep": [0, 1],  # Test with specific sweeps
        "first_dim": "auto",
        "reindex_coord": {
            "angle": {
                "start_angle": 0.0,
                "stop_angle": 360.0,
                "angle_res": 1.0,
                "direction": 1,  # Set a valid direction within reindex_angle
            }
        },
        "fix_second_angle": True,
        "site_as_coords": True,
    }

    # Call the function with an actual GAMIC HDF5 file
    dtree = closing(open_gamic_datatree(gamic_file, **kwargs))

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
    assert len(sweep_groups) == 2, "Expected two sweep groups in the DataTree"

    # Verify a sample variable in one of the sweep groups
    sample_sweep = sweep_groups[0]
    assert (
        len(dtree[sample_sweep].data_vars) == 18
    ), f"Expected data variables in {sample_sweep}"
    assert dtree[sample_sweep]["DBZH"].shape == (360, 360)
    assert (
        "DBZH" in dtree[sample_sweep].data_vars
    ), f"DBZH should be a data variable in {sample_sweep}"
    assert (
        "VRADV" in dtree[sample_sweep].data_vars
    ), f"VRADV should be a data variable in {sample_sweep}"

    # Station coords should be on root as coordinates, NOT on sweeps
    assert "latitude" in dtree.ds.coords
    assert "longitude" in dtree.ds.coords
    assert "altitude" in dtree.ds.coords
    assert "latitude" not in dtree.ds.data_vars

    # Validate attributes
    assert len(dtree.attrs) == 9
    assert dtree.attrs["source"] == "gamic", "Source should match expected value"


def test_open_gamic_datatree_optional_groups(gamic_file, closing):
    """Test that optional_groups=True includes metadata subgroups."""
    dtree = closing(open_gamic_datatree(gamic_file, optional_groups=True))
    assert "radar_parameters" in dtree.children
    assert "georeferencing_correction" in dtree.children
    assert "radar_calibration" in dtree.children


def test_gamic_radar_metadata(gamic_file, closing):
    dtree = closing(
        xd.open_datatree(gamic_file, engine="gamic", optional_groups=True, sweep=0)
    )
    # /how azimuth_beam (horizontal) and elevation_beam (vertical)
    params = dtree["radar_parameters"].ds
    np.testing.assert_allclose(params.radar_beam_width_h, 1.0)
    np.testing.assert_allclose(params.radar_beam_width_v, 1.0)
    # the scan how, stored as strings, as numbers; the antenna gain with the name
    # of the model, the two values of unknown meaning with their GAMIC names
    calib = dtree["radar_calibration"].ds
    assert set(calib.data_vars) == {
        "antenna_gain_h",
        "antenna_gain_v",
        "noise_power_h",
        "noise_power_v",
        "rx_loss_h",
        "rx_loss_v",
    }
    for var in calib.data_vars.values():
        assert var.dtype == np.float64
    np.testing.assert_allclose(calib.antenna_gain_h, 43.0)
    np.testing.assert_allclose(calib.noise_power_h, -3.8298)
    np.testing.assert_allclose(calib.rx_loss_h, 3.0)
    # they are no attributes of the root or the sweeps
    assert not {"antenna_gain_h", "radar_beam_width_h"} & set(dtree.attrs)
    assert not {"antenna_gain_h", "radar_beam_width_h"} & set(dtree["sweep_0"].attrs)


def test_gamic_dataset_attrs_are_numbers(gamic_file):
    # the calibration values are numbers with their model names (they were the
    # strings of the file with the GAMIC names), also for open_dataset
    with open_dataset(gamic_file, engine="gamic", group="sweep_0") as ds:
        assert ds.attrs["antenna_gain_h"] == 43.0
        assert "ant_gain_h" not in ds.attrs
        assert ds.attrs["radar_beam_width_h"] == 1.0


@pytest.fixture
def gamic_how_file(gamic_file, tmp_path):
    """Copy of ``gamic_file`` with other values in the ``how`` groups."""
    path = tmp_path / "gamic_how.h5"
    shutil.copy(gamic_file, path)
    with h5py.File(path, "a") as f:
        # no beam widths, a bytes value without unit, a value that is no number
        del f["how"].attrs["azimuth_beam"]
        del f["how"].attrs["elevation_beam"]
        how = f["scan0/how"].attrs
        how["ant_gain_h"] = np.bytes_("44.5")
        del how["ant_gain_h__unit"]
        how["rx_loss_h"] = "n/a"
        f["scan1/how"].attrs["ant_gain_h"] = "45"
    return path


def test_gamic_radar_metadata_variants(gamic_how_file, closing):
    # sweeps with other names: scan0 has no rx_loss_h (not a number), scan1 has
    # it; the antenna gain differs between the scans
    with pytest.warns(
        UserWarning, match=r"values differ between the sweeps \(antenna_gain_h"
    ):
        dtree = closing(
            xd.open_datatree(
                gamic_how_file, engine="gamic", optional_groups=True, sweep=[0, 1]
            )
        )
    assert "radar_beam_width_h" not in dtree["radar_parameters"].ds
    calib = dtree["radar_calibration"].ds
    # the first sweep that has the name, the first sweep for the differing one
    np.testing.assert_allclose(calib.antenna_gain_h, 44.5)
    np.testing.assert_allclose(calib.rx_loss_h, 3.0)
    np.testing.assert_allclose(calib.rx_loss_v, 3.0)


@pytest.mark.parametrize("sweep, expected", [(0, 44.5), ([1], 45.0)])
def test_gamic_radar_calibration_of_the_selected_sweep(
    gamic_how_file, sweep, expected, closing
):
    dtree = closing(
        xd.open_datatree(
            gamic_how_file, engine="gamic", optional_groups=True, sweep=sweep
        )
    )
    np.testing.assert_allclose(dtree["radar_calibration"].ds.antenna_gain_h, expected)


@pytest.mark.parametrize("kind", ["bytesio", "filelike"])
def test_gamic_radar_metadata_inputs(gamic_file, kind, closing):
    if kind == "bytesio":
        source = io.BytesIO(Path(gamic_file).read_bytes())
    else:
        source = closing(open(gamic_file, "rb"))
    dtree = closing(
        xd.open_datatree(source, engine="gamic", optional_groups=True, sweep=0)
    )
    np.testing.assert_allclose(dtree["radar_parameters"].ds.radar_beam_width_h, 1.0)
    np.testing.assert_allclose(dtree["radar_calibration"].ds.antenna_gain_h, 43.0)


def test_gamic_beam_width_no_number_is_skipped(gamic_file, tmp_path, closing):
    path = tmp_path / "gamic_beam.h5"
    shutil.copy(gamic_file, path)
    with h5py.File(path, "a") as f:
        f["how"].attrs["azimuth_beam"] = "n/a"
    dtree = closing(
        xd.open_datatree(path, engine="gamic", optional_groups=True, sweep=0)
    )
    params = dtree["radar_parameters"].ds
    assert "radar_beam_width_h" not in params
    np.testing.assert_allclose(params.radar_beam_width_v, 1.0)


def test_get_radar_calibration_of_sweeps_with_other_names():
    # values come from the first sweep which has the name, a difference warns
    names = model.radar_calibration_subgroup
    first = xr.Dataset(attrs={"antenna_gain_h": 43.0})
    second = xr.Dataset(attrs={"antenna_gain_h": 43.0, "radar_constant_h": 70.0})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        calib = _get_radar_calibration([first, second], names)
    assert calib.antenna_gain_h == 43.0 and calib.radar_constant_h == 70.0
    third = xr.Dataset(attrs={"antenna_gain_h": 44.0})
    with pytest.warns(
        UserWarning, match=r"differ between the sweeps \(antenna_gain_h\)"
    ):
        calib = _get_radar_calibration([first, third], names)
    assert calib.antenna_gain_h == 43.0
    assert not _get_radar_calibration([xr.Dataset()], names).data_vars


def test_same_value_of_numbers_nan_and_strings():
    assert _same_value(43.0, 43.0)
    assert _same_value(np.nan, np.nan)
    assert not _same_value(43.0, 44.0)
    # np.array_equal(equal_nan=True) raises for strings (e.g. model names)
    assert _same_value("GAMIC", "GAMIC")
    assert not _same_value("GAMIC", "other")
