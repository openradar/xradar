#!/usr/bin/env python
# Copyright (c) 2023-2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for `io.backends.furuno` module.

Ported from wradlib.
"""

import datetime
import gzip
import os
import shutil
import struct
import tempfile

import numpy as np
import pytest
from xarray import DataTree, Variable
from xarray.backends import CachingFileManager
from xarray.core.indexing import OuterIndexer
from xarray.core.utils import Frozen

from xradar.io.backends import furuno, open_furuno_datatree
from xradar.io.backends.common import _get_fmt_string
from xradar.io.backends.furuno import (
    YMDS_TIME,
    FurunoBackendEntrypoint,
    FurunoFile,
    FurunoStore,
    decode_time,
)
from xradar.util import _get_data_file


def test_open_scn(furuno_scn_file, file_or_filelike):
    with _get_data_file(furuno_scn_file, file_or_filelike) as furunofile:
        data = FurunoFile(furunofile, loaddata=False, obsmode=1)
    assert isinstance(data, furuno.FurunoFile)
    assert isinstance(data.fh, (np.memmap, np.ndarray))
    with _get_data_file(furuno_scn_file, file_or_filelike) as furunofile:
        data = furuno.FurunoFile(furunofile, loaddata=True, obsmode=1)
    assert len(data.data) == 11
    assert data.filename == furunofile
    assert data.version == 3
    assert data.a1gate == 796
    assert data.angle_resolution == 0.26
    assert data.first_dimension == "azimuth"
    assert data.fixed_angle == 7.8
    assert data.site_coords == (15.44729, 47.07734000000001, 407.9)
    assert data.header["scan_start_time"] == datetime.datetime(2021, 7, 30, 16, 0)
    assert list(data.data.keys()) == [
        "RATE",
        "DBZH",
        "VRADH",
        "ZDR",
        "KDP",
        "PHIDP",
        "RHOHV",
        "WRADH",
        "QUAL",
        "azimuth",
        "elevation",
    ]


def test_open_scn_filelike(furuno_scn_file):
    with pytest.raises(
        ValueError, match="Furuno `observation mode` can't be extracted"
    ):
        with _get_data_file(furuno_scn_file, "filelike") as furunofile:
            data = furuno.FurunoFile(furunofile, loaddata=False)
            print(data.first_dimension)


def test_open_scnx(furuno_scnx_file, file_or_filelike):
    with _get_data_file(furuno_scnx_file, file_or_filelike) as furunofile:
        data = furuno.FurunoFile(furunofile, loaddata=False)
    assert isinstance(data, furuno.FurunoFile)
    assert isinstance(data.fh, (np.memmap, np.ndarray))
    with _get_data_file(furuno_scnx_file, file_or_filelike) as furunofile:
        data = furuno.FurunoFile(furunofile, loaddata=True)
    assert data.filename == furunofile
    assert data.version == 10
    assert data.a1gate == 292
    assert data.angle_resolution == 0.5
    assert data.first_dimension == "azimuth"
    assert data.fixed_angle == 0.5
    assert data.site_coords == (13.243970000000001, 53.55478, 38.0)
    assert data.header["scan_start_time"] == datetime.datetime(2022, 3, 24, 0, 0, 1)
    assert len(data.data) == 11
    assert list(data.data.keys()) == [
        "RATE",
        "DBZH",
        "VRADH",
        "ZDR",
        "KDP",
        "PHIDP",
        "RHOHV",
        "WRADH",
        "QUAL",
        "azimuth",
        "elevation",
    ]


def test_decode_time_valid():
    # Define valid data for YMDS_TIME
    data = {"year": 2023, "month": 8, "day": 15, "hour": 14, "minute": 30, "second": 45}
    # Create binary buffer using struct.pack
    fmt = _get_fmt_string(YMDS_TIME)
    buffer = struct.pack(
        fmt,
        data["year"],
        data["month"],
        data["day"],
        data["hour"],
        data["minute"],
        data["second"],
        b"\x00",
    )

    # Call decode_time with binary data
    result = decode_time(buffer, YMDS_TIME)
    expected = datetime.datetime(2023, 8, 15, 14, 30, 45)
    assert result == expected, "Expected valid datetime object"


def test_decode_time_invalid_date():
    # Define invalid data (month 13)
    data = {
        "year": 2023,
        "month": 13,
        "day": 15,
        "hour": 14,
        "minute": 30,
        "second": 45,
    }
    # Create binary buffer for invalid date
    fmt = _get_fmt_string(YMDS_TIME)
    buffer = struct.pack(
        fmt,
        data["year"],
        data["month"],
        data["day"],
        data["hour"],
        data["minute"],
        data["second"],
        b"\x00",
    )

    # Call decode_time with invalid data
    result = decode_time(buffer, YMDS_TIME)
    assert result is None, "Expected None for an invalid date"


def prepare_temp_file_with_extension(furuno_scn_file, extension):
    # Create a temporary file with the specified extension
    fnameo = os.path.join(
        tempfile.gettempdir(), f"{os.path.basename(furuno_scn_file)[:-7]}{extension}"
    )
    with gzip.open(furuno_scn_file) as fin:
        with open(fnameo, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    return fnameo


def test_furuno_file_with_binary_file(furuno_scn_file, file_or_filelike):
    fnameo = os.path.join(
        tempfile.gettempdir(), f"{os.path.basename(furuno_scn_file)[:-3]}"
    )
    with gzip.open(furuno_scn_file) as fin:
        with open(fnameo, "wb") as fout:
            shutil.copyfileobj(fin, fout)

    with furuno.FurunoFile(fnameo, loaddata=False) as data:
        # Check that _fp and _fh were set properly
        assert data._fp is not None, "_fp should be an open file object"
        assert isinstance(data._fh, np.memmap), "_fh should be a numpy memmap object"
        assert data.loaddata is False, "Expected loaddata to be False"
        assert data.rawdata is False, "Expected rawdata to be False"
        assert data.debug is False, "Expected debug to be False"
        assert data.first_dimension == "azimuth"
        assert data.filepos == 0
        data.filepos = 10
        assert data.filepos == 10
    assert data._fp.closed, "_fp should be closed after exiting the context"


def test_furuno_file_with_sppi_file(furuno_scn_file):
    fnameo = prepare_temp_file_with_extension(furuno_scn_file, ".sppi")
    with furuno.FurunoFile(fnameo, loaddata=False) as data:
        assert data.first_dimension == "azimuth", "Expected 'azimuth' for .sppi files"
    assert data._fp.closed, "_fp should be closed after exiting the context"


def test_furuno_file_with_rhi_file(furuno_scn_file):
    fnameo = prepare_temp_file_with_extension(furuno_scn_file, ".rhi")
    with furuno.FurunoFile(fnameo, loaddata=False) as data:
        assert (
            data.first_dimension == "elevation"
        ), "Expected 'elevation' for .rhi files"
    assert data._fp.closed, "_fp should be closed after exiting the context"


def test_furuno_file_unknown_obsmode(furuno_scn_file):
    # Prepare a file without a recognized extension to leave obs_mode unset
    fnameo = prepare_temp_file_with_extension(furuno_scn_file, ".txt")
    # Initialize FurunoFile without a recognized extension and no obsmode
    with pytest.raises(TypeError, match="Unknown Furuno Observation Mode: None"):
        with furuno.FurunoFile(fnameo, loaddata=False) as data:
            _ = data.first_dimension  # Access first_dimension to trigger the TypeError


def test_furuno_array_wrapper_initialization():
    # Initialize with a sample array
    data = np.arange(20).reshape(4, 5)
    wrapper = furuno.FurunoArrayWrapper(data)

    # Check that data, shape, and dtype are correctly set
    assert np.array_equal(wrapper.data, data), "Data should match the input array"
    assert wrapper.shape == (4, 5), "Shape should match the input data's shape"
    assert wrapper.dtype == np.dtype("uint16"), "Expected dtype to be uint16"


def test_furuno_array_wrapper_getitem():
    # Initialize with a sample array
    data = np.arange(20).reshape(4, 5)
    wrapper = furuno.FurunoArrayWrapper(data)

    # Use __getitem__ to access specific elements and slices
    assert (
        wrapper[OuterIndexer((1, 2))] == data[1, 2]
    ), "Single element access should match"
    assert np.array_equal(
        wrapper[OuterIndexer((1, slice(None)))], data[1, :]
    ), "Row slice should match"
    assert np.array_equal(
        wrapper[OuterIndexer((slice(None), 2))], data[:, 2]
    ), "Column slice should match"


def test_furuno_array_wrapper_raw_indexing_method():
    # Initialize with a sample array
    data = np.arange(20).reshape(4, 5)
    wrapper = furuno.FurunoArrayWrapper(data)

    # Access elements using _raw_indexing_method directly
    assert (
        wrapper._raw_indexing_method((1, 2)) == data[1, 2]
    ), "Direct element access should match"
    assert np.array_equal(
        wrapper._raw_indexing_method((1, slice(None))), data[1, :]
    ), "Direct row slice should match"
    assert np.array_equal(
        wrapper._raw_indexing_method((slice(None), 2)), data[:, 2]
    ), "Direct column slice should match"


def test_furuno_store_initialization(furuno_scn_file):
    # Initialize FurunoStore with a CachingFileManager
    manager = CachingFileManager(furuno.FurunoFile, furuno_scn_file)
    store = furuno.FurunoStore(manager)
    assert isinstance(store, furuno.FurunoStore), "Expected FurunoStore instance"
    assert store._manager == manager, "Expected manager to be set"
    assert store._group is None, "Expected group to be None by default"
    assert isinstance(store.root, furuno.FurunoFile)
    assert store.filename == furuno_scn_file, "Filename should match input file"
    assert isinstance(
        store.ds, furuno.FurunoFile
    ), "Expected ds property to return a FurunoFile instance"
    # Assuming 'DBZH' is a variable in the dataset
    var_data = np.random.rand(10, 10)  # Dummy data for testing
    dim = store.root.first_dimension
    variable = store.open_store_variable("DBZH", var_data)

    assert isinstance(variable, Variable), "Expected a Variable instance"
    assert variable.dims == (dim, "range"), "Expected dimensions to match"
    assert "scale_factor" in variable.attrs, "Expected scale_factor in attributes"


def test_furuno_store_open(furuno_scn_file):
    # Use the `open` method to initialize a FurunoStore instance
    store = FurunoStore.open(furuno_scn_file, mode="r", group="test_group")
    # Check that the returned object is an instance of FurunoStore
    assert isinstance(store, FurunoStore), "Expected a FurunoStore instance"
    # Verify that manager is an instance of CachingFileManager
    assert isinstance(
        store._manager, CachingFileManager
    ), "Expected a CachingFileManager instance"
    # Use acquire_context to confirm it manages FurunoFile instances
    with store._manager.acquire_context() as file_obj:
        assert isinstance(
            file_obj, FurunoFile
        ), "Expected CachingFileManager to manage FurunoFile instances"
    # Ensure the filename and group are set correctly
    assert store._group == "test_group", "Expected group to be 'test_group'"
    assert store.filename == furuno_scn_file, "Expected filename to match input file"


def test_furuno_store_open_store_coordinates(furuno_scn_file):
    manager = CachingFileManager(FurunoFile, furuno_scn_file)
    store = FurunoStore(manager)
    coords = store.open_store_coordinates()

    assert isinstance(coords, dict), "Expected coordinates to be a dictionary"
    assert "latitude" in coords, "Expected 'latitude' in coordinates"
    assert "longitude" in coords, "Expected 'longitude' in coordinates"
    assert "altitude" in coords, "Expected 'altitude' in coordinates"
    assert isinstance(coords["latitude"], Variable), "Expected Variable for 'latitude'"


def test_furuno_store_get_variables(furuno_scn_file):
    manager = CachingFileManager(FurunoFile, furuno_scn_file)
    store = FurunoStore(manager)
    variables = store.get_variables()

    assert isinstance(
        variables, Frozen
    ), "Expected get_variables to return a dictionary"
    assert "latitude" in variables, "Expected 'latitude' in variables"
    assert "longitude" in variables, "Expected 'longitude' in variables"


def test_furuno_store_get_attrs(furuno_scn_file):
    manager = CachingFileManager(FurunoFile, furuno_scn_file)
    store = FurunoStore(manager)
    attrs = store.get_attrs()

    assert isinstance(attrs, Frozen), "Expected attributes to be a dictionary"
    assert attrs["source"] == "Furuno", "Expected source attribute to be 'Furuno'"
    assert "version" in attrs, "Expected version attribute in attrs"


def test_furuno_store_get_calibration_parameters(furuno_scn_file):
    manager = CachingFileManager(FurunoFile, furuno_scn_file)
    store = FurunoStore(manager)
    calibration_params = store.get_calibration_parameters()
    assert isinstance(
        calibration_params, Frozen
    ), "Expected calibration parameters to be a dictionary"


def test_open_store_coordinates_version_else(furuno_scn_file):
    # Set up a mocked version of FurunoFile with header values that activate the else statement
    manager = CachingFileManager(FurunoFile, furuno_scn_file)
    store = FurunoStore(manager)

    # Mock attributes including format_version
    store.ds._header = {
        "format_version": 2,  # Ensure format_version is included
        "resolution_range_direction": 100.0,
        "number_range_direction_data": 100,
        "scan_start_time": datetime.datetime(2021, 7, 30, 16, 0, 0),
        "number_sweep_direction_data": 360,
        "antenna_rotation_speed": 10,
        "observation_mode": 1,
        "longitude": 1544729,
        "latitude": 4707734,
        "altitude": 4079,
    }

    # Mock ds.version to a value not in [3, 103]
    store.ds._version = (
        2  # Any value not 3 or 103 to trigger the else condition for range_step
    )

    # Call the function
    coords = store.open_store_coordinates()

    # Check if the range step has been applied as expected
    range_step = store.ds.header["resolution_range_direction"]
    expected_rng = np.arange(
        range_step / 2,
        range_step * store.ds.header["number_range_direction_data"] + range_step / 2,
        range_step,
        dtype="float32",
    )
    assert np.array_equal(
        coords["range"].values, expected_rng
    ), "Range values do not match expected values"


def test_open_store_coordinates_time_else(furuno_scn_file):
    # Set up a mocked version of FurunoFile with header values that trigger time-based else
    manager = CachingFileManager(FurunoFile, furuno_scn_file)
    store = FurunoStore(manager)

    # Mock attributes with missing scan_stop_time
    store.ds._header = {
        "format_version": 2,  # Ensure format_version is included
        "resolution_range_direction": 100.0,
        "number_range_direction_data": 100,
        "scan_start_time": datetime.datetime(2021, 7, 30, 16, 0, 0),
        "number_sweep_direction_data": 360,
        "antenna_rotation_speed": 10,
        "observation_mode": 1,
        "longitude": 1544729,
        "latitude": 4707734,
        "altitude": 4079,
    }

    # Mock ds.version to a value that uses resolution_range_direction directly
    store.ds._version = 3

    # Call the function
    coords = store.open_store_coordinates()

    # Calculate expected time step if no stop_time
    expected_raytime = datetime.timedelta(
        seconds=store.ds.angle_resolution
        / (store.ds.header["antenna_rotation_speed"] * 1e-1 * 6)
    )

    # Apply the initial offset
    initial_offset = coords["time"].values[0]
    expected_raytimes = np.array(
        [
            initial_offset + (x * expected_raytime).total_seconds()
            for x in range(store.ds.header["number_sweep_direction_data"])
        ]
    )

    # Check if the calculated ray times match the returned time coordinates with tolerance
    np.testing.assert_almost_equal(
        coords["time"].values, expected_raytimes, decimal=6
    ), "Time values do not match expected values"


def test_open_store_coordinates_time_else_with_stop_time(furuno_scn_file):
    # Set up a mocked version of FurunoFile with header values that trigger the else condition with stop_time
    manager = CachingFileManager(FurunoFile, furuno_scn_file)
    store = FurunoStore(manager)

    # Mock attributes including scan_stop_time to activate the else condition
    store.ds._header = {
        "format_version": 2,
        "resolution_range_direction": 100.0,
        "number_range_direction_data": 100,
        "scan_start_time": datetime.datetime(2021, 7, 30, 16, 0, 0),
        "scan_stop_time": datetime.datetime(2021, 7, 30, 16, 10, 0),  # 10 minutes later
        "number_sweep_direction_data": 360,
        "antenna_rotation_speed": 10,
        "observation_mode": 1,
        "longitude": 1544729,
        "latitude": 4707734,
        "altitude": 4079,
    }

    # Mock ds.version to use the else condition for range_step
    store.ds._version = 3

    # Call the function
    coords = store.open_store_coordinates()

    # Calculate expected ray times using the scan_stop_time for the else case
    time = store.ds.header["scan_start_time"]
    stop_time = store.ds.header["scan_stop_time"]
    num_rays = store.ds.header["number_sweep_direction_data"]
    raytime = (stop_time - time) / num_rays

    # Generate expected raytimes based on raytime intervals
    expected_raytimes = np.array(
        [((x + 0.5) * raytime).total_seconds() for x in range(num_rays)]
    )

    # Check if the calculated ray times match the returned time coordinates with tolerance
    np.testing.assert_almost_equal(
        coords["time"].values, expected_raytimes, decimal=5
    ), "Time values do not match expected values"


def test_open_dataset(furuno_scn_file):
    # Initialize FurunoBackendEntrypoint
    backend_entrypoint = furuno.FurunoBackendEntrypoint()

    # Define parameters for opening the dataset
    mask_and_scale = True
    decode_times = True
    concat_characters = True
    decode_coords = True
    drop_variables = None
    use_cftime = None
    decode_timedelta = None
    group = None
    obsmode = 1

    # Call open_dataset directly
    ds = backend_entrypoint.open_dataset(
        furuno_scn_file,
        mask_and_scale=mask_and_scale,
        decode_times=decode_times,
        concat_characters=concat_characters,
        decode_coords=decode_coords,
        drop_variables=drop_variables,
        use_cftime=use_cftime,
        decode_timedelta=decode_timedelta,
        group=group,
        obsmode=obsmode,
    )

    # Assertions to verify dataset content
    assert "azimuth" in ds.coords, "Azimuth coordinate is missing."
    assert "elevation" in ds.coords, "Elevation coordinate is missing."
    assert "time" in ds.coords, "Time coordinate is missing."

    # Check encoding
    assert ds.encoding["engine"] == "furuno", "Engine encoding mismatch."

    # Check dimension order and size
    assert list(ds.dims) == [
        "azimuth",
        "range",
    ], "Dimensions do not match expected values."
    assert ds["DBZH"].shape == (1376, 602), "DBZH shape does not match expected values."

    # Check attributes for calibration parameters and source
    assert "source" in ds.attrs, "Source attribute missing."
    assert "version" in ds.attrs, "Version attribute missing."
    assert ds.attrs["source"] == "Furuno", "Source attribute mismatch."

    # Check if reindexing and coordinate assignments are handled
    assert "latitude" in ds.coords, "Latitude coordinate missing."
    assert "longitude" in ds.coords, "Longitude coordinate missing."
    assert "altitude" in ds.coords, "Altitude coordinate missing."


def test_open_dataset_with_reindex_and_dimensions(furuno_scn_file):
    # Initialize FurunoBackendEntrypoint
    backend_entrypoint = FurunoBackendEntrypoint()

    # Define parameters to trigger all conditions
    mask_and_scale = True
    decode_times = True
    concat_characters = True
    decode_coords = True
    drop_variables = None
    use_cftime = None
    decode_timedelta = None
    group = None
    obsmode = 1
    reindex_angle = {
        "start_angle": 0.0,
        "stop_angle": 360.0,
        "angle_res": 1.0,  # Set a valid angle resolution
        "direction": 1,
    }  # Not False to trigger the reindexing block
    first_dim = "auto"  # Set to "auto" initially to test first_dim handling

    # Call open_dataset directly
    ds = backend_entrypoint.open_dataset(
        furuno_scn_file,
        mask_and_scale=mask_and_scale,
        decode_times=decode_times,
        concat_characters=concat_characters,
        decode_coords=decode_coords,
        drop_variables=drop_variables,
        use_cftime=use_cftime,
        decode_timedelta=decode_timedelta,
        group=group,
        reindex_angle=reindex_angle,  # Pass reindex_angle to trigger reindexing
        first_dim=first_dim,  # Test first_dim="auto"
        obsmode=obsmode,
    )

    # Assertions to ensure reindexing and dimension handling was applied
    assert "azimuth" in ds.coords, "Azimuth coordinate is missing."
    assert "elevation" in ds.coords, "Elevation coordinate is missing."
    assert "time" in ds.coords, "Time coordinate is missing."

    # Check if azimuth coordinate matches the expected start, stop, and resolution from reindexing
    expected_azimuth = np.linspace(
        reindex_angle["start_angle"] + reindex_angle["angle_res"] / 2,
        reindex_angle["stop_angle"] - reindex_angle["angle_res"] / 2,
        len(ds["azimuth"]),
    )
    np.testing.assert_almost_equal(
        ds["azimuth"].values,
        expected_azimuth,
        decimal=5,
        err_msg="Azimuth coordinate does not match expected reindexed values",
    )

    # Test first_dim behavior
    assert "azimuth" in ds.DBZH.dims

    # Repeat with first_dim set to "time" to check other handling
    ds_time_dim = backend_entrypoint.open_dataset(
        furuno_scn_file,
        mask_and_scale=mask_and_scale,
        decode_times=decode_times,
        concat_characters=concat_characters,
        decode_coords=decode_coords,
        drop_variables=drop_variables,
        use_cftime=use_cftime,
        decode_timedelta=decode_timedelta,
        group=group,
        reindex_angle=reindex_angle,
        first_dim="time",  # Set to "time" to test alternative branch
        obsmode=obsmode,
    )

    # Ensure that time is now the primary dimension, if expected
    assert (
        "time" in ds_time_dim.DBZH.dims
    ), "Time dimension was not set as the primary dimension when expected."


def test_open_furuno_datatree(furuno_scn_file):
    # Define parameters to trigger all conditions
    kwargs = {
        "first_dim": "auto",
        "reindex_angle": {
            "start_angle": 0.0,
            "stop_angle": 360.0,
            "angle_res": 1.0,
            "direction": 1,
        },
        "fix_second_angle": True,
        "site_coords": True,
    }

    # Call the function with the actual file
    dtree = open_furuno_datatree(furuno_scn_file, **kwargs)

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
    assert len(sweep_groups) == 1, "Expected at least one sweep group in the DataTree"

    # Verify a sample variable in one of the sweep groups (adjust as needed based on expected variables)
    sample_sweep = sweep_groups[0]
    assert (
        "DBZH" in dtree[sample_sweep].data_vars
    ), f"DBZH should be a data variable in {sample_sweep}"
    assert (
        "VRADH" in dtree[sample_sweep].data_vars
    ), f"VRADH should be a data variable in {sample_sweep}"

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
    assert len(dtree[sample_sweep].variables) == 21
    assert dtree[sample_sweep]["DBZH"].shape == (360, 602)
    assert len(dtree.attrs) == 9
    assert dtree.attrs["version"] == 3
    assert dtree.attrs["source"] == "Furuno"
