"""
Tests for the MRR2 backend for xradar
"""

import gzip

import numpy as np
import xarray as xr
from open_radar_data import DATASETS

from xradar.io.backends import metek

test_arr_ave = np.array(
    [
        25.4,
        24.87,
        24.63,
        25.12,
        25.39,
        26.09,
        27.21,
        28.34,
        29.41,
        31.21,
        32.29,
        28.85,
        21.96,
        19.27,
        20.19,
        21.32,
        21.49,
        20.58,
        19.43,
        18.07,
        16.79,
        15.9,
        14.59,
        14.35,
        13.41,
        11.71,
        10.63,
        10.48,
        7.84,
        4.25,
        4.23,
    ]
)

test_arr = np.array(
    [
        24.46,
        25.31,
        26.33,
        26.31,
        26.85,
        27.93,
        29.12,
        30.17,
        30.99,
        32.58,
        33.13,
        28.84,
        22.16,
        19.81,
        21.26,
        21.33,
        20.33,
        18.93,
        17.92,
        18.04,
        16.86,
        14.46,
        13.17,
        13.13,
        11.75,
        10.53,
        9.3,
        5.92,
        -4.77,
        np.nan,
        6.74,
    ]
)

test_raw = np.array(
    [
        1.090e03,
        6.330e02,
        1.250e02,
        1.000e01,
        2.000e00,
        2.000e00,
        2.000e00,
        2.000e00,
        3.000e00,
        3.000e00,
        3.000e00,
        4.000e00,
        6.000e00,
        8.000e00,
        1.100e01,
        1.600e01,
        2.700e01,
        6.200e01,
        1.370e02,
        2.130e02,
        2.560e02,
        3.550e02,
        5.880e02,
        1.087e03,
        1.554e03,
        1.767e03,
        1.910e03,
        1.977e03,
        2.002e03,
        2.039e03,
        1.926e03,
        1.837e03,
        1.893e03,
        1.837e03,
        1.926e03,
        2.039e03,
        2.002e03,
        1.977e03,
        1.910e03,
        1.767e03,
        1.554e03,
        1.087e03,
        5.880e02,
        3.550e02,
        2.560e02,
        2.130e02,
        1.370e02,
        6.200e01,
        2.700e01,
        1.600e01,
        1.100e01,
        8.000e00,
        6.000e00,
        4.000e00,
        3.000e00,
        3.000e00,
        3.000e00,
        2.000e00,
        2.000e00,
        2.000e00,
        2.000e00,
        1.000e01,
        1.250e02,
        6.330e02,
    ]
)


def test_open_average():
    mrr_test_file = DATASETS.fetch("0308.ave.gz")
    with gzip.open(mrr_test_file, "rt") as test_file:
        ds = xr.open_dataset(test_file, engine="metek")
        assert "corrected_reflectivity" in ds.variables.keys()
        assert "velocity" in ds.variables.keys()
        rainfall = ds["rainfall_rate"].isel(range=0).cumsum() / 60.0
        np.testing.assert_allclose(rainfall.values[-1], 0.938)
        np.testing.assert_allclose(ds["reflectivity"].values[0], test_arr_ave)


def test_open_average_datatree():
    mrr_test_file = DATASETS.fetch("0308.ave.gz")
    with gzip.open(mrr_test_file, "rt") as test_file:
        ds = metek.open_metek_datatree(test_file)
        assert "corrected_reflectivity" in ds["sweep_0"].variables.keys()
        assert "velocity" in ds["sweep_0"].variables.keys()
        rainfall = ds["sweep_0"]["rainfall_rate"].isel(range=0).cumsum() / 60.0
        np.testing.assert_allclose(rainfall.values[-1], 0.938)
        np.testing.assert_allclose(
            ds["sweep_0"]["reflectivity"].values[0], test_arr_ave
        )


def test_open_processed():
    mrr_test_file = DATASETS.fetch("0308.pro.gz")
    with gzip.open(mrr_test_file, "rt") as test_file:
        ds = xr.open_dataset(test_file, engine="metek")
        assert "corrected_reflectivity" in ds.variables.keys()
        assert "velocity" in ds.variables.keys()
        rainfall = ds["rainfall_rate"].isel(range=0).cumsum() / 360.0
        np.testing.assert_allclose(rainfall.values[-1], 0.93)
        np.testing.assert_allclose(ds["reflectivity"].values[0], test_arr)


def test_open_processed_datatree():
    mrr_test_file = DATASETS.fetch("0308.pro.gz")
    with gzip.open(mrr_test_file, "rt") as test_file:
        ds = metek.open_metek_datatree(test_file)
        assert "corrected_reflectivity" in ds["sweep_0"].variables.keys()
        assert "velocity" in ds["sweep_0"].variables.keys()
        rainfall = ds["sweep_0"]["rainfall_rate"].isel(range=0).cumsum() / 360.0
        np.testing.assert_allclose(rainfall.values[-1], 0.93)
        np.testing.assert_allclose(ds["sweep_0"]["reflectivity"].values[0], test_arr)


def test_open_raw():
    mrr_test_file = DATASETS.fetch("0308.raw.gz")
    with gzip.open(mrr_test_file, "rt") as test_file:
        ds = xr.open_dataset(test_file, engine="metek")
        assert "raw_spectra_counts" in ds.variables.keys()
        np.testing.assert_allclose(ds["raw_spectra_counts"].values[0], test_raw)


def test_open_raw_datatree():
    mrr_test_file = DATASETS.fetch("0308.raw.gz")
    with gzip.open(mrr_test_file, "rt") as test_file:
        ds = metek.open_metek_datatree(test_file)
        assert "raw_spectra_counts" in ds["sweep_0"].variables.keys()
        np.testing.assert_allclose(
            ds["sweep_0"]["raw_spectra_counts"].values[0], test_raw
        )
