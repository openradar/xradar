#!/usr/bin/env python
# Copyright (c) 2022-2023, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

import xradar
from xradar.georeference import (
    antenna_to_cartesian,
    cartesian_to_geographic_aeqd,
    get_lat_lon_alt,
    get_x_y_z,
)


def test_antenna_to_cartesian():
    ranges = np.arange(0, 1000, 100)
    azimuths = np.arange(0, 300, 30)
    elevations = np.arange(0, 50, 5)

    # Apply georeferencing to this sample data
    x, y, z = antenna_to_cartesian(ranges, azimuths, elevations, site_altitude=375.0)

    # Check to see if the origin contains all 0s
    assert_almost_equal(x[0], 0)
    assert_almost_equal(y[0], 0)
    assert_almost_equal(z[0], 375.0)

    # Make sure that at 180 degrees, x is (close to) 0
    assert_almost_equal(x[np.where(azimuths == 180.0)], 0)

    # Make sure that at 270 degrees, y is (close to) 0
    assert_almost_equal(y[np.where(azimuths == 270.0)], 0)


def test_cartesian_to_geographic_aeqd():
    # Define test values
    x = np.array([0, 1000, -1000])
    y = np.array([0, 500, -500])
    lon_0 = -97.59
    lat_0 = 36.49
    earth_radius = 6371000  # Earth's radius in meters

    # Convert Cartesian to geographic
    lon, lat = cartesian_to_geographic_aeqd(x, y, lon_0, lat_0, earth_radius)

    # Check that the origin remains unchanged
    assert_almost_equal(lon[0], lon_0)
    assert_almost_equal(lat[0], lat_0)

    # Check that the coordinates are within a reasonable range
    assert np.all(np.abs(lon) <= 180)
    assert np.all(np.abs(lat) <= 90)


def test_get_x_y_z():
    # Create default xradar dataset
    ds = xradar.model.create_sweep_dataset()

    # apply georeferencing
    ds = get_x_y_z(ds)

    # Check to see if the mean of the first range bins are 0
    # we have ranges at center of bin
    origin = ds.isel(range=0).reset_coords().mean("time")
    assert_almost_equal(origin.x, 0)
    assert_almost_equal(origin.y, 0)
    # center of first bin has already some added elevation
    assert_almost_equal(origin.z, ds.altitude + 0.87276752)

    # Make sure that at 180 degrees, x is (close to) 0
    # select the two beams around 180deg and calculate mean
    sel = ds.where((ds.azimuth >= 179) & (ds.azimuth <= 181), drop=True)
    sel = sel.reset_coords().isel(range=0).mean("time")
    np.testing.assert_almost_equal(sel.x, 0)
    np.testing.assert_approx_equal(sel.y, -50, significant=3)

    # Make sure that at 270 degrees, y is (close to) 0
    # select the two beams around 270deg and calculate mean
    sel = ds.where((ds.azimuth >= 269) & (ds.azimuth <= 271), drop=True)
    sel = sel.reset_coords().isel(range=0).mean("time")
    np.testing.assert_almost_equal(sel.y, 0)
    np.testing.assert_approx_equal(sel.x, -50, significant=3)

    # Make sure spatial_ref has been added with the correct values
    assert ds.crs_wkt == 0
    crs = {
        "semi_major_axis": 6378137.0,
        "semi_minor_axis": 6356752.314245179,
        "inverse_flattening": 298.257223563,
        "reference_ellipsoid_name": "WGS 84",
        "longitude_of_prime_meridian": 0.0,
        "prime_meridian_name": "Greenwich",
        "geographic_crs_name": "unknown",
        "horizontal_datum_name": "World Geodetic System 1984",
        "projected_crs_name": "unknown",
        "grid_mapping_name": "azimuthal_equidistant",
        "latitude_of_projection_origin": 46.172541,
        "longitude_of_projection_origin": 8.7877271,
        "false_easting": 0.0,
        "false_northing": 0.0,
    }
    for key, value in crs.items():
        if isinstance(value, float):
            assert ds.crs_wkt.attrs[key] == pytest.approx(value)
        else:
            assert ds.crs_wkt.attrs[key] == value


def test_get_lat_lon_alt():
    # Create default xradar dataset
    ds = xradar.model.create_sweep_dataset()

    # Apply lat, lon, and alt georeferencing
    ds = get_lat_lon_alt(ds.swap_dims({"time": "azimuth"}))

    # Check that latitude, longitude, and altitude have been added
    assert "lon" in ds.coords
    assert "lat" in ds.coords
    assert "alt" in ds.coords

    # # Check that the first range bin latitude and longitude are close to the radar location
    origin = ds.isel(range=0).reset_coords().mean("azimuth")
    assert_almost_equal(ds.latitude, origin.lat)
    assert_almost_equal(
        ds.longitude,
        origin.lon,
    )
    assert_almost_equal(ds.altitude, origin.alt - 0.8727675)
