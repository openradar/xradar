#!/usr/bin/env python
# Copyright (c) 2022-2023, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

import xradar
from xradar.georeference import antenna_to_cartesian, get_x_y_z


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
    assert ds.spatial_ref == 0
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
        if type(value) == float:
            assert ds.spatial_ref.attrs[key] == pytest.approx(value)
        else:
            assert ds.spatial_ref.attrs[key] == value
