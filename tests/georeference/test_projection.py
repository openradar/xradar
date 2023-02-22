#!/usr/bin/env python
# Copyright (c) 2023, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

import numpy as np
import pyproj
import pytest

import xradar
from xradar.georeference import add_crs, get_crs, get_earth_radius


def test_get_earth_radius():
    proj_crs = pyproj.CRS(
        proj="aeqd",
        datum="WGS84",
        lon_0=8.7877271,
        lat_0=46.172541,
    )
    ea = get_earth_radius(proj_crs, [20, 22, 24])
    np.testing.assert_allclose(ea, [6375653.951276, 6375157.677218, 6374623.948138])


def test_get_crs():
    # Create default xradar dataset
    ds = xradar.model.create_sweep_dataset()

    proj_crs = get_crs(ds)
    proj_crs_cf = proj_crs.to_cf()

    assert isinstance(proj_crs, pyproj.CRS)

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
            assert proj_crs_cf[key] == pytest.approx(value)
        else:
            assert proj_crs_cf[key] == value


def test_write_crs():
    # Create default xradar dataset
    ds = xradar.model.create_sweep_dataset()

    ds = add_crs(ds)

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
