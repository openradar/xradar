import warnings

import numpy as np
from numpy.testing import assert_almost_equal

from xradar.georeference import antenna_to_cartesian, geographic_to_cartesian_aeqd


def test_antenna_to_cartesian():
    ranges = np.arange(0, 1000, 100)
    azimuths = np.arange(0, 300, 30)
    elevations = np.arange(0, 50, 5)

    # Apply georeferencing to this sample data
    x, y, z = antenna_to_cartesian(ranges, azimuths, elevations)

    # Check to see if the origin contains all 0s
    assert_almost_equal(x[0], 0)
    assert_almost_equal(y[0], 0)
    assert_almost_equal(z[0], 0)

    # Make sure that at 180 degrees, x is (close to) 0
    assert_almost_equal(x[np.where(azimuths == 180.)], 0)

    # Make sure that at 270 degrees, y is (close to) 0
    assert_almost_equal(y[np.where(azimuths == 270.)], 0)


def test_geographic_to_cartesian_aeqd():
    # Example taken from:
    # Snyder, J.P. Map Projections A Working Manual, 1987, page 338.
    R = 3.0
    lat_0 = 40.0        # 40 degrees North latitude
    lon_0 = -100.       # 100 degrees West longitude
    lat = -20.0         # 20 degrees S latitude
    lon = 100.0         # 100.0 E longitude
    x = -5.8311398
    y = 5.5444634

    with warnings.catch_warnings():  # invalid divide is handled by code
        warnings.simplefilter('ignore', category=RuntimeWarning)
        x, y = geographic_to_cartesian_aeqd(
            lon, lat, lon_0, lat_0, R)
    assert_almost_equal(x, -5.8311398, 7)
    assert_almost_equal(y, 5.5444634, 7)

    # edge case, distance from projection center is zero
    lat = 40.0
    lon = -100.
    with warnings.catch_warnings():  # invalid divide is handled by code
        # ignore division runtime warning
        warnings.simplefilter('ignore', category=RuntimeWarning)
        x, y = geographic_to_cartesian_aeqd(
            lon, lat, lon_0, lat_0, R)
    assert_almost_equal(x, 0.0, 5)
    assert_almost_equal(y, 0.0, 5)

    # edge case, sin(c) is zero
    with warnings.catch_warnings():  # invalid divide is handled by code
        warnings.simplefilter('ignore', category=RuntimeWarning)
        x, y = geographic_to_cartesian_aeqd(
            10.0, 90.0, 20.0, 90.0, 3.0)

    assert_almost_equal(x, 0.0, 5)
    assert_almost_equal(y, 0.0, 5)
