import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal

import xradar
from xradar.georeference import antenna_to_cartesian, get_x_y_z


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
    assert_almost_equal(x[np.where(azimuths == 180.0)], 0)

    # Make sure that at 270 degrees, y is (close to) 0
    assert_almost_equal(y[np.where(azimuths == 270.0)], 0)


def test_get_x_y_z():
    # Create an empty xradar dataset
    ds = xradar.model.Dataset()

    # Add similar data to before
    ds["time"] = pd.date_range(start="2018-11-10", end="2018-11-10T00:09:00", freq="T")
    ds["range"] = np.arange(0, 1000, 100)
    ds["azimuth"] = ("time", np.arange(0, 300, 30))
    ds["elevation"] = ("time", np.arange(0, 50, 5))

    ds = get_x_y_z(ds)

    # Check to see if the origin contains all 0s
    origin = ds.isel(range=0, time=0)
    assert_almost_equal(origin.x, 0)
    assert_almost_equal(origin.y, 0)
    assert_almost_equal(origin.z, 0)

    # Make sure that at 180 degrees, x is (close to) 0
    np.testing.assert_almost_equal(
        ds.isel(range=0, time=np.where(ds.azimuth == 180)[0]).x, 0
    )

    # Make sure that at 270 degrees, y is (close to) 0
    np.testing.assert_almost_equal(
        ds.isel(range=0, time=np.where(ds.azimuth == 270)[0]).y, 0
    )
