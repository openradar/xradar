import datatree as dt
import numpy as np
from numpy.testing import assert_almost_equal

import xradar as xd


def test_georeference_dataarray():
    radar = xd.model.create_sweep_dataset()
    radar["sample_field"] = radar.azimuth + radar.range
    geo = radar.sample_field.xradar.georeference()
    assert_almost_equal(
        geo.x.values[:3, 0], np.array([0.43626028, 1.30864794, 2.18063697])
    )
    assert_almost_equal(
        geo.y.values[:3, 0], np.array([49.99047607, 49.97524848, 49.94479795])
    )


def test_georeference_dataset():
    radar = xd.model.create_sweep_dataset()
    geo = radar.xradar.georeference()
    assert_almost_equal(
        geo.x.values[:3, 0], np.array([0.43626028, 1.30864794, 2.18063697])
    )
    assert_almost_equal(
        geo.y.values[:3, 0], np.array([49.99047607, 49.97524848, 49.94479795])
    )


def test_georeference_datatree():
    radar = xd.model.create_sweep_dataset()
    tree = dt.DataTree.from_dict({"sweep_0": radar})
    geo = tree.xradar.georeference()["sweep_0"]
    assert_almost_equal(
        geo["x"].values[:3, 0], np.array([0.43626028, 1.30864794, 2.18063697])
    )
    assert_almost_equal(
        geo["y"].values[:3, 0], np.array([49.99047607, 49.97524848, 49.94479795])
    )
