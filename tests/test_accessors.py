#!/usr/bin/env python
# Copyright (c) 2022-2023, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

import datatree as dt
import numpy as np
from numpy.testing import assert_almost_equal

import xradar as xd


def test_georeference_dataarray():
    radar = xd.model.create_sweep_dataset()
    radar["sample_field"] = radar.azimuth + radar.range

    geo = radar.sample_field.xradar.georeference()
    assert_almost_equal(geo.x.values[:3, 0], np.array([0.436241, 1.3085901, 2.1805407]))
    assert_almost_equal(
        geo.y.values[:3, 0], np.array([49.9882679, 49.973041, 49.9425919])
    )
    assert_almost_equal(
        geo.z.values[:3, 0], np.array([375.8727675, 375.8727675, 375.8727675])
    )


def test_georeference_dataset():
    radar = xd.model.create_sweep_dataset()
    geo = radar.xradar.georeference()
    assert_almost_equal(geo.x.values[:3, 0], np.array([0.436241, 1.3085901, 2.1805407]))
    assert_almost_equal(
        geo.y.values[:3, 0], np.array([49.9882679, 49.973041, 49.9425919])
    )
    assert_almost_equal(
        geo.z.values[:3, 0], np.array([375.8727675, 375.8727675, 375.8727675])
    )


def test_georeference_datatree():
    radar = xd.model.create_sweep_dataset()
    tree = dt.DataTree.from_dict({"sweep_0": radar})
    geo = tree.xradar.georeference()["sweep_0"]
    assert_almost_equal(
        geo["x"].values[:3, 0], np.array([0.436241, 1.3085901, 2.1805407])
    )
    assert_almost_equal(
        geo["y"].values[:3, 0], np.array([49.9882679, 49.973041, 49.9425919])
    )
    assert_almost_equal(
        geo["z"].values[:3, 0], np.array([375.8727675, 375.8727675, 375.8727675])
    )
