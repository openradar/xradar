#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Backends accept path-like and rewound file-like inputs."""

import io
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import xradar as xd


def test_uf_datatree_from_file_like(uf_file_1):
    # sweep discovery reads the object first; the reader rewinds it
    data = Path(uf_file_1).read_bytes()
    with (
        xd.open_datatree(uf_file_1, engine="uf", sweep=[0, 1]) as expected,
        xd.open_datatree(io.BytesIO(data), engine="uf", sweep=[0, 1]) as actual,
    ):
        xr.testing.assert_equal(actual, expected)


@pytest.mark.parametrize(
    "engine,fixture_name",
    [
        ("datamet", "datamet_file"),
        ("furuno", "furuno_scn_file"),
        ("iris", "iris0_file"),
        ("metek", "metek_ave_gz_file"),
    ],
)
def test_open_dataset_accepts_pathlike(engine, fixture_name, request):
    filename = request.getfixturevalue(fixture_name)
    kwargs = dict(engine=engine, group="sweep_0")
    with (
        xr.open_dataset(str(filename), **kwargs) as expected,
        xr.open_dataset(Path(filename), **kwargs) as actual,
    ):
        xr.testing.assert_equal(actual, expected)


def test_iris_datatree_accepts_pathlike(iris0_file):
    with (
        xd.open_datatree(str(iris0_file), engine="iris", sweep=[0]) as expected,
        xd.open_datatree(Path(iris0_file), engine="iris", sweep=[0]) as actual,
    ):
        xr.testing.assert_equal(actual, expected)


def test_cfradial1_fix_second_angle(cfradial1_file):
    kwargs = dict(engine="cfradial1", group="sweep_0", first_dim="auto")
    with (
        xr.open_dataset(cfradial1_file, **kwargs) as raw,
        xr.open_dataset(cfradial1_file, fix_second_angle=True, **kwargs) as fixed,
    ):
        # the secondary angle (elevation on a PPI) collapses onto its median
        assert np.unique(fixed.elevation.values).size == 1
        median = float(raw.elevation.median())
        assert fixed.elevation.values[0] == pytest.approx(median)
        expected = raw.DBZ.assign_coords(elevation=fixed.elevation)
        xr.testing.assert_equal(fixed.DBZ, expected)


def test_cfradial1_fix_second_angle_root_group(cfradial1_file):
    # the root group holds no sweep, the fix must not apply there
    with (
        xr.open_dataset(cfradial1_file, engine="cfradial1") as expected,
        xr.open_dataset(
            cfradial1_file, engine="cfradial1", fix_second_angle=True
        ) as actual,
    ):
        xr.testing.assert_equal(actual, expected)


def test_cfradial1_fix_second_angle_datatree(cfradial1_file):
    # open_datatree applies the fix like open_dataset does
    kwargs = dict(engine="cfradial1", sweep=0, first_dim="auto")
    with (
        xd.open_datatree(cfradial1_file, **kwargs) as raw,
        xd.open_datatree(cfradial1_file, fix_second_angle=True, **kwargs) as fixed,
    ):
        assert np.unique(fixed["sweep_0"].elevation.values).size == 1
        assert np.unique(raw["sweep_0"].elevation.values).size > 1
