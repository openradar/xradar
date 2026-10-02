#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for the ``reindex_coord`` backend kwarg (#407)."""

import numpy as np
import pytest
import xarray as xr

from xradar.io.backends.common import _get_reindex_coord

ANGLE = dict(start_angle=0, stop_angle=360, angle_res=1.0, direction=1)


def test_get_reindex_coord_none():
    assert _get_reindex_coord() is None
    assert _get_reindex_coord(None, False) is None


def test_get_reindex_coord_passthrough():
    coord = {"angle": ANGLE, "range": {"range_res": 100.0}}
    assert _get_reindex_coord(coord) is coord


def test_get_reindex_coord_deprecated_reindex_angle():
    with pytest.warns(FutureWarning, match="reindex_angle"):
        assert _get_reindex_coord(None, ANGLE) == {"angle": ANGLE}


def test_get_reindex_coord_both_given():
    coord = {"angle": ANGLE, "range": {"range_res": 100.0}}
    with pytest.warns(UserWarning, match="drop `reindex_angle`"):
        assert _get_reindex_coord(coord, {"angle_res": 2.0}) is coord


@pytest.mark.parametrize("value", [True, ANGLE["start_angle"], "angle"])
def test_get_reindex_coord_not_a_dict(value):
    with pytest.raises(TypeError, match="must be a dict"):
        _get_reindex_coord(value)


def test_get_reindex_coord_unknown_key():
    with pytest.raises(ValueError, match="Unknown key"):
        _get_reindex_coord({"azimuth": ANGLE})


def test_get_reindex_coord_inner_not_a_dict():
    with pytest.raises(TypeError, match=r"reindex_coord\['range'\]"):
        _get_reindex_coord({"range": 100.0})


def test_reindex_coord_angle_matches_deprecated_reindex_angle(gamic_file):
    kwargs = dict(group="sweep_0", engine="gamic")
    with xr.open_dataset(gamic_file, reindex_coord={"angle": ANGLE}, **kwargs) as ds:
        assert ds.sizes["azimuth"] == 360
        with pytest.warns(FutureWarning, match="reindex_angle"):
            with xr.open_dataset(gamic_file, reindex_angle=ANGLE, **kwargs) as old:
                xr.testing.assert_identical(ds, old)


def test_reindex_coord_range(gamic_file):
    kwargs = dict(group="sweep_0", engine="gamic")
    with xr.open_dataset(gamic_file, **kwargs) as ds0:
        rng = ds0.range
        res = rng.diff("range").median().item()
        start, stop = rng[0].item(), rng[-1].item() + 10 * res
    reindex_coord = {
        "angle": ANGLE,
        "range": dict(start_range=start, stop_range=stop, range_res=res),
    }
    with xr.open_dataset(gamic_file, reindex_coord=reindex_coord, **kwargs) as ds:
        assert ds.sizes == {"azimuth": 360, "range": rng.size + 10}
        np.testing.assert_allclose(ds.range[: rng.size], rng)
        # gates beyond the original range are filled
        assert np.isnan(ds.DBZH.isel(range=slice(rng.size, None))).all()


def test_reindex_coord_datatree(odim_file):
    from xradar.io import open_odim_datatree

    dtree = open_odim_datatree(odim_file, sweep=0, reindex_coord={"angle": ANGLE})
    assert dtree["sweep_0"].ds.sizes["azimuth"] == 360


def test_reindex_coord_hpl(hpl_file):
    reindex_coord = {"range": dict(range_res=60.0)}
    with xr.open_dataset(hpl_file, engine="hpl", group="sweep_0") as ds0:
        with xr.open_dataset(
            hpl_file, engine="hpl", group="sweep_0", reindex_coord=reindex_coord
        ) as ds:
            assert ds.range.attrs["spacing_is_constant"] == "true"
            assert ds.range.attrs["meters_between_gates"] == 60.0
            assert ds.range[0] == ds0.range[0]


def test_apply_reindex_coord_noop(gamic_file):
    from xradar.io.backends.common import _apply_reindex_coord

    with xr.open_dataset(gamic_file, group="sweep_0", engine="gamic") as ds:
        assert _apply_reindex_coord(ds, None) is ds
        assert _apply_reindex_coord(ds, {}) is ds


@pytest.mark.parametrize(
    "fixture, engine",
    [
        ("cfradial1_file", "cfradial1"),
        ("rainbow_file", "rainbow"),
        ("nexradlevel2_file", "nexradlevel2"),
        ("uf_file_1", "uf"),
    ],
)
def test_reindex_coord_open_dataset(request, fixture, engine):
    filename = request.getfixturevalue(fixture)
    with xr.open_dataset(
        filename, group="sweep_0", engine=engine, reindex_coord={"angle": ANGLE}
    ) as ds:
        assert ds.sizes["azimuth"] == 360
        np.testing.assert_allclose(ds.azimuth, np.arange(0.5, 360, 1.0))


def test_reindex_coord_nexrad_incomplete_pad_range(nexradlevel2_file):
    # incomplete sweeps get an auto-detected angle grid, range is reindexed too
    from xradar.io.backends.nexrad_level2 import open_sweeps_as_dict

    sweeps = open_sweeps_as_dict(nexradlevel2_file, sweeps=["sweep_0"])
    rng = sweeps["sweep_0"].range
    res = rng.diff("range").median().item()
    reindex_coord = {"range": dict(stop_range=rng[-1].item() + 4 * res)}
    padded = open_sweeps_as_dict(
        nexradlevel2_file,
        sweeps=["sweep_0"],
        incomplete_sweeps={0},
        reindex_coord=reindex_coord,
    )["sweep_0"]
    assert padded.sizes["range"] == rng.size + 4
    assert padded.range.attrs["spacing_is_constant"] == "true"
