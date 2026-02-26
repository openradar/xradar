#!/usr/bin/env python
# Copyright (c) 2024-2025, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for xarray-native open_datatree with engine= parameter.

Tests the unified ``xd.open_datatree()`` and ``xr.open_datatree()`` APIs,
``open_groups_as_dict()`` direct calls, backward compatibility with
deprecated standalone functions, and ``supports_groups`` attribute.
"""

import warnings

import pytest
import xarray as xr
from xarray import DataTree

import xradar as xd
from xradar.io.backends.cfradial1 import (
    CfRadial1BackendEntrypoint,
    open_cfradial1_datatree,
)
from xradar.io.backends.nexrad_level2 import (
    NexradLevel2BackendEntrypoint,
    open_nexradlevel2_datatree,
)
from xradar.io.backends.odim import OdimBackendEntrypoint, open_odim_datatree

# -- Fixtures ----------------------------------------------------------------


@pytest.fixture(
    params=[
        pytest.param("odim", id="odim"),
        pytest.param("nexradlevel2", id="nexradlevel2"),
    ]
)
def engine_and_file(request, odim_file, nexradlevel2_file):
    """Parametrize over engines that do not require netCDF4."""
    mapping = {
        "odim": odim_file,
        "nexradlevel2": nexradlevel2_file,
    }
    return request.param, mapping[request.param]


@pytest.fixture
def cfradial1_engine_file(cfradial1_file):
    return "cfradial1", cfradial1_file


# -- CfRadial2 structure keys -----------------------------------------------

REQUIRED_GROUPS = {
    "/",
    "/radar_parameters",
    "/georeferencing_correction",
    "/radar_calibration",
}


# -- Helper ------------------------------------------------------------------


def _assert_cfradial2_structure(dtree, optional_groups=False):
    """Verify that a DataTree has CfRadial2 group structure."""
    assert isinstance(dtree, DataTree)
    children = set(dtree.children.keys())
    # Metadata groups only present when optional_groups=True
    if optional_groups:
        for grp in [
            "radar_parameters",
            "georeferencing_correction",
            "radar_calibration",
        ]:
            assert grp in children, f"Missing group: {grp}"
    # Must have at least one sweep
    sweep_groups = [k for k in children if k.startswith("sweep_")]
    assert len(sweep_groups) > 0, "No sweep groups found"
    # Root must have key variables
    root_vars = set(dtree.ds.data_vars)
    assert "time_coverage_start" in root_vars
    assert "time_coverage_end" in root_vars


# -- xd.open_datatree integration tests (ODIM, NEXRAD) ----------------------


class TestXdOpenDatatree:
    """Test xd.open_datatree() for ODIM and NEXRAD."""

    def test_basic_open(self, engine_and_file):
        engine, filepath = engine_and_file
        dtree = xd.open_datatree(filepath, engine=engine)
        _assert_cfradial2_structure(dtree)

    def test_sweep_selection_list(self, engine_and_file):
        engine, filepath = engine_and_file
        dtree = xd.open_datatree(filepath, engine=engine, sweep=[0, 1])
        sweep_groups = [k for k in dtree.children if k.startswith("sweep_")]
        assert len(sweep_groups) == 2

    def test_sweep_selection_int(self, engine_and_file):
        engine, filepath = engine_and_file
        dtree = xd.open_datatree(filepath, engine=engine, sweep=0)
        sweep_groups = [k for k in dtree.children if k.startswith("sweep_")]
        assert len(sweep_groups) == 1

    def test_kwargs_flow_through(self, engine_and_file):
        engine, filepath = engine_and_file
        dtree = xd.open_datatree(
            filepath,
            engine=engine,
            first_dim="auto",
            site_coords=True,
            sweep=[0],
        )
        sweep_ds = dtree["sweep_0"].ds
        assert "latitude" in sweep_ds.coords
        assert "longitude" in sweep_ds.coords
        assert "altitude" in sweep_ds.coords

    def test_unknown_engine_raises(self, odim_file):
        with pytest.raises(ValueError, match="Unknown engine"):
            xd.open_datatree(odim_file, engine="nonexistent_engine")


# -- xd.open_datatree for CfRadial1 -----------------------------------------


class TestXdOpenDatatreeCfRadial1:
    """Test xd.open_datatree() for CfRadial1 (requires h5netcdf in this env)."""

    def test_basic_open(self, cfradial1_engine_file):
        _, filepath = cfradial1_engine_file
        backend = CfRadial1BackendEntrypoint()
        dtree = backend.open_datatree(
            filepath, engine="h5netcdf", decode_timedelta=False
        )
        _assert_cfradial2_structure(dtree)

    def test_sweep_selection(self, cfradial1_engine_file):
        _, filepath = cfradial1_engine_file
        backend = CfRadial1BackendEntrypoint()
        dtree = backend.open_datatree(
            filepath,
            engine="h5netcdf",
            decode_timedelta=False,
            sweep=[0, 1],
        )
        sweep_groups = [k for k in dtree.children if k.startswith("sweep_")]
        assert len(sweep_groups) == 2


# -- xr.open_datatree tests -------------------------------------------------


class TestXrOpenDatatree:
    """Test xr.open_datatree() with xradar engines."""

    def test_xr_open_datatree_odim(self, odim_file):
        dtree = xr.open_datatree(odim_file, engine="odim")
        _assert_cfradial2_structure(dtree)

    def test_xr_open_datatree_nexrad(self, nexradlevel2_file):
        dtree = xr.open_datatree(nexradlevel2_file, engine="nexradlevel2")
        _assert_cfradial2_structure(dtree)


# -- open_groups_as_dict direct tests ----------------------------------------


class TestOpenGroupsAsDict:
    """Test open_groups_as_dict() returns correct dict structure."""

    def test_odim_groups_dict(self, odim_file):
        backend = OdimBackendEntrypoint()
        groups = backend.open_groups_as_dict(
            odim_file, sweep=[0, 1], optional_groups=True
        )
        assert isinstance(groups, dict)
        assert "/" in groups
        assert "/radar_parameters" in groups
        assert "/georeferencing_correction" in groups
        assert "/radar_calibration" in groups
        assert "/sweep_0" in groups
        assert "/sweep_1" in groups
        for key, ds in groups.items():
            assert isinstance(ds, xr.Dataset), f"{key} is not a Dataset"

    def test_nexrad_groups_dict(self, nexradlevel2_file):
        backend = NexradLevel2BackendEntrypoint()
        groups = backend.open_groups_as_dict(nexradlevel2_file, sweep=[0, 1])
        assert isinstance(groups, dict)
        assert "/" in groups
        assert "/sweep_0" in groups
        assert "/sweep_1" in groups

    def test_cfradial1_groups_dict(self, cfradial1_file):
        backend = CfRadial1BackendEntrypoint()
        groups = backend.open_groups_as_dict(
            cfradial1_file,
            engine="h5netcdf",
            decode_timedelta=False,
            sweep=[0, 1],
        )
        assert isinstance(groups, dict)
        assert "/" in groups
        assert "/sweep_0" in groups
        assert "/sweep_1" in groups


# -- supports_groups attribute -----------------------------------------------


class TestSupportsGroups:
    """Verify supports_groups is True on all 3 backend classes."""

    def test_odim_supports_groups(self):
        assert OdimBackendEntrypoint.supports_groups is True

    def test_cfradial1_supports_groups(self):
        assert CfRadial1BackendEntrypoint.supports_groups is True

    def test_nexrad_supports_groups(self):
        assert NexradLevel2BackendEntrypoint.supports_groups is True


# -- Backward compatibility & deprecation tests ------------------------------


class TestDeprecation:
    """Test that standalone functions still work but emit FutureWarning."""

    def test_open_odim_datatree_deprecation(self, odim_file):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dtree = open_odim_datatree(odim_file, sweep=[0])
            deprecation_warnings = [
                x for x in w if issubclass(x.category, FutureWarning)
            ]
            assert len(deprecation_warnings) >= 1
            assert "open_odim_datatree" in str(deprecation_warnings[0].message)
        _assert_cfradial2_structure(dtree)

    def test_open_cfradial1_datatree_deprecation(self, cfradial1_file):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dtree = open_cfradial1_datatree(
                cfradial1_file,
                engine="h5netcdf",
                decode_timedelta=False,
                sweep=[0],
            )
            deprecation_warnings = [
                x for x in w if issubclass(x.category, FutureWarning)
            ]
            assert len(deprecation_warnings) >= 1
            assert "open_cfradial1_datatree" in str(deprecation_warnings[0].message)
        _assert_cfradial2_structure(dtree)

    def test_open_nexradlevel2_datatree_deprecation(self, nexradlevel2_file):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dtree = open_nexradlevel2_datatree(nexradlevel2_file, sweep=[0])
            deprecation_warnings = [
                x for x in w if issubclass(x.category, FutureWarning)
            ]
            assert len(deprecation_warnings) >= 1
            assert "open_nexradlevel2_datatree" in str(deprecation_warnings[0].message)
        _assert_cfradial2_structure(dtree)

    def test_odim_deprecated_output_matches_new_api(self, odim_file):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            old = open_odim_datatree(odim_file, sweep=[0, 1])
        new = xd.open_datatree(odim_file, engine="odim", sweep=[0, 1])
        # Same number of children
        assert set(old.children.keys()) == set(new.children.keys())

    def test_nexrad_deprecated_output_matches_new_api(self, nexradlevel2_file):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            old = open_nexradlevel2_datatree(nexradlevel2_file, sweep=[0, 1])
        new = xd.open_datatree(nexradlevel2_file, engine="nexradlevel2", sweep=[0, 1])
        assert set(old.children.keys()) == set(new.children.keys())
