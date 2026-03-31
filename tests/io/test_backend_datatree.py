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
from xradar.io import _ENGINE_REGISTRY

# -- Fixtures ----------------------------------------------------------------


@pytest.fixture(
    params=[
        pytest.param(("odim", "odim_file"), id="odim"),
        pytest.param(("gamic", "gamic_file"), id="gamic"),
        pytest.param(("iris", "iris0_file"), id="iris"),
        pytest.param(("nexradlevel2", "nexradlevel2_file"), id="nexradlevel2"),
        pytest.param(("furuno", "furuno_scn_file"), id="furuno"),
        pytest.param(("rainbow", "rainbow_file"), id="rainbow"),
        pytest.param(("datamet", "datamet_file"), id="datamet"),
        pytest.param(("hpl", "hpl_file"), id="hpl"),
        pytest.param(("metek", "metek_ave_gz_file"), id="metek"),
        pytest.param(("uf", "uf_file_1"), id="uf"),
    ]
)
def engine_and_file(request):
    """Parametrize over all engines with their fixture names."""
    engine, fixture_name = request.param
    filepath = request.getfixturevalue(fixture_name)
    return engine, filepath


@pytest.fixture
def cfradial1_engine_file(cfradial1_file):
    return "cfradial1", cfradial1_file


# -- Helper ------------------------------------------------------------------


def _assert_cfradial2_structure(dtree, optional_groups=False):
    """Verify that a DataTree has CfRadial2 group structure."""
    assert isinstance(dtree, DataTree)
    children = set(dtree.children.keys())
    if optional_groups:
        for grp in [
            "radar_parameters",
            "georeferencing_correction",
            "radar_calibration",
        ]:
            assert grp in children, f"Missing group: {grp}"
    sweep_groups = [k for k in children if k.startswith("sweep_")]
    assert len(sweep_groups) > 0, "No sweep groups found"
    root_vars = set(dtree.ds.data_vars)
    assert "time_coverage_start" in root_vars
    assert "time_coverage_end" in root_vars


# -- xd.open_datatree integration tests (all engines) -----------------------


class TestXdOpenDatatree:
    """Test xd.open_datatree() for all engines."""

    def test_basic_open(self, engine_and_file):
        engine, filepath = engine_and_file
        dtree = xd.open_datatree(filepath, engine=engine)
        _assert_cfradial2_structure(dtree)

    def test_sweep_selection_int(self, engine_and_file):
        engine, filepath = engine_and_file
        dtree = xd.open_datatree(filepath, engine=engine, sweep=0)
        sweep_groups = [k for k in dtree.children if k.startswith("sweep_")]
        assert len(sweep_groups) == 1

    def test_sweep_selection_string(self, engine_and_file):
        engine, filepath = engine_and_file
        dtree = xd.open_datatree(filepath, engine=engine, sweep="sweep_0")
        sweep_groups = [k for k in dtree.children if k.startswith("sweep_")]
        assert len(sweep_groups) == 1

    def test_kwargs_flow_through(self, engine_and_file):
        engine, filepath = engine_and_file
        dtree = xd.open_datatree(
            filepath, engine=engine, first_dim="auto", site_coords=True, sweep=0
        )
        # Station coords are on root (promoted by _assign_root)
        assert "latitude" in dtree.ds.coords
        assert "longitude" in dtree.ds.coords

    def test_unknown_engine_raises(self, odim_file):
        with pytest.raises(ValueError, match="Unknown engine"):
            xd.open_datatree(odim_file, engine="nonexistent_engine")

    def test_empty_sweep_list_raises(self, engine_and_file):
        engine, filepath = engine_and_file
        with pytest.raises(ValueError, match="sweep list is empty"):
            xd.open_datatree(filepath, engine=engine, sweep=[])


# -- xd.open_datatree for CfRadial1 -----------------------------------------


class TestXdOpenDatatreeCfRadial1:
    """Test xd.open_datatree() for CfRadial1."""

    def test_basic_open(self, cfradial1_engine_file):
        _, filepath = cfradial1_engine_file
        from xradar.io.backends.cfradial1 import CfRadial1BackendEntrypoint

        backend = CfRadial1BackendEntrypoint()
        dtree = backend.open_datatree(
            filepath, engine="h5netcdf", decode_timedelta=False
        )
        _assert_cfradial2_structure(dtree)

    def test_sweep_selection(self, cfradial1_engine_file):
        _, filepath = cfradial1_engine_file
        from xradar.io.backends.cfradial1 import CfRadial1BackendEntrypoint

        backend = CfRadial1BackendEntrypoint()
        dtree = backend.open_datatree(
            filepath, engine="h5netcdf", decode_timedelta=False, sweep=[0, 1]
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

    def test_xr_open_datatree_cfradial1(self, cfradial1_file):
        dtree = xr.open_datatree(
            cfradial1_file, engine="cfradial1", decode_timedelta=False
        )
        _assert_cfradial2_structure(dtree)

    def test_xr_open_datatree_gamic(self, gamic_file):
        dtree = xr.open_datatree(gamic_file, engine="gamic")
        _assert_cfradial2_structure(dtree)

    def test_xr_open_datatree_iris(self, iris0_file):
        dtree = xr.open_datatree(iris0_file, engine="iris")
        _assert_cfradial2_structure(dtree)

    def test_xr_open_datatree_furuno(self, furuno_scn_file):
        dtree = xr.open_datatree(furuno_scn_file, engine="furuno")
        _assert_cfradial2_structure(dtree)

    def test_xr_open_datatree_rainbow(self, rainbow_file):
        dtree = xr.open_datatree(rainbow_file, engine="rainbow")
        _assert_cfradial2_structure(dtree)

    def test_xr_open_datatree_datamet(self, datamet_file):
        dtree = xr.open_datatree(datamet_file, engine="datamet")
        _assert_cfradial2_structure(dtree)

    def test_xr_open_datatree_hpl(self, hpl_file):
        dtree = xr.open_datatree(hpl_file, engine="hpl")
        _assert_cfradial2_structure(dtree)

    def test_xr_open_datatree_metek(self, metek_ave_gz_file):
        dtree = xr.open_datatree(metek_ave_gz_file, engine="metek")
        _assert_cfradial2_structure(dtree)

    def test_xr_open_datatree_uf(self, uf_file_1):
        dtree = xr.open_datatree(uf_file_1, engine="uf")
        _assert_cfradial2_structure(dtree)


# -- supports_groups attribute -----------------------------------------------


class TestSupportsGroups:
    """Verify supports_groups is True on all backend classes."""

    @pytest.mark.parametrize(
        "engine",
        sorted(_ENGINE_REGISTRY.keys()),
    )
    def test_supports_groups(self, engine):
        backend_cls = _ENGINE_REGISTRY[engine]
        assert backend_cls.supports_groups is True


# -- Engine registry ---------------------------------------------------------


class TestEngineRegistry:
    """Verify _ENGINE_REGISTRY contains all expected engines."""

    def test_registry_contains_all_engines(self):
        expected = {
            "odim",
            "cfradial1",
            "nexradlevel2",
            "gamic",
            "iris",
            "furuno",
            "rainbow",
            "datamet",
            "hpl",
            "metek",
            "uf",
        }
        assert set(_ENGINE_REGISTRY.keys()) == expected


# -- Backward compatibility & deprecation tests ------------------------------

# Map of deprecated function names to (import_path, engine, fixture_name)
_DEPRECATED_FUNCTIONS = {
    "open_odim_datatree": ("xradar.io.backends.odim", "odim_file", {}),
    "open_gamic_datatree": ("xradar.io.backends.gamic", "gamic_file", {}),
    "open_iris_datatree": ("xradar.io.backends.iris", "iris0_file", {}),
    "open_nexradlevel2_datatree": (
        "xradar.io.backends.nexrad_level2",
        "nexradlevel2_file",
        {},
    ),
    "open_cfradial1_datatree": (
        "xradar.io.backends.cfradial1",
        "cfradial1_file",
        {"engine": "h5netcdf", "decode_timedelta": False},
    ),
    "open_furuno_datatree": ("xradar.io.backends.furuno", "furuno_scn_file", {}),
    "open_rainbow_datatree": ("xradar.io.backends.rainbow", "rainbow_file", {}),
    "open_datamet_datatree": ("xradar.io.backends.datamet", "datamet_file", {}),
    "open_hpl_datatree": ("xradar.io.backends.hpl", "hpl_file", {}),
    "open_metek_datatree": ("xradar.io.backends.metek", "metek_ave_gz_file", {}),
    "open_uf_datatree": ("xradar.io.backends.uf", "uf_file_1", {}),
}


class TestDeprecation:
    """Test that all standalone functions emit FutureWarning."""

    @pytest.mark.parametrize(
        "func_name,module_path,fixture_name,extra_kwargs",
        [
            (name, mod, fix, kw)
            for name, (mod, fix, kw) in _DEPRECATED_FUNCTIONS.items()
        ],
        ids=list(_DEPRECATED_FUNCTIONS.keys()),
    )
    def test_deprecated_function_warns(
        self, func_name, module_path, fixture_name, extra_kwargs, request
    ):
        import importlib

        filepath = request.getfixturevalue(fixture_name)
        module = importlib.import_module(module_path)
        func = getattr(module, func_name)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dtree = func(filepath, sweep=0, **extra_kwargs)
            deprecation_warnings = [
                x for x in w if issubclass(x.category, FutureWarning)
            ]
            assert len(deprecation_warnings) == 1, (
                f"{func_name} emitted {len(deprecation_warnings)} "
                f"FutureWarnings, expected 1"
            )
            assert func_name in str(deprecation_warnings[0].message)
        _assert_cfradial2_structure(dtree)
