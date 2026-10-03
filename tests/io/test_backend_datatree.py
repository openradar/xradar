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
from xradar.io.backends import open_imd_datatree

# -- Fixtures ----------------------------------------------------------------


@pytest.fixture(
    params=[
        pytest.param(("odim", "odim_file"), id="odim"),
        pytest.param(("gamic", "gamic_file"), id="gamic"),
        pytest.param(("iris", "iris0_file"), id="iris"),
        pytest.param(("nexradlevel2", "nexradlevel2_file"), id="nexradlevel2"),
        pytest.param(("cfradial2", "cfradial2_file"), id="cfradial2"),
        pytest.param(("furuno", "furuno_scn_file"), id="furuno"),
        pytest.param(("rainbow", "rainbow_file"), id="rainbow"),
        pytest.param(("datamet", "datamet_file"), id="datamet"),
        pytest.param(("hpl", "hpl_file"), id="hpl"),
        pytest.param(("metek", "metek_ave_gz_file"), id="metek"),
        pytest.param(("uf", "uf_file_1"), id="uf"),
        pytest.param(
            ("imd", "imd_file"),
            marks=pytest.mark.skip(
                reason="IMD is single-sweep-per-file; see TestIMDMultiFile",
            ),
            id="imd",
        ),
    ]
)
def engine_and_file(request):
    """Parametrize over all engines.

    See ``TestIMDMultiFile`` for IMD-specific coverage (the multi-file
    carve-out from the engine= API).
    """
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

    def test_xr_open_datatree_cfradial2(self, cfradial2_file):
        dtree = xr.open_datatree(cfradial2_file, engine="cfradial2")
        _assert_cfradial2_structure(dtree)

    def test_xr_open_datatree_imd(self, imd_file):
        dtree = xr.open_datatree(imd_file, engine="imd")
        _assert_cfradial2_structure(dtree)


# -- IMD: multi-file carve-out vs single-file engine -------------------------


class TestIMDMultiFile:
    """IMD is the documented multi-file carve-out from the engine= API.

    The single-file path uses ``engine="imd"``; multi-file volumes still
    go through the module-level ``xd.io.open_imd_datatree([files])``.
    """

    def test_engine_imd_handles_single_file(self, imd_file):
        dtree = xd.open_datatree(imd_file, engine="imd")
        _assert_cfradial2_structure(dtree)
        sweep_groups = [k for k in dtree.children if k.startswith("sweep_")]
        assert len(sweep_groups) == 1

    def test_module_level_handles_multi_file_volume(self, imd_volume_files):
        # Precondition: each fixture file in `imd_volume_files` contains
        # exactly one sweep, so the resulting volume has one sweep per file.
        dtree = open_imd_datatree(imd_volume_files)
        _assert_cfradial2_structure(dtree)
        sweep_groups = [k for k in dtree.children if k.startswith("sweep_")]
        assert len(sweep_groups) == len(imd_volume_files)


# -- CfRadial2 site_coords behavior ------------------------------------------


class TestCfRadial2SiteCoords:
    """`site_coords` honors True/False for the CfRadial2 entrypoint."""

    def test_site_coords_true_keeps_station_coords(self, cfradial2_file):
        dtree = xd.open_datatree(cfradial2_file, engine="cfradial2", site_coords=True)
        assert "latitude" in dtree.ds.coords
        assert "longitude" in dtree.ds.coords
        assert "altitude" in dtree.ds.coords

    def test_site_coords_false_drops_station_coords(self, cfradial2_file):
        dtree = xd.open_datatree(cfradial2_file, engine="cfradial2", site_coords=False)
        assert "latitude" not in dtree.ds.coords
        assert "longitude" not in dtree.ds.coords
        assert "altitude" not in dtree.ds.coords


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


# -- Docstring regression guard ---------------------------------------------


class TestDocstrings:
    """`open_groups_as_dict` / `open_datatree` must carry usable docstrings.

    The composed docstrings are assigned by module-level side effects
    (e.g. ``OdimBackendEntrypoint.open_groups_as_dict.__doc__ = ...``).
    Without this guard a future refactor could silently delete a
    docstring and no test would catch the regression.
    """

    @pytest.mark.parametrize(
        "engine",
        sorted(_ENGINE_REGISTRY.keys()),
    )
    def test_open_groups_as_dict_has_param_docstring(self, engine):
        doc = _ENGINE_REGISTRY[engine].open_groups_as_dict.__doc__
        assert doc, f"{engine} open_groups_as_dict has no docstring"
        assert "Parameters" in doc
        assert "Returns" in doc
        assert "optional_groups" in doc

    @pytest.mark.parametrize(
        "engine",
        sorted(_ENGINE_REGISTRY.keys()),
    )
    def test_open_datatree_references_groups_as_dict(self, engine):
        doc = _ENGINE_REGISTRY[engine].open_datatree.__doc__
        assert doc, f"{engine} open_datatree has no docstring"
        assert "open_groups_as_dict" in doc


def test_compose_docstring_structure():
    """`_compose_docstring` assembles summary + common block + extras + Returns."""
    from xradar.io.backends.common import REINDEX_PARAMS_DOC, _compose_docstring

    doc = _compose_docstring("Summary line.", REINDEX_PARAMS_DOC)
    assert doc.startswith("Summary line.")
    assert "Parameters" in doc
    assert "Returns" in doc
    assert "reindex_angle" in doc
    assert "filename_or_obj" in doc  # common block is always included
    assert "dict[str, xarray.Dataset]" in doc


def test_compose_docstring_skips_empty_extra_blocks():
    """Empty/None extra blocks must not double-insert section headers."""
    from xradar.io.backends.common import _compose_docstring

    doc = _compose_docstring("Summary.", "", None)
    assert doc.count("Parameters") == 1
    assert doc.count("Returns") == 1


# -- Engine registry ---------------------------------------------------------


class TestEngineRegistry:
    """Verify _ENGINE_REGISTRY contains all expected engines."""

    def test_registry_contains_all_engines(self):
        expected = {
            "odim",
            "cfradial1",
            "cfradial2",
            "nexradlevel2",
            "gamic",
            "iris",
            "furuno",
            "rainbow",
            "datamet",
            "hpl",
            "metek",
            "uf",
            "imd",
        }
        assert set(_ENGINE_REGISTRY.keys()) == expected

    def test_demo_notebook_lists_all_engines(self):
        """Bitrot guard: adding an engine to the registry must also be demoed."""
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        notebook = repo_root / "docs/notebooks/Open-Datatree-Engine.md"
        text = notebook.read_text()
        for engine in _ENGINE_REGISTRY:
            assert f'engine="{engine}"' in text, f"notebook missing engine={engine!r}"


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
    "open_cfradial2_datatree": (
        "xradar.io.backends.cfradial2",
        "cfradial2_file",
        {},
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
