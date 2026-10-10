#!/usr/bin/env python
# Copyright (c) 2024-2025, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for xarray-native open_datatree with engine= parameter.

Tests the unified ``xd.open_datatree()`` and ``xr.open_datatree()`` APIs,
``open_groups_as_dict()`` direct calls, backward compatibility with
deprecated standalone functions, and ``supports_groups`` attribute.
"""

import os
import shutil
import warnings

import numpy as np
import pytest
import xarray as xr
from xarray import DataTree

import xradar as xd
from xradar.io import _ENGINE_REGISTRY
from xradar.io.backends import imd as imd_backend
from xradar.io.backends import open_imd_datatree
from xradar.io.backends.common import _STATION_VARS, _resolve_sweeps

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


_ANGLE_GRID = {
    "azimuth": dict(start_angle=0, stop_angle=360, angle_res=1.0, direction=1),
    "elevation": dict(start_angle=0, stop_angle=90, angle_res=1.0, direction=1),
}


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

    @pytest.mark.parametrize(
        "sweep,exc,match",
        [
            (True, TypeError, "Unsupported sweep True"),
            (-1, TypeError, "Unsupported sweep -1"),
            ([True], ValueError, "Invalid type in 'sweep' list"),
            ([0, 1.0], ValueError, "Invalid type in 'sweep' list"),
        ],
    )
    def test_invalid_sweep_raises(self, engine_and_file, sweep, exc, match):
        engine, filepath = engine_and_file
        with pytest.raises(exc, match=match):
            xd.open_datatree(filepath, engine=engine, sweep=sweep)

    def test_reindex_coord(self, engine_and_file):
        engine, filepath = engine_and_file
        if engine in ("metek", "cfradial2"):
            pytest.skip(f"{engine} has no reindexing")
        plain = xd.open_datatree(filepath, engine=engine, sweep=0)["sweep_0"]
        dim = "elevation" if plain["sweep_mode"].item() == "rhi" else "azimuth"
        dtree = xd.open_datatree(
            filepath, engine=engine, sweep=0, reindex_coord={"angle": _ANGLE_GRID[dim]}
        )
        assert dtree["sweep_0"].sizes[dim] == (90 if dim == "elevation" else 360)


# -- xd.open_datatree for CfRadial1 -----------------------------------------


class TestXdOpenDatatreeCfRadial1:
    """Test xd.open_datatree() for CfRadial1."""

    def test_basic_open(self, cfradial1_engine_file):
        engine, filepath = cfradial1_engine_file
        dtree = xd.open_datatree(
            filepath, engine=engine, netcdf_engine="h5netcdf", decode_timedelta=False
        )
        _assert_cfradial2_structure(dtree)

    def test_sweep_selection(self, cfradial1_engine_file):
        engine, filepath = cfradial1_engine_file
        dtree = xd.open_datatree(
            filepath,
            engine=engine,
            netcdf_engine="h5netcdf",
            decode_timedelta=False,
            sweep=[0, 1],
        )
        sweep_groups = [k for k in dtree.children if k.startswith("sweep_")]
        assert len(sweep_groups) == 2

    @pytest.mark.parametrize(
        "sweep,exc,match",
        [
            (True, TypeError, "Unsupported sweep True"),
            ([0, 1.0], ValueError, "Invalid type in 'sweep' list"),
            (99, ValueError, r"Sweep\(s\) \['sweep_99'\] not found"),
        ],
    )
    def test_invalid_sweep_raises(self, cfradial1_file, sweep, exc, match):
        with pytest.raises(exc, match=match):
            xd.open_datatree(cfradial1_file, engine="cfradial1", sweep=sweep)

    def test_sweep_path(self, cfradial1_file):
        dtree = xd.open_datatree(cfradial1_file, engine="cfradial1", sweep="/sweep_0")
        assert list(dtree.match("sweep_*")) == ["sweep_0"]

    def test_reindex_coord(self, cfradial1_file):
        dtree = xd.open_datatree(
            cfradial1_file,
            engine="cfradial1",
            sweep=0,
            reindex_coord={"angle": _ANGLE_GRID["azimuth"]},
        )
        assert dtree["sweep_0"].sizes["azimuth"] == 360


@pytest.mark.parametrize("first_dim,dim0", [("auto", "azimuth"), ("time", "time")])
def test_hpl_angle_reindex(hpl_file, first_dim, dim0):
    # angle reindexing used to fail while time was still the first dimension
    dtree = xd.open_datatree(
        hpl_file,
        engine="hpl",
        sweep=0,
        first_dim=first_dim,
        reindex_coord={"angle": _ANGLE_GRID["azimuth"]},
    )
    sweep = dtree["sweep_0"]
    assert sweep["intensity"].dims[0] == dim0
    assert sweep.sizes[dim0] == 360


@pytest.mark.parametrize(
    "engine,fixture_name",
    [("furuno", "furuno_scn_file"), ("metek", "metek_ave_gz_file")],
)
@pytest.mark.parametrize("sweep", [1, "sweep_2", [0, 1]])
def test_single_sweep_engines_reject_other_sweeps(engine, fixture_name, sweep, request):
    filepath = request.getfixturevalue(fixture_name)
    with pytest.raises(ValueError, match="single sweep"):
        xd.open_datatree(filepath, engine=engine, sweep=sweep)


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

    @pytest.mark.parametrize("site_coords", [True, False])
    def test_engine_imd_accepts_site_coords(self, imd_file, odim_file, site_coords):
        # same keyword and same station-coord layout as the other engines
        def station_coords(dtree):
            stations = {"latitude", "longitude", "altitude"}
            return (
                sorted(stations & set(dtree.ds.coords)),
                sorted(stations & set(dtree["sweep_0"].to_dataset().variables)),
            )

        imd = xd.open_datatree(imd_file, engine="imd", site_coords=site_coords)
        odim = xd.open_datatree(
            odim_file, engine="odim", site_coords=site_coords, sweep=0
        )
        assert station_coords(imd) == station_coords(odim)

    def test_engine_imd_rejects_unknown_kwarg(self, imd_file):
        with pytest.raises(TypeError, match="site_as_coords"):
            xd.open_datatree(imd_file, engine="imd", site_as_coords=False)

    @pytest.mark.parametrize("sweep", [0, "sweep_0", "/sweep_0", [0]])
    def test_engine_imd_sweep_0(self, imd_file, sweep):
        dtree = xd.open_datatree(imd_file, engine="imd", sweep=sweep)
        assert list(dtree.match("sweep_*")) == ["sweep_0"]

    @pytest.mark.parametrize("sweep", [1, [0, 1], "sweep_2"])
    def test_engine_imd_other_sweep_raises(self, imd_file, sweep):
        with pytest.raises(ValueError, match="single sweep"):
            xd.open_datatree(imd_file, engine="imd", sweep=sweep)

    def test_legacy_site_coords_alias(self, imd_file, monkeypatch):
        seen = {}

        def fake_open(filename, **kwargs):
            seen.update(kwargs)
            return DataTree()

        monkeypatch.setattr(imd_backend, "_open_single_imd_datatree", fake_open)
        open_imd_datatree(imd_file, site_coords=False)
        assert seen == {"site_as_coords": False}

    def test_legacy_site_coords_both_raises(self, imd_file):
        with pytest.raises(TypeError, match="not both"):
            open_imd_datatree(imd_file, site_coords=False, site_as_coords=True)


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


def _no_discovery():
    raise AssertionError("discover_fn must not be called for an explicit sweep")


@pytest.mark.parametrize(
    "sweep,expected",
    [
        (0, ["sweep_0"]),
        (np.int64(2), ["sweep_2"]),
        ("sweep_1", ["sweep_1"]),
        ("/sweep_1", ["sweep_1"]),
        ([0, 2], ["sweep_0", "sweep_2"]),
        ((0, 2), ["sweep_0", "sweep_2"]),
        ([np.int32(0), 1], ["sweep_0", "sweep_1"]),
        (["sweep_0", "/sweep_3"], ["sweep_0", "sweep_3"]),
        ([0, "sweep_1"], ["sweep_0", "sweep_1"]),
    ],
)
def test_resolve_sweeps_valid(sweep, expected):
    assert _resolve_sweeps(sweep, _no_discovery) == expected


def test_resolve_sweeps_none_discovers():
    assert _resolve_sweeps(None, lambda: ["sweep_0", "sweep_1"]) == [
        "sweep_0",
        "sweep_1",
    ]


@pytest.mark.parametrize(
    "sweep,exc,match",
    [
        (True, TypeError, "Unsupported sweep True"),
        (1.0, TypeError, "Unsupported sweep 1.0"),
        (-1, TypeError, "Unsupported sweep -1"),
        ("/", TypeError, "Unsupported sweep '/'"),
        ({0}, TypeError, r"Unsupported sweep \{0\}"),
        ([0, -1], ValueError, "Invalid type in 'sweep' list"),
        ([], ValueError, "sweep list is empty"),
        ([True, False], ValueError, "Invalid type in 'sweep' list"),
        ([0, 1.5], ValueError, "Invalid type in 'sweep' list"),
        ([None], ValueError, "Invalid type in 'sweep' list"),
    ],
)
def test_resolve_sweeps_invalid(sweep, exc, match):
    with pytest.raises(exc, match=match):
        _resolve_sweeps(sweep, _no_discovery)


def test_resolve_sweeps_error_message_is_bounded():
    # untrusted input must not blow up error messages / logs
    with pytest.raises(ValueError) as excinfo:
        _resolve_sweeps([0] * 10_000 + [1.5], _no_discovery)
    assert len(str(excinfo.value)) < 300


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

    @pytest.mark.parametrize("func_name", list(_DEPRECATED_FUNCTIONS))
    def test_deprecated_function_warns(self, func_name, request):
        func, _, filepath, extra_kwargs, _ = _legacy_and_engine(func_name, request)

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
            # stacklevel must point at the caller's line, not xradar internals
            assert deprecation_warnings[0].filename == __file__
        _assert_cfradial2_structure(dtree)


# -- Legacy wrappers keep their pre-#335 behaviour ({issue}`481`) -------------


def _legacy_and_engine(func_name, request):
    """Return the legacy function, engine name, file and matching kwargs."""
    import importlib

    module_path, fixture_name, extra = _DEPRECATED_FUNCTIONS[func_name]
    func = getattr(importlib.import_module(module_path), func_name)
    engine = func_name.removeprefix("open_").removesuffix("_datatree")
    engine_extra = dict(extra)
    if "engine" in engine_extra:  # cfradial1's inner netCDF engine
        engine_extra["netcdf_engine"] = engine_extra.pop("engine")
    return func, engine, request.getfixturevalue(fixture_name), extra, engine_extra


def _call_legacy(func, *args, **kwargs):
    with pytest.warns(FutureWarning, match="is deprecated"):
        return func(*args, **kwargs)


@pytest.mark.parametrize("func_name", list(_DEPRECATED_FUNCTIONS))
@pytest.mark.parametrize(
    "legacy_kw,engine_kw",
    [
        ({}, {}),
        ({"site_as_coords": False}, {"site_coords": False}),
        ({"optional": False}, {"optional": False}),
        ({"optional_groups": True}, {"optional_groups": True}),
    ],
    ids=["default", "site_as_coords", "optional", "optional_groups"],
)
def test_legacy_matches_engine(func_name, legacy_kw, engine_kw, request):
    func, engine, filename, extra, engine_extra = _legacy_and_engine(func_name, request)
    legacy = _call_legacy(func, filename, sweep=0, **extra, **legacy_kw)
    expected = xd.open_datatree(
        filename, engine=engine, sweep=0, **engine_extra, **engine_kw
    )
    xr.testing.assert_identical(legacy, expected)


@pytest.mark.parametrize("func_name", list(_DEPRECATED_FUNCTIONS))
def test_legacy_chunks(func_name, request):
    func, _, filename, extra, _ = _legacy_and_engine(func_name, request)
    dtree = _call_legacy(func, filename, sweep=0, chunks={}, **extra)
    sweep = dtree["sweep_0"].to_dataset()
    lazy = [v for v in sweep.data_vars.values() if v.ndim]
    assert lazy and all(v.chunks is not None for v in lazy)


@pytest.mark.parametrize(
    "func_name",
    [
        name
        for name in _DEPRECATED_FUNCTIONS
        if name not in ("open_cfradial2_datatree", "open_metek_datatree")
    ],
)
def test_legacy_backend_kwargs_reach_backend(func_name, request):
    func, _, filename, extra, _ = _legacy_and_engine(func_name, request)
    via_backend_kwargs = _call_legacy(
        func, filename, sweep=0, backend_kwargs={"first_dim": "time"}, **extra
    )
    direct = _call_legacy(func, filename, sweep=0, first_dim="time", **extra)
    xr.testing.assert_identical(via_backend_kwargs, direct)
    assert "time" in via_backend_kwargs["sweep_0"].dims


@pytest.mark.parametrize(
    "func,engine,fixture_name,key",
    [
        (xd.io.open_odim_datatree, "odim", "odim_file", "optional"),
        # capital-O ``Optional`` is the old GAMIC spelling, accepted everywhere
        (xd.io.open_odim_datatree, "odim", "odim_file", "Optional"),
    ],
    ids=["optional", "Optional"],
)
def test_legacy_backend_kwargs_optional(func, engine, fixture_name, key, request):
    filename = request.getfixturevalue(fixture_name)
    legacy = _call_legacy(func, filename, sweep=0, backend_kwargs={key: False})
    expected = xd.open_datatree(filename, engine=engine, sweep=0, optional=False)
    xr.testing.assert_identical(legacy, expected)
    default = xd.open_datatree(filename, engine=engine, sweep=0)
    assert not legacy.identical(default)


def test_xd_matches_xr_and_indexes_dims(engine_and_file):
    engine, filename = engine_and_file
    dtree = xd.open_datatree(filename, engine=engine, sweep=0)
    xr.testing.assert_identical(
        dtree, xr.open_datatree(filename, engine=engine, sweep=0)
    )
    sweep = dtree["sweep_0"].to_dataset()
    for dim in sweep.dims:
        if dim in sweep.coords:
            assert dim in sweep.xindexes, f"{engine}: {dim} is not indexed"


def test_optional_groups_hold_no_station_vars(engine_and_file):
    engine, filename = engine_and_file
    dtree = xd.open_datatree(filename, engine=engine, sweep=0, optional_groups=True)
    for group in ("radar_parameters", "georeferencing_correction"):
        if group in dtree.children:  # cfradial2 copies only groups in the file
            assert not _STATION_VARS & set(dtree[group].variables), (engine, group)


def test_sweep_selection_returns_requested_sweep(engine_and_file):
    engine, filename = engine_and_file
    if engine in ("furuno", "metek", "hpl"):
        pytest.skip(f"{engine} test file holds a single sweep")
    reference = xd.open_datatree(filename, engine=engine, sweep=[0, 1])
    selected = xd.open_datatree(filename, engine=engine, sweep=[1])
    (name,) = [c for c in selected.children if c.startswith("sweep_")]
    np.testing.assert_equal(
        selected[name]["sweep_fixed_angle"].values,
        reference["sweep_1"]["sweep_fixed_angle"].values,
    )


@pytest.mark.parametrize("legacy", [False, True], ids=["engine", "legacy"])
def test_reindex_angle_warns_once_at_caller(odim_file, legacy):
    angle = _ANGLE_GRID["azimuth"]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        if legacy:
            xd.io.open_odim_datatree(odim_file, sweep=[0, 1], reindex_angle=angle)
        else:
            xd.open_datatree(
                odim_file, engine="odim", sweep=[0, 1], reindex_angle=angle
            )
    reindex = [x for x in w if "reindex_angle" in str(x.message)]
    assert len(reindex) == 1
    assert issubclass(reindex[0].category, FutureWarning)
    assert reindex[0].filename == __file__


def test_uf_legacy_signature():
    import inspect

    params = list(inspect.signature(xd.io.open_uf_datatree).parameters)
    assert params == [
        "filename_or_obj",
        "mask_and_scale",
        "decode_times",
        "concat_characters",
        "decode_coords",
        "drop_variables",
        "use_cftime",
        "decode_timedelta",
        "sweep",
        "first_dim",
        "reindex_coord",
        "reindex_angle",
        "fix_second_angle",
        "site_as_coords",
        "optional",
        "optional_groups",
        "lock",
        "kwargs",
    ]
    assert "site_as_coords" in xd.io.open_uf_datatree.__doc__


@pytest.mark.parametrize(
    "engine,absent",
    [
        ("imd", "optional :"),
        ("rainbow", "fix_second_angle"),
        ("datamet", "fix_second_angle"),
        ("iris", "group :"),
    ],
)
def test_docstrings_only_list_accepted_params(engine, absent):
    import inspect

    method = _ENGINE_REGISTRY[engine].open_groups_as_dict
    assert absent not in method.__doc__
    param = absent.removesuffix(" :")
    assert param not in inspect.signature(method).parameters


def test_legacy_site_as_coords_maps_to_site_coords(cfradial2_file):
    # cfradial2 is the engine where ``site_coords=False`` changes the tree
    legacy = _call_legacy(
        xd.io.open_cfradial2_datatree, cfradial2_file, site_as_coords=False
    )
    expected = xd.open_datatree(cfradial2_file, engine="cfradial2", site_coords=False)
    xr.testing.assert_identical(legacy, expected)
    assert not legacy.identical(xd.open_datatree(cfradial2_file, engine="cfradial2"))


@pytest.mark.parametrize("engine", ["cfradial1", "cfradial2"])
def test_netcdf_engine_reaches_reader(engine, cfradial1_file, cfradial2_file):
    filename = cfradial1_file if engine == "cfradial1" else cfradial2_file
    with pytest.raises(ValueError, match="not-an-engine"):
        xd.open_datatree(filename, engine=engine, netcdf_engine="not-an-engine")
    legacy = getattr(xd.io, f"open_{engine}_datatree")
    with pytest.warns(FutureWarning), pytest.raises(ValueError, match="not-an-engine"):
        legacy(filename, engine="not-an-engine")


def test_legacy_backend_kwargs_decoder(nexradlevel2_file):
    # NEXRAD's wrapper passes every decoder; one in backend_kwargs must not clash
    dtree = _call_legacy(
        xd.io.open_nexradlevel2_datatree,
        nexradlevel2_file,
        sweep=0,
        backend_kwargs={"decode_times": False},
    )
    assert not np.issubdtype(dtree["sweep_0"]["time"].dtype, np.datetime64)


def test_legacy_rejects_both_site_spellings(odim_file):
    with pytest.raises(TypeError, match="not both"):
        _call_legacy(
            xd.io.open_odim_datatree,
            odim_file,
            sweep=0,
            site_as_coords=False,
            site_coords=True,
        )


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="needs /proc/self/fd")
def test_cfradial2_close_releases_file(cfradial2_file, tmp_path):
    # private copy: no other test holds it in xarray's file cache
    filename = tmp_path / "cfradial2.nc"
    shutil.copyfile(cfradial2_file, filename)
    target = os.path.realpath(filename)

    def open_handles():
        fds = os.listdir("/proc/self/fd")
        return sum(
            os.path.realpath(f"/proc/self/fd/{fd}") == target
            for fd in fds
            if os.path.exists(f"/proc/self/fd/{fd}")
        )

    dtree = xd.open_datatree(filename, engine="cfradial2", sweep=0)
    dtree.load()
    dtree.close()
    assert open_handles() == 0
