#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""

CfRadial2
=========

This sub-module contains the CfRadial2 reader for hierarchical radar datasets.

The reader opens a grouped dataset via :py:func:`xarray.open_datatree` and applies
best-effort compatibility normalization to common institutional variations so the
result better matches xradar's FM301-oriented DataTree layout. Full FM301
validation is not currently performed.

Example::

    import xradar as xd
    dtree = xd.io.open_cfradial2_datatree(filename)

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}

"""

__all__ = ["open_cfradial2_datatree"]

__doc__ = __doc__.format("\n   ".join(__all__))

import warnings
from collections.abc import Iterable
from os import PathLike
from typing import Any

import numpy as np
from xarray import DataTree, Variable, open_datatree

from ...model import (
    georeferencing_correction_subgroup,
    get_altitude_attrs,
    get_azimuth_attrs,
    get_elevation_attrs,
    get_latitude_attrs,
    get_longitude_attrs,
    get_moment_attrs,
    get_range_attrs,
    get_time_attrs,
    optional_root_attrs,
    optional_root_vars,
    radar_calibration_subgroup,
    radar_parameters_subgroup,
    required_global_attrs,
    required_root_vars,
    sweep_vars_mapping,
)
from .common import _STATION_VARS, _apply_site_as_coords

_ROOT_ATTR_RENAMES = {
    "RadarName": "instrument_name",
}

_ROOT_VAR_FROM_ATTRS = {
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Height": "altitude",
}

_SWEEP_VAR_RENAMES = {
    "time_us": "time",
    "range_m": "range",
    "azimuth_deg": "azimuth",
    "elevation_deg": "elevation",
    "fixed_angle": "sweep_fixed_angle",
}

_SUBGROUPS = {
    "radar_parameters": radar_parameters_subgroup,
    "georeferencing_correction": georeferencing_correction_subgroup,
    "radar_calibration": radar_calibration_subgroup,
}


def _update_var_attrs(ds, name: str, attrs: dict[str, Any]):
    if name in ds:
        merged = dict(ds[name].attrs)
        merged.update(attrs)
        ds[name].attrs = merged
    return ds


def _fixed_angle_values_conflict(candidates):
    normalized = [np.asarray(value).reshape(-1) for _, value in candidates]
    first = normalized[0]
    for other in normalized[1:]:
        if first.shape != other.shape:
            return True
        try:
            if not np.allclose(first, other, equal_nan=True):
                return True
        except TypeError:
            if not np.array_equal(first, other):
                return True
    return False


def _get_sweep_fixed_angle(sw, sweep_name: str):
    candidates = []
    explicit_candidates = []
    if "sweep_fixed_angle" in sw:
        candidate = ("sweep_fixed_angle", sw["sweep_fixed_angle"].values)
        candidates.append(candidate)
        explicit_candidates.append(candidate)
    if "fixed_angle" in sw:
        candidate = ("fixed_angle", sw["fixed_angle"].values)
        candidates.append(candidate)
        explicit_candidates.append(candidate)
    if "elevation" in sw:
        candidates.append(
            ("elevation", np.asarray(sw["elevation"].median(skipna=True).values))
        )
    if "azimuth" in sw:
        candidates.append(
            ("azimuth", np.asarray(sw["azimuth"].median(skipna=True).values))
        )

    if len(explicit_candidates) > 1 and _fixed_angle_values_conflict(
        explicit_candidates
    ):
        ordered = ", ".join(name for name, _ in candidates)
        warnings.warn(
            f"CfRadial2 sweep `{sweep_name}` contains multiple fixed-angle candidates "
            f"({ordered}); using `{candidates[0][0]}`.",
            UserWarning,
            stacklevel=3,
        )

    if not candidates:
        return np.nan

    fixed = candidates[0][1]
    return np.asarray(fixed).item() if np.asarray(fixed).ndim == 0 else fixed


def _normalize_sweep_name(name: str) -> str:
    if name.startswith("sweep_"):
        suffix = name.split("_", 1)[1]
        try:
            return f"sweep_{int(suffix)}"
        except ValueError:
            return name
    return name


def _iter_selected_sweeps(tree: DataTree, sweep: Any) -> list[str]:
    available = sorted(
        (
            _normalize_sweep_name(child)
            for child in tree.children
            if child.startswith("sweep_")
        ),
        key=lambda x: int(x.split("_", 1)[1]),
    )
    if sweep is None:
        return available
    if isinstance(sweep, str):
        return [_normalize_sweep_name(sweep)]
    if isinstance(sweep, int):
        return [f"sweep_{sweep}"]
    if isinstance(sweep, Iterable):
        selected: list[str] = []
        for item in sweep:
            if isinstance(item, int):
                selected.append(f"sweep_{item}")
            else:
                selected.append(_normalize_sweep_name(item))
        return selected
    raise TypeError("sweep must be None, int, str or an iterable of ints/strings")


def _coerce_scalar_dataarray(ds, name: str, value: Any):
    if name in ds:
        return ds
    ds[name] = Variable((), value)
    return ds


def _normalize_root_dataset(
    root, sweep_names: list[str], sweep_datasets: list, optional: bool
):
    ds = root.copy()

    rename_attrs = {k: v for k, v in _ROOT_ATTR_RENAMES.items() if k in ds.attrs}
    if rename_attrs:
        ds.attrs.update({new: ds.attrs.pop(old) for old, new in rename_attrs.items()})

    for attr_name, var_name in _ROOT_VAR_FROM_ATTRS.items():
        if var_name not in ds and attr_name in ds.attrs:
            ds[var_name] = ds.attrs.pop(attr_name)

    ds = ds.drop_vars(["sweep_group_name", "sweep_fixed_angle"], errors="ignore")
    ds["sweep_group_name"] = ("sweep", np.array(sweep_names, dtype=str))

    fixed_angles = [
        _get_sweep_fixed_angle(sw, sweep_names[idx])
        for idx, sw in enumerate(sweep_datasets)
    ]
    ds["sweep_fixed_angle"] = ("sweep", np.asarray(fixed_angles, dtype="float32"))

    if "time_coverage_start" not in ds or "time_coverage_end" not in ds:
        times = [
            sw["time"] for sw in sweep_datasets if "time" in sw.coords or "time" in sw
        ]
        if times:
            start = min(t.min().values for t in times)
            end = max(t.max().values for t in times)
            ds = _coerce_scalar_dataarray(ds, "time_coverage_start", start)
            ds = _coerce_scalar_dataarray(ds, "time_coverage_end", end)

    defaults = {
        "volume_number": 0,
        "platform_type": "fixed",
        "instrument_type": "radar",
    }
    for name, value in defaults.items():
        ds = _coerce_scalar_dataarray(ds, name, value)

    ds.attrs.setdefault("Conventions", "CF-1.8, WMO CF-1.0, ACDD-1.3")
    ds.attrs.setdefault("wmo__cf_profile", "FM 301-XX")
    ds.attrs.setdefault("platform_is_mobile", "false")
    ds.attrs.setdefault("history", "")

    allowed_attrs = (
        set(required_global_attrs) | set(optional_root_attrs) | {"wmo__cf_profile"}
    )
    ds.attrs = {k: v for k, v in ds.attrs.items() if k in allowed_attrs}

    for name in ("sweep_group_name", "sweep_fixed_angle"):
        if name in ds and name in ds.coords:
            ds = ds.reset_coords(name)

    promote = _STATION_VARS & set(ds.data_vars)
    if promote:
        ds = ds.set_coords(list(promote))

    root_vars = set(required_root_vars) | set(optional_root_vars)
    keep_vars = (root_vars & set(ds.data_vars)) | (set(ds.coords) & _STATION_VARS)
    drop_vars = set(ds.data_vars) - keep_vars
    if drop_vars:
        ds = ds.drop_vars(drop_vars)

    if not optional:
        removable = set(optional_root_vars) & set(ds.data_vars)
        if removable:
            ds = ds.drop_vars(removable)

    ds = _normalize_root_attrs(ds)

    return ds


def _rename_using_mapping(ds, mapping: dict[str, str | None]):
    rename_map = {}
    for source, target in mapping.items():
        if source in ds and target and source != target and target not in ds:
            rename_map[source] = target
    if rename_map:
        ds = ds.rename_vars(rename_map)
    return ds


def _infer_sweep_mode(ds):
    mode = ds.attrs.get("sweep_mode")
    if mode is not None:
        return str(mode)
    scan_type = ds.attrs.get("ScanType")
    if scan_type in {1, "1"}:
        return "rhi"
    return "azimuth_surveillance"


def _derive_range_attrs(ds):
    if "range" not in ds:
        return ds
    attrs = dict(ds["range"].attrs)
    attrs.setdefault("units", "meters")
    attrs.setdefault("standard_name", "projection_range_coordinate")
    attrs.setdefault("long_name", "range_to_measurement_volume")
    attrs.setdefault("axis", "radial_range_coordinate")
    if ds["range"].size > 1:
        spacing = float(np.diff(ds["range"].values[:2])[0])
        attrs.setdefault("spacing_is_constant", "true")
        attrs.setdefault("meters_to_center_of_first_gate", float(ds["range"].values[0]))
        attrs.setdefault("meters_between_gates", spacing)
    ds["range"].attrs = attrs
    return ds


def _normalize_root_attrs(ds):
    for name, attrs in {
        "latitude": get_latitude_attrs(),
        "longitude": get_longitude_attrs(),
        "altitude": get_altitude_attrs(),
        "time_coverage_start": get_time_attrs(),
        "time_coverage_end": get_time_attrs(),
    }.items():
        ds = _update_var_attrs(ds, name, attrs)
    return ds


def _normalize_sweep_coord_attrs(ds):
    ds = _update_var_attrs(ds, "time", get_time_attrs())
    if "range" in ds:
        ds = _update_var_attrs(ds, "range", get_range_attrs(ds["range"].values))
    if "azimuth" in ds:
        attrs = (
            get_azimuth_attrs(ds["azimuth"].values)
            if ds["azimuth"].size > 2
            else get_azimuth_attrs()
        )
        ds = _update_var_attrs(ds, "azimuth", attrs)
    if "elevation" in ds:
        attrs = (
            get_elevation_attrs(ds["elevation"].values)
            if ds["elevation"].size > 2
            else get_elevation_attrs()
        )
        ds = _update_var_attrs(ds, "elevation", attrs)
    ds = _update_var_attrs(ds, "frequency", {"units": "s-1", "standard_name": ""})
    return ds


def _normalize_dataset_var_attrs(ds):
    for name, da in ds.data_vars.items():
        if name in {
            "sweep_number",
            "sweep_mode",
            "follow_mode",
            "prt_mode",
            "sweep_fixed_angle",
        }:
            continue
        if da.ndim == 2 and set(da.dims) == {"time", "range"}:
            attrs = dict(da.attrs)
            attrs.setdefault("coordinates", "elevation azimuth range")
            if name in sweep_vars_mapping:
                moment_attrs = get_moment_attrs(name)
                moment_attrs.update(attrs)
                attrs = moment_attrs
            da.attrs = attrs
    return ds


def _normalize_sweep_dataset(ds, sweep_name: str, first_dim: str, optional: bool):
    ds = ds.copy()

    rename_dims = {}
    if "ntime" in ds.dims and "time" not in ds.dims:
        rename_dims["ntime"] = "time"
    if "nrange" in ds.dims and "range" not in ds.dims:
        rename_dims["nrange"] = "range"
    if rename_dims:
        ds = ds.rename_dims(rename_dims)

    rename_vars = {}
    for source, target in _SWEEP_VAR_RENAMES.items():
        if source in ds and target not in ds:
            rename_vars[source] = target
    if rename_vars:
        ds = ds.rename_vars(rename_vars)

    if "azimuth" in ds.dims and "time" not in ds.dims:
        ds = ds.swap_dims({"azimuth": "time"})
    if "elevation" in ds.dims and "time" not in ds.dims:
        ds = ds.swap_dims({"elevation": "time"})

    for coord in ["time", "range", "azimuth", "elevation"]:
        if coord in ds and coord not in ds.coords:
            ds = ds.set_coords(coord)

    mode = _infer_sweep_mode(ds)
    dim0 = "elevation" if mode == "rhi" else "azimuth"
    fixed_coord = "azimuth" if mode == "rhi" else "elevation"

    if "sweep_number" not in ds:
        ds["sweep_number"] = Variable((), int(sweep_name.split("_", 1)[1]))
    if "sweep_mode" not in ds:
        ds["sweep_mode"] = Variable((), mode)
    if "follow_mode" not in ds:
        ds["follow_mode"] = Variable((), "none")
    if "prt_mode" not in ds:
        ds["prt_mode"] = Variable((), "fixed")

    if "sweep_fixed_angle" not in ds:
        if "fixed_angle" in ds:
            ds["sweep_fixed_angle"] = ds["fixed_angle"]
        elif fixed_coord in ds:
            ds["sweep_fixed_angle"] = ds[fixed_coord].median(skipna=True)
        elif fixed_coord in ds.coords:
            ds["sweep_fixed_angle"] = ds[fixed_coord].median(skipna=True)

    if "frequency" not in ds:
        if "carrier_frequency_hz" in ds:
            carrier = ds["carrier_frequency_hz"]
            value = carrier.isel(time=0) if "time" in carrier.dims else carrier
            ds["frequency"] = ("frequency", np.atleast_1d(value.values))
        else:
            ds["frequency"] = ("frequency", np.array([np.nan], dtype="float32"))

    for coord in ["time", "range", "azimuth", "elevation", "frequency"]:
        if coord in ds and coord not in ds.coords:
            ds = ds.set_coords(coord)

    ds = _derive_range_attrs(ds)
    ds = _normalize_sweep_coord_attrs(ds)
    ds = _normalize_dataset_var_attrs(ds)
    ds = _apply_site_as_coords(ds, True)

    if first_dim == "auto" and dim0 in ds.coords and "time" in ds.dims:
        ds = ds.swap_dims({"time": dim0}).sortby(dim0)
    elif "time" in ds.dims:
        ds = ds.sortby("time")

    mandatory = {
        "sweep_number",
        "sweep_mode",
        "follow_mode",
        "prt_mode",
        "sweep_fixed_angle",
        "azimuth",
        "elevation",
    }
    if optional:
        keep_vars = set(ds.data_vars)
    else:
        keep_vars = mandatory | {
            k
            for k, v in ds.data_vars.items()
            if any(d in v.dims for d in ["time", "range"])
        }
    drop_vars = set(ds.data_vars) - keep_vars
    if drop_vars:
        ds = ds.drop_vars(drop_vars)

    return ds


def _normalize_subgroup(node: DataTree, mapping: dict[str, str | None]):
    ds = _rename_using_mapping(node.to_dataset(), mapping)
    ds.attrs = {}
    return ds


def open_cfradial2_datatree(
    filename_or_obj: str | PathLike[str], **kwargs: Any
) -> DataTree:
    """Open a CfRadial2-like grouped dataset as :py:class:`xarray.DataTree`.

    The reader performs best-effort normalization of common CfRadial2/FM301
    naming and metadata differences. It is not a full FM301 validator.

    Parameters
    ----------
    filename_or_obj : str or PathLike
        Path or object understood by :py:func:`xarray.open_datatree`.

    Keyword Arguments
    -----------------
    sweep : int, str, iterable, optional
        Sweep selection. Defaults to all available sweeps.
    first_dim : str
        Can be ``time`` or ``auto``. Defaults to ``time``.
    optional : bool
        Keep optional root variables when available. Defaults to ``True``.
    optional_groups : bool
        Include root metadata subgroups if present. Defaults to ``False``.
    **kwargs : dict
        Additional keyword arguments passed to :py:func:`xarray.open_datatree`.

    Returns
    -------
    xarray.DataTree
        Normalized DataTree containing root metadata and sweep groups.
    """
    sweep = kwargs.pop("sweep", None)
    first_dim = kwargs.pop("first_dim", "time")
    optional = kwargs.pop("optional", True)
    optional_groups = kwargs.pop("optional_groups", False)
    kwargs.update(decode_timedelta=kwargs.pop("decode_timedelta", False))

    with open_datatree(filename_or_obj, **kwargs) as tree:
        raw_sweep_names = [name for name in tree.children if name.startswith("sweep_")]
        selected = _iter_selected_sweeps(tree, sweep)
        output_names = [f"sweep_{i}" for i in range(len(selected))]
        sweep_nodes = {
            _normalize_sweep_name(name): tree[name] for name in raw_sweep_names
        }

        missing = [name for name in selected if name not in sweep_nodes]
        if missing:
            raise ValueError(
                f"Sweep group(s) missing from file `{filename_or_obj}`: {missing}"
            )

        normalized_sweeps = []
        for output_name, source_name in zip(output_names, selected):
            ds = _normalize_sweep_dataset(
                sweep_nodes[source_name].to_dataset(inherit=True),
                output_name,
                first_dim=first_dim,
                optional=optional,
            )
            normalized_sweeps.append(ds)

        dtree = {
            "/": _normalize_root_dataset(
                tree["/"].to_dataset(),
                output_names,
                normalized_sweeps,
                optional=optional,
            )
        }

        if optional_groups:
            for name, mapping in _SUBGROUPS.items():
                if name in tree.children:
                    dtree[f"/{name}"] = _normalize_subgroup(tree[name], mapping)

        for i, ds in enumerate(normalized_sweeps):
            cleaned = ds.drop_vars(_STATION_VARS, errors="ignore")
            cleaned.attrs = {}
            dtree[f"sweep_{i}"] = cleaned

    normalized = selected != output_names or any(
        name != _normalize_sweep_name(name) for name in raw_sweep_names
    )
    if normalized:
        warnings.warn(
            "CfRadial2 sweep groups were renumbered into sequential `sweep_<n>` order.",
            UserWarning,
            stacklevel=2,
        )

    root_ds = dtree["/"]
    missing_root = required_root_vars - set(root_ds.data_vars) - set(root_ds.coords)
    if missing_root:
        warnings.warn(
            "CfRadial2 reader could not fully normalize FM301 root variables; "
            f"missing {sorted(missing_root)}.",
            UserWarning,
            stacklevel=2,
        )

    return DataTree.from_dict(dtree)
