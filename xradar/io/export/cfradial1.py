#!/usr/bin/env python
# Copyright (c) 2023-2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
CfRadial1 Output
================

This sub-module contains the writer for export of CfRadial1-based radar
data.

Example::

    import xradar as xd
    xd.io.to_cfradial1(dtree, filename, calibs=True)

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = [
    "to_cfradial1",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import warnings
from importlib.metadata import version

import numpy as np
import xarray as xr

#: Per-sweep metadata variables that CfRadial1 stores as scalars (one value
#: per sweep) rather than along a ray dimension.
SWEEP_METADATA_VARS = (
    "sweep_number",
    "sweep_mode",
    "polarization_mode",
    "prt_mode",
    "follow_mode",
    "sweep_fixed_angle",
    "sweep_start_ray_index",
    "sweep_end_ray_index",
)

#: Subset of :data:`SWEEP_METADATA_VARS` gathered into the ``sweep``-indexed
#: info dataset; the ray-index variables are computed separately.
SWEEP_INFO_VARS = (
    "sweep_number",
    "sweep_mode",
    "polarization_mode",
    "prt_mode",
    "follow_mode",
    "sweep_fixed_angle",
)


def _first_valid_scalar(data_array):
    """Collapse a metadata variable to its first non-missing scalar value."""
    if data_array.ndim == 0:
        if data_array.notnull().item():
            return data_array.to_numpy()[()]
        return np.nan

    flat = data_array.stack(_flat=data_array.dims)
    valid = flat.where(flat.notnull(), drop=True)
    if valid.size:
        return valid.isel(_flat=0).to_numpy()[()]
    return np.nan


def _normalize_sweep_metadata(sweep_ds):
    """Restore sweep metadata variables to scalar form before CfRadial1 export."""
    sweep_ds = sweep_ds.copy()
    for name in SWEEP_METADATA_VARS:
        if name not in sweep_ds:
            continue
        if sweep_ds[name].ndim == 0:
            continue
        sweep_ds[name] = xr.DataArray(
            _first_valid_scalar(sweep_ds[name]),
            attrs=sweep_ds[name].attrs,
        )

    return sweep_ds


def _sweep_group_names(dtree):
    """Return the names of the sweep groups in a radar ``DataTree``."""
    return [name for name in dtree.groups if "sweep" in name]


def _map_radar_calibration(calib_ds):
    """
    Map calibration parameters to the CfRadial1 ``r_calib_*`` layout.

    Parameters
    ----------
    calib_ds: xarray.Dataset
        Calibration parameters dataset.

    Returns
    -------
    xarray.Dataset
        New dataset with each variable renamed to ``r_calib_<var>`` and a
        leading ``r_calib`` dimension.
    """
    renamed_vars = {}
    for name in calib_ds.data_vars:
        data_array = calib_ds[name]
        renamed_vars["r_calib_" + name] = xr.DataArray(
            data=data_array.data[np.newaxis, ...],
            dims=["r_calib"] + list(data_array.dims),
            coords={"r_calib": [0]},
            attrs=data_array.attrs,
        )
    radar_calib = xr.Dataset(renamed_vars)
    return radar_calib.drop_vars("r_calib", errors="ignore")


def _extract_root_dataset(dtree):
    """
    Extract the root (volume-level) dataset from a radar ``DataTree``.

    Parameters
    ----------
    dtree: xarray.DataTree
        Radar xarray.DataTree.

    Returns
    -------
    xarray.Dataset
        The root dataset with ``sweep_group_name`` dropped and
        ``sweep_fixed_angle`` renamed to ``fixed_angle`` if present.
    """
    root_ds = dtree.root.to_dataset()
    root_ds = root_ds.drop_vars("sweep_group_name", errors="ignore")

    if "sweep_fixed_angle" in root_ds:
        root_ds = root_ds.rename({"sweep_fixed_angle": "fixed_angle"})

    return root_ds


def _combine_sweeps(dtree, dim0=None):
    """
    Combine all sweep groups into a single ray-indexed CfRadial1 dataset.

    Parameters
    ----------
    dtree: xarray.DataTree
        Radar xarray.DataTree.
    dim0: str, optional
        Either ``azimuth`` or ``elevation``. Inferred per sweep from the
        sweep mode when not provided.

    Returns
    -------
    xarray.Dataset
        Dataset with all sweeps concatenated along ``time``, merged with the
        per-sweep and volume-level metadata.
    """
    sweep_info = _collect_sweep_metadata(dtree)
    root_ds = _extract_root_dataset(dtree)
    if "fixed_angle" in root_ds:
        root_ds = root_ds.drop_vars("fixed_angle")

    sweep_datasets = []
    for group_name in _sweep_group_names(dtree):
        sweep_ds = _normalize_sweep_metadata(
            dtree[group_name].to_dataset(inherit="all_coords")
        )

        # handling first dimension
        if dim0 is None:
            dim0 = (
                "elevation"
                if str(sweep_ds.sweep_mode.load().values) == "rhi"
                else "azimuth"
            )
            if dim0 not in sweep_ds.dims:
                dim0 = "time"
                assert dim0 in sweep_ds.dims

        # swap dims, if needed
        if dim0 != "time" and dim0 in sweep_ds.dims:
            sweep_ds = sweep_ds.swap_dims({dim0: "time"})

        # sort in any case
        sweep_ds = sweep_ds.sortby("time")

        sweep_ds = sweep_ds.drop_vars(["x", "y", "z"], errors="ignore")

        # Strip per-sweep attrs that may vary across sweeps (e.g.
        # NEXRAD ICD waveform_type, super_resolution) to avoid
        # merge conflicts in combine_by_coords below.
        sweep_ds.attrs = {}

        sweep_datasets.append(sweep_ds)

    # need to use combine_by_coords to correctly test for
    # incompatible attrs on DataArray's
    combined = xr.combine_by_coords(
        sweep_datasets,
        data_vars="all",
        compat="no_conflicts",
        join="outer",
        coords="minimal",
        combine_attrs="no_conflicts",
    )

    # These are re-added from ``sweep_info`` below with a ``sweep`` dimension.
    combined = combined.drop_vars(SWEEP_INFO_VARS, errors="ignore")

    georef_coords = ["latitude", "longitude", "altitude", "spatial_ref", "crs_wkt"]
    combined = combined.drop_vars(georef_coords, errors="ignore")

    combined.update(sweep_info)
    combined.update(calculate_sweep_indices(dtree, combined))
    combined = combined.reset_coords(["elevation", "azimuth"])
    combined.update(root_ds)
    return combined


def _collect_sweep_metadata(dtree):
    """
    Collect per-sweep metadata variables into a ``sweep``-indexed dataset.

    Parameters
    ----------
    dtree: xarray.DataTree
        Radar xarray.DataTree.

    Returns
    -------
    xarray.Dataset
        Dataset with one value per sweep for each metadata variable.
    """
    sweep_info = xr.Dataset()
    sweep_groups = _sweep_group_names(dtree)

    for var_name in SWEEP_INFO_VARS:
        values = [
            (
                np.asarray([_first_valid_scalar(dtree[group][var_name])])
                if var_name in dtree[group]
                else np.array([np.nan])
            )
            for group in sweep_groups
        ]

        merged_attrs = {}
        for group in sweep_groups:
            if var_name in dtree[group]:
                merged_attrs.update(dtree[group][var_name].attrs)

        var_data = np.concatenate(values) if values else np.array([np.nan])
        sweep_info[var_name] = xr.DataArray(
            var_data, dims=("sweep",), attrs=merged_attrs
        )

    return sweep_info.rename({"sweep_fixed_angle": "fixed_angle"})


def calculate_sweep_indices(dtree, dataset=None):
    """
    Calculate sweep start and end ray indices for elevation
    values in a radar dataset.

    Parameters
    ----------
    dtree: xarray.DataTree
        Radar xarray.DataTree containing elevation values for different sweep groups.
    dataset: xarray.Dataset, optional
        An optional dataset to which the calculated indices will be added.
        If None, a new dataset will be created.

    Returns:
    xarray.Dataset
        Dataset with sweep start and end ray indices.
    """
    if dataset is None:
        dataset = xr.Dataset()

    sweep_start_ray_index = []
    sweep_end_ray_index = []
    cumulative_size = 0

    for group_name in _sweep_group_names(dtree):
        try:
            ele_size = dtree[group_name].elevation.size
        except AttributeError:
            warnings.warn(
                f"Sweep group '{group_name}' has no 'elevation' coordinate; "
                "skipping its ray-index calculation.",
                stacklevel=2,
            )
            continue
        sweep_start_ray_index.append(cumulative_size)
        sweep_end_ray_index.append(cumulative_size + ele_size - 1)
        cumulative_size += ele_size

    dataset["sweep_start_ray_index"] = xr.DataArray(
        sweep_start_ray_index,
        dims=("sweep",),
        attrs={"standard_name": "index_of_first_ray_in_sweep"},
    )

    dataset["sweep_end_ray_index"] = xr.DataArray(
        sweep_end_ray_index,
        dims=("sweep",),
        attrs={"standard_name": "index_of_last_ray_in_sweep"},
    )

    return dataset


def _build_cfradial1_dataset(dtree, calibs=True):
    """
    Build a single CfRadial1 ``Dataset`` from a radar ``DataTree``.

    This assembles the sweep variables, calibration, radar parameters and
    georeferencing correction, and sets the CfRadial1 global attributes.
    It is shared by :func:`to_cfradial1` (which writes to a file) and by
    :func:`xradar.transform.to_cfradial1` (which returns the dataset).

    Parameters
    ----------
    dtree: xarray.DataTree
        Radar xarray.DataTree object.
    calibs: bool, optional
        Whether to include calibration parameters.

    Returns
    -------
    xarray.Dataset
        The assembled CfRadial1 dataset.
    """
    if dtree is None:
        raise ValueError("`dtree` must be a radar xarray.DataTree, not None.")

    cfradial1_ds = _combine_sweeps(dtree)

    # Handle calibration parameters
    if calibs and "radar_calibration" in dtree:
        calib_ds = _map_radar_calibration(dtree["radar_calibration"].to_dataset())
        cfradial1_ds.update(calib_ds)

    # Add additional parameters if they exist in dtree
    if "radar_parameters" in dtree:
        radar_params = dtree["radar_parameters"].to_dataset().reset_coords()
        cfradial1_ds.update(radar_params)

    if "georeferencing_correction" in dtree:
        radar_georef = dtree["georeferencing_correction"].to_dataset().reset_coords()
        cfradial1_ds.update(radar_georef)

    # Ensure that the data type of sweep_mode and similar variables matches
    if "sweep_mode" in cfradial1_ds.variables:
        cfradial1_ds["sweep_mode"] = cfradial1_ds["sweep_mode"].astype("S")

    # Update global attributes
    cfradial1_ds.attrs = dict(dtree.attrs)
    cfradial1_ds.attrs["Conventions"] = "Cf/Radial"
    cfradial1_ds.attrs["version"] = "1.2"
    xradar_version = version("xradar")
    history = cfradial1_ds.attrs.get("history", "")
    cfradial1_ds.attrs["history"] = (
        f"{history}: xradar v{xradar_version} CfRadial1 export"
    )

    return cfradial1_ds


def to_cfradial1(dtree=None, filename=None, calibs=True):
    """
    Convert a radar xarray.DataTree to the CfRadial1 format
    and save it to a file. Ensure that the resulting dataset
    is well-formed and does not include specified extraneous variables.

    Parameters
    ----------
    dtree: xarray.DataTree
        Radar xarray.DataTree object.
    filename: str, optional
        The name of the output netCDF file. When omitted, a name is derived
        from the instrument name and first timestamp.
    calibs: bool, optional
        Whether to include calibration parameters.
    """
    cfradial1_ds = _build_cfradial1_dataset(dtree, calibs=calibs)

    if filename is None:
        time = str(cfradial1_ds.time[0].dt.strftime("%Y%m%d_%H%M%S").values)
        filename = f"cfrad1_{cfradial1_ds.instrument_name}_{time}.nc"

    cfradial1_ds.to_netcdf(filename, format="netcdf4")
