#!/usr/bin/env python
# Copyright (c) 2023, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""

CfRadial1 output
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

from importlib.metadata import version

import numpy as np
import xarray as xr


def _calib_mapper(calib_params):
    """
    Map calibration parameters to a new dataset format.

    Parameters
    ----------
    calib_params: xarray.Dataset
        Calibration parameters dataset.

    Returns
    -------
    xarray.Dataset
        New dataset with mapped calibration parameters.
    """
    new_data_vars = {}
    for var in calib_params.data_vars:
        data_array = calib_params[var]
        new_data_vars["r_calib_" + var] = xr.DataArray(
            data=data_array.data[np.newaxis, ...],
            dims=["r_calib"] + list(data_array.dims),
            coords={"r_calib": [0]},
            attrs=data_array.attrs,
        )
    radar_calib_renamed = xr.Dataset(new_data_vars)
    dummy_ds = radar_calib_renamed.rename_vars({"r_calib": "fake_coord"})
    del dummy_ds["fake_coord"]
    return dummy_ds


def _main_info_mapper(dtree):
    """
    Map main radar information from a radar datatree dataset.

    Parameters
    ----------
    dtree: datatree.DataTree
        Radar datatree.

    Returns
    -------
    xarray.Dataset
        Dataset containing the mapped radar information.
    """
    dataset = (
        dtree.root.to_dataset()
        .drop_vars("sweep_group_name", errors="ignore")
        .rename({"sweep_fixed_angle": "fixed_angle"})
    )
    return dataset


def _variable_mapper(dtree, dim0=None):
    """
    Map radar variables for different sweep groups.

    Parameters
    ----------
    dtree: datatree.DataTree
        Radar datatree.
    dim0: str
        Either `azimuth` or `elevation`

    Returns
    -------
    xarray.Dataset
        Dataset containing mapped radar variables.
    """

    sweep_info = _sweep_info_mapper(dtree)
    vol_info = _main_info_mapper(dtree).drop("fixed_angle")
    sweep_datasets = []
    for grp in dtree.groups:
        if "sweep" in grp:
            data = dtree[grp]

            # handling first dimension
            if dim0 is None:
                dim0 = (
                    "elevation"
                    if data.sweep_mode.load().astype(str) == "rhi"
                    else "azimuth"
                )
                if dim0 not in data.dims:
                    dim0 = "time"
                    assert dim0 in data.dims

            # swap dims, if needed
            if dim0 != "time" and dim0 in data.dims:
                data = data.swap_dims({dim0: "time"})

            # sort in any case
            data = data.sortby("time")

            data = data.drop_vars(["x", "y", "z"], errors="ignore")

            # Convert to a dataset and append to the list
            sweep_datasets.append(data.to_dataset())

    result_dataset = xr.concat(
        sweep_datasets,
        dim="time",
        compat="no_conflicts",
        join="right",
        combine_attrs="drop_conflicts",
    )

    # Check if specific variables exist before dropping them
    drop_variables = [
        "sweep_fixed_angle",
        "sweep_number",
        "sweep_mode",
        "prt_mode",
        "follow_mode",
    ]
    result_dataset = result_dataset.drop_vars(drop_variables, errors="ignore")

    drop_coords = ["latitude", "longitude", "altitude", "spatial_ref", "crs_wkt"]
    result_dataset = result_dataset.drop_vars(drop_coords, errors="ignore")

    result_dataset = result_dataset.update(sweep_info)
    sweep_indices = calculate_sweep_indices(dtree, result_dataset)
    result_dataset = result_dataset.update(sweep_indices)
    result_dataset = result_dataset.reset_coords(["elevation", "azimuth"])
    result_dataset = result_dataset.update(vol_info)
    return result_dataset


def _sweep_info_mapper(dtree):
    """
    Extract specified sweep information variables from a radar datatree

    Parameters
    ----------
    dtree: datatree.DataTree
        Radar datatree.

    Returns
    -------
    xarray.Dataset
        Dataset containing the specified sweep information variables.
    """
    dataset = xr.Dataset()

    sweep_vars = [
        "sweep_number",
        "sweep_mode",
        "polarization_mode",
        "prt_mode",
        "follow_mode",
        "sweep_fixed_angle",
        "sweep_start_ray_index",
        "sweep_end_ray_index",
    ]

    for var_name in sweep_vars:
        var_data_list = [
            np.unique(dtree[s][var_name].values[np.newaxis, ...])
            if var_name in dtree[s]
            else np.array([np.nan])
            for s in dtree.groups
            if "sweep" in s
        ]

        var_attrs_list = [
            dtree[s][var_name].attrs if var_name in dtree[s] else {}
            for s in dtree.groups
            if "sweep" in s
        ]

        if not var_data_list:
            var_data = np.array([np.nan])
        else:
            var_data = np.concatenate(var_data_list)

        var_attrs = {}
        for attrs in var_attrs_list:
            var_attrs.update(attrs)

        var_data_array = xr.DataArray(var_data, dims=("sweep",), attrs=var_attrs)
        dataset[var_name] = var_data_array

    dataset = dataset.rename({"sweep_fixed_angle": "fixed_angle"})

    return dataset


def calculate_sweep_indices(dtree, dataset=None):
    """
    Calculate sweep start and end ray indices for elevation
    values in a radar dataset.

    Parameters
    ----------
    dtree: datatree.DataTree
        Radar datatree containing elevation values for different sweep groups.
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

    try:
        for group_name in dtree.groups:
            if "sweep" in group_name:
                ele_size = dtree[group_name].elevation.size
                sweep_start_ray_index.append(cumulative_size)
                sweep_end_ray_index.append(cumulative_size + ele_size - 1)
                cumulative_size += ele_size

    except KeyError as e:
        print(
            f"Error: The sweep group '{e.args[0]}' was not found in radar datatree. Skipping..."
        )

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


def to_cfradial1(dtree=None, filename=None, calibs=True):
    """
    Convert a radar datatree.DataTree to the CFRadial1 format
    and save it to a file.

    Parameters
    ----------
    dtree: datatree.DataTree
        Radar datatree object.
    filename: str, optional
        The name of the output netCDF file.
    calibs: Bool, optional
        calibration parameters
    """
    dataset = _variable_mapper(dtree)

    # Check if radar_parameters, radar_calibration, and
    # georeferencing_correction exist in dtree
    if calibs:
        if "radar_calibration" in dtree:
            calib_params = dtree["radar_calibration"].to_dataset()
            calibs = _calib_mapper(calib_params)
            dataset.update(calibs)

    if "radar_parameters" in dtree:
        radar_params = dtree["radar_parameters"].to_dataset()
        dataset.update(radar_params)

    if "georeferencing_correction" in dtree:
        radar_georef = dtree["georeferencing_correction"].to_dataset()
        dataset.update(radar_georef)

    dataset.attrs = dtree.attrs

    dataset.attrs["Conventions"] = "Cf/Radial"
    dataset.attrs["version"] = "1.2"
    xradar_version = version("xradar")
    dataset.attrs["history"] += f": xradar v{xradar_version} CfRadial1 export"

    if filename is None:
        time = str(dataset.time[0].dt.strftime("%Y%m%d_%H%M%S").values)
        filename = f"cfrad1_{dataset.instrument_name}_{time}.nc"

    dataset.to_netcdf(filename, format="netcdf4")
