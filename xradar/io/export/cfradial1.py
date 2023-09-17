#!/usr/bin/env python
# Copyright (c) 2022, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""

CfRadial1 output
================

This sub-module contains the writer for export of CfRadial1-based radar
data.

Author: @syedhamidali.

Example::

    import xradar as xd
    dtree = xd.io.to_cfradial1(dtree, filename)

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}

"""

__all__ = [
    "to_cfradial1",
]

import numpy as np
import xarray as xr


def _calib_mapper(calib_params):
    """
    Map calibration parameters to a new dataset format.

    Parameters:
    - calib_params: xarray.Dataset
        Calibration parameters dataset.

    Returns:
    xarray.Dataset
        New dataset with mapped calibration parameters.
    """
    new_data_vars = {}
    for var in calib_params.data_vars:
        data_array = calib_params[var]
        new_data_vars['r_calib_' + var] = xr.DataArray(
            data=data_array.data[np.newaxis, ...],
            dims=['r_calib'] + list(data_array.dims),
            coords={'r_calib': [0]},
            attrs=data_array.attrs
        )
    radar_calib_renamed = xr.Dataset(new_data_vars)
    dummy_ds = radar_calib_renamed.rename_vars({'r_calib': "fake_coord"})
    del dummy_ds['fake_coord']
    return dummy_ds


def _main_info_mapper(dtree):
    """
    Map main radar information from a radar dtreeume dataset.

    Parameters:
    - dtree: xarray.Dataset
        Radar dtreeume dataset.

    Returns:
    xarray.Dataset
        Dataset containing the mapped radar information.
    """
    dataset = (
        dtree.root.to_dataset()
        .drop('sweep_group_name')
        .rename({'sweep_fixed_angle': 'fixed_angle'})
    )
    return dataset


def _variable_mapper(dtree, sweep_group_name):
    """
    Map radar variables for different sweep groups.

    Parameters:
    - dtree: xarray.Dataset
        Radar dtreeume dataset.
    - sweep_group_name: xarray.DataArray
        DataArray containing sweep group names.

    Returns:
    xarray.Dataset
        Dataset containing mapped radar variables.
    """
    sweep_group_names = sweep_group_name.values.tolist()

    sweep_datasets = [
        dtree[name]
        .drop_vars(['x', 'y', 'z'])
        .swap_dims({'azimuth': 'time'})
        .to_dataset()
        for name in sweep_group_names
    ]

    result_dataset = xr.concat(
        sweep_datasets,
        dim='time',
        compat='no_conflicts',
        join='right',
        combine_attrs='drop_conflicts'
    )

    result_dataset = result_dataset.drop(
        ["sweep_fixed_angle",
         "sweep_number",
         "sweep_mode",
         "prt_mode",
         "follow_mode"]
    )
    result_dataset = result_dataset.drop(
        ['latitude', 'longitude', 'altitude', 'spatial_ref']
    )

    return result_dataset


def _sweep_info_mapper(dtree):
    """
    Extract specified sweep information variables from a
    radar dtreeume dataset.

    Parameters:
    - dtree: xarray.Dataset
        Radar dtreeume dataset.

    Returns:
    xarray.Dataset
        Dataset containing the specified sweep information variables.
    """
    dataset = xr.Dataset()

    sweep_vars = [
        'sweep_number',
        'sweep_mode',
        'polarization_mode',
        'prt_mode',
        'follow_mode',
        'sweep_fixed_angle',
        'sweep_start_ray_index',
        'sweep_end_ray_index'
    ]

    for var_name in sweep_vars:
        var_data_list = [
            np.unique(
                dtree[s][var_name].values[np.newaxis, ...]
            ) if var_name in dtree[s] else np.array([np.nan])
            for s in dtree.sweep_group_name.values
        ]
        var_attrs_list = [
            dtree[s][var_name]
            .attrs if var_name in dtree[s] else {
            } for s in dtree.sweep_group_name.values
        ]
        var_data = np.concatenate(var_data_list)
        var_attrs = {}
        for attrs in var_attrs_list:
            var_attrs.update(attrs)
        var_data_array = xr.DataArray(
            var_data,
            dims=('sweep',),
            attrs=var_attrs
        )
        dataset[var_name] = var_data_array

    dataset = dataset.rename({'sweep_fixed_angle': 'fixed_angle'})

    return dataset


def calculate_sweep_indices(dtree, dataset=None):
    """
    Calculate sweep start and end ray indices for elevation
    values in a radar dataset.

    Parameters:
    - dtree: xarray.Dataset
        Radar dataset containing elevation values for different sweep groups.
    - dataset: xarray.Dataset, optional
        An optional dataset to which the calculated indices will be added.
        If None, a new dataset will be created.

    Returns:
    xarray.Dataset
        Dataset with sweep start and end ray indices.
    """
    if dataset is None:
        dataset = xr.Dataset()

    sweep_group_names = dtree['sweep_group_name'].values

    sweep_start_ray_index = np.zeros(len(sweep_group_names), dtype='int32')
    sweep_end_ray_index = np.zeros(len(sweep_group_names), dtype='int32')

    cumulative_size = 0

    for i, sweep_name in enumerate(sweep_group_names):
        elevation_var = dtree[sweep_name].elevation

        size = elevation_var.size

        sweep_start_ray_index[i] = cumulative_size
        sweep_end_ray_index[i] = cumulative_size + size - 1

        cumulative_size += size

    dataset['sweep_start_ray_index'] = xr.DataArray(
        sweep_start_ray_index,
        dims=('sweep',),
        attrs={'standard_name': 'index_of_first_ray_in_sweep'}
    )
    dataset['sweep_end_ray_index'] = xr.DataArray(
        sweep_end_ray_index,
        dims=('sweep',),
        attrs={'standard_name': 'index_of_last_ray_in_sweep'}
    )

    return dataset


def to_cfradial1(dtree, filename):
    """
    Convert a radar dtreeume dataset to the CFRadial1 format
    and save it to a file.

    Parameters:
    - dtree: xarray.Dataset
        Radar dtreeume dataset.
    - filename: str
        The name of the output netCDF file.
    """
    radar_params = dtree['radar_parameters'].to_dataset()
    calib_params = dtree['radar_calibration'].to_dataset()
    calibs = _calib_mapper(calib_params)
    radar_info = _main_info_mapper(dtree)
    variables = _variable_mapper(dtree, dtree.sweep_group_name)
    swp_info = _sweep_info_mapper(dtree)
    radar_georef = dtree['georeferencing_correction'].to_dataset()
    for params in [radar_params, calibs, radar_georef, radar_info, swp_info]:
        variables.update(params)
    variables = variables.reset_coords('elevation')
    variables = variables.reset_coords('azimuth')
    dataset = calculate_sweep_indices(dtree, variables)
    dataset.to_netcdf(filename, format='netcdf4')
