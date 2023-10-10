#!/usr/bin/env python
# Copyright (c) 2022, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""

CfRadial1
=========

This sub-module contains the CfRadial1 xarray backend for reading CfRadial1-based radar
data into Xarray structures as well as a reader to create a complete datatree.Datatree.

Code ported from wradlib.

Example::

    import xradar as xd
    dtree = xd.io.open_cfradial1_datatree(filename)

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}

"""

__all__ = [
    "CfRadial1BackendEntrypoint",
    "to_cfradial2",
    "to_cfradial1",
]

__doc__ = __doc__.format("\n   ".join(__all__))

from typing import Mapping, Optional
import numpy as np
from datatree import DataTree
from xarray import open_dataset
from xarray.backends import NetCDF4DataStore
from xarray.backends.common import BackendEntrypoint
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset

from .. import util
from ..model import (
    conform_cfradial2_sweep_group,
    georeferencing_correction_subgroup,
    optional_root_attrs,
    optional_root_vars,
    radar_calibration_subgroup,
    radar_parameters_subgroup,
    required_global_attrs,
    required_root_vars,
)
from ..io.backends.common import _attach_sweep_groups, _maybe_decode


def _get_required_root_dataset(ds, optional=True):
    """Extract Root Dataset."""
    # keep only defined mandatory and defined optional variables per default
    var = ds.variables.keys()
    remove_root = set(var) ^ set(required_root_vars)
    if optional:
        remove_root ^= set(optional_root_vars)
    remove_root ^= {"sweep_number", "fixed_angle"}
    remove_root &= var
    root = ds.drop_vars(remove_root)

    # rename variables
    # todo: find a more easy method not iterating over all variables
    for k in root.data_vars:
        rename = optional_root_vars.get(k, None)
        if rename:
            root = root.rename_vars({k: rename})

    # keep only defined mandatory and defined optional attributes per default
    attrs = root.attrs.keys()
    remove_attrs = set(attrs) ^ set(required_global_attrs)
    if optional:
        remove_attrs ^= set(optional_root_attrs)
    for k in remove_attrs:
        root.attrs.pop(k, None)

    # handle sweep variables and dimension
    root = root.rename_vars(
        {"sweep_number": "sweep_group_name", "fixed_angle": "sweep_fixed_angle"}
    )
    root["sweep_group_name"].values = np.array(
        [f"sweep_{i}" for i in root["sweep_group_name"].values]
    )
    # fix dtype in encoding
    root.sweep_group_name.encoding["dtype"] = root.sweep_group_name.dtype
    # remove cf standard name
    root.sweep_group_name.attrs = []

    return root


def _get_sweep_groups(
    obj, sweep=None, first_dim="auto", optional=True, site_coords=True
):
    """Extract Sweep Groups.

    Parameters
    ----------
    obj : xarray.Dataset
        CfRadial1 Dataset

    Keyword Arguments
    -----------------
    sweep : str, int, optional
        Sweep/Group to extract, default to None.
    first_dim : str
        Can be ``time`` or ``auto`` first dimension. If set to ``auto``,
        first dimension will be either ``azimuth`` or ``elevation`` depending on
        type of sweep. Defaults to ``auto``.
    optional : bool
        Import optional mandatory data and metadata, defaults to ``True``.
    site_coords : bool
        Attach radar site-coordinates to Dataset, defaults to ``True``.

    Ported from wradlib.
    """
    # get hold of sweep start/stop indices and remove variables
    start_idx = obj.sweep_start_ray_index.values.astype(int)
    end_idx = obj.sweep_end_ray_index.values.astype(int)
    root = obj.drop_vars(["sweep_start_ray_index", "sweep_end_ray_index"])

    ray_n_gates = root.get("ray_n_gates", False)
    ray_start_index = root.get("ray_start_index", False)

    # strip variables and attributes
    anc_dims = list(set(root.dims) ^ {"time", "range", "sweep"})
    root = root.drop_dims(anc_dims)

    root = root.rename({"fixed_angle": "sweep_fixed_angle"})

    # conform to cfradial2 standard
    data = conform_cfradial2_sweep_group(root, optional, "time")

    # which sweeps to load
    # sweep is assumed a list of strings with elements like "sweep_0"
    sweep_groups = {}
    if isinstance(sweep, str):
        sweep = [sweep]
    elif isinstance(sweep, int):
        sweep = [f"sweep_{sweep}"]

    # iterate over sweeps
    for i in range(root.dims["sweep"]):
        sw = f"sweep_{i}"
        if sweep is not None and not (sw in sweep or i in sweep):
            continue

        # slice time and sweep dimension
        tslice = slice(start_idx[i], end_idx[i] + 1)
        swslice = slice(i, i + 1)
        ds = data.isel(time=tslice, sweep=swslice).squeeze("sweep")

        sweep_mode = _maybe_decode(ds.sweep_mode).compute()
        dim0 = "elevation" if sweep_mode == "rhi" else "azimuth"

        # check and extract for variable number of gates
        if ray_n_gates is not False:
            # swap dimensions to correctly stack/unstack n_points = ["time", "range"]
            ds = ds.swap_dims({"time": dim0})

            current_ray_n_gates = ray_n_gates.isel(time=tslice)
            current_rays_sum = current_ray_n_gates.sum().values.astype(int)
            nslice = slice(
                ray_start_index[start_idx[i]].values.astype(int),
                ray_start_index[start_idx[i]].values.astype(int) + current_rays_sum,
            )
            rslice = slice(0, current_ray_n_gates[0].values.astype(int))
            ds = ds.isel(range=rslice)
            ds = ds.isel(n_points=nslice)
            ds = ds.stack(n_points=[dim0, "range"])
            ds = ds.unstack("n_points")
            # fix elevation/time additional range dimension in coordinate
            ds = ds.assign_coords({"elevation": ds.elevation.isel(range=0, drop=True)})

        # handling first dimension
        # for CfRadial1 first dimension is time
        if first_dim == "auto":
            ds = ds.swap_dims({"time": dim0})
            ds = ds.sortby(dim0)

        # reassign azimuth/elevation coordinates
        ds = ds.assign_coords({"azimuth": ds.azimuth})
        ds = ds.assign_coords({"elevation": ds.elevation})

        # assign site_coords
        if site_coords:
            ds = ds.assign_coords(
                {
                    "latitude": root.latitude,
                    "longitude": root.longitude,
                    "altitude": root.altitude,
                }
            )

        sweep_groups[sw] = ds

    return sweep_groups


def _get_cfradial2_group(
    ds, sweep="sweep_0", first_dim="time", optional=True, site_coords=False
):
    """Assign CfRadial2 group from CfRadial1 data structure.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset of CfRadial1 file
    sweep : str, int, optional
        Sweep/Group to extract, default to first sweep.
    first_dim : str
        Can be ``time`` or ``auto`` first dimension. If set to ``auto``,
        first dimension will be either ``azimuth`` or ``elevation`` depending on
        type of sweep. Defaults to ``auto``.
    optional : bool
        Import optional mandatory data and metadata, defaults to ``True``.
    site_coords : bool
        Attach radar site-coordinates to Dataset, defaults to ``True``.

    Returns
    -------
    sweep : xarray.Dataset
        CfRadial2 group.
    """
    if sweep in ["/", "root"]:
        return _get_required_root_dataset(ds, optional=optional)
    elif sweep in ["radar_calibration", "calib", "r_calib"]:
        return _get_radar_calibration(ds)
    elif sweep in ["radar_parameters"]:
        return _get_subgroup(ds, radar_parameters_subgroup)
    elif sweep in ["georeferencing_correction"]:
        return _get_subgroup(ds, georeferencing_correction_subgroup)
    elif "sweep" in sweep:
        return list(
            _get_sweep_groups(
                ds,
                sweep,
                first_dim=first_dim,
                optional=optional,
                site_coords=site_coords,
            ).values()
        )[0]
    else:
        return None


def _get_subgroup(ds, subdict):
    """Get CfRadial2 root metadata group.

    Variables are fetched from the provided Dataset according to the subdict dictionary.
    """
    meta_vars = subdict
    extract_vars = set(ds.data_vars) & set(meta_vars)
    subgroup = ds[extract_vars]
    for k in subgroup.data_vars:
        rename = meta_vars[k]
        if rename:
            subgroup = subgroup.rename_vars({k: rename})
    subgroup.attrs = {}
    return subgroup


def _get_radar_calibration(ds):
    """Get radar calibration root metadata group."""
    # radar_calibration is connected with calib-dimension
    calib = [anc for anc in ds.dims if "calib" in anc]
    if calib:
        # extract group variables
        subgroup = ds.drop_dims(list(set(ds.dims) ^ {calib[0]}))
        drop = [k for k, v in subgroup.variables.items() if calib[0] not in v.dims]
        subgroup = subgroup.drop_vars(drop).squeeze(calib[0])
        calib_vars = {}
        for name in subgroup.data_vars:
            item = next(
                filter(lambda x: x[0] in name, radar_calibration_subgroup.items())
            )
            item = item[1] if item[1] else item[0]
            calib_vars[name] = item
        subgroup = subgroup.rename_vars(calib_vars)
        subgroup.attrs = {}
        return subgroup


class Cfradial2(DataTree):
    def __init__(self, data: Dataset | DataArray | None = None, parent: DataTree | None = None, children: Mapping[str, DataTree] | None = None, name: str | None = None):
        super().__init__(data, parent, children, name)
    def to_cfradial1_dataset(self):
        ds = to_cfradial1(self)
        return ds


def to_cfradial2(ds,  **kwargs):
    """Open CfRadial1 dataset as :py:class:`datatree.DataTree`.

    Parameters
    ----------
    filename_or_obj : str, Path, file-like or xarray.DataStore
        Strings and Path objects are interpreted as a path to a local or remote
        radar file

    Keyword Arguments
    -----------------
    sweep : int, list of int, optional
        Sweep number(s) to extract, default to first sweep. If None, all sweeps are
        extracted into a list.
    first_dim : str
        Can be ``time`` or ``auto`` first dimension. If set to ``auto``,
        first dimension will be either ``azimuth`` or ``elevation`` depending on
        type of sweep. Defaults to ``auto``.
    reindex_angle : bool or dict
        Defaults to False, no reindexing. Given dict should contain the kwargs to
        reindex_angle. Only invoked if `decode_coord=True`.
    fix_second_angle : bool
        If True, fixes erroneous second angle data. Defaults to ``False``.
    optional : bool
        Import optional mandatory data and metadata, defaults to ``True``.
    site_coords : bool
        Attach radar site-coordinates to Dataset, defaults to ``True``.

    Returns
    -------
    dtree: datatree.DataTree
        DataTree with CfRadial2 groups.
    """
    # handle kwargs, extract first_dim
    first_dim = kwargs.pop("first_dim", "auto")
    optional = kwargs.pop("optional", True)
    site_coords = kwargs.pop("site_coords", True)
    sweep = kwargs.pop("sweep", None)

    # open root group, cfradial1 only has one group
    # open_cfradial1_datatree only opens the file once using netcdf4
    # and retrieves the different groups from the loaded object
    # ds = open_dataset(filename_or_obj, engine="netcdf4", **kwargs)

    # create datatree root node with required data
    root = _get_required_root_dataset(ds, optional=optional)
    # dtree = DataTree(data=root, name="root")
    dtree = Cfradial2(data=root, name="root")

    # additional root metadata groups
    # radar_parameters
    subgroup = _get_subgroup(ds, radar_parameters_subgroup)
    DataTree(subgroup, name="radar_parameters", parent=dtree)

    # radar_calibration (connected with calib-dimension)
    calib = _get_radar_calibration(ds)
    if calib:
        DataTree(calib, name="radar_calibration", parent=dtree)

    # georeferencing_correction
    subgroup = _get_subgroup(ds, georeferencing_correction_subgroup)
    DataTree(subgroup, name="georeferencing_correction", parent=dtree)

    # return datatree with attached sweep child nodes
    dtree = _attach_sweep_groups(
        dtree,
        list(
            _get_sweep_groups(
                ds,
                sweep=sweep,
                first_dim=first_dim,
                optional=optional,
                site_coords=False,
            ).values()
        ),
    )

    return dtree


class CfRadial1BackendEntrypoint(BackendEntrypoint):
    """Xarray BackendEntrypoint for CfRadial1 data.

    Keyword Arguments
    -----------------
    first_dim : str
        Can be ``time`` or ``auto`` first dimension. If set to ``auto``,
        first dimension will be either ``azimuth`` or ``elevation`` depending on
        type of sweep. Defaults to ``auto``.
    reindex_angle : bool or dict
        Defaults to False, no reindexing. Given dict should contain the kwargs to
        reindex_angle. Only invoked if `decode_coord=True`.
    fix_second_angle : bool
        For PPI only. If True, fixes erroneous second angle data. Defaults to False.
    site_coords : bool
        Attach radar site-coordinates to Dataset, defaults to ``True``.
    kwargs : dict
        Additional kwargs are fed to :py:func:`xarray.open_dataset`.
    """

    description = "Open CfRadial1 (.nc, .nc4) using netCDF4 in Xarray"
    url = "https://xradar.rtfd.io/en/latest/io.html#cfradial1"

    def open_dataset(
        self,
        filename_or_obj,
        *,
        mask_and_scale=True,
        decode_times=True,
        concat_characters=True,
        decode_coords=True,
        drop_variables=None,
        use_cftime=None,
        decode_timedelta=None,
        format=None,
        group="/",
        first_dim="auto",
        reindex_angle=False,
        fix_second_angle=False,
        site_coords=True,
        optional=True,
    ):
        store = NetCDF4DataStore.open(
            filename_or_obj,
            format=format,
            group=None,
        )

        store_entrypoint = StoreBackendEntrypoint()

        ds0 = store_entrypoint.open_dataset(
            store,
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            concat_characters=concat_characters,
            decode_coords=decode_coords,
            drop_variables=drop_variables,
            use_cftime=use_cftime,
            decode_timedelta=decode_timedelta,
        )

        ds = _get_cfradial2_group(
            ds0,
            sweep=group,
            first_dim=first_dim,
            optional=optional,
            site_coords=site_coords,
        )
        if ds is None:
            raise ValueError(f"Group `{group}` missing from file `{filename_or_obj}`.")

        if decode_coords and reindex_angle is not False:
            ds = ds.pipe(util.remove_duplicate_rays)
            ds = ds.pipe(util.reindex_angle, **reindex_angle)
            ds = ds.pipe(util.ipol_time, **reindex_angle)

        return ds



from collections.abc import Mapping
from importlib.metadata import version
from typing import Any

import numpy as np
import xarray as xr
from xarray import Dataset

class Cfradial1(Dataset):
    __slots__ = ("__all__")

    def __init__(self, data_vars: Mapping[Any, Any] | None = None, coords: Mapping[Any, Any] | None = None, attrs: Mapping[Any, Any] | None = None) -> None:
        super().__init__(data_vars, coords, attrs)
    def to_cfradial2_datatree(self):
        dtree = to_cfradial2(self)
        return dtree


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
    vol_info = _main_info_mapper(dtree).drop_vars("fixed_angle",)
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


def to_cfradial1(dtree=None, calibs=True):
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
    dataset.attrs["version"] = "1.4"
    dataset = Cfradial1(dataset)
    return dataset