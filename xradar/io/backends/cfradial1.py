#!/usr/bin/env python
# Copyright (c) 2022-2024, openradar developers.
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
    "open_cfradial1_datatree",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import numpy as np
from xarray import Dataset, DataTree, merge, open_dataset
from xarray.backends import NetCDF4DataStore
from xarray.backends.common import BackendEntrypoint
from xarray.backends.store import StoreBackendEntrypoint

from ... import util
from ...model import (
    conform_cfradial2_sweep_group,
    georeferencing_correction_subgroup,
    optional_root_attrs,
    optional_root_vars,
    radar_calibration_subgroup,
    radar_parameters_subgroup,
    required_global_attrs,
    required_root_vars,
)
from .common import _attach_sweep_groups, _maybe_decode


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
    anc_dims = set(root.dims) ^ {"time", "range", "sweep", "n_points"}
    anc_dims &= set(root.dims)

    root = root.drop_dims(anc_dims)

    root = root.rename({"fixed_angle": "sweep_fixed_angle"})

    # conform to cfradial2 standard
    data = conform_cfradial2_sweep_group(root, optional, "time")
    data_vars = {
        k
        for k, v in data.data_vars.items()
        if any(d in v.dims for d in ["range", "n_points"])
    }

    # which sweeps to load
    # sweep is assumed a list of strings with elements like "sweep_0"
    sweep_groups = {}
    if isinstance(sweep, str):
        sweep = [sweep]
    elif isinstance(sweep, int):
        sweep = [f"sweep_{sweep}"]

    # iterate over sweeps
    for i in range(root.sizes["sweep"]):
        sw = f"sweep_{i}"
        if sweep is not None and not (sw in sweep or i in sweep):
            continue

        # slice time and sweep dimension
        tslice = slice(start_idx[i], end_idx[i] + 1)
        swslice = slice(i, i + 1)
        ds = data.isel(time=tslice, sweep=swslice).squeeze("sweep")

        ds["sweep_mode"] = _maybe_decode(ds.sweep_mode).compute()
        dim0 = "elevation" if ds["sweep_mode"] == "rhi" else "azimuth"

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
            ds_vars = ds[data_vars]
            ds_vars = merge([ds_vars, ds[[dim0, "range"]]])
            ds_vars = ds_vars.stack(n_points=[dim0, "range"])
            ds_vars = ds_vars.unstack("n_points")
            ds = ds.drop_vars(ds_vars.data_vars)
            ds = merge([ds, ds_vars])

        # assign site_coords
        if site_coords:

            ds = ds.assign_coords(
                {
                    "latitude": root.latitude,
                    "longitude": root.longitude,
                    "altitude": root.altitude,
                }
            )

        # handling first dimension
        # for CfRadial1 first dimension is time
        if first_dim == "auto":
            if "time" in ds.dims:
                ds = ds.swap_dims({"time": dim0})
            ds = ds.sortby(dim0)
        else:
            if "time" not in ds.dims:
                ds = ds.swap_dims({dim0: "time"})
            ds = ds.sortby("time")

        # reassign azimuth/elevation coordinates
        ds = ds.set_coords(["azimuth", "elevation"])

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
    else:
        return Dataset()


def open_cfradial1_datatree(filename_or_obj, **kwargs):
    """Open CfRadial1 dataset as :py:class:`xarray.DataTree`.

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
    engine: str
        Engine that will be passed to Xarray.open_dataset, defaults to "netcdf4"

    Returns
    -------
    dtree: xarray.DataTree
        DataTree with CfRadial2 groups.
    """

    # handle kwargs, extract first_dim
    first_dim = kwargs.pop("first_dim", "auto")
    optional = kwargs.pop("optional", True)
    site_coords = kwargs.pop("site_coords", True)
    sweep = kwargs.pop("sweep", None)
    engine = kwargs.pop("engine", "netcdf4")

    # open root group, cfradial1 only has one group
    # open_cfradial1_datatree only opens the file once using netcdf4
    # and retrieves the different groups from the loaded object
    ds = open_dataset(filename_or_obj, engine=engine, **kwargs)

    # create datatree root node additional root metadata groups
    dtree: dict = {
        "/": _get_required_root_dataset(ds, optional=optional),
        "/radar_parameters": _get_subgroup(ds, radar_parameters_subgroup),
        "/georeferencing_correction": _get_subgroup(
            ds, georeferencing_correction_subgroup
        ),
        "/radar_calibration": _get_radar_calibration(ds),
    }

    # radar_calibration (connected with calib-dimension)
    dtree = _attach_sweep_groups(
        dtree,
        list(
            _get_sweep_groups(
                ds,
                sweep=sweep,
                first_dim=first_dim,
                optional=optional,
                site_coords=site_coords,
            ).values()
        ),
    )
    return DataTree.from_dict(dtree)


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
