#!/usr/bin/env python
# Copyright (c) 2022-2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""

CfRadial2
=========

This sub-module contains the CfRadial2 xarray backend for reading CfRadial2-group-based radar
data into Xarray structures as well as a reader to create a complete datatree.Datatree.

Code ported from wradlib.

Example::

    import xradar as xd
    dtree = xd.io.open_cfradial2_datatree(filename)

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}

"""

__all__ = [
    "CfRadial2BackendEntrypoint",
    "open_cfradial2_datatree",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import numpy as np
from pandas import to_datetime, to_timedelta
from xarray import DataArray, Dataset, DataTree, open_groups
from xarray.backends import NetCDF4DataStore
from xarray.backends.common import BackendEntrypoint
from xarray.backends.store import StoreBackendEntrypoint

from ... import util
from ...model import (
    georeferencing_correction_subgroup,
    optional_root_attrs,
    optional_root_vars,
    radar_calibration_subgroup,
    radar_parameters_subgroup,
    required_global_attrs,
    required_root_vars,
)
from .common import _attach_sweep_groups


def _get_required_root_dataset(tree, optional=True):
    """Extract Root Dataset."""
    # keep only defined mandatory and defined optional variables per default

    ds = tree["/"]
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

    try:
        root["latitude"] = DataArray(
            float(root.attrs["Latitude"]),
            attrs={"standard_name": "latitude", "units": "degrees_north"},
        )
        root["longitude"] = DataArray(
            float(root.attrs["Longitude"]),
            attrs={"standard_name": "longitude", "units": "degrees_east"},
        )

        root["altitude"] = DataArray(
            float(root.attrs["Height"]),
            attrs={"standard_name": "altitude", "units": "meters", "positive": "up"},
        )
    except KeyError:
        print("Dataset does not contain Latitude, Londitude, and Altitude")
        pass

    for k in remove_attrs:
        root.attrs.pop(k, None)

    sweep_names = [f"sweep_{i[-1]}" for i in root["sweep_group_name"].values]
    root = root.drop_vars("sweep_group_name")
    sn = sweep_names[0] if len(sweep_names) == 1 else sweep_names
    # handle sweep variables and dimension
    root["sweep_group_name"] = DataArray(
        np.array(sn, dtype="<U7"),
    )
    # fix dtype in encoding
    root.sweep_group_name.encoding["dtype"] = root.sweep_group_name.dtype
    # remove cf standard name
    root.sweep_group_name.attrs = []

    time = root.attrs["history"].split("on ")[-1]
    time_range = tree["/sweep_0000"].ntime
    base_time = to_datetime(time).tz_localize(None)
    time_series = base_time + to_timedelta(time_range.values, unit="s")
    time_array = DataArray(
        time_series,
        dims=["time"],
        attrs={
            "standard_name": "time",
            "long_name": "time in seconds since volume start",
            "comment": "times are relative to the volume start_time",
        },
    )
    timestamp = time_array[0].values
    iso_string = to_datetime(timestamp).strftime("%Y-%m-%dT%H:%M:%SZ")
    root["time_coverage_start"] = DataArray(
        np.array(iso_string, dtype="S32"),
        attrs={
            "standard_name": "data_volume_start_time_utc",
            "comment": "ray times are relative to start time in secs",
        },
    )
    timestamp = time_array[-1].values
    iso_string = to_datetime(timestamp).strftime("%Y-%m-%dT%H:%M:%SZ")
    root["time_coverage_end"] = DataArray(
        np.array(iso_string, dtype="S32"),
        attrs={
            "standard_name": "data_volume_end_time_utc",
        },
    )
    root["sweep_fixed_angle"] = DataArray(
        float(tree["/sweep_0000"].elevation_deg.mean().values),
        attrs={"standard_name": "beam_target_fixed_angle", "units": "degrees"},
    )
    try:
        root.attrs["scan_name"] = tree["/sweep_0000"].attrs["ScanType"]
    except KeyError:
        pass
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
    root = obj.pop("/")
    for idx, sweep in enumerate(obj.keys()):
        ds = obj[sweep]
        ds = ds.rename(
            {
                "ntime": "time",
                "nrange": "range",
                "elevation_deg": "elevation",
                "azimuth_deg": "azimuth",
            }
        )
        # strip variables and attributes
        anc_dims = set(ds.dims) ^ {"time", "range", "sweep", "n_points"}
        anc_dims &= set(ds.dims)

        ds = ds.drop_dims(anc_dims)
        # conform to cfradial2 standard
        # data = conform_cfradial2_sweep_group(ds, optional, "time")
        # data_vars = {
        #     k
        #     for k, v in data.data_vars.items()
        #     if any(d in v.dims for d in ["range", "n_points"])
        # }

        # which sweeps to load
        # sweep is assumed a list of strings with elements like "sweep_0"
        sweep_groups = {}
        sweep = f"sweep_{idx}"
        try:
            dim0 = "elevation" if ds["sweep_mode"] == "rhi" else "azimuth"
        except KeyError:
            dim0 = "azimuth"

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
        ds.attrs = {}
        sweep_groups[sweep] = ds

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


def open_cfradial2_datatree(filename_or_obj, **kwargs):
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
    # needed for new xarray literal timedelta decoding
    kwargs.update(decode_timedelta=kwargs.pop("decode_timedelta", False))

    # open root group, cfradial1 only has one group
    # open_cfradial1_datatree only opens the file once using netcdf4
    # and retrieves the different groups from the loaded object
    # ds = open_dataset(filename_or_obj, engine=engine, **kwargs)

    tree_groups = open_groups(filename_or_obj, engine=engine, **kwargs)
    tree_groups["/"] = _get_required_root_dataset(tree_groups, optional=optional)
    # create datatree root node additional root metadata groups
    dtree: dict = {
        "/": tree_groups["/"],
        "/radar_parameters": _get_subgroup(
            tree_groups["/sweep_0000"], radar_parameters_subgroup
        ),
        "/georeferencing_correction": _get_subgroup(
            tree_groups["/sweep_0000"], georeferencing_correction_subgroup
        ),
        "/radar_calibration": _get_radar_calibration(tree_groups["/sweep_0000"]),
    }
    dtree = _attach_sweep_groups(
        dtree,
        list(
            _get_sweep_groups(
                tree_groups,
                sweep=sweep,
                first_dim=first_dim,
                optional=optional,
                site_coords=site_coords,
            ).values()
        ),
    )
    return DataTree.from_dict(dtree)


class CfRadial2BackendEntrypoint(BackendEntrypoint):
    """Xarray BackendEntrypoint for CfRadial2 data.

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

    description = "Open CfRadial2 (.nc, .nc4) using netCDF4 in Xarray"
    url = "https://xradar.rtfd.io/en/latest/io.html#cfradial2"

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
        decode_timedelta=False,
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
