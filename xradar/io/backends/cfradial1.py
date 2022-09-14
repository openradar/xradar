#!/usr/bin/env python
# Copyright (c) 2022, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""

CfRadial1
=========

CfRadial1 Backend for reading CfRadial1-based radar data into Xarray structures.

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

from datatree import DataTree
from xarray import open_dataset
from xarray.backends import NetCDF4DataStore
from xarray.backends.common import BackendEntrypoint
from xarray.backends.store import StoreBackendEntrypoint

from ...model import (
    non_standard_sweep_dataset_vars,
    required_global_attrs,
    required_root_vars,
    required_sweep_metadata_vars,
    sweep_coordinate_vars,
    sweep_dataset_vars,
)
from .common import _attach_sweep_groups, _maybe_decode


def _get_required_root_dataset(ds):
    """Extract Root Dataset."""
    # keep only mandatory variables
    var = ds.variables.keys()
    remove_root = var ^ required_root_vars
    remove_root &= var
    root = ds.drop_vars(remove_root)
    attrs = root.attrs.keys()

    # keep only mandatory attributes
    remove_attrs = (attrs ^ required_global_attrs) & attrs
    for k in remove_attrs:
        root.attrs.pop(k)
    return root


def _get_sweep_groups(root, sweep=None, first_dim="time"):
    """Extract Sweep Groups.

    Ported from wradlib.
    """
    # get hold of sweep start/stop indices
    start_idx = root.sweep_start_ray_index.values.astype(int)
    end_idx = root.sweep_end_ray_index.values.astype(int)
    ray_n_gates = root.get("ray_n_gates", False)
    ray_start_index = root.get("ray_start_index", False)

    # strip variables and attributes
    var = root.variables.keys()
    keep_vars = (
        sweep_coordinate_vars
        | required_sweep_metadata_vars
        | sweep_dataset_vars
        | non_standard_sweep_dataset_vars
    )
    remove_vars = var ^ keep_vars
    remove_vars &= var
    data = root.drop_vars(remove_vars)
    data.attrs = {}
    sweep_groups = []
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
        if first_dim == "auto":
            if "time" in ds.dims:
                ds = ds.swap_dims({"time": dim0})
            ds = ds.sortby(dim0)
        else:
            if "time" not in ds.dims:
                ds = ds.swap_dims({dim0: "time"})
            ds = ds.sortby("time")

        # reassign azimuth/elevation coordinates
        ds = ds.assign_coords({"azimuth": ds.azimuth})
        ds = ds.assign_coords({"elevation": ds.elevation})

        # assign geo-coords
        ds = ds.assign_coords(
            {
                "latitude": root.latitude,
                "longitude": root.longitude,
                "altitude": root.altitude,
            }
        )

        sweep_groups.append(ds)

    return sweep_groups


def _assign_data_radial(root, sweep="sweep_0", first_dim="time"):
    """Assign from CfRadial1 data structure.

    Parameters
    ----------
    root : xarray.Dataset
        Dataset of CfRadial1 file
    sweep : int, optional
        Sweep number to extract, default to first sweep. If None, all sweeps are
        extracted into a list.

    Returns
    -------
    sweeps : list
        List of Sweep Datasets
    """
    sweeps = _get_sweep_groups(root, sweep, first_dim=first_dim)
    return sweeps


def open_cfradial1_datatree(filename_or_obj, **kwargs):
    """Open CfRadial1 dataset as xradar Datatree.

    Parameters
    ----------
    filename_or_obj : str, Path, file-like or DataStore
        Strings and Path objects are interpreted as a path to a local or remote
        radar file

    Keyword Arguments
    -----------------
    first_dim : str
        Default to 'time' as first dimension. If set to 'auto', first dimension will
        be either 'azimuth' or 'elevation' depending on type of sweep.
    sweep : int, list of int, optional
        Sweep number(s) to extract, default to first sweep. If None, all sweeps are
        extracted into a list.
    kwargs :  kwargs
        Additional kwargs are fed to `xr.open_dataset`.

    Returns
    -------
    dtree: DataTree
        DataTree
    """
    # handle kwargs, extract first_dim
    first_dim = kwargs.get("first_dim", None)
    sweep = kwargs.pop("sweep", None)

    # open root group, cfradial1 only has one group
    ds = open_dataset(filename_or_obj, engine="cfradial1", **kwargs)
    # create datatree root node with required data
    dtree = DataTree(data=_get_required_root_dataset(ds), name="root")
    # return datatree with attached sweep child nodes
    return _attach_sweep_groups(
        dtree, _get_sweep_groups(ds, sweep=sweep, first_dim=first_dim)
    )


class CfRadial1BackendEntrypoint(BackendEntrypoint):
    """Xarray BackendEntrypoint for CfRadial1 data.

    Keyword Arguments
    -----------------
    first_dim : str
        Default to 'time' as first dimension. If set to 'auto', first dimension will
        be either 'azimuth' or 'elevation' depending on type of sweep.

    Ported from wradlib.
    """

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
        first_dim="time",
    ):

        store = NetCDF4DataStore.open(
            filename_or_obj,
            format=format,
            group=None,
        )

        store_entrypoint = StoreBackendEntrypoint()

        ds = store_entrypoint.open_dataset(
            store,
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            concat_characters=concat_characters,
            decode_coords=decode_coords,
            drop_variables=drop_variables,
            use_cftime=use_cftime,
            decode_timedelta=decode_timedelta,
        )

        if group != "/":
            ds = _assign_data_radial(ds, sweep=group, first_dim=first_dim)
            if not ds:
                raise ValueError(
                    f"Group `{group}` missing from file `{filename_or_obj}`."
                )
            ds = ds[0]
        return ds
