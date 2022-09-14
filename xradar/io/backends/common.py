#!/usr/bin/env python
# Copyright (c) 2022, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

import numpy as np
import xarray as xr
from datatree import DataTree


def _maybe_decode(attr):
    try:
        return attr.decode()
    except AttributeError:
        return attr


def _remove_duplicate_rays(ds, store=None):
    dimname = list(ds.dims)[0]
    # find exact duplicates and remove
    _, idx = np.unique(ds[dimname], return_index=True)
    if len(idx) < len(ds[dimname]):
        ds = ds.isel({dimname: idx})
        # if ray_time was erroneously created from wrong dimensions
        # we need to recalculate it
        if store and store._need_time_recalc:
            ray_times = store._get_ray_times(nrays=len(idx))
            # need to decode only if ds is decoded
            if "units" in ds.rtime.encoding:
                ray_times = xr.decode_cf(xr.Dataset({"rtime": ray_times})).rtime
            ds = ds.assign({"rtime": ray_times})
    return ds


def _fix_angle(da):
    # fix elevation outliers
    if len(set(da.values)) > 1:
        med = da.median(skipna=True)
        da = da.where(da == med).fillna(med)
    return da


def _reindex_angle(ds, store=None, force=False, tol=None):
    # Todo: The current code assumes to have PPI's of 360deg and RHI's of 90deg,
    #       make this work also for sectorized (!!!) measurements
    #       this needs refactoring, it's too complex
    if tol is True or tol is None:
        tol = 0.4
    # disentangle different functionality
    full_range = {"azimuth": 360, "elevation": 90}
    dimname = list(ds.dims)[0]
    # sort in any case, to prevent unsorted errors
    ds = ds.sortby(dimname)
    # fix angle range for rhi
    if hasattr(ds, "elevation_upper_limit"):
        ul = np.rint(ds.elevation_upper_limit)
        full_range["elevation"] = ul

    secname = {"azimuth": "elevation", "elevation": "azimuth"}.get(dimname)
    dim = ds[dimname]
    diff = dim.diff(dimname)
    # this captures different angle spacing
    # catches also missing rays and double rays
    # and other erroneous ray alignments which result in different diff values
    diffset = set(diff.values)
    non_uniform_angle_spacing = len(diffset) > 1
    # this captures missing and additional rays in case the angle differences
    # are equal
    non_full_circle = False
    if not non_uniform_angle_spacing:
        res = list(diffset)[0]
        non_full_circle = ((res * ds.dims[dimname]) % full_range[dimname]) != 0

    # fix issues with ray alignment
    if force | non_uniform_angle_spacing | non_full_circle:
        # create new array and reindex
        if store and hasattr(store, "angle_resolution"):
            res = store.angle_resolution
        elif hasattr(ds[dimname], "angle_res"):
            res = ds[dimname].angle_res
        else:
            res = diff.median(dimname).values
        new_rays = int(np.round(full_range[dimname] / res, decimals=0))
        # find exact duplicates and remove
        ds = _remove_duplicate_rays(ds, store=store)

        # do we have all needed rays?
        if non_uniform_angle_spacing | len(ds[dimname]) != new_rays:
            # todo: check if assumption that beam center points to
            #       multiples of res/2. is correct in any case
            # it might fail for cfradial1 data which already points to beam centers
            azr = np.arange(res / 2.0, new_rays * res, res, dtype=diff.dtype)
            fill_value = {
                k: np.asarray(v._FillValue).astype(v.dtype)
                for k, v in ds.items()
                if hasattr(v, "_FillValue")
            }
            ds = ds.reindex(
                {dimname: azr},
                method="nearest",
                tolerance=tol,
                fill_value=fill_value,
            )

        # check other coordinates
        # check secondary angle coordinate (no nan)
        # set nan values to reasonable median
        if hasattr(ds, secname) and np.count_nonzero(np.isnan(ds[secname])):
            ds[secname] = ds[secname].fillna(ds[secname].median(skipna=True))
        # todo: rtime is also affected, might need to be treated accordingly

    return ds


def _attach_sweep_groups(dtree, sweeps):
    """Attach sweep groups to DataTree."""
    for i, sw in enumerate(sweeps):
        DataTree(sw, name=f"sweep_{i}", parent=dtree)
    return dtree
