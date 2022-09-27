#!/usr/bin/env python
# Copyright (c) 2022, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Common Backend Functions
========================

This submodule contains helper functions for data and metadata alignment.

Currently all private and not part of the public API.

"""

import struct
from collections import OrderedDict

import io

import h5netcdf
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


def _calculate_angle_res(dim):
    # need to sort dim first
    angle_diff = np.diff(sorted(dim))
    angle_diff2 = np.abs(np.diff(angle_diff))

    # only select angle_diff, where angle_diff2 is less than 0.1 deg
    # Todo: currently 0.05 is working in most cases
    #  make this robust or parameterisable
    angle_diff_wanted = angle_diff[:-1][angle_diff2 < 0.05]
    return np.round(np.nanmean(angle_diff_wanted), decimals=2)


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


def _get_h5group_names(filename, engine):
    if engine == "odim":
        groupname = "dataset"
    elif engine == "gamic":
        groupname = "scan"
    elif engine == "cfradial2":
        groupname = "sweep"
    else:
        raise ValueError(f"xradar: unknown engine `{engine}`.")
    with h5netcdf.File(filename, "r", decode_vlen_strings=True) as fh:
        groups = ["/".join(["", grp]) for grp in fh.groups if groupname in grp.lower()]
    if isinstance(filename, io.BytesIO):
        filename.seek(0)
    return groups


def _assign_root(sweeps):
    """(Re-)Create root object according CfRadial2 standard"""
    # extract time coverage
    times = np.array(
        [[ts.time.values.min(), ts.time.values.max()] for ts in sweeps[1:]]
    ).flatten()
    time_coverage_start = min(times)
    time_coverage_end = max(times)

    time_coverage_start_str = str(time_coverage_start)[:19] + "Z"
    time_coverage_end_str = str(time_coverage_end)[:19] + "Z"

    # create root group from scratch
    root = xr.Dataset()  # data_vars=wrl.io.xarray.global_variables,
    # attrs=wrl.io.xarray.global_attrs)

    # take first dataset/file for retrieval of location
    # site = self.site

    # assign root variables
    root = root.assign(
        {
            "volume_number": 0,
            "platform_type": str("fixed"),
            "instrument_type": "radar",
            "time_coverage_start": time_coverage_start_str,
            "time_coverage_end": time_coverage_end_str,
            "latitude": sweeps[1]["latitude"].data,
            "longitude": sweeps[1]["longitude"].data,
            "altitude": sweeps[1]["altitude"].data,
        }
    )

    # assign root attributes
    attrs = {}
    attrs["Conventions"] = sweeps[0].attrs.get("Conventions", "None")
    attrs.update(
        {
            "version": "None",
            "title": "None",
            "institution": "None",
            "references": "None",
            "source": "None",
            "history": "None",
            "comment": "im/exported using xradar",
            "instrument_name": "None",
        }
    )
    root = root.assign_attrs(attrs)
    # todo: pull in only CF attributes
    root = root.assign_attrs(sweeps[1].attrs)
    return root


def _get_fmt_string(dictionary, retsub=False, byte_order="<"):
    """Get Format String from given dictionary.

    Parameters
    ----------
    dictionary : dict
        Dictionary containing data structure with fmt-strings.
    retsub : bool
        If True, return sub structures.

    Returns
    -------
    fmt : str
        struct format string
    sub : dict
        Dictionary containing substructure
    """
    fmt = f"{byte_order}"
    if retsub:
        sub = OrderedDict()
    for k, v in dictionary.items():
        try:
            fmt += v["fmt"]
        except KeyError:
            # remember sub-structures
            if retsub:
                sub[k] = v
            if "size" in v:
                fmt += v["size"]
            else:
                fmt += f"{struct.calcsize(_get_fmt_string(v))}s"
    if retsub:
        return fmt, sub
    else:
        return fmt


def _unpack_dictionary(buffer, dictionary, rawdata=False):
    """Unpacks binary data using the given dictionary structure.

    Parameters
    ----------
    buffer : array-like
    dictionary : dict
        data structure in dictionary, keys are names and values are structure formats

    Returns
    -------
    data : dict
        Ordered Dictionary with unpacked data
    """
    # get format and substructures of dictionary
    fmt, sub = _get_fmt_string(dictionary, retsub=True)

    # unpack into OrderedDict
    data = OrderedDict(zip(dictionary, struct.unpack(fmt, buffer)))

    # remove spares
    if not rawdata:
        keys_to_remove = [k for k in data.keys() if k.startswith("spare")]
        keys_to_remove.extend([k for k in data.keys() if k.startswith("reserved")])
        for k in keys_to_remove:
            data.pop(k, None)

    # iterate over sub dictionary and unpack/read/decode
    for k, v in sub.items():
        if not rawdata:
            # read/decode data
            for k1 in ["read", "func"]:
                try:
                    # print("K/V:", k, v)
                    data[k] = v[k1](data[k], **v[k1[0] + "kw"])
                except KeyError:
                    pass
                except UnicodeDecodeError:
                    pass
        # unpack sub dictionary
        try:
            data[k] = _unpack_dictionary(data[k], v, rawdata=rawdata)
        except TypeError:
            pass

    return data


# IRIS Data Types and corresponding python struct format characters
# 4.2 Scalar Definitions, Page 23
# https://docs.python.org/3/library/struct.html#format-characters
# also used for Furuno data types
SINT2 = {"fmt": "h", "dtype": "int16"}
SINT4 = {"fmt": "i", "dtype": "int32"}
UINT1 = {"fmt": "B", "dtype": "unit8"}
UINT2 = {"fmt": "H", "dtype": "uint16"}
UINT4 = {"fmt": "I", "dtype": "unint32"}




