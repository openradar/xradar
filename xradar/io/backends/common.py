#!/usr/bin/env python
# Copyright (c) 2022-2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Common Backend Functions
========================

This submodule contains helper functions for data and metadata alignment.

Currently, all private and not part of the public API.

"""

import io
import struct
from collections import OrderedDict

import h5netcdf
import numpy as np
import xarray as xr

from ...model import (
    optional_root_attrs,
    optional_root_vars,
    required_global_attrs,
    required_root_vars,
)


def _maybe_decode(attr):
    try:
        # Decode the xr.DataArray differently than a byte string
        if type(attr) is xr.core.dataarray.DataArray:
            decoded_attr = attr.astype(str).str.rstrip()
        else:
            decoded_attr = attr.decode()
        return decoded_attr
    except AttributeError:
        return attr


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


def _attach_sweep_groups(dtree, sweeps):
    """Attach sweep groups to DataTree."""
    for i, sw in enumerate(sweeps):
        # remove attributes only from Dataset's not DataArrays
        dtree[f"sweep_{i}"] = xr.DataTree(sw.drop_attrs(deep=False))
    return dtree


def _get_h5group_names(filename, engine):
    if engine == "odim":
        groupname = "dataset"
        off = 1
    elif engine == "gamic":
        groupname = "scan"
        off = 0
    else:
        raise ValueError(f"xradar: unknown engine `{engine}`.")
    with h5netcdf.File(filename, "r", decode_vlen_strings=True) as fh:
        groups = ["/".join(["", grp]) for grp in fh.groups if groupname in grp.lower()]
        # h5py/h5netcdf might return groups with alphanumeric sorting
        # just sort in any case
        groups = sorted(groups, key=lambda x: int(x[len(groupname) + 1 :]))
        groups = [f"sweep_{int(sw[len(groupname) + 1 :]) - off}" for sw in groups]
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
    # reset_coords as root doesn't have coordinates
    root = root.assign(
        {
            "volume_number": 0,
            "platform_type": "fixed",
            "instrument_type": "radar",
            "time_coverage_start": time_coverage_start_str,
            "time_coverage_end": time_coverage_end_str,
            "latitude": sweeps[1]["latitude"],
            "longitude": sweeps[1]["longitude"],
            "altitude": sweeps[1]["altitude"],
        }
    ).reset_coords()

    # assign root attributes
    attrs = {}
    attrs["Conventions"] = sweeps[0].attrs.get("Conventions", "None")
    attrs["instrument_name"] = sweeps[0].attrs.get("instrument_name", "None")
    comment = sweeps[0].attrs.get("comment", None)
    attrs.update(
        {
            "version": "None",
            "title": "None",
            "institution": "None",
            "references": "None",
            "source": "None",
            "history": "None",
            "comment": "im/exported using xradar",
        }
    )
    if comment is not None:
        attrs["comment"] = attrs["comment"] + ",\n" + comment
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


def _get_required_root_dataset(ls_ds, optional=True):
    """Extract Root Dataset."""
    # keep only defined mandatory and defined optional variables per default
    # by checking in all nodes
    data_var = {x for xs in [sweep.variables.keys() for sweep in ls_ds] for x in xs}
    remove_root = set(data_var) ^ set(required_root_vars)
    if optional:
        remove_root ^= set(optional_root_vars)
    remove_root ^= {"sweep_number", "fixed_angle"}
    remove_root &= data_var
    # ignore errors for variables which exist in one sweep but not the other
    root = [sweep.drop_vars(remove_root, errors="ignore") for sweep in ls_ds]
    root_vars = {x for xs in [sweep.variables.keys() for sweep in root] for x in xs}
    # rename variables
    # todo: find a more easy method not iterating over all variables
    for k in root_vars:
        rename = optional_root_vars.get(k, None)
        if rename:
            root = [sweep.rename_vars({k: rename}) for sweep in root]

    ds_vars = [sweep[root_vars] for sweep in ls_ds]
    _vars = xr.concat(ds_vars, dim="sweep").reset_coords()

    # Creating the root group using _assign_root function
    ls = ls_ds.copy()
    ls.insert(0, xr.Dataset())
    root = _assign_root(ls)

    # merging both the created and the variables within each dataset
    root = xr.merge([root, _vars], compat="override")

    attrs = root.attrs.keys()
    remove_attrs = set(attrs) ^ set(required_global_attrs)
    if optional:
        remove_attrs ^= set(optional_root_attrs)
    for k in remove_attrs:
        root.attrs.pop(k, None)
    # Renaming variable
    if "sweep_number" in data_var and "sweep_group_name" not in data_var:
        root = root.rename_vars({"sweep_number": "sweep_group_name"})
    elif "sweep_group_name" in data_var:
        root["sweep_group_name"].values = np.array(
            [f"sweep_{i}" for i in range(len(root["sweep_group_name"].values))]
        )
    return root


def _get_subgroup(ls_ds: list[xr.Dataset], subdict):
    """Get iris-sigmet root metadata group.
    Variables are fetched from the provided Dataset according to the subdict dictionary.
    """
    meta_vars = subdict
    data_vars = {x for xs in [ds.variables.keys() for ds in ls_ds] for x in xs}
    extract_vars = set(data_vars) & set(meta_vars)
    subgroup = xr.merge([ds[extract_vars] for ds in ls_ds])
    for k in subgroup.data_vars:
        rename = meta_vars[k]
        if rename:
            subgroup = subgroup.rename_vars({k: rename})
    subgroup.attrs = {}
    return subgroup


def _get_radar_calibration(ls_ds: list[xr.Dataset], subdict: dict) -> xr.Dataset:
    """Get radar calibration root metadata group."""
    meta_vars = subdict
    data_vars = {x for xs in [ds.attrs for ds in ls_ds] for x in xs}
    extract_vars = set(data_vars) & set(meta_vars)
    if extract_vars:
        var_dict = {var: ls_ds[0].attrs[var] for var in extract_vars}
        return xr.Dataset({key: xr.DataArray(value) for key, value in var_dict.items()})
    else:
        return xr.Dataset()


# IRIS Data Types and corresponding python struct format characters
# 4.2 Scalar Definitions, Page 23
# https://docs.python.org/3/library/struct.html#format-characters
# also used for Furuno data types
SINT2 = {"fmt": "h", "dtype": "int16"}
SINT4 = {"fmt": "i", "dtype": "int32"}
UINT1 = {"fmt": "B", "dtype": "unit8"}
UINT2 = {"fmt": "H", "dtype": "uint16"}
UINT4 = {"fmt": "I", "dtype": "unint32"}
