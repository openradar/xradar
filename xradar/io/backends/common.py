#!/usr/bin/env python
# Copyright (c) 2022, openradar developers.
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
from datatree import DataTree


def _maybe_decode(attr):
    try:
        return attr.decode()
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
        DataTree(sw, name=f"sweep_{i}", parent=dtree)
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
