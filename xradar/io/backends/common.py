#!/usr/bin/env python
# Copyright (c) 2022, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Common Backend Functions
========================

This submodule contains helper functions for data and metadata alignment.

Currently, all private and not part of the public API.

"""

import bz2
import gzip
import io
import itertools
import struct
from collections import OrderedDict
from collections.abc import MutableMapping

import fsspec
import h5netcdf
import numpy as np
import xarray as xr
from datatree import DataTree


def _maybe_decode(attr):
    try:
        # Decode the xr.DataArray differently than a byte string
        if type(attr) == xr.core.dataarray.DataArray:
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


def prepare_for_read(filename, storage_options={"anon": True}):
    """
    Return a file like object read for reading.

    Open a file for reading in binary mode with transparent decompression of
    Gzip and BZip2 files. The resulting file-like object should be closed.

    Parameters
    ----------
    filename : str or file-like object
        Filename or file-like object which will be opened. File-like objects
        will not be examined for compressed data.

    storage_options : dict, optional
        Parameters passed to the backend file-system such as Google Cloud Storage,
        Amazon Web Service S3.

    Returns
    -------
    file_like : file-like object
        File like object from which data can be read.

    """
    # if a file-like object was provided, return
    if hasattr(filename, "read"):  # file-like object
        return filename

    # look for compressed data by examining the first few bytes
    fh = fsspec.open(filename, mode="rb", compression="infer", **storage_options).open()
    magic = fh.read(3)
    fh.close()

    # If the data is still compressed, use gunzip/bz2 to uncompress the data
    if magic.startswith(b"\x1f\x8b"):
        return gzip.GzipFile(filename, "rb")

    if magic.startswith(b"BZh"):
        return bz2.BZ2File(filename, "rb")

    return fsspec.open(
        filename, mode="rb", compression="infer", **storage_options
    ).open()


def make_time_unit_str(dtobj):
    """Return a time unit string from a datetime object."""
    return "seconds since " + dtobj.strftime("%Y-%m-%dT%H:%M:%SZ")


class LazyLoadDict(MutableMapping):
    """
    A dictionary-like class supporting lazy loading of specified keys.

    Keys which are lazy loaded are specified using the set_lazy method.
    The callable object which produces the specified key is provided as the
    second argument to this method. This object gets called when the value
    of the key is loaded. After this initial call the results is cached
    in the traditional dictionary which is used for supplemental access to
    this key.

    Testing for keys in this dictionary using the "key in d" syntax will
    result in the loading of a lazy key, use "key in d.keys()" to prevent
    this evaluation.

    The comparison methods, __cmp__, __ge__, __gt__, __le__, __lt__, __ne__,
    nor the view methods, viewitems, viewkeys, viewvalues, are implemented.
    Neither is the the fromkeys method.

    This is from Py-ART.

    Parameters
    ----------
    dic : dict
        Dictionary containing key, value pairs which will be stored and
        evaluated traditionally. This dictionary referenced not copied into
        the LazyLoadDictionary and hence changed to this dictionary may change
        the original. If this behavior is not desired copy dic in the
        initalization.

    Examples
    --------
    >>> d = LazyLoadDict({'key1': 'value1', 'key2': 'value2'})
    >>> d.keys()
    ['key2', 'key1']
    >>> lazy_func = lambda : 999
    >>> d.set_lazy('lazykey1', lazy_func)
    >>> d.keys()
    ['key2', 'key1', 'lazykey1']
    >>> d['lazykey1']
    999

    """

    def __init__(self, dic):
        """initalize."""
        self._dic = dic
        self._lazyload = {}

    # abstract methods
    def __setitem__(self, key, value):
        """Set a key which will not be stored and evaluated traditionally."""
        self._dic[key] = value
        if key in self._lazyload:
            del self._lazyload[key]

    def __getitem__(self, key):
        """Get the value of a key, evaluating a lazy key if needed."""
        if key in self._lazyload:
            value = self._lazyload[key]()
            self._dic[key] = value
            del self._lazyload[key]
        return self._dic[key]

    def __delitem__(self, key):
        """Remove a lazy or traditional key from the dictionary."""
        if key in self._lazyload:
            del self._lazyload[key]
        else:
            del self._dic[key]

    def __iter__(self):
        """Iterate over all lazy and traditional keys."""
        return itertools.chain(self._dic.copy(), self._lazyload.copy())

    def __len__(self):
        """Return the number of traditional and lazy keys."""
        return len(self._dic) + len(self._lazyload)

    # additional class to mimic dict behavior
    def __str__(self):
        """Return a string representation of the object."""
        if len(self._dic) == 0 or len(self._lazyload) == 0:
            seperator = ""
        else:
            seperator = ", "
        lazy_reprs = [(repr(k), repr(v)) for k, v in self._lazyload.items()]
        lazy_strs = ["{}: LazyLoad({})".format(*r) for r in lazy_reprs]
        lazy_str = ", ".join(lazy_strs) + "}"
        return str(self._dic)[:-1] + seperator + lazy_str

    def has_key(self, key):
        """True if dictionary has key, else False."""
        return key in self

    def copy(self):
        """
        Return a copy of the dictionary.

        Lazy keys are not evaluated in the original or copied dictionary.
        """
        dic = self.__class__(self._dic.copy())
        # load all lazy keys into the copy
        for key, value_callable in self._lazyload.items():
            dic.set_lazy(key, value_callable)
        return dic

    # lazy dictionary specific methods
    def set_lazy(self, key, value_callable):
        """Set a lazy key to load from a callable object."""
        if key in self._dic:
            del self._dic[key]
        self._lazyload[key] = value_callable
