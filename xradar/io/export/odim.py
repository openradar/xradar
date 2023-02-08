#!/usr/bin/env python
# Copyright (c) 2022-2023, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""

ODIM_H5 output
==============

This sub-module contains the writer for export of ODIM_H5-based radar
data.

Code ported from wradlib.

Example::

    import xradar as xd
    dtree = xd.io.to_odim(dtree, filename)

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}

"""

__all__ = [
    "to_odim",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import datetime as dt

import h5py
import numpy as np


def _write_odim(source, destination):
    """Writes ODIM_H5 Attributes.

    Parameters
    ----------
    source : dict
        Attributes to write
    destination : handle
        h5py-group handle
    """
    for key, value in source.items():
        if key in destination.attrs:
            continue
        if isinstance(value, str):
            tid = h5py.h5t.C_S1.copy()
            tid.set_size(len(value) + 1)
            H5T_C_S1_NEW = h5py.Datatype(tid)
            destination.attrs.create(key, value, dtype=H5T_C_S1_NEW)
        else:
            destination.attrs[key] = value


def _write_odim_dataspace(source, destination):
    """Write ODIM_H5 Dataspaces.

    Parameters
    ----------
    source : dict
        Moments to write
    destination : handle
        h5py-group handle
    """
    # todo: check bottom-up/top-down rhi
    dim0 = "elevation" if source.sweep_mode == "rhi" else "azimuth"

    # only assume as radar moments when dimensions fit
    keys = [key for key, val in source.items() if {dim0, "range"} == set(val.dims)]

    data_list = [f"data{i + 1}" for i in range(len(keys))]
    data_idx = np.argsort(data_list)
    for idx in data_idx:
        value = source[keys[idx]]
        h5_data = destination.create_group(data_list[idx])
        enc = value.encoding
        dtype = enc.get("dtype", value.dtype)

        # p. 21 ff
        h5_what = h5_data.create_group("what")
        # get maximum value for dtype for undetect if not available
        undetect = float(enc.get("_Undetect", np.ma.minimum_fill_value(dtype)))

        # set some defaults, if not available
        scale_factor = float(enc.get("scale_factor", 1.0))
        add_offset = float(enc.get("add_offset", 0.0))
        _fillvalue = float(enc.get("_FillValue", undetect))
        what = {
            "quantity": value.name,
            "gain": scale_factor,
            "offset": add_offset,
            "nodata": _fillvalue,
            "undetect": undetect,
        }
        _write_odim(what, h5_what)

        # moments handling
        val = value.sortby(dim0).values
        fillval = _fillvalue * scale_factor
        fillval += add_offset
        val[np.isnan(val)] = fillval
        val = (val - add_offset) / scale_factor
        if np.issubdtype(dtype, np.integer):
            val = np.rint(val).astype(dtype)
        # todo: compression is chosen totally arbitrary here
        #  maybe parameterizing it?
        ds = h5_data.create_dataset(
            "data",
            data=val,
            compression="gzip",
            compression_opts=6,
            fillvalue=_fillvalue,
            dtype=dtype,
        )
        if enc["dtype"] == "uint8":
            image = "IMAGE"
            version = "1.2"
            tid1 = h5py.h5t.C_S1.copy()
            tid1.set_size(len(image) + 1)
            H5T_C_S1_IMG = h5py.Datatype(tid1)
            tid2 = h5py.h5t.C_S1.copy()
            tid2.set_size(len(version) + 1)
            H5T_C_S1_VER = h5py.Datatype(tid2)
            ds.attrs.create("CLASS", image, dtype=H5T_C_S1_IMG)
            ds.attrs.create("IMAGE_VERSION", version, dtype=H5T_C_S1_VER)


def to_odim(dtree, filename):
    """Save DataTree to ODIM_H5/V2_2 compliant file.

    Parameters
    ----------
    dtree : :class:`datatree.DataTree`
    filename : str
        output filename
    """
    root = dtree["/"]

    h5 = h5py.File(filename, "w")

    # root group, only Conventions for ODIM_H5
    _write_odim({"Conventions": "ODIM_H5/V2_2"}, h5)

    # how group
    how = {}
    how.update({"_modification_program": "xradar"})

    h5_how = h5.create_group("how")
    _write_odim(how, h5_how)

    grps = dtree.groups[1:]

    # what group, object, version, date, time, source, mandatory
    # p. 10 f
    what = {}
    if len(grps) > 1:
        what["object"] = "PVOL"
    else:
        what["object"] = "SCAN"
    # todo: parameterize version
    what["version"] = "H5rad 2.2"
    what["date"] = str(root["time_coverage_start"].values)[:10].replace("-", "")
    what["time"] = str(root["time_coverage_end"].values)[11:19].replace(":", "")
    what["source"] = root.attrs["instrument_name"]

    h5_what = h5.create_group("what")
    _write_odim(what, h5_what)

    # where group, lon, lat, height, mandatory
    where = {
        "lon": root["longitude"].values,
        "lat": root["latitude"].values,
        "height": root["altitude"].values,
    }
    h5_where = h5.create_group("where")
    _write_odim(where, h5_where)

    # datasets
    ds_list = [f"dataset{i + 1}" for i in range(len(grps))]
    for idx in range(len(ds_list)):
        ds = dtree[grps[idx]].ds
        dim0 = "elevation" if ds.sweep_mode == "rhi" else "azimuth"

        # datasetN group
        h5_dataset = h5.create_group(ds_list[idx])

        # what group p. 21 ff.
        h5_ds_what = h5_dataset.create_group("what")
        ds_what = {}
        # skip NaT values
        valid_times = ~np.isnat(ds.time.values)
        t = sorted(ds.time.values[valid_times])
        start = dt.datetime.utcfromtimestamp(np.rint(t[0].astype("O") / 1e9))
        end = dt.datetime.utcfromtimestamp(np.rint(t[-1].astype("O") / 1e9))
        ds_what["product"] = "SCAN"
        ds_what["startdate"] = start.strftime("%Y%m%d")
        ds_what["starttime"] = start.strftime("%H%M%S")
        ds_what["enddate"] = end.strftime("%Y%m%d")
        ds_what["endtime"] = end.strftime("%H%M%S")
        _write_odim(ds_what, h5_ds_what)

        # where group, p. 11 ff. mandatory
        h5_ds_where = h5_dataset.create_group("where")
        rscale = ds.range.values[1] / 1.0 - ds.range.values[0]
        rstart = (ds.range.values[0] - rscale / 2.0) / 1000.0
        a1gate = np.argsort(ds.sortby(dim0).time.values)[0]
        ds_where = {
            "elangle": ds["sweep_fixed_angle"].values,
            "nbins": ds.range.shape[0],
            "rstart": rstart,
            "rscale": rscale,
            "nrays": ds.azimuth.shape[0],
            "a1gate": a1gate,
        }
        _write_odim(ds_where, h5_ds_where)

        # how group, p. 14 ff.
        h5_ds_how = h5_dataset.create_group("how")
        tout = [tx.astype("O") / 1e9 for tx in ds.sortby(dim0).time.values]
        tout_sorted = sorted(tout)

        # handle non-uniform times (eg. only second-resolution)
        if np.count_nonzero(np.diff(tout_sorted)) < (len(tout_sorted) - 1):
            tout = np.roll(
                np.linspace(tout_sorted[0], tout_sorted[-1], len(tout)), a1gate
            )
            tout_sorted = sorted(tout)

        difft = np.diff(tout_sorted) / 2.0
        difft = np.insert(difft, 0, difft[0])
        azout = ds.sortby(dim0).azimuth
        diffa = np.diff(azout) / 2.0
        diffa = np.insert(diffa, 0, diffa[0])
        elout = ds.sortby(dim0).elevation
        diffe = np.diff(elout) / 2.0
        diffe = np.insert(diffe, 0, diffe[0])

        # ODIM_H5 datasetN numbers are 1-based
        sweep_number = ds.sweep_number + 1
        ds_how = {
            "scan_index": sweep_number,
            "scan_count": len(grps),
            "startazT": tout - difft,
            "stopazT": tout + difft,
            "startazA": azout - diffa,
            "stopazA": azout + diffa,
            "startelA": elout - diffe,
            "stopelA": elout + diffe,
        }
        _write_odim(ds_how, h5_ds_how)

        # write moments
        _write_odim_dataspace(ds, h5_dataset)

    h5.close()
