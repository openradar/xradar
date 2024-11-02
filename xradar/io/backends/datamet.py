#!/usr/bin/env python
# Copyright (c) 2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""

Datamet
=======

This sub-module contains the Datamet xarray backend for reading Datamet-based radar
data into Xarray structures.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}

"""

__all__ = ["DataMetBackendEntrypoint", "open_datamet_datatree"]

__doc__ = __doc__.format("\n   ".join(__all__))

import gzip
import io
import os
import tarfile
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from xarray import DataTree
from xarray.backends.common import AbstractDataStore, BackendArray, BackendEntrypoint
from xarray.backends.file_manager import CachingFileManager
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.utils import FrozenDict
from xarray.core.variable import Variable

from ... import util
from ...model import (
    georeferencing_correction_subgroup,
    get_altitude_attrs,
    get_azimuth_attrs,
    get_elevation_attrs,
    get_latitude_attrs,
    get_longitude_attrs,
    get_range_attrs,
    get_time_attrs,
    moment_attrs,
    radar_calibration_subgroup,
    radar_parameters_subgroup,
    sweep_vars_mapping,
)
from .common import (
    _attach_sweep_groups,
    _get_radar_calibration,
    _get_required_root_dataset,
    _get_subgroup,
)

#: mapping from DataMet names to CfRadial2/ODIM
datamet_mapping = {
    "UZ": "DBTH",
    "CZ": "DBZH",
    "V": "VRADH",
    "W": "WRADH",
    "ZDR": "ZDR",
    "PHIDP": "PHIDP",
    "RHOHV": "RHOHV",
    "KDP": "KDP",
}


def convert_value(value):
    """
    Try to convert the value to an int or float. If conversion is not possible,
    return the original value.
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        pass

    try:
        return float(value)
    except (ValueError, TypeError):
        pass

    return value


class DataMetFile:
    def __init__(self, filename):
        self.filename = filename
        # Handle .tar.gz case
        if self.filename.endswith(".gz"):
            with gzip.open(self.filename, "rb") as gzip_file:
                tar_bytes = io.BytesIO(gzip_file.read())
            self.tarfile = tarfile.open(fileobj=tar_bytes)
        else:
            self.tarfile = tarfile.open(self.filename, "r")
        self.first_dimension = "azimuth"  # No idea if other scan types are available
        self.scan_metadata = self.get_scan_metadata()
        self.moments = self.scan_metadata["measure"]
        self.data = dict()

    def extract_parameters(self, parameter_path):
        # Extract set of parameters from a file in the tarball
        member = self.tarfile.getmember(parameter_path)
        file = self.tarfile.extractfile(member)
        labels = np.loadtxt(file, delimiter="=", dtype=str, usecols=[0])
        file.seek(0)  # Reset file pointer
        values = np.loadtxt(file, delimiter="=", dtype=str, usecols=[1])
        values = np.array(values, ndmin=1)
        labels = np.array(labels, ndmin=1)
        # Iterate over the labels and values, appending values to the appropriate label key
        parameters = defaultdict(list)
        for label, value in zip(labels, values):
            parameters[label.strip()].append(value.strip())
        parameters = dict(parameters)
        # Convert lists with a single element to individual values
        parameters = {
            k: convert_value(v[0]) if len(v) == 1 else v for k, v in parameters.items()
        }
        return parameters

    def extract_data(self, data_path, dtype):
        # Extract moment data from a file in the tarball
        member = self.tarfile.getmember(data_path + "/SCAN.dat")
        file = self.tarfile.extractfile(member)
        data = np.frombuffer(file.read(), dtype=dtype)
        return data

    def get_scan_metadata(self):
        # Get all metadata at scan level (valid for all sweeps/moments)
        navigation = self.extract_parameters(os.path.join(".", "navigation.txt"))
        archiviation = self.extract_parameters(os.path.join(".", "archiviation.txt"))
        return {**navigation, **archiviation}

    def get_mom_metadata(self, mom, sweep):
        # Get all metadata that is moment and/or sweep dependent
        # Note that DataMet uses 1-indexed but we switch to 0-indexed
        # as is done in other backends
        mom_path = os.path.join(".", mom)
        sweep_path = os.path.join(".", mom, str(sweep + 1))

        generic = self.extract_parameters(os.path.join(sweep_path, "generic.txt"))
        calibration_momlvl = self.extract_parameters(
            os.path.join(mom_path, "calibration.txt")
        )
        calibration_sweeplvl = self.extract_parameters(
            os.path.join(sweep_path, "calibration.txt")
        )
        navigation_var = self.extract_parameters(
            os.path.join(sweep_path, "navigation.txt")
        )
        return {
            **navigation_var,
            **generic,
            **calibration_sweeplvl,
            **calibration_momlvl,
        }

    def get_moment(self, mom, sweep):
        # Get the data for a moment and apply byte to float conversion
        # Note that DataMet uses 1-indexed but we switch to 0-indexed
        # as is done in other backends
        mom_path = os.path.join(".", mom, str(sweep + 1))
        mom_medata = self.get_mom_metadata(mom, sweep)

        bitplanes = 16 if mom == "PHIDP" else int(mom_medata["bitplanes"] or 8)
        nazim = int(mom_medata.get("nlines"))
        nrange = int(mom_medata.get("ncols"))

        dtype = np.uint16 if bitplanes == 16 else np.uint8
        data = self.extract_data(mom_path, dtype)
        data = np.reshape(data, (nazim, nrange))

        return data

    def get_sweep(self, sweep):
        # Get data for all moments and sweeps
        moment_names = self.moments
        if sweep not in self.data:
            self.data[sweep] = dict()

        for moment in moment_names:
            if moment not in self.data[sweep]:
                self.data[sweep][moment] = self.get_moment(moment, sweep)

    def close(self):
        # Clase tarfile
        if self.tarfile is not None:
            self.tarfile.close()


class DataMetArrayWrapper(BackendArray):
    """Wraps array of DataMet RAW data."""

    def __init__(self, datastore, variable_name):
        self.datastore = datastore
        self.group = datastore._group
        self.variable_name = variable_name

        array = self.get_array()
        self.dtype = array.dtype
        self.shape = array.shape

    def _getitem(self, key):
        # read the data and put it into dict
        key = tuple(list(k) if isinstance(k, np.ndarray) else k for k in key)
        array = self.get_array()
        return array[key]

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._getitem,
        )

    def get_array(self):
        ds = self.datastore._acquire()
        ds.get_sweep(self.group)
        return ds.data[self.group][self.variable_name]


class DataMetStore(AbstractDataStore):
    """Store for reading DataMet sweeps."""

    def __init__(self, manager, group=None):
        self._manager = manager
        self._group = int(group[6:])
        self._filename = self.filename
        self._need_time_recalc = False

    @classmethod
    def open(cls, filename, mode="r", group=None, **kwargs):
        manager = CachingFileManager(DataMetFile, filename, kwargs=kwargs)
        return cls(manager, group=group)

    @property
    def filename(self):
        with self._manager.acquire_context(False) as root:
            return root.filename

    @property
    def root(self):
        with self._manager.acquire_context(False) as root:
            return root

    def _acquire(self):
        # Does it need a lock as other backends ?
        ds = self.root
        return ds

    @property
    def ds(self):
        return self._acquire()

    def open_store_variable(self, mom):
        dim = self.root.first_dimension

        data = indexing.LazilyOuterIndexedArray(DataMetArrayWrapper(self, mom))
        encoding = {"group": self._group, "source": self._filename}

        mom_metadata = self.root.get_mom_metadata(mom, self._group)
        add_offset = float(mom_metadata.get("offset") or 0.0)
        scale_factor = float(mom_metadata.get("slope") or 1.0)
        maxval = mom_metadata.get("maxval", None)

        if maxval is not None:
            maxval = float(maxval)
            top = 255
            bottom = float(mom_metadata["bottom"])
            scale_factor = (maxval + add_offset) / (top - bottom)

        mname = datamet_mapping.get(mom, mom)
        mapping = sweep_vars_mapping.get(mname, {})
        attrs = {key: mapping[key] for key in moment_attrs if key in mapping}
        attrs["add_offset"] = add_offset
        attrs["scale_factor"] = scale_factor
        attrs["_FillValue"] = 0
        attrs["coordinates"] = (
            "elevation azimuth range latitude longitude altitude time"
        )
        return {mname: Variable((dim, "range"), data, attrs, encoding)}

    def open_store_coordinates(self, mom):
        scan_mdata = self.root.scan_metadata
        mom_mdata = self.root.get_mom_metadata(mom, self._group)
        dim = self.root.first_dimension

        # Radar coordinates
        lat = scan_mdata["orig_lat"]
        lon = scan_mdata["orig_lon"]
        alt = scan_mdata["orig_alt"]

        rng = (
            mom_mdata["Rangeoff"]
            + np.arange(mom_mdata["ncols"]) * mom_mdata["Rangeres"]
        )
        azimuth = (
            mom_mdata["Azoff"] + np.arange(mom_mdata["nlines"]) * mom_mdata["Azres"]
        )
        # Unravel azimuth
        azimuth[azimuth > 360] -= 360
        elevation = mom_mdata["Eloff"] + np.arange(mom_mdata["nlines"]) * 0

        sweep_mode = "azimuth_surveillance" if dim == "azimuth" else "rhi"
        sweep_number = self._group
        prt_mode = "not_set"
        follow_mode = "not_set"

        raytime = 0  # TODO find out raytime
        raytimes = np.array(
            [
                timedelta(seconds=x * raytime).total_seconds()
                for x in range(azimuth.shape[0] + 1)
            ]
        )
        diff = np.diff(raytimes) / 2.0
        rtime = raytimes[:-1] + diff

        timestr = scan_mdata["dt_acq"]
        time = datetime.strptime(timestr, "%Y-%m-%d-%H%M")

        time_attrs = get_time_attrs(f"{time.isoformat()}Z")

        encoding = {"group": self._group}
        rng = Variable(("range",), rng, get_range_attrs())
        azimuth = Variable((dim,), azimuth, get_azimuth_attrs(), encoding)
        elevation = Variable((dim,), elevation, get_elevation_attrs(), encoding)
        time = Variable((dim,), rtime, time_attrs, encoding)

        coords = {
            "azimuth": azimuth,
            "elevation": elevation,
            "range": rng,
            "time": time,
            "sweep_mode": Variable((), sweep_mode),
            "sweep_number": Variable((), sweep_number),
            "prt_mode": Variable((), prt_mode),
            "follow_mode": Variable((), follow_mode),
            "sweep_fixed_angle": Variable((), float(elevation[0])),
            "longitude": Variable((), lon, get_longitude_attrs()),
            "latitude": Variable((), lat, get_latitude_attrs()),
            "altitude": Variable((), alt, get_altitude_attrs()),
        }

        return coords

    def get_variables(self):
        return FrozenDict(
            (k1, v1)
            for k1, v1 in {
                **{
                    k: v
                    for list_item in [
                        self.open_store_variable(k) for k in self.ds.moments
                    ]
                    for (k, v) in list_item.items()
                },
                **self.open_store_coordinates(self.ds.moments[0]),
            }.items()
        )

    def get_attrs(self):
        attributes = {
            "scan_name": self.root.scan_metadata["scan_type"],
            "instrument_name": self.root.scan_metadata["origin"],
            "source": "Datamet",
        }
        return FrozenDict(attributes)


class DataMetBackendEntrypoint(BackendEntrypoint):
    """Xarray BackendEntrypoint for DataMet data."""

    description = "Open DataMet files in Xarray"
    url = "https://xradar.rtfd.io/latest/io.html#datamet-data-i-o"

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
        group=None,
        reindex_angle=False,
        first_dim="auto",
        site_coords=True,
    ):
        store = DataMetStore.open(
            filename_or_obj,
            group=group,
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

        # reassign azimuth/elevation/time coordinates
        ds = ds.assign_coords({"azimuth": ds.azimuth})
        ds = ds.assign_coords({"elevation": ds.elevation})
        ds = ds.assign_coords({"time": ds.time})

        ds.encoding["engine"] = "datamet"

        # handle duplicates and reindex
        if decode_coords and reindex_angle is not False:
            ds = ds.pipe(util.remove_duplicate_rays)
            ds = ds.pipe(util.reindex_angle, **reindex_angle)
            ds = ds.pipe(util.ipol_time, **reindex_angle)

        # handling first dimension
        dim0 = "elevation" if ds.sweep_mode.load() == "rhi" else "azimuth"
        if first_dim == "auto":
            if "time" in ds.dims:
                ds = ds.swap_dims({"time": dim0})
            ds = ds.sortby(dim0)
        else:
            if "time" not in ds.dims:
                ds = ds.swap_dims({dim0: "time"})
            ds = ds.sortby("time")

        # assign geo-coords
        if site_coords:
            ds = ds.assign_coords(
                {
                    "latitude": ds.latitude,
                    "longitude": ds.longitude,
                    "altitude": ds.altitude,
                }
            )

        return ds


def open_datamet_datatree(filename_or_obj, **kwargs):
    """Open DataMet dataset as :py:class:`xarray.DataTree`.

    Parameters
    ----------
    filename_or_obj : str, Path, file-like or DataStore
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
    site_coords : bool
        Attach radar site-coordinates to Dataset, defaults to ``True``.
    kwargs : dict
        Additional kwargs are fed to :py:func:`xarray.open_dataset`.

    Returns
    -------
    dtree: xarray.DataTree
        DataTree
    """
    # handle kwargs, extract first_dim
    backend_kwargs = kwargs.pop("backend_kwargs", {})
    optional = kwargs.pop("optional", True)
    kwargs["backend_kwargs"] = backend_kwargs

    sweep = kwargs.pop("sweep", None)
    sweeps = []
    kwargs["backend_kwargs"] = backend_kwargs

    if isinstance(sweep, str):
        sweeps = [sweep]
    elif isinstance(sweep, int):
        sweeps = [f"sweep_{sweep}"]
    elif isinstance(sweep, list):
        if isinstance(sweep[0], int):
            sweeps = [f"sweep_{i}" for i in sweep]
        else:
            sweeps.extend(sweep)
    else:
        # Get number of sweeps from data
        dmet = DataMetFile(filename_or_obj)
        sweeps = [
            f"sweep_{i}" for i in range(0, dmet.scan_metadata["elevation_number"])
        ]

    ls_ds: list[xr.Dataset] = [
        xr.open_dataset(
            filename_or_obj, group=swp, engine=DataMetBackendEntrypoint, **kwargs
        )
        for swp in sweeps
    ]

    dtree: dict = {
        "/": _get_required_root_dataset(ls_ds, optional=optional),
        "/radar_parameters": _get_subgroup(ls_ds, radar_parameters_subgroup),
        "/georeferencing_correction": _get_subgroup(
            ls_ds, georeferencing_correction_subgroup
        ),
        "/radar_calibration": _get_radar_calibration(ls_ds, radar_calibration_subgroup),
    }
    dtree = _attach_sweep_groups(dtree, ls_ds)
    return DataTree.from_dict(dtree)
