#!/usr/bin/env python
# Copyright (c) 2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""

StreamLine HPL
==============

This sub-module contains the StreamLine HPL xarray backend for reading StreamLine-based lidar
data into Xarray structures.

Import of StreamLine .hpl (txt) files and save locally in directory. Therefore
the data is converted into matrices with dimension "number of range gates" x "time stamp/rays".
In newer versions of the StreamLine software, the spectral width can be
stored as additional parameter in the .hpl files.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}

"""

__all__ = [
    "HPLBackendEntrypoint",
    "open_hpl_datatree",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import io
from collections import OrderedDict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
from xarray import DataTree
from xarray.backends.common import AbstractDataStore, BackendArray, BackendEntrypoint
from xarray.backends.file_manager import CachingFileManager
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.utils import FrozenDict

from ...model import (
    georeferencing_correction_subgroup,
    get_altitude_attrs,
    get_azimuth_attrs,
    get_elevation_attrs,
    get_latitude_attrs,
    get_longitude_attrs,
    get_time_attrs,
    radar_calibration_subgroup,
    radar_parameters_subgroup,
)
from .common import (
    _attach_sweep_groups,
    _get_radar_calibration,
    _get_required_root_dataset,
    _get_subgroup,
)

variable_attr_dict = {}
variable_attr_dict["intensity"] = {
    "long_name": "Backscatter intensity",
    "standard_name": "intensity",
    "units": "SNR+1",
    "dims": ("time", "range"),
}

variable_attr_dict["backscatter"] = {
    "long_name": "Attenuated backscatter",
    "standard_name": "backscatter",
    "units": "m-1 sr-1",
    "dims": ("time", "range"),
}

variable_attr_dict["mean_doppler_velocity"] = {
    "long_name": "Mean Doppler velocity",
    "standard_name": "doppler_velocity",
    "units": "m s-1",
    "dims": ("time", "range"),
}

variable_attr_dict["spectral_width"] = {
    "long_name": "Spectral width",
    "standard_name": "spectral_width",
    "units": "m s-1",
    "dims": ("time", "range"),
}

variable_attr_dict["azimuth"] = {
    "long_name": "Azimuth angle",
    "standard_name": "azimuth",
    "units": "degrees",
    "dims": ("time",),
}
variable_attr_dict["elevation"] = {
    "long_name": "Elevation angle",
    "standard_name": "elevation",
    "units": "degrees",
    "dims": ("time",),
}
variable_attr_dict["antenna_transition"] = {
    "long_name": "Antenna transition flag",
    "standard_name": "antenna_transition",
    "units": "1 = transition 0 = sweep",
    "dims": ("time",),
}
variable_attr_dict["pitch"] = {
    "long_name": "Lidar Pitch angle",
    "standard_name": "pitch",
    "units": "degrees",
    "dims": ("time",),
}

variable_attr_dict["roll"] = {
    "long_name": "Lidar Roll angle",
    "standard_name": "roll",
    "units": "degrees",
    "dims": ("time",),
}

variable_attr_dict["range"] = {
    "long_name": "Range from lidar",
    "standard_name": "range",
    "units": "m",
    "dims": ("range",),
}


variable_attr_dict["sweep_mode"] = {"dims": ()}
variable_attr_dict["sweep_fixed_angle"] = {"dims": ()}
variable_attr_dict["sweep_group_name"] = {"dims": ()}
variable_attr_dict["sweep_number"] = {"dims": ()}
variable_attr_dict["time"] = get_time_attrs()
variable_attr_dict["time"]["dims"] = ("time",)
variable_attr_dict["azimuth"] = get_azimuth_attrs()
variable_attr_dict["azimuth"]["dims"] = ("time",)
variable_attr_dict["elevation"] = get_elevation_attrs()
variable_attr_dict["elevation"]["dims"] = ("time",)
variable_attr_dict["latitude"] = get_latitude_attrs()
variable_attr_dict["latitude"]["dims"] = ()
variable_attr_dict["longitude"] = get_longitude_attrs()
variable_attr_dict["longitude"]["dims"] = ()
variable_attr_dict["altitude"] = get_altitude_attrs()
variable_attr_dict["altitude"]["dims"] = ()

HALO_DEFAULT_MOMENTS = [
    "intensity",
    "mean_doppler_velocity",
    "spectral_width",
    "backscatter",
]
HALO_CONSTANT_PARAMETERS = [
    k for k in variable_attr_dict if k not in HALO_DEFAULT_MOMENTS
]


def _convert_to_hours_minutes_seconds(decimal_hour, initial_time):
    delta = timedelta(hours=decimal_hour)
    return datetime(initial_time.year, initial_time.month, initial_time.day) + delta


def _hpl2dict(file_buf):
    # import hpl files into intercal storage
    lines = file_buf.readlines()

    # write lines into Dictionary
    data_temp = dict()

    header_n = 17  # length of header
    data_temp["filename"] = lines[0].split()[-1]
    data_temp["system_id"] = int(lines[1].split()[-1])
    data_temp["number_of_gates"] = int(lines[2].split()[-1])
    data_temp["range_gate_length_m"] = float(lines[3].split()[-1])
    data_temp["gate_length_pts"] = int(lines[4].split()[-1])
    data_temp["pulses_per_ray"] = int(lines[5].split()[-1])
    data_temp["number_of_waypoints_in_file"] = int(lines[6].split()[-1])
    rays_n = (len(lines) - header_n) / (data_temp["number_of_gates"] + 1)

    """
    number of lines does not match expected format if the number of range gates
    was changed in the measuring period of the data file (especially possible for stare data)
    """
    if not rays_n.is_integer():
        print("Number of lines does not match expected format")
        return np.nan

    data_temp["no_of_rays_in_file"] = int(rays_n)
    data_temp["scan_type"] = " ".join(lines[7].split()[2:])
    data_temp["focus_range"] = lines[8].split()[-1]
    data_temp["start_time"] = pd.to_datetime(" ".join(lines[9].split()[-2:]))
    data_temp["resolution"] = lines[10].split()[-1] + " m s-1"
    data_temp["range_gates"] = np.arange(0, data_temp["number_of_gates"])
    data_temp["center_of_gates"] = (data_temp["range_gates"] + 0.5) * data_temp[
        "range_gate_length_m"
    ]

    # dimensions of data set
    gates_n = data_temp["number_of_gates"]
    rays_n = data_temp["no_of_rays_in_file"]

    # item of measurement variables are predefined as symetric numpy arrays filled with NaN values
    data_temp["radial_velocity"] = np.full([gates_n, rays_n], np.nan)  # m s-1
    data_temp["intensity"] = np.full([gates_n, rays_n], np.nan)  # SNR+1
    data_temp["beta"] = np.full([gates_n, rays_n], np.nan)  # m-1 sr-1
    data_temp["spectral_width"] = np.full([gates_n, rays_n], np.nan)
    data_temp["elevation"] = np.full(rays_n, np.nan)  # degrees
    data_temp["azimuth"] = np.full(rays_n, np.nan)  # degrees
    data_temp["decimal_time"] = np.full(rays_n, np.nan)  # hours
    data_temp["pitch"] = np.full(rays_n, np.nan)  # degrees
    data_temp["roll"] = np.full(rays_n, np.nan)  # degrees

    for ri in range(0, rays_n):  # loop rays
        lines_temp = lines[
            header_n
            + (ri * gates_n)
            + ri
            + 1 : header_n
            + (ri * gates_n)
            + gates_n
            + ri
            + 1
        ]
        header_temp = np.asarray(
            lines[header_n + (ri * gates_n) + ri].split(), dtype=float
        )
        data_temp["decimal_time"][ri] = header_temp[0]
        data_temp["azimuth"][ri] = header_temp[1]
        data_temp["elevation"][ri] = header_temp[2]
        data_temp["pitch"][ri] = header_temp[3]
        data_temp["roll"][ri] = header_temp[4]
        for gi in range(0, gates_n):  # loop range gates
            line_temp = np.asarray(lines_temp[gi].split(), dtype=float)
            data_temp["radial_velocity"][gi, ri] = line_temp[1]
            data_temp["intensity"][gi, ri] = line_temp[2]
            data_temp["beta"][gi, ri] = line_temp[3]
            if line_temp.size > 4:
                data_temp["spectral_width"][gi, ri] = line_temp[4]

    return data_temp


class HplFile:
    def __init__(self, filename, **kwargs):
        """
        Opens a Halo Photonics .hpl file
        """

        transition_threshold_azi = kwargs.pop("transition_threshold_azi", 0.01)
        transition_threshold_el = kwargs.pop("transition_threshold_el", 0.005)
        round_azi = kwargs.pop("round_azi", 1)
        round_el = kwargs.pop("round_el", 1)
        latitude = kwargs.pop("latitude", 0)
        longitude = kwargs.pop("longitude", 0)
        altitude = kwargs.pop("altitude", 0)
        if isinstance(filename, str):
            self._fp = open(filename)
            self._filename = filename
        elif isinstance(filename, io.IOBase):
            filename.seek(0)
            self._fp = filename
            self._filename = None
        data_temp = _hpl2dict(self._fp)
        initial_time = pd.to_datetime(data_temp["start_time"])

        time = pd.to_datetime(
            [
                _convert_to_hours_minutes_seconds(x, initial_time)
                for x in data_temp["decimal_time"]
            ]
        )
        data_unsorted = {}
        data_unsorted["time"] = time
        data_unsorted["intensity"] = data_temp["intensity"].T
        data_unsorted["mean_doppler_velocity"] = data_temp["radial_velocity"].T
        data_unsorted["backscatter"] = data_temp["beta"].T
        data_unsorted["spectral_width"] = data_temp["spectral_width"].T
        data_unsorted["range"] = data_temp["center_of_gates"]
        data_unsorted["pitch"] = data_temp["pitch"]
        data_unsorted["roll"] = data_temp["roll"]
        data_unsorted["azimuth"] = np.round(data_temp["azimuth"], round_azi)
        data_unsorted["elevation"] = np.round(data_temp["elevation"], round_el)
        data_unsorted["azimuth"] = np.where(
            data_unsorted["azimuth"] >= 360.0,
            data_unsorted["azimuth"] - 360.0,
            data_unsorted["azimuth"],
        )
        diff_azimuth = np.diff(data_unsorted["azimuth"], axis=0)
        diff_elevation = np.pad(
            np.diff(data_unsorted["elevation"], axis=0), pad_width=(1, 0)
        )
        where_unique = np.argwhere(diff_elevation <= transition_threshold_el)
        unique_elevations = np.unique(data_unsorted["elevation"][where_unique])
        unique_elevations = unique_elevations[np.isfinite(unique_elevations)]
        counts = np.zeros_like(unique_elevations)
        for i in range(len(unique_elevations)):
            counts[i] = np.sum(data_unsorted["elevation"] == unique_elevations[i])

        for i in range(len(unique_elevations)):
            counts[i] = np.sum(data_unsorted["elevation"] == unique_elevations[i])

        if np.sum(np.abs(diff_azimuth) > transition_threshold_azi) <= 2 and not np.all(
            data_unsorted["elevation"] == 90.0
        ):
            data_unsorted["sweep_mode"] = "rhi"
            n_sweeps = 1
        elif np.all(data_unsorted["elevation"] == 90.0):
            data_unsorted["sweep_mode"] = "vertical_pointing"
            n_sweeps = 1
        else:
            # We will filter out the transitions between sweeps
            data_unsorted["sweep_mode"] = "azimuth_surveillance"
            n_sweeps = len(unique_elevations)

        data_unsorted["azimuth"] = np.where(
            data_unsorted["azimuth"] < 360.0,
            data_unsorted["azimuth"],
            data_unsorted["azimuth"] - 360.0,
        )
        if data_unsorted["sweep_mode"] == "rhi":
            non_transitions = (
                np.argwhere(np.abs(diff_azimuth) <= transition_threshold_azi) + 1
            )
            unique_elevations = np.unique(
                data_unsorted["azimuth"][np.squeeze(non_transitions)]
            )
            data_unsorted["fixed_angle"] = unique_elevations
        elif (
            data_unsorted["sweep_mode"] == "azimuth_surveillance"
            or data_unsorted["sweep_mode"] == "vertical_pointing"
        ):
            data_unsorted["fixed_angle"] = np.array(unique_elevations)
        data_unsorted["sweep_number"] = np.arange(0, n_sweeps)

        # These are not provided in the .hpl file, so user must specify
        data_unsorted["latitude"] = np.array(latitude)
        data_unsorted["longitude"] = np.array(longitude)
        data_unsorted["altitude"] = np.array(altitude)
        data_unsorted["sweep_mode"] = np.array(
            data_unsorted["sweep_mode"], dtype="|S32"
        )
        start_indicies = []
        end_indicies = []
        for i, t in enumerate(data_unsorted["fixed_angle"]):
            if data_unsorted["sweep_mode"] == b"rhi":
                where_in_sweep = np.argwhere(data_unsorted["azimuth"] == t)
            else:
                where_in_sweep = np.argwhere(data_unsorted["elevation"] == t)
            start_indicies.append(int(where_in_sweep.min()))
            end_indicies.append(int(where_in_sweep.max()))
        end_indicies = np.array(end_indicies)
        transitions = np.pad(np.abs(diff_elevation) > transition_threshold_el, (1, 0))
        data_unsorted["sweep_start_ray_index"] = np.array(start_indicies)
        data_unsorted["sweep_end_ray_index"] = np.array(end_indicies)
        data_unsorted["antenna_transition"] = transitions
        self._data = OrderedDict()
        for i in range(len(data_unsorted["fixed_angle"])):
            sweep_dict = OrderedDict()
            time_inds = slice(
                data_unsorted["sweep_start_ray_index"][i],
                data_unsorted["sweep_end_ray_index"][i],
            )
            for k in data_unsorted.keys():
                if k == "sweep_start_ray_index" or k == "sweep_end_ray_index":
                    continue
                if k == "fixed_angle":
                    sweep_dict["sweep_fixed_angle"] = data_unsorted["fixed_angle"][i]
                elif k == "sweep_number":
                    sweep_dict["sweep_group_name"] = np.array(f"sweep_{i - 1}")
                    sweep_dict["sweep_number"] = np.array(i - 1)
                elif len(variable_attr_dict[k]["dims"]) == 0:
                    sweep_dict[k] = data_unsorted[k]
                elif variable_attr_dict[k]["dims"][0] == "time":
                    sweep_dict[k] = data_unsorted[k][time_inds]
                else:
                    sweep_dict[k] = data_unsorted[k]
            self._data[f"sweep_{i}"] = sweep_dict
        self._data["sweep_number"] = data_unsorted["sweep_number"]
        self._data["fixed_angle"] = data_unsorted["fixed_angle"]

    def get_sweep(self, sweep_number, moments=None):
        if moments is None:
            moments = HALO_DEFAULT_MOMENTS
        self.data["sweep_data"] = OrderedDict()
        for m in moments:
            self.data["sweep_data"][m] = self._data[sweep_number][m]
        for k in HALO_CONSTANT_PARAMETERS:
            self.data["sweep_data"][k] = self._data[sweep_number][k]

    def close(self):
        if self._fp is not None:
            self._fp.close()

    __del__ = close

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    @property
    def data(self):
        return self._data

    @property
    def filename(self):
        return self._filename


class HplArrayWrapper(BackendArray):
    def __init__(self, data, name):
        self.data = data
        self.shape = data.shape
        self.name = name
        self.dtype = np.dtype("float32")

    def __getitem__(self, key: tuple):
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.OUTER_1VECTOR,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple):
        return self.data[key]


class HplStore(AbstractDataStore):
    """Store for reading Furuno sweeps via wradlib."""

    def __init__(self, manager, group=None):
        self._manager = manager
        self._group = group
        self._filename = self.filename

    @classmethod
    def open(cls, filename, mode="r", group=None, **kwargs):
        manager = CachingFileManager(HplFile, filename, mode=mode, kwargs=kwargs)
        return cls(manager, group=group)

    @property
    def filename(self):
        with self._manager.acquire_context(False) as root:
            return root.filename

    @property
    def root(self):
        with self._manager.acquire_context(False) as root:
            return root

    def _acquire(self, needs_lock=True):
        with self._manager.acquire_context(needs_lock) as root:
            root.get_sweep(self._group)
            ds = root.data[self._group]
        return ds

    @property
    def ds(self):
        return self._acquire()

    def open_store_variable(self, name, var):
        data = indexing.LazilyOuterIndexedArray(HplArrayWrapper(var, name))
        encoding = {"group": self._group, "source": self._filename}
        attrs = variable_attr_dict[name].copy()
        dims = attrs["dims"]
        del attrs["dims"]
        return xr.Variable(dims, data, attrs, encoding)

    def open_store_coordinates(self):
        coord_keys = ["time", "azimuth", "range"]
        coords = {}
        for k in coord_keys:
            attrs = variable_attr_dict[k].copy()
            dims = attrs["dims"]
            del attrs["dims"]
            coords[k] = xr.Variable(dims, self.ds[k], attrs=attrs)

        return coords

    def get_variables(self):
        return FrozenDict(
            (k1, v1)
            for k1, v1 in {
                **{k: self.open_store_variable(k, v) for k, v in self.ds.items()},
                **self.open_store_coordinates(),
            }.items()
        )

    def get_attrs(self):
        return FrozenDict()


class HPLBackendEntrypoint(BackendEntrypoint):
    """Xarray BackendEntrypoint for Halo Photonics Doppler processed lidar data.

    Keyword Arguments
    -----------------
    first_dim : str
        Can be ``time`` or ``auto`` first dimension. If set to ``auto``,
        first dimension will be either ``azimuth`` or ``elevation`` depending on
        type of sweep. Defaults to ``auto``.
    site_coords : bool
        Attach radar site-coordinates to Dataset, defaults to ``True``.
    kwargs : dict
        Additional kwargs are fed to :py:func:`xarray.open_dataset`.
    """

    description = "Backend for reading Halo Photonics Doppler lidar processed data"
    url = "https://xradar.rtfd.io/en/latest/io.html#metek"

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
        group="sweep_0",
        invalid_netcdf=None,
        phony_dims="access",
        decode_vlen_strings=True,
        first_dim="auto",
        site_coords=True,
        optional=True,
        latitude=0,
        longitude=0,
        altitude=0,
        transition_threshold_azi=0.05,
        transition_threshold_el=0.001,
    ):
        store_entrypoint = StoreBackendEntrypoint()

        store = HplStore.open(
            filename_or_obj,
            format=format,
            group=group,
            invalid_netcdf=invalid_netcdf,
            phony_dims=phony_dims,
            decode_vlen_strings=decode_vlen_strings,
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            transition_threshold_azi=transition_threshold_azi,
            transition_threshold_el=transition_threshold_el,
        )

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
        if site_coords is True:
            ds = ds.assign_coords({"longitude": ds.longitude})
            ds = ds.assign_coords({"latitude": ds.latitude})
            ds = ds.assign_coords({"altitude": ds.altitude})

        ds.encoding["engine"] = "hpl"
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

        return ds


def _get_h5group_names(filename_or_obj):
    store = HplStore.open(filename_or_obj)
    return [f"sweep_{i}" for i in store.root.data["sweep_number"]]


def open_hpl_datatree(filename_or_obj, **kwargs):
    """Open Halo Photonics processed Doppler lidar dataset as :py:class:`xarray.DataTree`.

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
    optional = backend_kwargs.pop("optional", None)
    sweep = kwargs.pop("sweep", None)
    sweeps = []
    kwargs["backend_kwargs"] = backend_kwargs

    if isinstance(sweep, str):
        sweeps = [sweep]
    elif isinstance(sweep, int):
        sweeps = [f"sweep_{sweep}"]
    elif isinstance(sweep, list):
        if isinstance(sweep[0], int):
            sweeps = [f"sweep_{i + 1}" for i in sweep]
        else:
            sweeps.extend(sweep)
    else:
        sweeps = _get_h5group_names(filename_or_obj)

    ls_ds: list[xr.Dataset] = [
        xr.open_dataset(filename_or_obj, group=swp, engine="hpl", **kwargs)
        for swp in sweeps
    ]

    dtree: dict = {
        "/": _get_required_root_dataset(ls_ds, optional=optional).rename(
            {"sweep_fixed_angle": "fixed_angle"}
        ),
        "/radar_parameters": _get_subgroup(ls_ds, radar_parameters_subgroup),
        "/georeferencing_correction": _get_subgroup(
            ls_ds, georeferencing_correction_subgroup
        ),
        "/radar_calibration": _get_radar_calibration(ls_ds, radar_calibration_subgroup),
    }
    dtree = _attach_sweep_groups(dtree, ls_ds)
    return DataTree.from_dict(dtree)
