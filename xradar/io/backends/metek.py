#!/usr/bin/env python
# Copyright (c) 2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Metek MRR2 raw and processed data
=================================
Read data from METEK's MRR-2 raw (.raw) and processed (.pro, .avg) files.

Example::

   import xradar as xd
   ds = xr.open_dataset('0308.pro', engine='metek')

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

import io
import warnings
from datetime import datetime

import numpy as np
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

__all__ = [
    "MRRBackendEntrypoint",
    "open_metek_datatree",
]

__doc__ = __doc__.format("\n   ".join(__all__))

variable_attr_dict = dict(
    transfer_function={
        "long_name": "Transfer function",
        "standard_name": "transfer_function",
        "units": "1",
        "dims": ("time", "range"),
    },
    spectral_reflectivity={
        "long_name": "Spectral reflectivity",
        "standard_name": "equivalent_reflectivity_factor",
        "units": "dB",
        "dims": ("index", "sample"),
    },
    raw_spectra_counts={
        "long_name": "Raw spectra counts",
        "standard_name": "raw_spectra",
        "units": "",
        "dims": ("index", "sample"),
    },
    drop_size={
        "long_name": "Drop size",
        "standard_name": "drop_size",
        "units": "mm",
        "dims": ("index", "sample"),
    },
    drop_number_density={
        "long_name": "Raindrop number density",
        "standard_name": "raindrop_number_density",
        "units": "m-4",
        "dims": ("index", "sample"),
    },
    rainfall_rate={
        "long_name": "Rainfall rate",
        "standard_name": "rainfall_rate",
        "units": "mm hr-1",
        "dims": ("time", "range"),
    },
    liquid_water_content={
        "long_name": "Liquid water content",
        "standard_name": "liquid_water_content",
        "units": "g m-3",
        "dims": ("time", "range"),
    },
    path_integrated_attenuation={
        "long_name": "Path integrated attenuation",
        "standard_name": "path_integrated_attenuation",
        "units": "dB",
        "dims": ("time", "range"),
    },
    corrected_reflectivity={
        "long_name": "Attenuation-corrected Radar reflectivity factor",
        "standard_name": "equivalent_radar_reflectivity_factor",
        "units": "dBZ",
        "dims": ("time", "range"),
    },
    reflectivity={
        "long_name": "Radar reflectivity factor",
        "standard_name": "equivalent_radar_reflectivity_factor",
        "units": "dBZ",
        "dims": ("time", "range"),
    },
    spectrum_index={
        "long_name": "Spectrum index",
        "standard_name": "spectrum_index",
        "units": "1",
        "dims": ("time", "range"),
    },
    percentage_valid_spectra={
        "long_name": "Percentage of spectra that are valid",
        "standard_name": "percentage_valid_spectra",
        "units": "percent",
        "dims": ("time"),
    },
    number_valid_spectra={
        "long_name": "number of spectra that are valid",
        "standard_name": "number_valid_spectra",
        "units": "1",
        "dims": ("time"),
    },
    total_number_spectra={
        "long_name": "Total number of spectra",
        "standard_name": "total_number_spectra",
        "units": "1",
        "dims": ("time"),
    },
    velocity_bins={
        "long_name": "Doppler velocity bins",
        "standard_name": "doppler_velocity_bins",
        "units": "m s-1",
        "dims": ("sample"),
    },
    range={
        "long_name": "Range from radar",
        "standard_name": "range",
        "units": "m",
        "dims": ("range",),
    },
    time=get_time_attrs(),
    azimuth=get_azimuth_attrs(),
    elevation=get_elevation_attrs(),
    velocity={
        "long_name": "Radial velocity of scatterers toward instrument",
        "standard_name": "radial_velocity_of_scatterers_toward_instrument",
        "units": "m s-1",
    },
    latitude=get_latitude_attrs(),
    longitude=get_longitude_attrs(),
    altitude=get_altitude_attrs(),
)

variable_attr_dict["time"]["dims"] = ("time",)
variable_attr_dict["azimuth"]["dims"] = ("time",)
variable_attr_dict["elevation"]["dims"] = ("time",)
variable_attr_dict["velocity"]["dims"] = ("time", "range")
variable_attr_dict["latitude"]["dims"] = ()
variable_attr_dict["longitude"]["dims"] = ()
variable_attr_dict["altitude"]["dims"] = ()


def _parse_spectra_line(input_str, num_gates):
    out_array = np.zeros(num_gates)
    increment = {32: 9, 31: 7}[num_gates]
    for i, pos in enumerate(range(3, len(input_str) - increment, increment)):
        input_num_str = input_str[pos : pos + increment]
        try:
            out_array[i] = float(input_num_str)
        except ValueError:
            out_array[i] = np.nan

    return out_array


class MRR2File:
    def __init__(self, file_name="", **kwargs):
        self.vel_bin_spacing = 0.1887
        self.nyquist_velocity = self.vel_bin_spacing * 64
        self._data = {}
        self._data["velocity_bins"] = np.arange(
            0, 64 * self.vel_bin_spacing, self.vel_bin_spacing
        )
        self._data["range"] = []
        self._data["transfer_function"] = []
        self._data["spectral_reflectivity"] = []
        self._data["raw_spectra_counts"] = []
        self._data["drop_size"] = []
        self._data["drop_number_density"] = []
        self._data["time"] = []
        self.filetype = ""
        self.device_version = ""
        self.device_serial_number = ""
        self.bandwidth = 0
        self._fp = None
        self.calibration_constant = []
        self._data["percentage_valid_spectra"] = []
        self._data["number_valid_spectra"] = []
        self._data["total_number_spectra"] = []
        self.spectra_index = 0
        self.altitude = None
        self.sampling_rate = 125000.0
        self._data["path_integrated_attenuation"] = []
        self._data["corrected_reflectivity"] = []
        self._data["reflectivity"] = []
        self._data["rainfall_rate"] = []
        self._data["liquid_water_content"] = []
        self._data["velocity"] = []
        self._data["altitude"] = np.array(np.nan)
        self._data["longitude"] = np.array(np.nan)
        self._data["latitude"] = np.array(np.nan)
        self.filename = None
        self.n_gates = 32
        if not file_name == "":
            self.filename = file_name
            self.open(file_name)

    def open(self, filename_or_obj):
        if isinstance(filename_or_obj, io.IOBase):
            filename_or_obj.seek(0)
            self._fp = filename_or_obj

        if isinstance(filename_or_obj, str):
            self.filename = filename_or_obj
            self._fp = open(filename_or_obj)

        num_times = 0
        temp_spectra = np.zeros((self.n_gates, 64))
        temp_drops = np.zeros((self.n_gates, 64))
        temp_number = np.zeros((self.n_gates, 64))
        spec_var = ""
        for file_line in self._fp:
            if isinstance(file_line, bytes):
                file_line = file_line.decode("utf-8")
            if file_line[:3] == "MRR":
                if num_times > 0:
                    self._data[spec_var].append(temp_spectra)
                    self._data["drop_number_density"].append(temp_number)
                    self._data["drop_size"].append(temp_drops)

                string_split = file_line.split()
                time_str = string_split[1]
                parsed_datetime = datetime.strptime(time_str, "%y%m%d%H%M%S")
                self._data["time"] = self._data["time"] + [parsed_datetime]
                self.filetype = string_split[-1]
                if self.filetype == "RAW":
                    self.device_version = string_split[4]
                    self.device_serial_number = string_split[6]
                    self.bandwidth = int(string_split[8])
                    self.calibration_constant.append(int(string_split[10]))
                    self._data["percentage_valid_spectra"].append(int(string_split[12]))
                    self._data["number_valid_spectra"].append(int(string_split[13]))
                    self._data["total_number_spectra"].append(int(string_split[14]))
                    self.n_gates = 32
                    spec_var = "raw_spectra_counts"
                elif self.filetype == "AVE" or self.filetype == "PRO":
                    self._data["altitude"] = np.array(float(string_split[8]))
                    self.sampling_rate = float(string_split[10])
                    self.mrr_service_version = string_split[12]
                    self.device_version = string_split[14]
                    self.calibration_constant.append(int(string_split[16]))
                    self._data["percentage_valid_spectra"].append(int(string_split[18]))
                    self.n_gates = 31
                    spec_var = "spectral_reflectivity"
                else:
                    raise OSError(
                        "Invalid file type flag in file! Must be RAW, AVG, or PRO!"
                    )
                temp_spectra = np.zeros((self.n_gates, 64))
                temp_drops = np.zeros((self.n_gates, 64))
                temp_number = np.zeros((self.n_gates, 64))
                num_times = num_times + 1

            if file_line[0] == "H":
                in_array = _parse_spectra_line(file_line, self.n_gates)
                if num_times > 1:
                    after_res = in_array[1] - in_array[0]
                    before_res = self._data["range"][1] - self._data["range"][0]
                    if not after_res == before_res:
                        warnings.warn(
                            f"MRR2 resolution was changed mid file. Before time period "
                            f"{parsed_datetime} the resolution was {before_res}, "
                            f"and {after_res} after.",
                            UserWarning,
                        )
                self._data["range"] = in_array

            if file_line[0:2] == "TF":
                self._data["transfer_function"].append(
                    _parse_spectra_line(file_line, self.n_gates)
                )
            if file_line[0] == "F":
                spectra_bin_no = int(file_line[1:3])
                temp_spectra[:, spectra_bin_no] = _parse_spectra_line(
                    file_line, self.n_gates
                )
            if file_line[0] == "D":
                spectra_bin_no = int(file_line[1:3])
                temp_drops[:, spectra_bin_no] = _parse_spectra_line(
                    file_line, self.n_gates
                )
            if file_line[0] == "N":
                spectra_bin_no = int(file_line[1:3])
                temp_number[:, spectra_bin_no] = _parse_spectra_line(
                    file_line, self.n_gates
                )
            if file_line[0:3] == "PIA":
                self._data["path_integrated_attenuation"] = self._data[
                    "path_integrated_attenuation"
                ] + [_parse_spectra_line(file_line, self.n_gates)]
            if file_line[0:3] == "z  ":
                self._data["reflectivity"] = self._data["reflectivity"] + [
                    _parse_spectra_line(file_line, self.n_gates)
                ]
            if file_line[0:3] == "Z  ":
                self._data["corrected_reflectivity"] = self._data[
                    "corrected_reflectivity"
                ] + [_parse_spectra_line(file_line, self.n_gates)]
            if file_line[0:3] == "RR ":
                self._data["rainfall_rate"] = self._data["rainfall_rate"] + [
                    _parse_spectra_line(file_line, self.n_gates)
                ]
            if file_line[0:3] == "LWC":
                self._data["liquid_water_content"] = self._data[
                    "liquid_water_content"
                ] + [_parse_spectra_line(file_line, self.n_gates)]
            if file_line[0:3] == "W  ":
                self._data["velocity"] = self._data["velocity"] + [
                    _parse_spectra_line(file_line, self.n_gates)
                ]

        self._data[spec_var].append(temp_spectra)
        self._data["drop_number_density"].append(temp_number)
        self._data["drop_size"].append(temp_drops)
        self._data["transfer_function"] = np.stack(
            self._data["transfer_function"], axis=0
        )
        self._data[spec_var] = np.stack(self._data[spec_var], axis=0)
        self._data["drop_number_density"] = np.stack(
            self._data["drop_number_density"], axis=0
        )
        self._data["drop_size"] = np.stack(self._data["drop_size"], axis=0)

        if self.filetype == "RAW":
            self._data["total_number_spectra"] = np.stack(
                self._data["total_number_spectra"], axis=0
            )
            self._data["number_valid_spectra"] = np.stack(
                self._data["number_valid_spectra"], axis=0
            )
            self._data["reflectivity"] = None
            self._data["corrected_reflectivity"] = None
            self._data["liquid_water_content"] = None
            self._data["rainfall_rate"] = None
            self._data["percentage_valid_spectra"] = None
            self._data["drop_number_density"] = None
            self._data["drop_size"] = None
            self._data["path_integrated_attenuation"] = None
            self._data["velocity"] = None
            self._data["spectral_reflectivity"] = None

            del self._data["reflectivity"]
            del self._data["corrected_reflectivity"]
            del self._data["liquid_water_content"]
            del self._data["rainfall_rate"]
            del self._data["percentage_valid_spectra"]
            del self._data["drop_number_density"]
            del self._data["drop_size"]
            del self._data["path_integrated_attenuation"]
            del self._data["velocity"]
            del self._data["spectral_reflectivity"]
        else:
            del self._data["total_number_spectra"], self._data["number_valid_spectra"]
            self._data["reflectivity"] = np.stack(self._data["reflectivity"], axis=0)
            self._data["path_integrated_attenuation"] = np.stack(
                self._data["path_integrated_attenuation"], axis=0
            )
            self._data["corrected_reflectivity"] = np.stack(
                self._data["corrected_reflectivity"], axis=0
            )

            self._data["liquid_water_content"] = np.stack(
                self._data["liquid_water_content"], axis=0
            )
            self._data["velocity"] = np.stack(self._data["velocity"], axis=0)
            self._data["rainfall_rate"] = np.stack(self._data["rainfall_rate"], axis=0)
            self._data["percentage_valid_spectra"] = np.stack(
                self._data["percentage_valid_spectra"], axis=0
            )
            self._data["drop_number_density"] = np.stack(
                self._data["drop_number_density"], axis=0
            )
            self._data["drop_size"] = np.stack(self._data["drop_size"], axis=0)
            self._data["raw_spectra_counts"] = None
            del self._data["raw_spectra_counts"]

        self._data["range"] = np.squeeze(self._data["range"])
        # Now we compress the spectrum variables to remove invalid spectra
        self._data[spec_var] = self._data[spec_var].reshape(
            self._data[spec_var].shape[0] * self._data[spec_var].shape[1],
            self._data[spec_var].shape[2],
        )
        where_valid_spectra = np.any(np.isfinite(self._data[spec_var]), axis=1)
        inds = np.where(where_valid_spectra, 1, -1)

        self._data[spec_var] = self._data[spec_var][where_valid_spectra]
        if self.filetype == "PRO" or self.filetype == "AVE":
            self._data["drop_number_density"] = self._data[
                "drop_number_density"
            ].reshape(
                self._data["drop_number_density"].shape[0]
                * self._data["drop_number_density"].shape[1],
                self._data["drop_number_density"].shape[2],
            )
            self._data["drop_number_density"] = self._data["drop_number_density"][
                where_valid_spectra
            ]
            self._data["drop_size"] = self._data["drop_size"].reshape(
                self._data["drop_size"].shape[0] * self._data["drop_size"].shape[1],
                self._data["drop_size"].shape[2],
            )
            self._data["drop_size"] = self._data["drop_size"][where_valid_spectra]
        cur_index = 0
        for i in range(len(inds)):
            if inds[i] > -1:
                inds[i] = cur_index
                cur_index += 1
        self._data["spectrum_index"] = inds.reshape(
            (len(self._data["time"]), len(self._data["range"]))
        )

        self._data["azimuth"] = np.zeros_like(self._data["time"])
        self._data["elevation"] = 90 * np.ones_like(self._data["time"])
        self._data["time"] = np.array(self._data["time"])
        temp_drops = None
        temp_number = None
        temp_spectra = None

    def close(self):
        if self._fp is not None:
            self._fp.close()

    __del__ = close

    @property
    def data(self):
        return self._data

    @property
    def coordinates(self):
        return self._coordinates

    @property
    def fixed_angle(self):
        return self._data["elevation"]

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


class MRR2ArrayWrapper(BackendArray):
    def __init__(
        self,
        data,
    ):
        self.data = data
        self.shape = data.shape
        self.dtype = np.dtype("float64")

    def __getitem__(self, key: tuple):
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.OUTER_1VECTOR,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple):
        return self.data[key]


class MRR2DataStore(AbstractDataStore):
    def __init__(self, manager, group=None):
        self._manager = manager
        self._group = group
        self._filename = self.filename
        self._need_time_recalc = False

    @classmethod
    def open(cls, filename, mode="r", group=None, **kwargs):
        manager = CachingFileManager(MRR2File, filename, mode=mode, kwargs=kwargs)
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
            return root

    @property
    def ds(self):
        return self._acquire()

    def open_store_variable(self, name, var):
        data = indexing.LazilyOuterIndexedArray(MRR2ArrayWrapper(var))
        encoding = {"group": self._group, "source": self._filename}
        attrs = variable_attr_dict[name].copy()
        dims = attrs["dims"]
        del attrs["dims"]
        return xr.Variable(dims, data, attrs, encoding)

    def open_store_coordinates(self):
        coord_keys = ["time", "range", "velocity_bins"]
        coords = {}
        for k in coord_keys:
            attrs = variable_attr_dict[k].copy()
            dims = attrs["dims"]
            del attrs["dims"]
            coords[k] = xr.Variable(dims, self.ds.data[k], attrs=attrs)

        return coords

    def get_variables(self):
        return FrozenDict(
            (k1, v1)
            for k1, v1 in {
                **{k: self.open_store_variable(k, v) for k, v in self.ds.data.items()},
                **self.open_store_coordinates(),
            }.items()
        )

    def get_attrs(self):
        return FrozenDict()


class MRRBackendEntrypoint(BackendEntrypoint):
    """Xarray BackendEntrypoint for Metek MRR2 data.

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

    description = "Backend for reading Metek MRR2 processed and raw data"
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
        group="/",
        invalid_netcdf=None,
        phony_dims="access",
        decode_vlen_strings=True,
        first_dim="auto",
        site_coords=True,
        optional=True,
    ):
        store_entrypoint = StoreBackendEntrypoint()

        store = MRR2DataStore.open(
            filename_or_obj,
            format=format,
            group=group,
            invalid_netcdf=invalid_netcdf,
            phony_dims=phony_dims,
            decode_vlen_strings=decode_vlen_strings,
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

        ds = ds.assign_coords({"range": ds.range})
        ds = ds.assign_coords({"time": ds.time})
        ds = ds.assign_coords({"velocity_bins": ds.velocity_bins})
        ds.encoding["engine"] = "metek"

        return ds


def open_metek_datatree(filename_or_obj, **kwargs):
    """Open Metek MRR2 dataset as :py:class:`xarray.DataTree`.

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
    optional = backend_kwargs.pop("optional", True)
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
        sweeps = ["sweep_0"]

    ls_ds: list[xr.Dataset] = [
        xr.open_dataset(filename_or_obj, group=swp, engine="metek", **kwargs)
        for swp in sweeps
    ].copy()
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
