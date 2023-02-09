#!/usr/bin/env python
# Copyright (c) 2022, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""

Furuno binary data
==================

Reads data from Furuno's binary data formats

To read from Furuno files :class:`numpy:numpy.memmap` is used to get access to
the data. The Furuno header is read in any case into dedicated OrderedDict's.
Reading sweep data can be skipped by setting `loaddata=False`.
By default, the data is decoded on the fly.

Using `rawdata=True` the data will be kept undecoded.

Code ported from wradlib.

Example::

    import xradar as xd
    dtree = xd.io.open_furuno_datatree(filename)

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}

"""

__all__ = [
    "FurunoBackendEntrypoint",
    "open_furuno_datatree",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import datetime as dt
import gzip
import io
import struct
from collections import OrderedDict

import lat_lon_parser
import numpy as np
import xarray as xr
from datatree import DataTree
from xarray.backends.common import AbstractDataStore, BackendArray, BackendEntrypoint
from xarray.backends.file_manager import CachingFileManager
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.utils import FrozenDict
from xarray.core.variable import Variable

from ... import util
from ...model import (
    get_altitude_attrs,
    get_azimuth_attrs,
    get_elevation_attrs,
    get_latitude_attrs,
    get_longitude_attrs,
    get_range_attrs,
    get_time_attrs,
    moment_attrs,
    sweep_vars_mapping,
)
from .common import (
    SINT2,
    SINT4,
    UINT1,
    UINT2,
    UINT4,
    _assign_root,
    _attach_sweep_groups,
    _calculate_angle_res,
    _get_fmt_string,
    _unpack_dictionary,
)


def decode_time(data, time_struct=None):
    """Decode `YMDS_TIME` into datetime object."""
    time = _unpack_dictionary(data, time_struct)
    try:
        t = dt.datetime(
            time["year"],
            time["month"],
            time["day"],
            time["hour"],
            time["minute"],
            time["second"],
        )
        return t
    except ValueError:
        return None


def decode_geo_angle(data):
    angle = _unpack_dictionary(data, GEO_ANGLE)
    angle = lat_lon_parser.to_dec_deg(
        angle["degree"], angle["minute"], angle["second"] / 1000.0
    )
    return angle


def decode_altitude(data):
    alt = _unpack_dictionary(data, ALTITUDE)
    return alt["upper"] * 100 + alt["lower"] / 100


def decode_radar_constant(data):
    rc = _unpack_dictionary(data, RADAR_CONSTANT)
    return rc["mantissa"] * 10 ** rc["exponent"]


YMDS_TIME = OrderedDict(
    [
        ("year", UINT2),
        ("month", UINT1),
        ("day", UINT1),
        ("hour", UINT1),
        ("minute", UINT1),
        ("second", UINT1),
        ("spare", {"fmt": "1s"}),
    ]
)

YMDS2_TIME = OrderedDict(
    [
        ("year", UINT2),
        ("month", UINT2),
        ("day", UINT2),
        ("hour", UINT2),
        ("minute", UINT2),
        ("second", UINT2),
    ]
)

GEO_ANGLE = OrderedDict(
    [
        ("degree", SINT2),
        ("minute", UINT2),
        ("second", UINT2),
    ]
)

ALTITUDE = OrderedDict(
    [
        ("upper", UINT2),
        ("lower", UINT2),
    ]
)

RADAR_CONSTANT = OrderedDict(
    [
        ("mantissa", SINT4),
        ("exponent", SINT2),
    ]
)


LEN_YMDS_TIME = struct.calcsize(_get_fmt_string(YMDS_TIME))
LEN_YMDS2_TIME = struct.calcsize(_get_fmt_string(YMDS2_TIME))
LEN_GEO_ANGLE = struct.calcsize(_get_fmt_string(GEO_ANGLE))
LEN_ALTITUDE = struct.calcsize(_get_fmt_string(ALTITUDE))
LEN_RADAR_CONSTANT = struct.calcsize(_get_fmt_string(RADAR_CONSTANT))

_YMDS_TIME = {
    "size": f"{LEN_YMDS_TIME}s",
    "func": decode_time,
    "fkw": {"time_struct": YMDS_TIME},
}
_YMDS2_TIME = {
    "size": f"{LEN_YMDS2_TIME}s",
    "func": decode_time,
    "fkw": {"time_struct": YMDS2_TIME},
}
_GEO_ANGLE = {"size": f"{LEN_GEO_ANGLE}s", "func": decode_geo_angle, "fkw": {}}
_ALTITUDE = {"size": f"{LEN_ALTITUDE}s", "func": decode_altitude, "fkw": {}}
_RADAR_CONSTANT = {
    "size": f"{LEN_RADAR_CONSTANT}s",
    "func": decode_radar_constant,
    "fkw": {},
}

# Furuno Operator's Manual WR2120
# data file type 3 binary v10
# 7.3 pp. 61-66
HEADER_HEAD = OrderedDict(
    [
        ("size_of_header", UINT2),
        ("format_version", UINT2),
    ]
)

SCNX_HEADER = OrderedDict(
    [
        ("size_of_header", UINT2),
        ("format_version", UINT2),
        ("scan_start_time", _YMDS_TIME),
        ("scan_stop_time", _YMDS_TIME),
        ("time_zone", SINT2),
        ("product_number", UINT2),
        ("model_type", UINT2),
        ("latitude", SINT4),
        ("longitude", SINT4),
        ("altitude", SINT4),
        ("azimuth_offset", UINT2),
        ("tx_frequency", UINT4),
        ("polarization_mode", UINT2),
        ("antenna_gain_h", UINT2),
        ("antenna_gain_v", UINT2),
        ("half_power_beam_width_h", UINT2),
        ("half_power_beam_width_v", UINT2),
        ("tx_power_h", UINT2),
        ("tx_power_v", UINT2),
        ("radar_constant_h", SINT2),
        ("radar_constant_v", SINT2),
        ("noise_power_short_pulse_h", SINT2),
        ("noise_power_long_pulse_h", SINT2),
        ("threshold_power_short_pulse", SINT2),
        ("threshold_power_long_pulse", SINT2),
        ("tx_pulse_specification", UINT2),
        ("prf_mode", UINT2),
        ("prf_1", UINT2),
        ("prf_2", UINT2),
        ("prf_3", UINT2),
        ("nyquist_velocity", UINT2),
        ("sample_number", UINT2),
        ("tx_pulse_blind_length", UINT2),
        ("short_pulse_width", UINT2),
        ("short_pulse_modulation_bandwidth", UINT2),
        ("long_pulse_width", UINT2),
        ("long_pulse_modulation_bandwidth", UINT2),
        ("pulse_switch_point", UINT2),
        ("observation_mode", UINT2),
        ("antenna_rotation_speed", UINT2),
        ("number_sweep_direction_data", UINT2),
        ("number_range_direction_data", UINT2),
        ("resolution_range_direction", UINT2),
        ("current_scan_number", UINT2),
        ("total_number_scans_volume", UINT2),
        ("rainfall_intensity_estimation_method", UINT2),
        ("z_r_coefficient_b", UINT2),
        ("z_r_coefficient_beta", UINT2),
        ("kdp_r_coefficient_a", UINT2),
        ("kdp_r_coefficient_b", UINT2),
        ("kdp_r_coefficient_c", UINT2),
        ("zh_attenuation_correction_method", UINT2),
        ("zh_attenuation_coefficient_b1", UINT2),
        ("zh_attenuation_coefficient_b2", UINT2),
        ("zh_attenuation_coefficient_d1", UINT2),
        ("zh_attenuation_coefficient_d2", UINT2),
        ("air_attenuation_one_way", UINT2),
        ("output_threshold_rain", UINT2),
        ("record_item", UINT2),
        ("signal_processing_flag", UINT2),
        ("clutter_reference_file", _YMDS_TIME),
        ("reserved", {"fmt": "8s"}),
    ]
)

SCN_HEADER = OrderedDict(
    [
        ("size_of_header", UINT2),
        ("format_version", UINT2),
        ("dpu_log_time", _YMDS2_TIME),
        ("latitude", _GEO_ANGLE),
        ("longitude", _GEO_ANGLE),
        ("altitude", _ALTITUDE),
        ("antenna_rotation_speed", UINT2),
        ("prf_1", UINT2),
        ("prf_2", UINT2),
        ("noise_level_pulse_modulation_h", SINT2),
        ("noise_level_frequency_modulation_h", SINT2),
        ("number_sweep_direction_data", UINT2),
        ("number_range_direction_data", UINT2),
        ("resolution_range_direction", UINT2),
        ("radar_constant_h", _RADAR_CONSTANT),
        ("radar_constant_v", _RADAR_CONSTANT),
        ("azimuth_offset", UINT2),
        ("scan_start_time", _YMDS2_TIME),
        ("record_item", UINT2),
        ("tx_pulse_blind_length", UINT2),
        ("tx_pulse_specification", UINT2),
    ]
)


class FurunoFile:
    """FurunoFile class"""

    def __init__(self, filename, **kwargs):
        self._debug = kwargs.get("debug", False)
        self._rawdata = kwargs.get("rawdata", False)
        self._loaddata = kwargs.get("loaddata", True)
        self._obsmode = kwargs.get("obsmode", None)
        self._fp = None
        self._filename = filename
        if isinstance(filename, str):
            if filename.endswith(".gz"):
                filename = gzip.open(filename)
        if isinstance(filename, str):
            self._fp = open(filename, "rb")
            self._fh = np.memmap(self._fp, mode="r")
        else:
            if isinstance(filename, (io.BytesIO, gzip.GzipFile)):
                filename.seek(0)
                filename = filename.read()
            self._fh = np.frombuffer(filename, dtype=np.uint8)
        self._filepos = 0
        self._data = None
        # read header
        len = struct.calcsize(_get_fmt_string(HEADER_HEAD))
        head = _unpack_dictionary(self.read_from_file(len), HEADER_HEAD)
        if head["format_version"] == 10:
            header = SCNX_HEADER
        elif head["format_version"] in [3, 103]:
            header = SCN_HEADER
        self._filepos = 0
        self.get_header(header)
        self._filepos = 0
        if self._loaddata:
            self.get_data()

    def get_data(self):
        if self._data is None:
            moments = [
                "RATE",
                "DBZH",
                "VRADH",
                "ZDR",
                "KDP",
                "PHIDP",
                "RHOHV",
                "WRADH",
                "QUAL",
                "RES1",
                "RES2",
                "RES3",
                "RES4",
                "RES5",
                "RES6",
                "FIX",
            ]
            # check available moments
            items = dict()
            for i in range(9):
                if (self.header["record_item"] & 2**i) == 2**i:
                    items[i] = moments[i]
            # claim available moments
            rays = self.header["number_sweep_direction_data"]
            rng = self.header["number_range_direction_data"]
            start = self.header["size_of_header"]
            raw_data = self._fh[start:].view(dtype="uint16").reshape(rays, -1)
            data = raw_data[:, 4:].reshape(rays, len(items), rng)
            self._data = dict()
            for i, item in enumerate(items.values()):
                self._data[item] = data[:, i, :]
            # get angles
            angles = raw_data[:, :4].reshape(rays, 4)
            self._data["azimuth"] = np.fmod(
                angles[:, 1] + self.header["azimuth_offset"], 36000
            )
            # elevation angles are dtype "int16"
            # which was tested against a sweep with -1deg elevation
            # https://github.com/openradar/xradar/pull/82
            self._data["elevation"] = angles[:, 2].view(dtype="int16")
        return self._data

    def close(self):
        if self._fp is not None:
            self._fp.close()

    __del__ = close

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    @property
    def header(self):
        """Returns ingest_header dictionary."""
        return self._header

    @property
    def version(self):
        return self.header["format_version"]

    @property
    def site_coords(self):
        if self.version in [3, 103]:
            lon = self.header["longitude"]
            lat = self.header["latitude"]
            alt = self.header["altitude"]
        else:
            lon = self.header["longitude"] * 1e-5
            lat = self.header["latitude"] * 1e-5
            alt = self.header["altitude"] * 1e-2
        return lon, lat, alt

    @property
    def data(self):
        return self._data

    @property
    def loaddata(self):
        """Returns `loaddata` switch."""
        return self._loaddata

    @property
    def rawdata(self):
        """Returns `rawdata` switch."""
        return self._rawdata

    @property
    def debug(self):
        return self._debug

    @property
    def filename(self):
        return self._filename

    @property
    def first_dimension(self):
        obs_mode = None
        if self.version in [3, 103]:
            # extract mode from filename
            if ".scn" in self.filename:
                obs_mode = 1
            elif ".sppi" in self.filename:
                obs_mode = 1
            elif ".rhi" in self.filename:
                obs_mode = 2
            elif isinstance(self.filename, io.BytesIO):
                if self._obsmode is None:
                    raise ValueError(
                        "Furuno `observation mode` can't be extracted from `io.BytesIO`. "
                        "Please use kwarg `obsmode=1` for PPI or `obsmode=2` "
                        "for RHI sweeps."
                    )
                obs_mode = self._obsmode
            else:
                pass
        else:
            obs_mode = self.header["observation_mode"]
        if obs_mode in [1, 3, 4]:
            return "azimuth"
        elif obs_mode == 2:
            return "elevation"
        else:
            raise TypeError(f"Unknown Furuno Observation Mode: {obs_mode}")

    @property
    def fixed_angle(self):
        dim = "azimuth" if self.first_dimension == "elevation" else "elevation"

        return self._data[dim][0] * 1e-2

    @property
    def a1gate(self):
        return np.argmin(self._data[self.first_dimension][::-1])

    @property
    def angle_resolution(self):
        return _calculate_angle_res(self._data[self.first_dimension] / 100.0)

    @property
    def fh(self):
        return self._fh

    @property
    def filepos(self):
        return self._filepos

    @filepos.setter
    def filepos(self, pos):
        self._filepos = pos

    def read_from_file(self, size):
        """Read from file.

        Parameters
        ----------
        size : int
            Number of data words to read.

        Returns
        -------
        data : array-like
            numpy array of data
        """
        start = self._filepos
        self._filepos += size
        return self._fh[start : self._filepos]

    def get_header(self, header):
        len = struct.calcsize(_get_fmt_string(header))
        self._header = _unpack_dictionary(
            self.read_from_file(len), header, self._rawdata
        )


class FurunoArrayWrapper(BackendArray):
    def __init__(
        self,
        data,
    ):
        self.data = data
        self.shape = data.shape
        self.dtype = np.dtype("uint16")

    def __getitem__(self, key: tuple):
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.OUTER_1VECTOR,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple):
        return self.data[key]


class FurunoStore(AbstractDataStore):
    """Store for reading Furuno sweeps via wradlib."""

    def __init__(self, manager, group=None):
        self._manager = manager
        self._group = group
        self._filename = self.filename
        self._need_time_recalc = False

    @classmethod
    def open(cls, filename, mode="r", group=None, **kwargs):
        manager = CachingFileManager(FurunoFile, filename, mode=mode, kwargs=kwargs)
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
        dim = self.root.first_dimension

        data = indexing.LazilyOuterIndexedArray(FurunoArrayWrapper(var))
        encoding = {"group": self._group, "source": self._filename}
        if name == "PHIDP":
            add_offset = 360 * -32768 / 65535
            scale_factor = 360 / 65535
        elif name == "RHOHV":
            add_offset = 2 * -1 / 65534
            scale_factor = 2 / 65534
        elif name == "WRADH":
            add_offset = -1e-2
            scale_factor = 1e-2
        elif name in ["azimuth", "elevation"]:
            add_offset = 0.0
            scale_factor = 1e-2
        else:
            add_offset = -327.68
            scale_factor = 1e-2

        mapping = sweep_vars_mapping.get(name, {})
        attrs = {key: mapping[key] for key in moment_attrs if key in mapping}
        if name in ["azimuth", "elevation"]:
            attrs = get_azimuth_attrs() if name == "azimuth" else get_elevation_attrs()
            attrs["add_offset"] = add_offset
            attrs["scale_factor"] = scale_factor
            # choose maximum of dtype here, because it is out of the valid range
            attrs["_FillValue"] = np.ma.minimum_fill_value(data.dtype)
            dims = (dim,)
            # do not propagate a1gate and angle_res for now
            # if name == self.ds.first_dimension:
            #    attrs["a1gate"] = self.ds.a1gate
            #    attrs["angle_res"] = self.ds.angle_resolution

        else:
            if name != "QUAL":
                attrs["add_offset"] = add_offset
                attrs["scale_factor"] = scale_factor
                attrs["_FillValue"] = 0.0
            dims = (dim, "range")
        attrs[
            "coordinates"
        ] = "elevation azimuth range latitude longitude altitude time rtime sweep_mode"
        return Variable(dims, data, attrs, encoding)

    def open_store_coordinates(self):
        dim = self.ds.first_dimension

        # range
        start_range = 0
        if self.ds.version in [3, 103]:
            range_step = self.ds.header["resolution_range_direction"] / 100
        else:
            range_step = self.ds.header["resolution_range_direction"]
        stop_range = range_step * self.ds.header["number_range_direction_data"]
        rng = np.arange(
            start_range + range_step / 2,
            stop_range + range_step / 2,
            range_step,
            dtype="float32",
        )
        range_attrs = get_range_attrs(rng)
        rng = Variable(("range",), rng, range_attrs)

        # making-up ray times
        time = self.ds.header["scan_start_time"]
        stop_time = self.ds.header.get("scan_stop_time", time)
        num_rays = self.ds.header["number_sweep_direction_data"]

        # if no stop_time is available, get time from rotation speed
        if time == stop_time:
            raytime = self.ds.angle_resolution / (
                self.ds.header["antenna_rotation_speed"] * 1e-1 * 6
            )
            raytime = dt.timedelta(seconds=raytime)
        # otherwise, calculate from time difference
        else:
            raytime = (stop_time - time) / num_rays

        raytimes = np.array(
            [(x * raytime).total_seconds() for x in range(num_rays + 1)]
        )

        diff = np.diff(raytimes) / 2.0
        rtime = raytimes[:-1] + diff

        rtime_attrs = get_time_attrs(f"{time.isoformat()}Z")

        encoding = {}
        rng = Variable(("range",), rng, range_attrs)
        time = Variable((dim,), rtime, rtime_attrs, encoding)

        # get coordinates from Furuno File
        sweep_mode = "azimuth_surveillance" if dim == "azimuth" else "rhi"
        sweep_number = 0
        prt_mode = "not_set"
        follow_mode = "not_set"

        lon, lat, alt = self.ds.site_coords

        coords = {
            "range": rng,
            "time": time,
            "sweep_mode": Variable((), sweep_mode),
            "sweep_number": Variable((), sweep_number),
            "prt_mode": Variable((), prt_mode),
            "follow_mode": Variable((), follow_mode),
            "sweep_fixed_angle": Variable((), self.ds.fixed_angle),
            "longitude": Variable((), lon, get_longitude_attrs()),
            "latitude": Variable((), lat, get_latitude_attrs()),
            "altitude": Variable((), alt, get_altitude_attrs()),
        }

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
        # attributes = {"fixed_angle": float(self.ds.fixed_angle)}
        return FrozenDict()


class FurunoBackendEntrypoint(BackendEntrypoint):
    """Xarray BackendEntrypoint for Furuno data."""

    description = "Open FURUNO (.scn, .scnx) in Xarray"
    url = "https://xradar.rtfd.io/en/latest/io.html#furuno-binary-data"

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
        first_dim="auto",
        reindex_angle=False,
        fix_second_angle=False,
        site_coords=True,
        obsmode=None,
    ):
        store = FurunoStore.open(
            filename_or_obj,
            group=group,
            loaddata=True,
            obsmode=obsmode,
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

        ds.encoding["engine"] = "furuno"

        # handle duplicates and reindex
        if decode_coords and reindex_angle is not False:
            ds = ds.pipe(util.remove_duplicate_rays)
            ds = ds.pipe(util.reindex_angle, **reindex_angle)
            ds = ds.pipe(util.ipol_time)

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


def open_furuno_datatree(filename_or_obj, **kwargs):
    """Open FURUNO dataset as xradar Datatree.

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
        If True, fixes erroneous second angle data. Defaults to False.
    site_coords : bool
        Attach radar site-coordinates to Dataset, defaults to ``True``.
    kwargs :  kwargs
        Additional kwargs are fed to `xr.open_dataset`.

    Returns
    -------
    dtree: DataTree
        DataTree
    """
    # handle kwargs, extract first_dim
    backend_kwargs = kwargs.pop("backend_kwargs", {})
    # first_dim = backend_kwargs.pop("first_dim", None)
    kwargs["backend_kwargs"] = backend_kwargs

    ds = [xr.open_dataset(filename_or_obj, engine="furuno", **kwargs)]

    ds.insert(0, xr.Dataset())

    # create datatree root node with required data
    dtree = DataTree(data=_assign_root(ds), name="root")
    # return datatree with attached sweep child nodes
    return _attach_sweep_groups(dtree, ds[1:])
