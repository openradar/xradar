#!/usr/bin/env python
# Copyright (c) 2024-2025, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Universal Format UF
^^^^^^^^^^^^^^^^^^^

Reads data from UF data format

IRIS (Vaisala Sigmet Interactive Radar Information System)

See M211318EN-F Programming Guide ftp://ftp.sigmet.com/outgoing/manuals/
or
https://www.eol.ucar.edu/sites/default/files/files_live/private/UfDoc.txt

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = [
    "UFBackendEntrypoint",
]
__doc__ = __doc__.format("\n   ".join(__all__))

import datetime as dt
import struct
from collections import OrderedDict, defaultdict

import dateutil
import numpy as np
from xarray.backends.common import AbstractDataStore, BackendArray, BackendEntrypoint
from xarray.backends.file_manager import CachingFileManager
from xarray.backends.locks import SerializableLock, ensure_lock
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.utils import FrozenDict
from xarray.core.variable import Variable

from xradar import util
from xradar.model import (
    get_altitude_attrs,
    get_azimuth_attrs,
    get_elevation_attrs,
    get_latitude_attrs,
    get_longitude_attrs,
    get_range_attrs,
    moment_attrs,
    sweep_vars_mapping,
)

from .iris import (
    _get_fmt_string,
    _unpack_dictionary,
    string_dict,
)

UF_LOCK = SerializableLock()


#: mapping from UF names to CfRadial2/ODIM
uf_mapping = {
    "VR": "VRADH",
    "SW": "WRADH",
    "DR": "ZDR",
    "CZ": "DBZH",
    "DZ": "DBTH",
    "ZT": "DBM",
    "PH": "PHIDP",
    "RH": "RHOHV",
    "KD": "KDP",
    "LH": "LDRH",
    "LV": "LDRV",
}

# UF file structures and sizes
# format of structure elements
UINT1 = {"fmt": "B", "dtype": "unit8"}
UINT2 = {"fmt": "H", "dtype": "uint16"}
UINT4 = {"fmt": "I", "dtype": "uint32"}
SINT1 = {"fmt": "b", "dtype": "int8"}
SINT2 = {"fmt": "h", "dtype": "int16"}
SINT4 = {"fmt": "i", "dtype": "int32"}
FLT4 = {"fmt": "f", "dtype": "float32"}
CODE1 = {"fmt": "B"}
CODE2 = {"fmt": "H"}


def _get_endianness(fh):
    # this check depends on the equality of
    # 32bit raylength (bytes 0-4) and the 16bit reclength (bytes(6-8)
    size = np.array(fh[0:4], dtype=np.uint8)
    size_little = size.view(dtype="<u4")[0]
    size_big = size.view(dtype=">u4")[0]
    rsize = np.array(fh[6:8], dtype=np.uint8)
    rsize_little = rsize.view(dtype="<u2")[0]
    rsize_big = rsize.view(dtype=">u2")[0]
    if rsize_big * 2 == size_big:
        return "big"
    if rsize_little * 2 == size_little:
        return "little"
    raise OSError("Endianess error, size does not match record size in either layout!")


def _sweep_mode(value):
    return {
        0: "Cal",
        1: "PPI",
        2: "Coplane",
        3: "RHI",
        4: "Vertical",
        5: "Target",
        6: "Manual",
        7: "Idle",
    }[value]


class UFFile:
    """
    Class for accessing data in a UF file.

    Parameters
    ----------
    filename : str
        Filename of UF file to read.

    """

    def __init__(self, filename, mode="r", loaddata=False):
        """initalize the object."""
        self._fp = None
        self._filename = filename
        # read in the volume header and compression_record
        if hasattr(filename, "read"):
            self._fh = filename
        else:
            self._fp = open(filename, "rb")
            self._fh = np.memmap(self._fp, mode=mode)
        self._filepos = 0
        self._rawdata = False
        self._loaddata = loaddata
        self._ray_indices = None
        self._ray_headers = None
        self._endianness = None
        self._byteorder = None
        self._data = defaultdict(dict)
        self.get_ray_headers()
        self._moments = self._get_moments()
        # self.volume_header = self.get_header(VOLUME_HEADER)

    @property
    def endianness(self):
        if self._endianness is None:
            self._endianness = _get_endianness(self._fh)
        return self._endianness

    @property
    def byteorder(self):
        if self._byteorder is None:
            self._byteorder = "<" if self.endianness == "little" else ">"
        return self._byteorder

    def get_header(self, offset, header):
        length = struct.calcsize(_get_fmt_string(header))
        head = _unpack_dictionary(
            self.read_from_file(offset, length), header, self._rawdata, self.byteorder
        )
        return head

    def get_ray_data_header(self, head):
        offset = head["file_offset"] + (head["mhead"]["DataHeaderPosition"] - 1) * 2 + 4
        data_header = self.get_header(offset, UF_DATA_HEADER)
        off = offset + LEN_UF_DATA_HEADER
        fields = dict()
        for i in range(data_header["FieldsThisRay"]):
            dsi2 = self.get_header(off, UF_DSI2)
            off += LEN_UF_DSI2
            data_type = dsi2.pop("DataType")
            fields.update({data_type: dsi2})
        for mom, fdict in fields.items():
            foff = (fdict["FieldHeaderPosition"] - 1) * 2 + 4
            off = head["file_offset"] + foff
            field_header = self.get_header(off, UF_FIELD_HEADER)
            off += LEN_UF_FIELD_HEADER
            doff = head["file_offset"] + (field_header["DataPosition"] - 1) * 2 + 4
            diff = doff - off
            # if there are additional field extensions, we read them
            # skipping the check for moments for the time being
            # if mom in ["VE", "VF", "VP", "VR", "VT"] and diff == 4:
            if diff == 4:
                field_header.update(VE=self.get_header(off, UF_FIELD_VE))
                off += LEN_UF_FIELD_VE
            # elif mom in ["DR", "KD", "PH", "RH", "ZD"] and diff == 12:
            elif diff == 12:
                field_header.update(DM=self.get_header(off, UF_FIELD_DM))
                off += LEN_UF_FIELD_DM
            fields[mom].update(field_header)
        data_header.update(fields=fields)
        return data_header

    def get_ray_field_data(self, ray_header, moment):
        field_header = ray_header["dhead"]["fields"][moment]
        off = ray_header["file_offset"] + (field_header["DataPosition"] - 1) * 2 + 4
        nsize = field_header["BinCount"] * 2
        dtype = f"{self.byteorder}i2"
        data = self.read_from_file(off, nsize).view(dtype=dtype)
        return data

    def get_ray_headers(self):
        ray_headers = defaultdict(list)
        for i, idx in enumerate(self.ray_indices):
            head = dict()
            head["file_offset"] = idx
            off = idx + 4
            mhead = self.get_header(off, UF_MANDATORY_HEADER)
            head.update(mhead=mhead)
            off += LEN_UF_MANDATORY_HEADER
            if mhead["OptionalHeaderPosition"] != mhead["DataHeaderPosition"]:
                head.update(ohead=self.get_header(off, UF_OPTIONAL_HEADER))
                off += LEN_UF_OPTIONAL_HEADER
            head.update(dhead=self.get_ray_data_header(head))
            sweep_number = mhead["SweepNumber"]
            ray_headers[sweep_number].append(head)
        self._ray_headers = ray_headers

    def get_moment(self, sweep_number, moment):
        _data = []
        for i, rh in enumerate(self._ray_headers[sweep_number]):
            _data.append(self.get_ray_field_data(rh, moment))
        unique = np.unique([len(x) for x in _data])
        if unique.size > 1:
            fill_value = rh["mhead"]["NoDataValue"]
            data = np.full((len(_data), max(unique)), fill_value)
            for i, arr in enumerate(_data):
                data[i, : len(arr)] = arr
        else:
            data = np.stack(_data, axis=0)
        mom = dict(data=data)
        self._data[sweep_number]["sweep_data"][moment].update(mom)

    @property
    def nsweeps(self):
        return len(self._ray_headers.keys())

    def _get_moments(self):
        sweep_range = list(self._ray_headers.keys())
        for sw in sweep_range:
            first_ray = self._ray_headers[sw][0]
            self._data[sw]["mhead"] = first_ray["mhead"]
            self._data[sw]["ohead"] = first_ray.get("ohead", None)
            self._data[sw]["sweep_data"] = first_ray["dhead"]["fields"]
            self._data[sw]["nrays"] = len(self._ray_headers[sw])

        return {
            k: list(v["sweep_data"].keys())
            for k, v in self._data.items()
            if isinstance(v, dict)
        }

    @property
    def moments(self):
        return self._moments

    @property
    def filename(self):
        return self._filename

    @property
    def fh(self):
        return self._fh

    @property
    def filepos(self):
        return self._filepos

    @filepos.setter
    def filepos(self, pos):
        self._filepos = pos

    def read_from_file(self, offset, size):
        """Read from file.

        Parameters
        ----------
        offset : int
            Offset into data/file
        size : int
            Number of data words to read.

        Returns
        -------
        data : array-like
            numpy array of data
        """
        return self._fh[offset : offset + size]

    @property
    def data(self):
        return self._data

    @property
    def ray_indices(self):
        """Get file offsets of individual rays (a.k.a. records)"""
        if self._ray_indices is None:
            # find UF/PF in data stream, located in bytes 4 and 5
            seq = np.array(self._fh[4:6], dtype=np.uint8)
            len_seq = 8
            rd = util.rolling_dim(self._fh, len_seq)
            mask = (rd[:, 4] == seq[0]) & (rd[:, 5] == seq[1])
            matches = rd[mask]
            # to only find real first 8 bytes, we check the equality of
            # raylength and reclength
            sizes = matches[:, 0:4].view(f"{self.byteorder}u4").flatten()
            rsizes = matches[:, 6:8].view(f"{self.byteorder}u2").flatten()
            valid_mask = sizes == rsizes * 2
            self._ray_indices = np.nonzero(mask)[0][valid_mask]

        return self._ray_indices

    def close(self):
        if self._fp is not None:
            self._fp.close()

    __del__ = close

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


def decode_64(num):
    """Decode BIN angle.

    See Appendix D UF Format

    Parameters
    ----------
    num : array-like

    Returns
    -------
    out : array-like
        decoded num
    """
    return num / 64


BIN2 = {"name": "BIN2", "dtype": "int16", "size": "h", "func": decode_64, "fkw": {}}

UF_MANDATORY_HEADER = OrderedDict(
    [
        ("ID", string_dict(2)),
        ("RecordSize", SINT2),
        ("OptionalHeaderPosition", SINT2),
        # Offset to start of optional header, origin 1
        ("LocalUseHeaderPosition", SINT2),
        # Offset to start of local use header, origin 1
        ("DataHeaderPosition", SINT2),  # Offset to start of data header, origin 1
        ("RecordNumber", SINT2),  # Record number within the file, origin 1
        ("VolumeNumber", SINT2),  # Volume scan number on the tape, N/A for disk
        ("RayNumber", SINT2),
        ("RecordInRay", SINT2),  # Record number within the ray
        ("SweepNumber", SINT2),
        ("RadarName", string_dict(8)),
        ("SiteName", string_dict(8)),
        ("LatDegrees", SINT2),
        ("LatMinutes", SINT2),
        ("LatSeconds", BIN2),
        ("LonDegrees", SINT2),
        ("LonMinutes", SINT2),
        ("LonSeconds", BIN2),  # Seconds in 1/64 ths
        ("Altitude", SINT2),  # Height of antenna above sea level in meters
        ("Year", SINT2),  # Time of the ray
        ("Month", SINT2),
        ("Day", SINT2),
        ("Hour", SINT2),
        ("Minute", SINT2),
        ("Second", SINT2),
        ("TimeZone", string_dict(2)),
        ("Azimuth", BIN2),  # Azimuth in 1/64 of a degree
        ("Elevation", BIN2),  # Elevation in 1/64 of a degree
        ("SweepMode", SINT2),  # Sweep mode, 1=PPI
        ("FixedAngle", BIN2),  # Sweep desired angle in 1/64 of a degree
        ("SweepRate", BIN2),  # Antenna scan rate in 1/64 of degrees/second
        ("ConvertYear", SINT2),  # Current time when the converter was run, GMT
        ("ConvertMonth", SINT2),
        ("ConvertDay", SINT2),
        ("ConvertName", string_dict(8)),
        ("NoDataValue", SINT2),
    ]
)

UF_OPTIONAL_HEADER = OrderedDict(
    [
        ("ProjectName", string_dict(8)),
        ("BaselineAzimuth", SINT2),
        ("BaselineElevation", SINT2),
        ("VolumeScanHour", SINT2),  # Time of start of current volume scan
        ("VolumeScanMinute", SINT2),
        ("VolumeScanSecond", SINT2),
        ("FieldTapeName", string_dict(8)),
        ("Flag", SINT2),
    ]
)

UF_DSI2 = OrderedDict(
    [
        ("DataType", string_dict(2)),
        ("FieldHeaderPosition", SINT2),
    ]
)

UF_DATA_HEADER = OrderedDict(
    [
        ("FieldsThisRay", SINT2),
        ("RecordsThisRay", SINT2),
        ("FieldsThisRecord", SINT2),
    ]
)
UF_FIELD_VE = OrderedDict(
    [
        ("NyquistVelocity", SINT2),
        ("pad1", SINT2),
    ]
)

UF_FIELD_DM = OrderedDict(
    [
        ("RadarConstant", SINT2),
        ("NoisePower", SINT2),
        ("ReceiverGain", SINT2),
        ("PeakPower", SINT2),
        ("AntennaGain", SINT2),
        ("PulseDuration", BIN2),
    ]
)

UF_FSI_VE = OrderedDict(
    [
        ("VE", UF_FIELD_VE),
    ]
)

UF_FSI_DM = OrderedDict(
    [
        ("DM", UF_FIELD_DM),
    ]
)


UF_FIELD_HEADER = OrderedDict(
    [
        ("DataPosition", SINT2),  # Origin 1 offset of data from start of record
        ("ScaleFactor", SINT2),  # Met units = file value/ScaleFactor
        ("StartRangeKm", SINT2),
        ("StartRangeMeters", SINT2),
        ("BinSpacing", SINT2),  # Bin spacing in meters
        ("BinCount", SINT2),
        ("PulseWidth", SINT2),  # Pulse width in meters
        ("BeamWidthH", BIN2),  # Horizontal Beam width in 1/64 of degree
        ("BeamWidthV", BIN2),
        ("BandWidth", BIN2),  # Receiver bandwidth in 1/64 Mhz
        ("Polarization", SINT2),
        ("WaveLength", BIN2),  # Wavelength in 1/64 of a cm
        ("SampleSize", SINT2),  # Sample size
        ("ThresholdData", string_dict(2)),  # Type of data used to threshold
        ("ThresholdValue", SINT2),
        ("Scale", SINT2),
        ("EditCode", string_dict(2)),
        ("PRT", SINT2),  # PRT in microseconds
        ("BitsPerBin", SINT2),  # Must be 16
    ]
)

LEN_UF_DATA_HEADER = struct.calcsize(_get_fmt_string(UF_DATA_HEADER))
LEN_UF_DSI2 = struct.calcsize(_get_fmt_string(UF_DSI2))
LEN_UF_FIELD_HEADER = struct.calcsize(_get_fmt_string(UF_FIELD_HEADER))
LEN_UF_FIELD_VE = struct.calcsize(_get_fmt_string(UF_FIELD_VE))
LEN_UF_FIELD_DM = struct.calcsize(_get_fmt_string(UF_FIELD_DM))
LEN_UF_MANDATORY_HEADER = struct.calcsize(_get_fmt_string(UF_MANDATORY_HEADER))
LEN_UF_OPTIONAL_HEADER = struct.calcsize(_get_fmt_string(UF_OPTIONAL_HEADER))


class UFArrayWrapper(BackendArray):
    """Wraps array of UF Universal Format data."""

    def __init__(self, datastore, name, var):
        self.datastore = datastore
        self.group = datastore._group
        self.name = name
        # get rays and bins
        nrays = datastore.ds["nrays"]
        nbins = var["BinCount"]
        # for all data use int16 in UF
        self.dtype = np.dtype("int16")
        self.shape = (nrays, nbins)

    def _getitem(self, key):
        with self.datastore.lock:
            # read the data and put it into dict
            self.datastore.root.get_moment(self.group, self.name)
            return self.datastore.ds["sweep_data"][self.name]["data"][key]

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._getitem,
        )


class UFStore(AbstractDataStore):
    """Store for reading UF sweeps via xradar."""

    def __init__(self, manager, group=None, lock=UF_LOCK):
        self._manager = manager
        self._group = int(group[6:]) + 1
        self._filename = self.filename
        self._need_time_recalc = False
        self.lock = ensure_lock(lock)

    @classmethod
    def open(cls, filename, mode="r", group=None, lock=None, **kwargs):
        if lock is None:
            lock = UF_LOCK
        manager = CachingFileManager(UFFile, filename, mode=mode, kwargs=kwargs)
        return cls(manager, group=group, lock=lock)

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
            ds = root.data[self._group]
        return ds

    @property
    def ds(self):
        return self._acquire()

    def open_store_variable(self, name, var):
        sm = _sweep_mode(self.ds["mhead"]["SweepMode"])
        nodata = np.int16(self.ds["mhead"]["NoDataValue"])
        if sm == "PPI":
            dim = "azimuth"
        elif sm == "RHI":
            dim = "elevation"
        else:
            dim = None
        data = indexing.LazilyOuterIndexedArray(UFArrayWrapper(self, name, var))
        encoding = {"group": self._group, "source": self._filename}

        mname = uf_mapping.get(name, name)
        mapping = sweep_vars_mapping.get(mname, {})
        attrs = {key: mapping[key] for key in moment_attrs if key in mapping}
        attrs["scale_factor"] = 1.0 / var["ScaleFactor"]
        attrs["_FillValue"] = nodata
        attrs["coordinates"] = (
            "elevation azimuth range latitude longitude altitude time"
        )
        return mname, Variable((dim, "range"), data, attrs, encoding)

    def open_store_coordinates(self):
        mhead = self.ds["mhead"]
        sm = _sweep_mode(mhead["SweepMode"])
        sweep_number = mhead["SweepNumber"]
        fixed_angle = mhead["FixedAngle"]
        prt_mode = "not_set"
        follow_mode = "not_set"
        if sm == "PPI":
            sweep_mode = "azimuth_surveillance"
            dim = "azimuth"
        elif sm == "RHI":
            sweep_mode = "rhi"
            dim = "elevation"
        else:
            dim = None

        lat_deg = mhead["LatDegrees"]
        lat_min = mhead["LatMinutes"]
        lat_sec = mhead["LatSeconds"]
        lon_deg = mhead["LonDegrees"]
        lon_min = mhead["LonMinutes"]
        lon_sec = mhead["LonSeconds"]
        lat = lat_deg + (lat_min + lat_sec / 60.0) / 60.0
        lon = lon_deg + (lon_min + lon_sec / 60.0) / 60.0
        alt = mhead["Altitude"]

        headers = self.root._ray_headers[self._group]
        azimuth = [hdr["mhead"]["Azimuth"] for hdr in headers]
        elevation = [hdr["mhead"]["Elevation"] for hdr in headers]

        azimuth = Variable((dim,), azimuth, get_azimuth_attrs())
        elevation = Variable((dim,), elevation, get_elevation_attrs())

        # range
        first_key = next(iter(self.ds["sweep_data"].values()))
        start_range = first_key["StartRangeMeters"]
        range_step = first_key["BinSpacing"]
        stop_range = range_step * first_key["BinCount"]
        rng = np.arange(
            start_range + range_step / 2,
            stop_range + range_step / 2,
            range_step,
            dtype="float32",
        )
        range_attrs = get_range_attrs(rng)
        rng = Variable(("range",), rng, range_attrs)

        # making-up ray times
        mhead = self.ds["mhead"]
        year = mhead["Year"]
        month = mhead["Month"]
        day = mhead["Day"]
        hour = mhead["Hour"]
        minute = mhead["Minute"]
        second = mhead["Second"]
        tz = mhead["TimeZone"]
        start = f"{year}-{month}-{day} {hour}:{minute}:{second} {tz}"
        tzinfos = {
            "UT": dateutil.tz.tzutc(),  # “Universal Time” ≡ UTC (offset 0)
        }
        start = dateutil.parser.parse(start, tzinfos=tzinfos)
        # convert to utc
        start = start.replace(tzinfo=dt.timezone.utc)
        # strip tzinfo
        start = start.replace(tzinfo=None)
        start2 = np.array(start).astype("=M8[us]")
        # because no stop_time is available, get time from rotation speed
        if dim == "azimuth":
            angle = azimuth
        else:
            angle = elevation
        rtime = np.full(angle.shape, start2)
        ctime = np.cumulative_sum(np.diff(angle) * 1e6 / mhead["SweepRate"]).astype(
            "=m8[us]"
        )
        rtime[1:] += ctime
        time_prefix = "micro"
        rtime_attrs = {
            "units": f"{time_prefix}seconds since {start.replace(tzinfo=None).isoformat()}Z",
            "standard_name": "time",
        }
        rtime = Variable((dim,), rtime, rtime_attrs)

        coords = {
            "azimuth": azimuth,
            "elevation": elevation,
            "range": rng,
            "time": rtime,
            "sweep_mode": Variable((), sweep_mode),
            "sweep_number": Variable((), sweep_number),
            "prt_mode": Variable((), prt_mode),
            "follow_mode": Variable((), follow_mode),
            "sweep_fixed_angle": Variable((), fixed_angle),
            "longitude": Variable((), lon, get_longitude_attrs()),
            "latitude": Variable((), lat, get_latitude_attrs()),
            "altitude": Variable((), alt, get_altitude_attrs()),
        }

        return coords

    def get_variables(self):
        return FrozenDict(
            (k1, v1)
            for k1, v1 in {
                **dict(
                    self.open_store_variable(k, v)
                    for k, v in self.ds["sweep_data"].items()
                ),
                **self.open_store_coordinates(),
            }.items()
        )

    def get_attrs(self):
        attributes = {}
        attributes["source"] = "Sigmet/UF"
        attributes["site_name"] = self.ds["mhead"]["SiteName"].strip()
        attributes["instrument_name"] = self.ds["mhead"]["RadarName"].strip()
        attributes["comment"] = self.ds["mhead"]["ConvertName"].strip()
        return FrozenDict(attributes)


class UFBackendEntrypoint(BackendEntrypoint):
    """Xarray BackendEntrypoint for Universal Format (UF) data."""

    description = "Open Universal Format (UF) files in Xarray"
    url = "https://xradar.rtfd.io/latest/io.html#uf-data-i-o"

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
        lock=None,
        first_dim="auto",
        reindex_angle=False,
        fix_second_angle=False,
        site_coords=True,
        optional=True,
    ):
        store = UFStore.open(
            filename_or_obj,
            group=group,
            lock=lock,
            loaddata=False,
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

        ds.encoding["engine"] = "uf"

        # handle duplicates and reindex
        if decode_coords and reindex_angle is not False:
            ds = ds.pipe(util.remove_duplicate_rays)
            ds = ds.pipe(util.reindex_angle, **reindex_angle)
            ds = ds.pipe(util.ipol_time, **reindex_angle)

        # handling first dimension
        dim0 = "elevation" if ds.sweep_mode.load() == "rhi" else "azimuth"

        # todo: could be optimized
        if first_dim == "time":
            ds = ds.swap_dims({dim0: "time"})
            ds = ds.sortby("time")
        else:
            ds = ds.sortby(dim0)

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
