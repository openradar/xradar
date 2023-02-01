#!/usr/bin/env python
# Copyright (c) 2022, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Rainbow Data I/O
^^^^^^^^^^^^^^^^

This sub-module contains the Rainbow xarray backend for reading data from Leonardo's
Rainbow5 data formats into Xarray structures as well as a reader to create a complete
datatree.Datatree. For this `mmap.mmap` is utilized.

Code ported from wradlib.

Example::

    import xradar as xd
    dtree = xd.io.open_rainbow_datatree(filename)

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}

"""

__all__ = [
    "RainbowBackendEntrypoint",
    "open_rainbow_datatree",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import datetime as dt
import sys
import zlib

import numpy as np
import xarray as xr
import xmltodict
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
from .common import _attach_sweep_groups
from .odim import _assign_root

#: mapping of rainbow moment names to CfRadial2/ODIM names
rainbow_mapping = {
    "dBuZ": "DBTH",
    "dBZ": "DBZH",
    "dBuZv": "DBTV",
    "dBZv": "DBZV",
    "V": "VRADH",
    "W": "WRADH",
    "ZDR": "ZDR",
    "KDP": "KDP",
    "PhiDP": "PHIDP",
    "SQI": "SQIH",
    "SNR": "SNR",
    "RhoHV": "RHOHV",
}


def _get_dict_value(d, k1, k2):
    v = d.get(k1, None)
    if v is None:
        v = d[k2]
    return v


def find_key(key, dictionary):
    """Searches for given key in given (nested) dictionary.

    Returns all found parent dictionaries in a list.

    Parameters
    ----------
    key : str
        the key to be searched for in the nested dict
    dictionary : dict
        the dictionary to be searched

    Yields
    -------
    output : dict
        a dictionary or list of dictionaries
    """
    for k, v in dictionary.items():
        if k == key:
            yield dictionary
        elif isinstance(v, dict):
            yield from find_key(key, v)
        elif isinstance(v, list):
            for d in v:
                if isinstance(d, dict):
                    yield from find_key(key, d)


def decompress(data):
    """Decompression of data

    Parameters
    ----------
    data : str
        (from xml) data string containing compressed data.
    """
    return zlib.decompress(data)


def get_rb_data_layout(datadepth):
    """Calculates DataWidth and DataType from given DataDepth of
    RAINBOW radar data

    Parameters
    ----------
    datadepth : int
        DataDepth as read from the Rainbow xml metadata.

    Returns
    -------
    datawidth : int
        Width in Byte of data.
    datatype : str
        conversion string .
    """

    if sys.byteorder != "big":
        byteorder = ">"
    else:
        byteorder = "<"

    datawidth = int(datadepth / 8)

    if datawidth in [1, 2, 4]:
        datatype = byteorder + "u" + str(datawidth)
    else:
        raise ValueError(
            f"Wrong DataDepth: {datadepth}. Conversion only for depth 8, 16, 32."
        )

    return datawidth, datatype


def get_rb_data_attribute(xmldict, attr):
    """Get Attribute `attr` from dict `xmldict`

    Parameters
    ----------
    xmldict : dict
        Blob Description Dictionary
    attr : str
        Attribute key

    Returns
    -------
    sattr : int
        Attribute Values
    """

    try:
        sattr = int(xmldict["@" + attr])
    except KeyError:
        raise KeyError(
            f"Attribute @{attr} is missing from "
            "Blob Description. There may be some "
            "problems with your file"
        )
    return sattr


def get_rb_blob_attribute(blobdict, attr):
    """Get Attribute `attr` from dict `blobdict`

    Parameters
    ----------
    blobdict : dict
        Blob Description Dictionary
    attr : str
        Attribute key

    Returns
    -------
    ret : Attribute Value
    """
    try:
        value = blobdict["BLOB"]["@" + attr]
    except KeyError:
        raise KeyError(
            "Attribute @"
            + attr
            + " is missing from Blob."
            + "There may be some problems with your file"
        )

    return value


def get_rb_blob_data(datastring, blobid):
    """Read BLOB data from datastring and return it

    Parameters
    ----------
    datastring : str
        Blob Description String
    blobid : int
        Number of requested blob

    Returns
    -------
    data : str
        Content of blob
    """
    start = 0
    search_string = f'<BLOB blobid="{blobid}"'
    start = datastring.find(search_string.encode(), start)
    if start == -1:
        raise EOFError(f"Blob ID {blobid} not found!")
    end = datastring.find(b">", start)
    xmlstring = datastring[start : end + 1]

    # cheat the xml parser by making xml well-known
    xmldict = xmltodict.parse(xmlstring.decode() + "</BLOB>")
    cmpr = get_rb_blob_attribute(xmldict, "compression")
    size = int(get_rb_blob_attribute(xmldict, "size"))
    data = datastring[end + 2 : end + 2 + size]  # read blob data to string

    # decompress if necessary
    if cmpr == "qt":
        # the first 4 bytes contain the uncompressed size in big endian
        usize = int.from_bytes(data[:4], "big")
        data = decompress(data[4:])
        if len(data) != usize:
            raise ValueError(
                f"Data size mismatch. {usize} bytes expected, "
                f"{len(data)} bytes read."
            )

    return data


def map_rb_data(data, datadepth, datashape=0):
    """Map BLOB data to correct DataWidth and Type and convert it
    to numpy array

    Parameters
    ----------
    data : str
        Blob Data
    datadepth : int
        bit depth of Blob data
    datashape : tuple
        expected data shape, only used for the flags to truncate

    Returns
    -------
    data : :py:class:`numpy:numpy.ndarray`
        Content of blob
    """
    flagdepth = None
    if datadepth < 8:
        flagdepth = datadepth
        datadepth = 8

    datawidth, datatype = get_rb_data_layout(datadepth)

    # import from data buffer well aligned to data array
    data = np.ndarray(shape=(int(len(data) / datawidth),), dtype=datatype, buffer=data)

    if flagdepth:
        data = np.unpackbits(data)[0 : np.prod(datashape)]

    return data


def get_rb_data_shape(blobdict):
    """Retrieve correct BLOB data shape from blobdict

    Parameters
    ----------
    blobdict : dict
        Blob Description Dict

    Returns
    -------
    shape : tuple
        shape of data
    """
    # this is a bit hacky, but we do not know beforehand,
    # so we extract this on the run
    try:
        dim0 = get_rb_data_attribute(blobdict, "rays")
        dim1 = get_rb_data_attribute(blobdict, "bins")
        # if rays and bins are found, return both
        return dim0, dim1
    except KeyError as e1:
        try:
            # if only rays is found, return rays
            return dim0
        except UnboundLocalError:
            try:
                # if both rays and bins not found assuming pixmap
                dim0 = get_rb_data_attribute(blobdict, "rows")
                dim1 = get_rb_data_attribute(blobdict, "columns")
                dim2 = get_rb_data_attribute(blobdict, "depth")
                if dim2 < 8:
                    # if flagged data return rows x columns x depth
                    return dim0, dim1, dim2
                else:
                    # otherwise just rows x columns
                    return dim0, dim1
            except KeyError as e2:
                # if no some keys are missing, print errors and raise
                print(e1)
                print(e2)
                raise


def get_rb_blob_from_string(datastring, blobdict):
    """Read BLOB data from datastring and return it as numpy array with correct
    dataWidth and shape

    Parameters
    ----------
    datastring : str
        Blob Description String
    blobdict : dict
        Blob Description Dict

    Returns
    -------
    data : :py:class:`numpy:numpy.ndarray`
        Content of blob as numpy array
    """

    blobid = get_rb_data_attribute(blobdict, "blobid")
    data = get_rb_blob_data(datastring, blobid)
    # map data to correct datatype and width
    datadepth = get_rb_data_attribute(blobdict, "depth")
    datashape = get_rb_data_shape(blobdict)
    data = map_rb_data(data, datadepth, datashape)

    # reshape data
    data.shape = datashape

    return data


def get_rb_header(fid):
    """Read Rainbow Header from filename, converts it to a dict and returns it

    Parameters
    ----------
    fid : object
        File handle of Data File

    Returns
    -------
    object : dict
        Rainbow File Contents
    """

    # load the header lines, i.e. the XML part
    end_xml_marker = b"<!-- END XML -->"
    header = b""
    line = b""

    while not line.startswith(end_xml_marker):
        header += line[:-1]
        line = fid.readline()
        if len(line) == 0:
            raise OSError("WRADLIB: Rainbow Fileheader Corrupt")

    return xmltodict.parse(header)


class RainbowFileBase:
    """Base class for Rainbow Files."""

    def __init__(self, **kwargs):
        super().__init__()


class RainbowFile(RainbowFileBase):
    """RainbowFile class"""

    def __init__(self, filename, **kwargs):
        self._debug = kwargs.get("debug", False)
        self._rawdata = kwargs.get("rawdata", False)
        self._loaddata = kwargs.get("loaddata", True)

        self._fp = None
        self._filename = filename
        if isinstance(filename, str):
            self._fp = open(filename, "rb")
            import mmap

            self._fh = mmap.mmap(self._fp.fileno(), 0, access=mmap.ACCESS_READ)
        else:
            raise TypeError(
                "Rainbow5 reader currently doesn't support file-like objects"
            )
        self._data = None
        super().__init__(**kwargs)
        # read rainbow header
        self._header = get_rb_header(self._fh)["volume"]
        self._coordinates = None
        slices = self._header["scan"]["slice"]
        if not isinstance(slices, list):
            slices = [slices]
        else:
            self._update_volume_slices()
        self._blobs = [list(find_key("@blobid", slc)) for slc in slices]
        if self._loaddata:
            for i, slc in enumerate(self._blobs):
                for blob in slc:
                    blobid = get_rb_data_attribute(blob, "blobid")
                    self.get_blob(blobid, i)

    def close(self):
        if self._fp is not None:
            self._fp.close()

    __del__ = close

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    @property
    def filename(self):
        return self._filename

    @property
    def version(self):
        return self._header["@version"]

    @property
    def type(self):
        return self._header["@type"]

    @property
    def datetime(self):
        return dt.datetime.strptime(self._header["@datetime"], "%Y-%m-%dT%H:%M:%S")

    @property
    def first_dimension(self):
        if self.type in ["vol", "azi"]:
            return "azimuth"
        elif self.type in ["ele"]:
            return "elevation"
        elif self.type in ["poi"]:
            raise NotImplementedError(
                "Rainbow5 data of type `poi` (pointmode) is currently not supported."
            )
        else:
            raise TypeError(f"Unknown Rainbow File Type: {self.type}")

    @property
    def header(self):
        return self._header

    @property
    def blobs(self):
        return self._blobs

    @property
    def slices(self):
        slices = self._header["scan"]["slice"]
        if not isinstance(slices, list):
            slices = [slices]
        return slices

    @property
    def pargroup(self):
        return self._header["scan"]["pargroup"]

    @property
    def sensorinfo(self):
        try:
            return self.header["sensorinfo"]
        except KeyError:
            return self.header.get("radarinfo", None)

    @property
    def history(self):
        return self.header.get("history", None)

    @property
    def site_coords(self):
        si = self.sensorinfo
        return (
            float(_get_dict_value(si, "lon", "@lon")),
            float(_get_dict_value(si, "lat", "@lat")),
            float(_get_dict_value(si, "alt", "@alt")),
        )

    def _get_rbdict_value(self, rbdict, name, dtype=None, default=None):
        value = rbdict.get(name, None)
        if value is None:
            value = self.pargroup.get(name, default)
        if dtype is not None:
            value = dtype(value)
        return value

    def _update_volume_slices(self):
        if isinstance(self._header["scan"]["slice"], list):
            slice0 = self._header["scan"]["slice"][0]
            for i, slice in enumerate(self._header["scan"]["slice"][1:]):
                newdict = dict(list(slice0.items()) + list(slice.items()))
                self._header["scan"]["slice"][i + 1] = newdict

    def get_blob(self, blobid, slc):
        self._fh.seek(0)
        blob = next(filter(lambda x: int(x["@blobid"]) == blobid, self._blobs[slc]))
        if blob.get("data", False) is False:
            data = get_rb_blob_from_string(self._fh, blob)
            # azimuth
            if blob.get("@refid", "") in ["startangle", "stopangle"]:
                # anglestep = self._get_rbdict_value(self.slices[slc], "anglestep", None, float)
                # anglestep = self.slices[slc].get("anglestep", None)
                # if anglestep is None:
                #     anglestep = self.pargroup["anglestep"]
                # anglestep = float(anglestep)
                # todo: correctly decode elevation angles
                #   elevation can decode negative values
                data = data * 360.0 / 2 ** float(blob["@depth"])
            blob["data"] = data


class RainbowArrayWrapper(BackendArray):
    """Wraps array of RAINBOW5 data."""

    def __init__(self, datastore, name, var):
        self.datastore = datastore
        self.name = name

        # get rays and bins
        nrays = int(var.get("@rays", False))
        nbins = int(var.get("@bins", False))
        dtype = np.dtype(f"uint{var.get('@depth')}")
        self.dtype = dtype
        if nbins:
            self.shape = (nrays, nbins)
        else:
            self.shape = (nrays,)
        self.blobid = int(var["@blobid"])

    def _getitem(self, key):
        # read the data and put it into dict
        self.datastore.root.get_blob(self.blobid, self.datastore._group)
        if isinstance(self.name, int):
            return self.datastore.ds["slicedata"]["rayinfo"][self.name]["data"][key]
        else:
            return self.datastore.ds["slicedata"]["rawdata"]["data"][key]

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._getitem,
        )


class RainbowStore(AbstractDataStore):
    """Store for reading RAINBOW5 sweeps via wradlib."""

    def __init__(self, manager, group=None):
        self._manager = manager
        self._group = int(group[6:])
        self._filename = self.filename
        self._need_time_recalc = False

    @classmethod
    def open(cls, filename, mode="r", group=None, **kwargs):
        manager = CachingFileManager(RainbowFile, filename, mode=mode, kwargs=kwargs)
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
            try:
                ds = root.header["scan"]["slice"][self._group]
            except KeyError:
                ds = root.header["scan"]["slice"]
        return ds

    @property
    def ds(self):
        return self._acquire()

    def open_store_variable(self, var):
        dim = self.root.first_dimension
        raw = var["slicedata"]["rawdata"]
        name = raw["@type"]

        data = indexing.LazilyOuterIndexedArray(RainbowArrayWrapper(self, name, raw))
        encoding = {"group": self._group, "source": self._filename}

        vmin = float(raw.get("@min"))
        vmax = float(raw.get("@max"))
        depth = int(raw.get("@depth"))
        scale_factor = (vmax - vmin) / (2**depth - 2)
        mname = rainbow_mapping.get(name, name)
        mapping = sweep_vars_mapping.get(mname, {})
        attrs = {key: mapping[key] for key in moment_attrs if key in mapping}
        attrs["add_offset"] = vmin - scale_factor
        attrs["scale_factor"] = scale_factor
        attrs["_FillValue"] = 0
        attrs[
            "coordinates"
        ] = "elevation azimuth range latitude longitude altitude time"
        return {mname: Variable((dim, "range"), data, attrs, encoding)}

    def open_store_coordinates(self, var):
        dim = self.root.first_dimension
        ray = var["slicedata"]["rayinfo"]

        if not isinstance(ray, list):
            var["slicedata"]["rayinfo"] = [ray]
            ray = var["slicedata"]["rayinfo"]

        start = next(filter(lambda x: x["@refid"] == "startangle", ray), False)
        start_idx = ray.index(start)
        stop = next(filter(lambda x: x["@refid"] == "stopangle", ray), False)

        anglestep = self.root._get_rbdict_value(var, "anglestep", dtype=float)
        antdirection = self.root._get_rbdict_value(
            var, "antdirection", default=0, dtype=bool
        )

        encoding = {"group": self._group}
        startangle = indexing.LazilyOuterIndexedArray(
            RainbowArrayWrapper(self, start_idx, start)
        )
        step = anglestep
        # antdirection == True ->> negative angles
        # antdirection == False ->> positive angles
        if antdirection:
            step = -anglestep

        if dim == "azimuth":
            startaz = Variable((dim,), startangle, get_azimuth_attrs(), encoding)
            if stop:
                stop_idx = ray.index(stop)
                stopangle = indexing.LazilyOuterIndexedArray(
                    RainbowArrayWrapper(self, stop_idx, stop)
                )
                stopaz = Variable((dim,), stopangle, get_azimuth_attrs(), encoding)
                zero_index = np.where(startaz - stopaz > 5)
                stopazv = stopaz.values
                stopazv[zero_index[0]] += 360
                azimuth = (startaz + stopazv) / 2.0
                azimuth[azimuth >= 360] -= 360
            else:
                azimuth = startaz + step / 2.0
                azimuth[azimuth < 0] += 360

            elevation = np.ones_like(azimuth) * float(var["posangle"])
        else:
            startel = Variable((dim,), startangle, get_azimuth_attrs(), encoding)

            if stop:
                stop_idx = ray.index(stop)
                stopangle = indexing.LazilyOuterIndexedArray(
                    RainbowArrayWrapper(self, stop_idx, stop)
                )
                stopel = Variable((dim,), stopangle, get_elevation_attrs(), encoding)
                elevation = (startel + stopel) / 2.0
            else:
                elevation = startel + step / 2.0

            azimuth = np.ones_like(elevation) * float(var["posangle"])

        dstr = var["slicedata"]["@date"]
        tstr = var["slicedata"]["@time"]

        timestr = f"{dstr}T{tstr}Z"
        time = dt.datetime.strptime(timestr, "%Y-%m-%dT%H:%M:%SZ")

        # range is in km
        start_range = self.root._get_rbdict_value(
            var, "startrange", default=0, dtype=float
        )
        start_range *= 1000.0

        stop_range = self.root._get_rbdict_value(var, "stoprange", dtype=float)
        stop_range *= 1000.0

        range_step = self.root._get_rbdict_value(var, "rangestep", dtype=float)
        range_step *= 1000.0
        rng = np.arange(
            start_range + range_step / 2,
            stop_range + range_step / 2,
            range_step,
            dtype="float32",
        )[: int(var["slicedata"]["rawdata"]["@bins"])]

        range_attrs = get_range_attrs(rng)

        # making-up ray times
        antspeed = self.root._get_rbdict_value(var, "antspeed", dtype=float)
        raytime = anglestep / antspeed
        raytimes = np.array(
            [
                dt.timedelta(seconds=x * raytime).total_seconds()
                for x in range(azimuth.shape[0] + 1)
            ]
        )

        diff = np.diff(raytimes) / 2.0
        rtime = raytimes[:-1] + diff
        time_attrs = get_time_attrs(f"{time.isoformat()}Z")

        rng = Variable(("range",), rng, range_attrs)
        azimuth = Variable((dim,), azimuth, get_azimuth_attrs(), encoding)
        elevation = Variable((dim,), elevation, get_elevation_attrs(), encoding)
        time = Variable((dim,), rtime, time_attrs, encoding)

        # get coordinates from RainbowFile
        sweep_mode = "azimuth_surveillance" if dim == "azimuth" else "rhi"
        sweep_number = self._group
        prt_mode = "not_set"
        follow_mode = "not_set"

        lon, lat, alt = self.root.site_coords

        coords = {
            "azimuth": azimuth,
            "elevation": elevation,
            "range": rng,
            "time": time,
            "sweep_mode": Variable((), sweep_mode),
            "sweep_number": Variable((), sweep_number),
            "prt_mode": Variable((), prt_mode),
            "follow_mode": Variable((), follow_mode),
            "sweep_fixed_angle": Variable((), float(var["posangle"])),
            "longitude": Variable((), lon, get_longitude_attrs()),
            "latitude": Variable((), lat, get_latitude_attrs()),
            "altitude": Variable((), alt, get_altitude_attrs()),
        }

        return coords

    def get_variables(self):
        return FrozenDict(
            (k1, v1)
            for k1, v1 in {
                **self.open_store_variable(self.ds),
                **self.open_store_coordinates(self.ds),
            }.items()
        )

    def get_attrs(self):
        return FrozenDict()


class RainbowBackendEntrypoint(BackendEntrypoint):
    """Xarray BackendEntrypoint for Rainbow5 data."""

    description = "Open Rainbow5 files in Xarray"
    url = "https://xradar.rtfd.io/latest/io.html#rainbow-data-i-o"

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
        store = RainbowStore.open(
            filename_or_obj,
            group=group,
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

        ds.encoding["engine"] = "rainbow"

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


def _get_rainbow_group_names(filename):
    with RainbowFile(filename, loaddata=False) as fh:
        cnt = len(fh.slices)
    return [f"sweep_{i}" for i in range(cnt)]


def open_rainbow_datatree(filename_or_obj, **kwargs):
    """Open ODIM_H5 dataset as xradar Datatree.

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
        sweeps = _get_rainbow_group_names(filename_or_obj)

    ds = [
        xr.open_dataset(filename_or_obj, group=swp, engine="rainbow", **kwargs)
        for swp in sweeps
    ]

    ds.insert(0, xr.Dataset())  # open_dataset(filename_or_obj, group="/"))

    # create datatree root node with required data
    dtree = DataTree(data=_assign_root(ds), name="root")
    # return datatree with attached sweep child nodes
    return _attach_sweep_groups(dtree, ds[1:])
