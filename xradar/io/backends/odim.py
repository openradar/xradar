#!/usr/bin/env python
# Copyright (c) 2022, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""

ODIM_H5
=======

This sub-module contains the ODIM_H5 xarray backend for reading ODIM_H5-based radar
data into Xarray structures as well as a reader to create a complete datatree.Datatree.

Code ported from wradlib.

Example::

    import xradar as xd
    dtree = xd.io.open_odim_datatree(filename)

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}

"""

__all__ = [
    "OdimBackendEntrypoint",
    "open_odim_datatree",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import datetime as dt
import io

import h5netcdf
import numpy as np
import xarray as xr
from datatree import DataTree
from packaging.version import Version
from xarray.backends.common import (
    AbstractDataStore,
    BackendArray,
    BackendEntrypoint,
    find_root_and_group,
)
from xarray.backends.file_manager import CachingFileManager, DummyFileManager
from xarray.backends.locks import SerializableLock, ensure_lock
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.utils import FrozenDict, is_remote_uri
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
    _assign_root,
    _attach_sweep_groups,
    _fix_angle,
    _get_h5group_names,
    _maybe_decode,
)

HDF5_LOCK = SerializableLock()


def _calculate_angle_res(dim):
    # need to sort dim first
    angle_diff = np.diff(sorted(dim))
    angle_diff2 = np.abs(np.diff(angle_diff))

    # only select angle_diff, where angle_diff2 is less than 0.1 deg
    # Todo: currently 0.05 is working in most cases
    #  make this robust or parameterisable
    angle_diff_wanted = angle_diff[:-1][angle_diff2 < 0.05]
    return np.round(np.nanmean(angle_diff_wanted), decimals=2)


class _OdimH5NetCDFMetadata:
    """Wrapper around OdimH5 data fileobj for easy access of metadata.

    Parameters
    ----------
    fileobj : file-like
        h5netcdf filehandle.
    group : str
        odim group to acquire

    Returns
    -------
    object : metadata object
    """

    def __init__(self, fileobj, group):
        self._root = fileobj
        self._group = group

    @property
    def first_dim(self):
        dim, _ = self._get_fixed_dim_and_angle()
        return dim

    def get_variable_dimensions(self, dims):
        dimensions = []
        for n, _ in enumerate(dims):
            if n == 0:
                dimensions.append(self.first_dim)
            elif n == 1:
                dimensions.append("range")
            else:
                pass
        return tuple(dimensions)

    @property
    def coordinates(self):
        azimuth = self.azimuth
        elevation = self.elevation
        a1gate = self.a1gate
        rtime = self.ray_times
        dim, angle = self.fixed_dim_and_angle
        angle_res = _calculate_angle_res(locals()[dim])
        dims = ("azimuth", "elevation")
        if dim == dims[1]:
            dims = (dims[1], dims[0])

        az_attrs = get_azimuth_attrs()
        el_attrs = get_elevation_attrs()

        # do not forward a1gate and angle_res for now
        # az_attrs["a1gate"] = a1gate
        #
        # if dim == "azimuth":
        #     az_attrs["angle_res"] = angle_res
        # else:
        #     el_attrs["angle_res"] = angle_res

        sweep_mode = "azimuth_surveillance" if dim == "azimuth" else "rhi"
        sweep_number = int(self._group.split("/")[0][7:]) - 1
        prt_mode = "not_set"
        follow_mode = "not_set"

        rtime_attrs = {
            "units": "seconds since 1970-01-01T00:00:00Z",
            "standard_name": "time",
        }

        range_data, cent_first, bin_range = self.range
        range_attrs = get_range_attrs(range_data)

        lon_attrs = get_longitude_attrs()
        lat_attrs = get_latitude_attrs()
        alt_attrs = get_altitude_attrs()

        time_attrs = get_time_attrs("1970-01-01T00:00:00Z")

        lon, lat, alt = self.site_coords

        # todo: add CF attributes where not yet available
        coordinates = {
            "azimuth": Variable((dims[0],), azimuth, az_attrs),
            "elevation": Variable((dims[0],), elevation, el_attrs),
            "time": Variable((dims[0],), rtime, rtime_attrs),
            "range": Variable(("range",), range_data, range_attrs),
            # "time": Variable((), self.time, time_attrs),
            "sweep_mode": Variable((), sweep_mode),
            "sweep_number": Variable((), sweep_number),
            "prt_mode": Variable((), prt_mode),
            "follow_mode": Variable((), follow_mode),
            "sweep_fixed_angle": Variable((), angle),
            "longitude": Variable((), lon, lon_attrs),
            "latitude": Variable((), lat, lat_attrs),
            "altitude": Variable((), alt, alt_attrs),
        }
        return coordinates

    @property
    def site_coords(self):
        return self._get_site_coords()

    @property
    def time(self):
        return self._get_time()

    @property
    def fixed_dim_and_angle(self):
        return self._get_fixed_dim_and_angle()

    @property
    def range(self):
        return self._get_range()

    @property
    def what(self):
        return self._get_dset_what()

    def _get_azimuth_how(self):
        grp = self._group.split("/")[0]
        how = self._root[grp]["how"].attrs
        startaz = how["startazA"]
        stopaz = how.get("stopazA", False)
        if stopaz is False:
            # stopazA missing
            # create from startazA
            stopaz = np.roll(startaz, -1)
            stopaz[-1] += 360
        zero_index = np.where(stopaz < startaz)
        stopaz[zero_index[0]] += 360
        azimuth_data = (startaz + stopaz) / 2.0
        azimuth_data[azimuth_data >= 360] -= 360
        return azimuth_data

    def _get_azimuth_where(self):
        grp = self._group.split("/")[0]
        nrays = self._root[grp]["where"].attrs["nrays"]
        res = 360.0 / nrays
        azimuth_data = np.arange(res / 2.0, 360.0, res, dtype="float32")
        return azimuth_data

    def _get_fixed_dim_and_angle(self):
        grp = self._group.split("/")[0]
        dim = "elevation"

        # try RHI first
        angle_keys = ["az_angle", "azangle"]
        angle = None
        for ak in angle_keys:
            angle = self._root[grp]["where"].attrs.get(ak, None)
            if angle is not None:
                break
        if angle is None:
            dim = "azimuth"
            angle = self._root[grp]["where"].attrs["elangle"]

        # do not round angle
        # angle = np.round(angle, decimals=1)
        return dim, angle

    def _get_elevation_how(self):
        grp = self._group.split("/")[0]
        how = self._root[grp]["how"].attrs
        startaz = how.get("startelA", False)
        stopaz = how.get("stopelA", False)
        if startaz is not False and stopaz is not False:
            elevation_data = (startaz + stopaz) / 2.0
        else:
            elevation_data = how["elangles"]
        return elevation_data

    def _get_elevation_where(self):
        grp = self._group.split("/")[0]
        nrays = self._root[grp]["where"].attrs["nrays"]
        elangle = self._root[grp]["where"].attrs["elangle"]
        elevation_data = np.ones(nrays, dtype="float32") * elangle
        return elevation_data

    def _get_time_how(self):
        grp = self._group.split("/")[0]
        startT = self._root[grp]["how"].attrs["startazT"]
        stopT = self._root[grp]["how"].attrs["stopazT"]
        time_data = (startT + stopT) / 2.0
        return time_data

    def _get_time_what(self, nrays=None):
        grp = self._group.split("/")[0]
        what = self._root[grp]["what"].attrs
        startdate = _maybe_decode(what["startdate"])
        starttime = _maybe_decode(what["starttime"])
        # take care for missing enddate/endtime
        # see https://github.com/wradlib/wradlib/issues/563
        enddate = _maybe_decode(what.get("enddate", what["startdate"]))
        endtime = _maybe_decode(what.get("endtime", what["starttime"]))
        start = dt.datetime.strptime(startdate + starttime, "%Y%m%d%H%M%S")
        end = dt.datetime.strptime(enddate + endtime, "%Y%m%d%H%M%S")
        start = start.replace(tzinfo=dt.timezone.utc).timestamp()
        end = end.replace(tzinfo=dt.timezone.utc).timestamp()
        if nrays is None:
            nrays = self._root[grp]["where"].attrs["nrays"]
        if start == end:
            import warnings

            warnings.warn(
                "xradar: Equal ODIM `starttime` and `endtime` "
                "values. Can't determine correct sweep start-, "
                "end- and raytimes.",
                UserWarning,
            )

            time_data = np.ones(nrays) * start
        else:
            delta = (end - start) / nrays
            time_data = np.arange(start + delta / 2.0, end, delta)
            time_data = np.roll(time_data, shift=+self.a1gate)
        return time_data

    def _get_ray_times(self, nrays=None):
        try:
            time_data = self._get_time_how()
            self._need_time_recalc = False
        except (AttributeError, KeyError, TypeError):
            time_data = self._get_time_what(nrays=nrays)
            self._need_time_recalc = True
        return time_data

    def _get_range(self):
        grp = self._group.split("/")[0]
        where = self._root[grp]["where"].attrs
        ngates = where["nbins"]
        range_start = where["rstart"] * 1000.0
        bin_range = where["rscale"]
        cent_first = range_start + bin_range / 2.0
        range_data = np.arange(
            cent_first, range_start + bin_range * ngates, bin_range, dtype="float32"
        )
        return range_data, cent_first, bin_range

    def _get_time(self, point="start"):
        grp = self._group.split("/")[0]
        what = self._root[grp]["what"].attrs
        startdate = _maybe_decode(what[f"{point}date"])
        starttime = _maybe_decode(what[f"{point}time"])
        start = dt.datetime.strptime(startdate + starttime, "%Y%m%d%H%M%S")
        start = start.replace(tzinfo=dt.timezone.utc).timestamp()
        return start

    def _get_a1gate(self):
        grp = self._group.split("/")[0]
        a1gate = self._root[grp]["where"].attrs["a1gate"]
        return a1gate

    def _get_site_coords(self):
        lon = self._root["where"].attrs["lon"]
        lat = self._root["where"].attrs["lat"]
        alt = self._root["where"].attrs["height"]
        return lon, lat, alt

    def _get_dset_what(self):
        attrs = {}
        what = self._root[self._group]["what"].attrs
        gain = what.get("gain", 1.0)
        offset = what.get("offset", 0.0)
        if gain != 1.0 and offset != 0.0:
            attrs["scale_factor"] = gain
            attrs["add_offset"] = offset
            attrs["_FillValue"] = what.get("nodata", None)
            attrs["_Undetect"] = what.get("undetect", 0.0)
        # if no quantity is given, use the group-name
        attrs["quantity"] = _maybe_decode(
            what.get("quantity", self._group.split("/")[-1])
        )
        return attrs

    @property
    def a1gate(self):
        return self._get_a1gate()

    @property
    def azimuth(self):
        try:
            azimuth = self._get_azimuth_how()
        except (AttributeError, KeyError, TypeError):
            azimuth = self._get_azimuth_where()
        return azimuth

    @property
    def elevation(self):
        try:
            elevation = self._get_elevation_how()
        except (AttributeError, KeyError, TypeError):
            elevation = self._get_elevation_where()
        return elevation

    @property
    def ray_times(self):
        return self._get_ray_times()


class H5NetCDFArrayWrapper(BackendArray):
    """H5NetCDFArrayWrapper

    adapted from https://github.com/pydata/xarray/
    """

    __slots__ = ("datastore", "dtype", "shape", "variable_name")

    def __init__(self, variable_name, datastore):
        self.datastore = datastore
        self.variable_name = variable_name

        array = self.get_array()
        self.shape = array.shape

        dtype = array.dtype
        if dtype is str:
            # use object dtype because that's the only way in numpy to
            # represent variable length strings; it also prevents automatic
            # string concatenation via conventions.decode_cf_variable
            dtype = np.dtype("O")
        self.dtype = dtype

    def __setitem__(self, key, value):
        with self.datastore.lock:
            data = self.get_array(needs_lock=False)
            data[key] = value
            if self.datastore.autoclose:
                self.datastore.close(needs_lock=False)

    def get_array(self, needs_lock=True):
        ds = self.datastore._acquire(needs_lock)
        return ds.variables[self.variable_name]

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.OUTER_1VECTOR, self._getitem
        )

    def _getitem(self, key):
        # h5py requires using lists for fancy indexing:
        # https://github.com/h5py/h5py/issues/992
        key = tuple(list(k) if isinstance(k, np.ndarray) else k for k in key)
        with self.datastore.lock:
            array = self.get_array(needs_lock=False)
            return array[key]


def _get_h5netcdf_encoding(self, var):
    """get encoding from h5netcdf Variable

    adapted from https://github.com/pydata/xarray/
    """
    import h5py

    # netCDF4 specific encoding
    encoding = {
        "chunksizes": var.chunks,
        "fletcher32": var.fletcher32,
        "shuffle": var.shuffle,
    }

    # Convert h5py-style compression options to NetCDF4-Python
    # style, if possible
    if var.compression == "gzip":
        encoding["zlib"] = True
        encoding["complevel"] = var.compression_opts
    elif var.compression is not None:
        encoding["compression"] = var.compression
        encoding["compression_opts"] = var.compression_opts

    # save source so __repr__ can detect if it's local or not
    encoding["source"] = self._filename
    encoding["original_shape"] = var.shape

    vlen_dtype = h5py.check_dtype(vlen=var.dtype)
    if vlen_dtype is str:
        encoding["dtype"] = str
    elif vlen_dtype is not None:  # pragma: no cover
        # xarray doesn't support writing arbitrary vlen dtypes yet.
        pass
    else:
        encoding["dtype"] = var.dtype
    return encoding


def _get_odim_variable_name_and_attrs(name, attrs):
    if "data" in name:
        name = attrs.pop("quantity")
        # handle non-standard moment names
        try:
            mapping = sweep_vars_mapping[name]
        except KeyError:
            pass
        else:
            attrs.update({key: mapping[key] for key in moment_attrs})
        attrs["coordinates"] = "elevation azimuth range"
    return name, attrs


class OdimSubStore(AbstractDataStore):
    """Store for reading ODIM data-moments via h5netcdf."""

    def __init__(
        self,
        store,
        group=None,
        lock=False,
    ):
        if not isinstance(store, OdimStore):
            raise TypeError(
                f"Wrong type {type(store)} for parameter store, "
                f"expected 'OdimStore'."
            )

        self._manager = store._manager
        self._group = group
        self._filename = store.filename
        self.is_remote = is_remote_uri(self._filename)
        self.lock = ensure_lock(lock)

    @property
    def root(self):
        with self._manager.acquire_context(False) as root:
            return _OdimH5NetCDFMetadata(root, self._group.lstrip("/"))

    def _acquire(self, needs_lock=True):
        with self._manager.acquire_context(needs_lock) as root:
            ds = root[self._group.lstrip("/")]
        return ds

    @property
    def ds(self):
        return self._acquire()

    def open_store_variable(self, name, var):
        dimensions = self.root.get_variable_dimensions(var.dimensions)
        data = indexing.LazilyOuterIndexedArray(H5NetCDFArrayWrapper(name, self))
        encoding = _get_h5netcdf_encoding(self, var)
        encoding["group"] = self._group
        name, attrs = _get_odim_variable_name_and_attrs(name, self.root.what)

        return name, Variable(dimensions, data, attrs, encoding)

    def open_store_coordinates(self):
        return self.root.coordinates

    def get_variables(self):
        return FrozenDict(
            (k1, v1)
            for k1, v1 in {
                **dict(
                    [
                        self.open_store_variable(k, v)
                        for k, v in self.ds.variables.items()
                    ]
                ),
            }.items()
        )


class OdimStore(AbstractDataStore):
    """Store for reading ODIM dataset groups via h5netcdf."""

    def __init__(self, manager, group=None, lock=False):
        if isinstance(manager, (h5netcdf.File, h5netcdf.Group)):
            if group is None:
                root, group = find_root_and_group(manager)
            else:
                if type(manager) is not h5netcdf.File:
                    raise ValueError(
                        "must supply a h5netcdf.File if the group "
                        "argument is provided"
                    )
                root = manager
            manager = DummyFileManager(root)

        self._manager = manager
        self._group = f"/dataset{int(group[6:])+1}"
        self._filename = self.filename
        self.is_remote = is_remote_uri(self._filename)
        self.lock = ensure_lock(lock)
        self._substore = None
        self._need_time_recalc = False

    @classmethod
    def open(
        cls,
        filename,
        mode="r",
        format=None,
        group=None,
        lock=None,
        invalid_netcdf=None,
        phony_dims=None,
        decode_vlen_strings=True,
    ):
        if isinstance(filename, bytes):
            raise ValueError(
                "can't open netCDF4/HDF5 as bytes "
                "try passing a path or file-like object"
            )

        if format not in [None, "NETCDF4"]:
            raise ValueError("invalid format for h5netcdf backend")

        kwargs = {"invalid_netcdf": invalid_netcdf}
        if phony_dims is not None:
            if Version(h5netcdf.__version__) >= Version("0.8.0"):
                kwargs["phony_dims"] = phony_dims
            else:
                raise ValueError(
                    "h5netcdf backend keyword argument 'phony_dims' needs "
                    "h5netcdf >= 0.8.0."
                )
        if Version(h5netcdf.__version__) >= Version("0.10.0") and Version(
            h5netcdf.core.h5py.__version__
        ) >= Version("3.0.0"):
            kwargs["decode_vlen_strings"] = decode_vlen_strings

        if lock is None:
            if util.has_import("dask"):
                lock = HDF5_LOCK
            else:
                lock = False

        manager = CachingFileManager(h5netcdf.File, filename, mode=mode, kwargs=kwargs)
        return cls(manager, group=group, lock=lock)

    @property
    def filename(self):
        with self._manager.acquire_context(False) as root:
            return root.filename

    @property
    def substore(self):
        if self._substore is None:
            with self._manager.acquire_context(False) as root:
                subgroups = [
                    "/".join([self._group, k])
                    for k in root[self._group].groups
                    # get data and quality groups
                    if "data" or "quality" in k
                ]
                substore = []
                substore.extend(
                    [
                        OdimSubStore(
                            self,
                            group=group,
                            lock=self.lock,
                        )
                        for group in subgroups
                    ]
                )
                self._substore = substore

        return self._substore

    def open_store_coordinates(self):
        return self.substore[0].open_store_coordinates()

    def get_variables(self):
        return FrozenDict(
            (k1, v1)
            for k1, v1 in {
                **{
                    k: v
                    for substore in self.substore
                    for k, v in substore.get_variables().items()
                },
                **self.open_store_coordinates(),
            }.items()
        )

    def get_attrs(self):
        dim, angle = self.substore[0].root.fixed_dim_and_angle
        attributes = {}
        # attributes["fixed_angle"] = angle.item()
        return FrozenDict(attributes)


class OdimBackendEntrypoint(BackendEntrypoint):
    """Xarray BackendEntrypoint for ODIM data.

    Keyword Arguments
    -----------------
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
    """

    description = "Open ODIM_H5 (.h5, .hdf5) using h5netcdf in Xarray"
    url = "https://xradar.rtfd.io/en/latest/io.html#odim-h5"

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
        reindex_angle=False,
        fix_second_angle=False,
        site_coords=True,
    ):
        if isinstance(filename_or_obj, io.IOBase):
            filename_or_obj.seek(0)

        store = OdimStore.open(
            filename_or_obj,
            format=format,
            group=group,
            invalid_netcdf=invalid_netcdf,
            phony_dims=phony_dims,
            decode_vlen_strings=decode_vlen_strings,
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

        ds.encoding["engine"] = "odim"

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

        # fix second angle
        if fix_second_angle and first_dim == "auto":
            dim1 = {"azimuth": "elevation", "elevation": "azimuth"}[dim0]
            ds = ds.assign_coords({dim1: ds[dim1].pipe(_fix_angle)})

        # assign geo-coords
        ds = ds.assign_coords(
            {
                "latitude": ds.latitude,
                "longitude": ds.longitude,
                "altitude": ds.altitude,
            }
        )

        return ds


def open_odim_datatree(filename_or_obj, **kwargs):
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
            sweeps = [f"sweep_{i+1}" for i in sweep]
        else:
            sweeps.extend(sweep)
    else:
        sweeps = _get_h5group_names(filename_or_obj, "odim")

    ds = [
        xr.open_dataset(filename_or_obj, group=swp, engine="odim", **kwargs)
        for swp in sweeps
    ]

    # todo: apply CfRadial2 group structure below
    ds.insert(0, xr.open_dataset(filename_or_obj, group="/"))

    # create datatree root node with required data
    dtree = DataTree(data=_assign_root(ds), name="root")
    # return datatree with attached sweep child nodes
    return _attach_sweep_groups(dtree, ds[1:])
