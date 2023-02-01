#!/usr/bin/env python
# Copyright (c) 2022, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""

GAMIC HDF5
==========

This sub-module contains the GAMIC HDF5 xarray backend for reading GAMIC HDF5-based radar
data into Xarray structures as well as a reader to create a complete datatree.Datatree.

Code ported from wradlib.

Example::

    import xradar as xd
    dtree = xd.io.open_gamic_datatree(filename)

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}

"""

__all__ = [
    "GamicBackendEntrypoint",
    "open_gamic_datatree",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import datetime as dt
import io

import dateutil
import h5netcdf
import numpy as np
import xarray as xr
from datatree import DataTree
from packaging.version import Version
from xarray.backends.common import (
    AbstractDataStore,
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
from .common import _assign_root, _attach_sweep_groups, _fix_angle, _get_h5group_names
from .odim import H5NetCDFArrayWrapper, _get_h5netcdf_encoding

HDF5_LOCK = SerializableLock()

gamic_mapping = {
    "zh": "DBZH",
    "zv": "DBZV",
    "uh": "DBTH",
    "uzh": "DBTH",
    "uv": "DBTV",
    "uzv": "DBTV",
    "vh": "VRADH",
    "vv": "VRADV",
    "wh": "WRADH",
    "uwh": "UWRADH",
    "wv": "WRADV",
    "uwv": "UWRADV",
    "zdr": "ZDR",
    "uzdr": "UZDR",
    "ldr": "LDR",
    "phidp": "PHIDP",
    "uphidp": "UPHIDP",
    "kdp": "KDP",
    "rhohv": "RHOHV",
    "urhohv": "URHOHV",
    "cmap": "CMAP",
}


def _get_gamic_variable_name_and_attrs(attrs, dtype):
    name = attrs.pop("moment").lower()
    try:
        name = gamic_mapping[name]
        mapping = sweep_vars_mapping[name]
    except KeyError:
        # ds = ds.drop_vars(mom)
        pass
    else:
        attrs.update({key: mapping[key] for key in moment_attrs})

    dmax = np.iinfo(dtype).max
    dmin = np.iinfo(dtype).min
    minval = attrs.pop("dyn_range_min")
    maxval = attrs.pop("dyn_range_max")
    dtype = minval.dtype
    dyn_range = maxval - minval
    if maxval != minval:
        gain = dyn_range / (dmax - 1)
        minval -= gain
    else:
        gain = (dmax - dmin) / dmax
        minval = dmin
    # ensure numpy type
    gain = np.array([gain])[0].astype(dtype)
    minval = np.array([minval])[0].astype(dtype)
    undetect = np.array([dmin])[0].astype(dtype)
    attrs["scale_factor"] = gain
    attrs["add_offset"] = minval
    attrs["_FillValue"] = undetect
    attrs["_Undetect"] = undetect

    attrs[
        "coordinates"
    ] = "elevation azimuth range latitude longitude altitude time rtime sweep_mode"

    return name, attrs


def _get_ray_header_data(dimensions, data, encoding):
    ray_header = Variable(dimensions, data, {}, encoding)

    azstart = ray_header.values["azimuth_start"]
    azstop = ray_header.values["azimuth_stop"]
    zero_index = np.where(azstop < azstart)
    azstop[zero_index[0]] += 360
    azimuth = (azstart + azstop) / 2.0

    elstart = ray_header.values["elevation_start"]
    elstop = ray_header.values["elevation_stop"]
    elevation = (elstart + elstop) / 2.0

    time = ray_header.values["timestamp"] / 1e6

    return {"azimuth": azimuth, "elevation": elevation, "time": time}


class _GamicH5NetCDFMetadata:
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

    def coordinates(self, dimensions, data, encoding):
        ray_header = _get_ray_header_data(dimensions, data, encoding)
        dim, angle = self.fixed_dim_and_angle

        dims = ("azimuth", "elevation")
        if dim == dims[1]:
            dims = (dims[1], dims[0])

        sweep_mode = "azimuth_surveillance" if dim == "azimuth" else "rhi"
        sweep_number = int(self._group.split("/")[0][4:])
        prt_mode = "not_set"
        follow_mode = "not_set"

        range_data, cent_first, bin_range = self.range
        range_attrs = get_range_attrs(range_data)

        lon, lat, alt = self.site_coords

        coordinates = {
            "azimuth": Variable(
                (dims[0],),
                ray_header["azimuth"],
                get_azimuth_attrs(),
            ),
            "elevation": Variable(
                (dims[0],),
                ray_header["elevation"],
                get_elevation_attrs(),
            ),
            "time": Variable(
                (dims[0],), ray_header["time"], get_time_attrs("1970-01-01T00:00:00Z")
            ),
            "range": Variable(("range",), range_data, range_attrs),
            "sweep_mode": Variable((), sweep_mode),
            "sweep_number": Variable((), sweep_number),
            "prt_mode": Variable((), prt_mode),
            "follow_mode": Variable((), follow_mode),
            "sweep_fixed_angle": Variable((), angle),
            "longitude": Variable((), lon, get_longitude_attrs()),
            "latitude": Variable((), lat, get_latitude_attrs()),
            "altitude": Variable((), alt, get_altitude_attrs()),
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

    def _get_fixed_dim_and_angle(self):
        how = self._root[self._group]["how"].attrs
        dims = {0: "elevation", 1: "azimuth"}
        try:
            dim = 1
            angle = np.round(how[dims[0]], decimals=1)
        except KeyError:
            dim = 0
            angle = np.round(how[dims[1]], decimals=1)

        return dims[dim], angle

    def _get_range(self):
        how = self._root[self._group]["how"].attrs
        range_samples = how["range_samples"]
        range_step = how["range_step"]
        ngates = how["bin_count"]
        bin_range = range_step * range_samples
        cent_first = bin_range / 2.0
        range_data = np.arange(
            cent_first,
            bin_range * ngates,
            bin_range,
            dtype="float32",
        )
        return range_data, cent_first, bin_range

    def _get_time(self):
        start = self._root[self._group]["how"].attrs["timestamp"]
        start = dateutil.parser.parse(start)
        start = start.replace(tzinfo=dt.timezone.utc).timestamp()
        return start

    def _get_site_coords(self):
        lon = self._root["where"].attrs["lon"]
        lat = self._root["where"].attrs["lat"]
        alt = self._root["where"].attrs["height"]
        return lon, lat, alt


class GamicStore(AbstractDataStore):
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
        self._group = f"scan{int(group[6:])}"
        self._filename = self.filename
        self.is_remote = is_remote_uri(self._filename)
        self.lock = ensure_lock(lock)
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
    def root(self):
        with self._manager.acquire_context(False) as root:
            return _GamicH5NetCDFMetadata(root, self._group.lstrip("/"))

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
        # cheat attributes
        if "moment" in name:
            name, attrs = _get_gamic_variable_name_and_attrs({**var.attrs}, var.dtype)
        elif "ray_header" in name:
            return self.root.coordinates(dimensions, data, encoding)
        else:
            return {}
        return {name: Variable(dimensions, data, attrs, encoding)}

    def get_variables(self):
        return FrozenDict(
            (k1, v1)
            for k, v in self.ds.variables.items()
            for k1, v1 in {
                **self.open_store_variable(k, v),
            }.items()
        )

    def get_attrs(self):
        #     dim, angle = self.root.fixed_dim_and_angle
        #     attributes = {"fixed_angle": angle.item()}
        return FrozenDict()


class GamicBackendEntrypoint(BackendEntrypoint):
    """Xarray BackendEntrypoint for GAMIC data.

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
        For PPI only. If True, fixes erroneous second angle data. Defaults to False.
    site_coords : bool
        Attach radar site-coordinates to Dataset, defaults to ``True``.
    kwargs :  kwargs
        Additional kwargs are fed to `xr.open_dataset`.
    """

    description = "Open GAMIC HDF5 (.h5, .hdf5, .mvol) using h5netcdf in Xarray"
    url = "https://xradar.rtfd.io/en/latest/io.html#gamic-hdf5"

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

        store = GamicStore.open(
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

        ds.encoding["engine"] = "gamic"

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
        if site_coords:
            ds = ds.assign_coords(
                {
                    "latitude": ds.latitude,
                    "longitude": ds.longitude,
                    "altitude": ds.altitude,
                }
            )

        return ds


def open_gamic_datatree(filename_or_obj, **kwargs):
    """Open GAMIC HDF5 dataset as xradar Datatree.

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
            sweeps = [f"sweep_{i}" for i in sweep]
        else:
            sweeps.extend(sweep)
    else:
        sweeps = _get_h5group_names(filename_or_obj, "gamic")

    ds = [
        xr.open_dataset(filename_or_obj, group=swp, engine="gamic", **kwargs)
        for swp in sweeps
    ]

    ds.insert(0, xr.open_dataset(filename_or_obj, group="/"))

    # create datatree root node with required data
    dtree = DataTree(data=_assign_root(ds), name="root")
    # return datatree with attached sweep child nodes
    return _attach_sweep_groups(dtree, ds[1:])
