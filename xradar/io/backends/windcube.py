#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Vaisala WindCube
================

This sub-module contains the Vaisala (formerly Leosphere) WindCube scanning
Doppler lidar xarray backend for reading WindCube NetCDF-4 files into Xarray
structures, as well as a reader to create a complete :py:class:`xarray.DataTree`.

WindCube files declare ``Conventions = "CF/Radial 2.0 , CF-1.7"`` and follow
the CfRadial2 group layout: the sweep groups (e.g. ``Sweep_152468-1``) are
listed in the root variable ``sweep_group_name``, the fixed angles in the root
``sweep_fixed_angle`` (azimuth for RHI, elevation for all other modes). Each
sweep group holds the scan geometry and the measurements, documented in the
files through ``long_name``, ``units`` and ``comments`` attributes, e.g.

- ``radial_wind_speed`` (m s-1, positive away from the instrument),
  ``radial_wind_speed_ci`` (confidence index, percent) and
  ``radial_wind_speed_status`` (0 rejected, 1 accepted),
- ``cnr`` (carrier-to-noise ratio, dB), ``relative_beta`` (attenuated
  relative backscatter, m-1 sr-1), ``doppler_spectrum_width`` (m s-1),
- ``time``: end of each ray measurement, ``ray_accumulation_time`` in ms.

``range`` is either a dimension or, for DBS/VAD scans, a variable
``range(time, gate_index)``; such sweeps are split into one sweep per distinct
gate geometry (e.g. inclined and vertical DBS beams). Vendor sweep modes are
mapped to CfRadial 2.1 sweep modes, the original mode is kept in the
``windcube_sweep_mode`` attribute of ``sweep_mode``. Over-the-top RHIs are
unfolded to elevations from 0 to 180 degrees (rays at ``fixed_angle + 180``
get ``180 - elevation``), rays without valid angles are dropped.

Example::

    import xradar as xd
    dtree = xd.io.open_windcube_datatree(filename)

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = [
    "WindCubeBackendEntrypoint",
    "open_windcube_datatree",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import numpy as np
import xarray as xr
from xarray import DataTree
from xarray.backends.common import BackendEntrypoint

from ...model import (
    get_azimuth_attrs,
    get_elevation_attrs,
    get_range_attrs,
)
from .common import (
    _apply_site_as_coords,
    _attach_sweep_groups,
    _get_required_root_dataset,
)

#: WindCube sweep modes mapped to CfRadial 2.1 sweep modes; None: decided
#: from the geometry (see ``_cfradial_sweep_mode``)
_SWEEP_MODES = {
    "sector": "sector",
    "coplane": "coplane",
    "rhi": "rhi",
    "manual_rhi": "rhi",
    "vertical_pointing": "vertical_pointing",
    "idle": "idle",
    "azimuth_surveillance": "azimuth_surveillance",
    "elevation_surveillance": "elevation_surveillance",
    "sunscan": "sunscan",
    "manual_ppi": "manual_ppi",
    "dbs": "doppler_beam_swinging",
    "segment": "complex_trajectory",
    "ppi": None,
    "volume": None,
    "vad": None,
    "fixed": None,
    "multifixed": None,
}

#: variables without data content (binary scan/settings/resolution files)
_DROP_VARS = ["res_file", "scan_file", "settings_file", "timestamp", "timestamp_local"]


def _decode(value):
    value = np.asarray(value).ravel()[0]
    return (value.decode() if isinstance(value, bytes) else str(value)).strip()


def _cfradial_sweep_mode(mode, azimuth, elevation):
    """Map a WindCube sweep mode to a CfRadial 2.1 sweep mode."""
    mapped = _SWEEP_MODES.get(mode, mode)
    if mapped is not None:
        return mapped
    if mode in ("fixed", "multifixed"):
        return "vertical_pointing" if np.nanmin(elevation) > 89.5 else "pointing"
    # ppi, volume, vad: full circle or sector
    # full circle unless a gap exceeds both twice the typical azimuth spacing
    # and 30 degrees (a few dropped rays don't make a sector)
    az = np.sort(np.asarray(azimuth) % 360)
    gaps = np.diff(np.concatenate([az, az[:1] + 360]))
    if gaps.size > 1 and gaps.max() <= max(2 * np.median(gaps), 30.0) + 1e-6:
        return "azimuth_surveillance"
    return "sector"


def _fix_attrs(ds):
    """Fix vendor attribute spelling and add missing ``long_name``."""
    for var in ds.variables.values():
        if "ancilliary_variables" in var.attrs:
            var.attrs["ancillary_variables"] = var.attrs.pop("ancilliary_variables")
        if "comments" in var.attrs and "comment" not in var.attrs:
            var.attrs["comment"] = var.attrs.pop("comments")
    for name in ds.data_vars:
        ds[name].attrs.setdefault("long_name", name)
    return ds


#: first dimension per CfRadial sweep mode (``time`` otherwise)
_DIM0 = {
    "rhi": "elevation",
    "manual_rhi": "elevation",
    "elevation_surveillance": "elevation",
    "azimuth_surveillance": "azimuth",
    "sector": "azimuth",
    "manual_ppi": "azimuth",
}


def _unfold_rhi(ds, fixed_angle):
    """Unfold an over-the-top RHI into elevations from 0 to 180 degrees.

    Rays measured at the opposite azimuth (``fixed_angle + 180``) get the
    elevation ``180 - elevation``; the measured azimuths are kept.
    """
    opposite = np.abs(((ds.azimuth.values - fixed_angle) + 180) % 360 - 180) > 90
    if not opposite.any():
        return ds
    elevation = np.where(opposite, 180.0 - ds.elevation.values, ds.elevation.values)
    return ds.assign(elevation=ds.elevation.copy(data=elevation))


def _split_gate_geometry(ds):
    """Split rays by gate geometry if ``range`` is ``(time, gate_index)``."""
    if "gate_index" not in ds.dims:
        return [ds.drop_vars("gate_index", errors="ignore")]
    rng = ds["range"].values
    rows, index = np.unique(rng, axis=0, return_inverse=True)
    sweeps = []
    for i, row in enumerate(rows):
        sub = ds.isel(time=np.flatnonzero(index.ravel() == i))
        sub = sub.drop_vars(["range", "gate_index"], errors="ignore")
        sub = sub.rename_dims({"gate_index": "range"})
        sweeps.append(sub.assign_coords(range=("range", row)))
    return sweeps


def _normalize_sweep(ds, fixed_angle, first_dim="auto"):
    """Convert a WindCube sweep group into an xradar sweep Dataset."""
    ds = ds.drop_vars(_DROP_VARS, errors="ignore")
    mode = _decode(ds["sweep_mode"].values)
    ds = ds.drop_vars("sweep_mode")

    # rays without valid angles (scanning head transitions) are dropped
    valid = np.isfinite(ds["azimuth"].values) & np.isfinite(ds["elevation"].values)
    ds = ds.isel(time=np.flatnonzero(valid))

    sweeps = []
    parts = _split_gate_geometry(ds)
    for sub in parts:
        cf_mode = _cfradial_sweep_mode(mode, sub.azimuth.values, sub.elevation.values)
        # split DBS/VAD parts (e.g. vertical beam) get their own fixed angle
        angle = (
            float(np.nanmedian(sub.elevation.values)) if len(parts) > 1 else fixed_angle
        )
        if cf_mode == "rhi":
            sub = _unfold_rhi(sub, angle)
        rng = sub["range"].values.astype("float32")
        sub = sub.assign_coords(range=("range", rng, get_range_attrs(rng)))
        sub["azimuth"].attrs.update(get_azimuth_attrs())
        sub["elevation"].attrs.update(get_elevation_attrs())
        sub["sweep_mode"] = xr.Variable((), cf_mode, {"windcube_sweep_mode": mode})
        sub["sweep_fixed_angle"] = xr.Variable((), np.float32(angle))
        sub["follow_mode"] = xr.Variable((), "none")
        sub["prt_mode"] = xr.Variable((), "not_set")
        sub = sub.set_coords(["azimuth", "elevation"])
        sub = _fix_attrs(sub)

        # angles don't change along pointing/DBS sweeps, keep time there
        dim0 = _DIM0.get(cf_mode, "time")
        if first_dim == "auto" and dim0 != "time":
            sub = sub.swap_dims({"time": dim0}).sortby(dim0)
        else:
            sub = sub.sortby("time")
        sweeps.append(sub)
    return sweeps


def _decode_times(ds):
    """Decode times, resolving ``seconds since time_reference``.

    Older WindCube files (e.g. WindCube Lidar server 3.3.3) give the epoch
    in a ``time_reference`` string variable of the sweep group and refer to
    it by name in the ``units`` of ``time``.
    """
    if "time_reference" in ds:
        reference = _decode(ds["time_reference"].values)
        for var in ds.variables.values():
            units = var.attrs.get("units")
            if isinstance(units, str) and units.endswith("since time_reference"):
                var.attrs["units"] = units.replace("time_reference", reference)
    return xr.decode_cf(ds, decode_timedelta=False)


def _read_windcube(filename_or_obj, first_dim="auto"):
    """Read all sweeps and the root group of a WindCube file."""
    with xr.open_datatree(
        filename_or_obj,
        engine="h5netcdf",
        decode_times=False,
        decode_timedelta=False,
    ) as tree:
        root = _decode_times(tree.ds.load())
        names = [_decode(n) for n in np.atleast_1d(root["sweep_group_name"].values)]
        fixed = np.atleast_1d(root["sweep_fixed_angle"].values)
        sweeps = []
        for name, angle in zip(names, fixed, strict=False):
            if name not in tree.children:
                continue
            # no inherited root coordinates (older files have a root "sweep")
            group = _decode_times(tree[name].to_dataset(inherit=False).load())
            # interrupted scans leave groups without measurements
            if "time" not in group.dims or group.sizes["time"] == 0:
                continue
            sweeps.extend(_normalize_sweep(group, angle, first_dim=first_dim))

    if not sweeps:
        raise ValueError(f"No WindCube sweeps found in `{filename_or_obj}`.")

    site = {k: root[k] for k in ("latitude", "longitude", "altitude") if k in root}
    attrs = {
        k: v.decode() if isinstance(v, bytes) else v for k, v in root.attrs.items()
    }
    for i, sweep in enumerate(sweeps):
        sweep["sweep_number"] = xr.Variable((), np.int32(i))
        sweeps[i] = sweep.assign(site)
        sweeps[i].attrs = dict(attrs)
    return sweeps


class WindCubeBackendEntrypoint(BackendEntrypoint):
    """Xarray BackendEntrypoint for Vaisala WindCube lidar data.

    Keyword Arguments
    -----------------
    group : str
        Sweep to read, ``sweep_<n>`` with ``n`` counting the sweeps after
        splitting DBS/VAD gate geometries. Defaults to ``sweep_0``.
    first_dim : str
        Can be ``time`` or ``auto`` first dimension. If set to ``auto``,
        first dimension will be either ``azimuth`` or ``elevation`` depending on
        type of sweep. Defaults to ``auto``.
    site_as_coords : bool
        Attach radar site-coordinates to Dataset, defaults to ``True``.
    """

    description = "Open Vaisala WindCube lidar NetCDF-4 files in Xarray"
    url = "https://xradar.rtfd.io/en/latest/io.html#windcube"

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        group="sweep_0",
        first_dim="auto",
        site_as_coords=True,
    ):
        sweeps = _read_windcube(filename_or_obj, first_dim=first_dim)
        index = int(str(group).rsplit("_", 1)[-1])
        try:
            ds = sweeps[index]
        except IndexError as err:
            raise ValueError(
                f"Group `{group}` missing from file `{filename_or_obj}`."
            ) from err
        if drop_variables:
            ds = ds.drop_vars(drop_variables, errors="ignore")
        ds = _apply_site_as_coords(ds, site_as_coords)
        ds.encoding["engine"] = "windcube"
        return ds

    def guess_can_open(self, filename_or_obj):
        try:
            with xr.open_dataset(filename_or_obj, engine="h5netcdf") as ds:
                # "WindCube data", older files "Leosphere Windcube data"
                return "windcube" in str(ds.attrs.get("title", "")).lower()
        except Exception:
            return False


def open_windcube_datatree(filename_or_obj, **kwargs):
    """Open Vaisala WindCube lidar dataset as :py:class:`xarray.DataTree`.

    Parameters
    ----------
    filename_or_obj : str, Path, file-like or DataStore
        Strings and Path objects are interpreted as a path to a local or remote
        WindCube NetCDF-4 file.

    Keyword Arguments
    -----------------
    sweep : int, list of int, optional
        Sweep number(s) to extract (counting sweeps after splitting DBS/VAD gate
        geometries). If None (default), all sweeps are extracted.
    first_dim : str
        Can be ``time`` or ``auto`` first dimension. If set to ``auto``,
        first dimension will be either ``azimuth`` or ``elevation`` depending on
        type of sweep. Defaults to ``auto``.
    optional : bool
        Import optional mandatory data and metadata, defaults to ``True``.

    Returns
    -------
    dtree: xarray.DataTree
        DataTree with CfRadial2 groups.
    """
    sweep = kwargs.pop("sweep", None)
    first_dim = kwargs.pop("first_dim", "auto")
    optional = kwargs.pop("optional", True)

    sweeps = _read_windcube(filename_or_obj, first_dim=first_dim)
    if sweep is not None:
        sweep = [sweep] if isinstance(sweep, int) else list(sweep)
        sweeps = [sweeps[i] for i in sweep]

    dtree = {"/": _get_required_root_dataset(sweeps, optional=optional)}
    dtree = _attach_sweep_groups(dtree, sweeps)
    return DataTree.from_dict(dtree)
