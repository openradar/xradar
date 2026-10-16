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
gate geometry (e.g. inclined and vertical DBS beams), in the order the beams
were measured. Masked gates of that ``range`` match any value, they do not
make a ray a geometry of its own. Vendor sweep modes are mapped to CfRadial 2.1
sweep modes, the original mode and the name of the group in the file are kept
in the attributes ``windcube_sweep_mode`` and ``windcube_group_name`` of
``sweep_mode``. Over-the-top RHIs are unfolded to elevations from 0 to 180
degrees (rays at ``fixed_angle + 180`` get ``180 - elevation``), rays without
valid angles are dropped. The measurements stay lazy, the geometry (angles,
range, time) is read when the file is opened.

Example::

    import xradar as xd
    dtree = xd.open_datatree(filename, engine="windcube")

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = [
    "WindCubeBackendEntrypoint",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import re
import warnings
from dataclasses import dataclass

import numpy as np
import xarray as xr
from xarray.backends.common import (
    BackendEntrypoint,
    datatree_from_dict_with_io_cleanup,
)

from ...model import (
    get_azimuth_attrs,
    get_elevation_attrs,
    get_range_attrs,
)
from .common import (
    SITE_COORDS_PARAM_DOC,
    _apply_site_as_coords,
    _build_groups_dict,
    _compose_docstring,
    _external_stacklevel,
    _input_kind,
    _read_nc_header,
    _resolve_sweeps,
)

#: WindCube sweep modes which have another name in CfRadial 2.1
_SWEEP_MODES = {
    "manual_rhi": "rhi",
    "dbs": "doppler_beam_swinging",
    "segment": "complex_trajectory",
}

#: WindCube sweep modes with the same name in CfRadial 2.1
_CFRADIAL_MODES = {
    "sector",
    "coplane",
    "rhi",
    "vertical_pointing",
    "idle",
    "azimuth_surveillance",
    "elevation_surveillance",
    "sunscan",
    "manual_ppi",
}

#: WindCube sweep modes decided from the geometry
_POINTING_MODES = {"fixed", "multifixed"}
_PPI_MODES = {"ppi", "volume", "vad"}

#: variables without data content (binary scan/settings/resolution files)
_DROP_VARS = ["res_file", "scan_file", "settings_file", "timestamp", "timestamp_local"]

#: small variables read into memory when the file is opened, the geometry is
#: needed to drop invalid rays, split the gate geometries and detect the mode
_GEOMETRY_VARS = [
    "azimuth",
    "elevation",
    "time",
    "range",
    "gate_index",
    "sweep_mode",
    "time_reference",
]

#: first dimension per CfRadial sweep mode (``time`` otherwise)
_DIM0 = {
    "rhi": "elevation",
    "elevation_surveillance": "elevation",
    "azimuth_surveillance": "azimuth",
    "sector": "azimuth",
    "manual_ppi": "azimuth",
}


def _decode(value):
    value = np.asarray(value).ravel()[0]
    return (value.decode() if isinstance(value, bytes) else str(value)).strip()


def _cfradial_sweep_mode(mode, azimuth, elevation):
    """Map a WindCube sweep mode to a CfRadial 2.1 sweep mode.

    Modes which are not known are decided from the geometry like ``ppi``, with
    a warning.
    """
    name = mode.lower()
    if name in _SWEEP_MODES:
        return _SWEEP_MODES[name]
    if name in _CFRADIAL_MODES:
        return name
    if name in _POINTING_MODES:
        return "vertical_pointing" if np.nanmin(elevation) > 89.5 else "pointing"
    if name not in _PPI_MODES:
        warnings.warn(
            f"xradar: unknown WindCube sweep mode {mode!r}, deriving the "
            "CfRadial sweep mode from the scan geometry.",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )
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


def _fill_masked_gates(row):
    """Fill gates of a range row which are masked in all rays by extrapolation."""
    missing = np.isnan(row)
    if not missing.any() or (~missing).sum() < 2:
        return row
    index = np.arange(row.size)
    fit = np.polyfit(index[~missing], row[~missing], 1)
    return np.where(missing, np.polyval(fit, index), row)


def _gate_geometries(rng):
    """Group the rays of a ``range(time, gate_index)`` by gate geometry.

    Masked gates (NaN) match any value, so a ray with a masked gate stays with
    the rays of its geometry. Returns a list of ``(rays, range)`` in the
    order of the first ray of each geometry, i.e. in acquisition order.
    """
    filled = np.where(np.isnan(rng), -1.0, rng)
    rows, inverse = np.unique(filled, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    templates = []
    row_geometry = np.empty(len(rows), dtype=int)
    # rows without masked gates first, they define the geometries
    for i in np.argsort((rows < 0).sum(axis=1), kind="stable"):
        row = np.where(rows[i] < 0, np.nan, rows[i])
        for k, template in enumerate(templates):
            known = ~np.isnan(row) & ~np.isnan(template)
            if np.array_equal(row[known], template[known]):
                templates[k] = np.where(np.isnan(template), row, template)
                row_geometry[i] = k
                break
        else:
            templates.append(row)
            row_geometry[i] = len(templates) - 1
    labels = row_geometry[inverse]
    geometries = [
        (np.flatnonzero(labels == k), _fill_masked_gates(template))
        for k, template in enumerate(templates)
    ]
    return sorted(geometries, key=lambda geometry: geometry[0][0])


@dataclass
class _Sweep:
    """A sweep of the file: the rays of a group with one gate geometry."""

    group: str  # name of the group in the file
    ds: xr.Dataset  # decoded group, geometry in memory, measurements lazy
    mode: str  # vendor sweep mode
    rays: np.ndarray  # indices into the time axis of ``ds``
    range: np.ndarray | None  # 1-D range if ``ds`` has ``range(time, gate_index)``
    fixed_angle: float
    split: bool  # the group has more than one gate geometry


def _decode_times(ds, **decoders):
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
    return xr.decode_cf(ds, **decoders)


def _open_tree(filename_or_obj):
    """Open the file lazily, the caller closes it."""
    return xr.open_datatree(
        filename_or_obj,
        engine="h5netcdf",
        decode_times=False,
        decode_timedelta=False,
        # the returned datasets are cached by xarray's open_datatree already
        cache=False,
    )


def _scan_tree(tree, filename_or_obj, decoders):
    """Find the sweeps of an opened file, only reading their geometry.

    Returns the sweeps in the order of the file and the root dataset. Groups
    without rays (interrupted scans) or without any ray with valid angles are
    left out.
    """
    root = tree.ds
    names = [_decode(name) for name in np.atleast_1d(root["sweep_group_name"].values)]
    fixed_angles = np.atleast_1d(root["sweep_fixed_angle"].values)
    if len(names) != len(fixed_angles):
        raise ValueError(
            f"`sweep_group_name` ({len(names)}) and `sweep_fixed_angle` "
            f"({len(fixed_angles)}) differ in length in `{filename_or_obj}`."
        )
    missing = [name for name in names if name not in tree.children]
    if missing:
        warnings.warn(
            f"Sweep group(s) listed in `sweep_group_name` but missing: {missing}",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )

    sweeps = []
    for name, fixed_angle in zip(names, fixed_angles, strict=True):
        if name in missing:
            continue
        # no inherited root coordinates (older files have a root "sweep")
        group = tree[name].to_dataset(inherit=False)
        # interrupted scans leave groups without measurements
        if "time" not in group.dims or group.sizes["time"] == 0:
            continue
        group = group.drop_vars(_DROP_VARS, errors="ignore")
        for var in _GEOMETRY_VARS:
            if var in group.variables:
                group[var].variable.load()
        group = _decode_times(group, **decoders)
        mode = _decode(group["sweep_mode"].values)

        # rays without valid angles (scanning head transitions) are dropped
        valid = np.isfinite(group["azimuth"].values) & np.isfinite(
            group["elevation"].values
        )
        if not valid.any():
            continue
        rays = np.flatnonzero(valid)
        if "gate_index" in group.dims:
            geometries = [
                (rays[sub], row)
                for sub, row in _gate_geometries(group["range"].values[rays])
            ]
        else:
            geometries = [(rays, None)]
        for part_rays, row in geometries:
            sweeps.append(
                _Sweep(
                    group=name,
                    ds=group,
                    mode=mode,
                    rays=part_rays,
                    range=row,
                    fixed_angle=float(fixed_angle),
                    split=len(geometries) > 1,
                )
            )
    return sweeps, root


def _site(root):
    """Site variables of the root, NaN if they are missing."""
    site = {}
    for name in ("latitude", "longitude", "altitude"):
        if name in root:
            site[name] = root[name].load()
    missing = [
        name for name in ("latitude", "longitude", "altitude") if name not in site
    ]
    if missing:
        warnings.warn(
            f"xradar: WindCube file without {missing} in the root group, "
            "the site coordinates are NaN.",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )
        for name in missing:
            site[name] = xr.DataArray(np.float64(np.nan))
    return site


def _build_sweep(sweep, root, site, number, first_dim, drop_variables):
    """Make the xradar sweep Dataset of a sweep of the file."""
    ds = sweep.ds.drop_vars(["sweep_mode"])
    ds = ds.isel(time=sweep.rays)
    if sweep.range is None:
        ds = ds.drop_vars("gate_index", errors="ignore")
    else:
        ds = ds.drop_vars(["range", "gate_index"], errors="ignore")
        ds = ds.rename_dims({"gate_index": "range"})
        ds = ds.assign_coords(range=("range", sweep.range))

    cf_mode = _cfradial_sweep_mode(sweep.mode, ds.azimuth.values, ds.elevation.values)
    # split DBS/VAD parts (e.g. vertical beam) get their own fixed angle, an
    # RHI keeps its azimuth
    angle = sweep.fixed_angle
    if sweep.split and cf_mode != "rhi":
        angle = float(np.nanmedian(ds.elevation.values))
    if cf_mode == "rhi":
        ds = _unfold_rhi(ds, angle)

    rng = ds["range"].values.astype("float32")
    ds = ds.assign_coords(range=("range", rng, get_range_attrs(rng)))
    ds["azimuth"].attrs.update(get_azimuth_attrs())
    ds["elevation"].attrs.update(get_elevation_attrs())
    ds["sweep_mode"] = xr.Variable(
        (),
        cf_mode,
        {"windcube_sweep_mode": sweep.mode, "windcube_group_name": sweep.group},
    )
    ds["sweep_number"] = xr.Variable((), np.int32(number))
    ds["sweep_fixed_angle"] = xr.Variable((), np.float32(angle))
    if "follow_mode" not in ds:
        ds["follow_mode"] = xr.Variable((), "none")
    if "prt_mode" not in ds:
        ds["prt_mode"] = xr.Variable((), "fixed")
    ds = ds.set_coords(["azimuth", "elevation"])
    ds = _fix_attrs(ds)

    # angles don't change along pointing/DBS sweeps, keep time there
    dim0 = _DIM0.get(cf_mode, "time")
    if first_dim == "auto" and dim0 != "time":
        ds = ds.swap_dims({"time": dim0}).sortby(dim0)
    else:
        ds = ds.sortby("time")

    if drop_variables:
        ds = ds.drop_vars(drop_variables, errors="ignore")
    ds = ds.assign(site)
    ds.attrs = {
        k: v.decode() if isinstance(v, bytes) else v for k, v in root.attrs.items()
    }
    return ds


_SWEEP_NAME = re.compile(r"sweep_(\d+)$")


def _sweep_indices(names, nsweeps, filename_or_obj):
    """Indices of the sweeps ``sweep_<n>`` in ``names``, missing ones raise."""
    indices, missing = [], []
    for name in names:
        match = _SWEEP_NAME.match(name)
        if match is not None and int(match.group(1)) < nsweeps:
            indices.append(int(match.group(1)))
        else:
            missing.append(name)
    if missing:
        raise ValueError(
            f"Sweep group(s) missing from file `{filename_or_obj}`: {missing}"
        )
    return indices


def _decoders(
    mask_and_scale,
    decode_times,
    concat_characters,
    decode_coords,
    use_cftime,
    decode_timedelta,
):
    return dict(
        mask_and_scale=mask_and_scale,
        decode_times=decode_times,
        concat_characters=concat_characters,
        decode_coords=decode_coords,
        use_cftime=use_cftime,
        decode_timedelta=False if decode_timedelta is None else decode_timedelta,
    )


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
    supports_groups = True

    def guess_can_open(self, filename_or_obj):
        """WindCube NetCDF-4 file (root ``title`` with "WindCube": "WindCube data",
        older files "Leosphere Windcube data")."""
        if _input_kind(filename_or_obj) not in ("path", "file"):
            return False
        header = _read_nc_header(filename_or_obj)
        if header is None:
            return False
        return "windcube" in str(header[0].get("title", "")).lower()

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
        group="sweep_0",
        first_dim="auto",
        site_as_coords=True,
    ):
        decoders = _decoders(
            mask_and_scale,
            decode_times,
            concat_characters,
            decode_coords,
            use_cftime,
            decode_timedelta,
        )
        tree = _open_tree(filename_or_obj)
        try:
            sweeps, root = _scan_tree(tree, filename_or_obj, decoders)
            if not sweeps:
                raise ValueError(f"No WindCube sweeps found in `{filename_or_obj}`.")
            (name,) = _resolve_sweeps(
                "sweep_0" if group is None else group,
                lambda: [f"sweep_{i}" for i in range(len(sweeps))],
            )
            (index,) = _sweep_indices([name], len(sweeps), filename_or_obj)
            ds = _build_sweep(
                sweeps[index],
                root,
                _site(root),
                index,
                first_dim,
                drop_variables,
            )
            ds = _apply_site_as_coords(ds, site_as_coords)
        except Exception:
            tree.close()
            raise
        ds.encoding["engine"] = "windcube"
        ds.set_close(tree._close)
        return ds

    def open_groups_as_dict(
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
        first_dim="auto",
        site_coords=True,
        sweep=None,
        optional=True,
        optional_groups=False,
    ):
        decoders = _decoders(
            mask_and_scale,
            decode_times,
            concat_characters,
            decode_coords,
            use_cftime,
            decode_timedelta,
        )
        tree = _open_tree(filename_or_obj)
        try:
            sweeps, root = _scan_tree(tree, filename_or_obj, decoders)
            if not sweeps:
                raise ValueError(f"No WindCube sweeps found in `{filename_or_obj}`.")
            names = _resolve_sweeps(
                sweep, lambda: [f"sweep_{i}" for i in range(len(sweeps))]
            )
            indices = _sweep_indices(names, len(sweeps), filename_or_obj)
            site = _site(root)
            ls_ds = []
            for number, index in enumerate(indices):
                ds = _build_sweep(
                    sweeps[index], root, site, number, first_dim, drop_variables
                )
                ds = _apply_site_as_coords(ds, site_coords)
                # the file is closed with the tree (once, whichever node closes)
                ds.set_close(tree._close)
                ls_ds.append(ds)
            return _build_groups_dict(
                ls_ds, optional=optional, optional_groups=optional_groups
            )
        except Exception:
            tree.close()
            raise

    def open_datatree(self, filename_or_obj, **kwargs):
        groups_dict = self.open_groups_as_dict(filename_or_obj, **kwargs)
        return datatree_from_dict_with_io_cleanup(groups_dict)


WindCubeBackendEntrypoint.open_groups_as_dict.__doc__ = _compose_docstring(
    "Open a Vaisala WindCube lidar file as a\n"
    "    CfRadial2-shaped dict of group datasets.",
    SITE_COORDS_PARAM_DOC,
)
WindCubeBackendEntrypoint.open_datatree.__doc__ = (
    "Open a Vaisala WindCube lidar file as :py:class:`xarray.DataTree`. "
    "See :meth:`open_groups_as_dict` for keyword arguments.\n"
)
