#!/usr/bin/env python
# Copyright (c) 2024-2025, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
IMD
===

This sub-module contains the xarray backend for reading India Meteorological
Department (IMD) radar NetCDF files, which use a NetCDF4 container with an
IRIS-inspired variable layout. Each IMD file stores a single sweep; a
complete volume is typically assembled from 2-3 files (long-range PPI) up to
9-10 files (short-range, high-resolution PPI).

Code ported from radarx[https://github.com/syedhamidali/radarx].

Example::

    import xradar as xd

    # Single sweep via xarray engine
    ds = xr.open_dataset("sweep_0.nc", engine="imd")

    # Single-sweep DataTree
    dtree = xd.io.open_imd_datatree("sweep_0.nc")

    # Multi-sweep volume DataTree
    dtree = xd.io.open_imd_datatree(["sweep_0.nc", "sweep_1.nc", ...])

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = [
    "IMDBackendEntrypoint",
    "open_imd_datatree",
    "open_imd_volumes",
    "group_imd_files",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import glob as _glob
import os
import re

import numpy as np
import xarray as xr
from xarray import DataTree
from xarray.backends import NetCDF4DataStore
from xarray.backends.common import BackendEntrypoint
from xarray.backends.store import StoreBackendEntrypoint

from ... import util
from ...model import (
    georeferencing_correction_subgroup,
    moment_attrs,
    radar_parameters_subgroup,
    sweep_vars_mapping,
)
from .common import (
    _STATION_VARS,
    _apply_site_as_coords,
    _get_subgroup,
)

#: IMD moment codes to CfRadial2 names. IMD's published NetCDF format ships
#: a limited set of moments: T/Z/V/W (base) and -- on dual-pol sites --
#: ZDR and HCLASS (hydrometeor classification). RHOHV / PHIDP / KDP are
#: *not* distributed by IMD at the time of writing.
imd_mapping = {
    "T": "DBTH",
    "Z": "DBZH",
    "V": "VRADH",
    "W": "WRADH",
    "ZDR": "ZDR",
    "HCLASS": "HCLASS",
}

#: IMD non-moment variable names to CfRadial2 names. The radar_* / pulse_width
#: entries map IMD's metadata scalars onto the canonical names so the
#: ``/radar_parameters`` subgroup is populated when ``optional_groups=True``.
_imd_var_mapping = {
    "radialAzim": "azimuth",
    "radialElev": "elevation",
    "elevationAngle": "fixed_angle",
    "radialTime": "time",
    "nyquist": "nyquist_velocity",
    "unambigRange": "unambiguous_range",
    "siteLat": "latitude",
    "siteLon": "longitude",
    "siteAlt": "altitude",
    "groundHeight": "altitude_agl",
    # Radar parameters subgroup
    "beamWidthHori": "radar_beam_width_h",
    "beamWidthVert": "radar_beam_width_v",
    "bandWidth": "radar_receiver_bandwidth",
    # pulseWidth is handled specially (unit conversion + broadcast to time)
    # in _conform_imd_sweep, not via plain rename.
}

#: IMD-specific calibration-related scalars. These don't map cleanly to
#: CfRadial2's polarization-aware canonicals (``noise_hc``, ``tx_power_h``,
#: etc.), so they're exposed passthrough under ``/radar_calibration``
#: when ``optional_groups=True``.
_IMD_CALIBRATION_VARS = {
    "calibConst",
    "radarConst",
    "calI0",
    "calNoise",
    "logNoise",
    "linNoise",
    "inphaseOffset",
    "quadratureOffset",
}


#: IMD scanType integer -> (sweep_mode, scan_type_name)
_scan_type_lookup = {
    0: ("unknown", "unknown"),
    1: ("sector", "ppi_sector"),
    2: ("manual_rhi", "rhi_sector"),
    4: ("azimuth_surveillance", "ppi"),
    7: ("elevation_surveillance", "rhi"),
}


def _as_scalar(da):
    """Return a python scalar from a 0-d (or single-element) DataArray."""
    return da.values.item()


def _build_imd_calibration(sweep_ds):
    """Collect IMD's cal-related scalar data_vars into a subgroup Dataset.

    IMD's ``calibConst`` / ``radarConst`` / ``calI0`` / ``calNoise`` /
    noise and I/Q offsets don't map to CfRadial2's polarization-aware
    cal dict, so they stay under their IMD names.
    """
    present = _IMD_CALIBRATION_VARS & set(sweep_ds.variables)
    if not present:
        return xr.Dataset()
    return sweep_ds[sorted(present)].reset_coords(drop=True)


def _conform_imd_sweep(ds, first_dim="auto", site_as_coords=True):
    """Transform a raw IMD NetCDF Dataset into a CfRadial2 sweep Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset as produced by opening an IMD NetCDF file directly.
    first_dim : str
        ``"auto"`` (default) sets the first dimension to ``azimuth`` for PPI
        and ``elevation`` for RHI. ``"time"`` keeps the CfRadial2 time
        dimension.
    site_as_coords : bool
        Promote ``latitude``/``longitude``/``altitude`` to coords.

    Returns
    -------
    xarray.Dataset
        Sweep Dataset in CfRadial2 layout.
    """
    # Drop the raw file's `sweep` dim -- in IMD files it describes the parent
    # volume (e.g. size 2 = volume has 2 sweeps), not this sweep's dimensions.
    # The CfRadial2 sweep group must not carry a sweep dim.
    if "sweep" in ds.dims:
        ds = ds.drop_dims("sweep", errors="ignore")

    # Dimension renames
    rename_dims = {k: v for k, v in {"radial": "azimuth", "bin": "range"}.items() if k in ds.dims}
    if rename_dims:
        ds = ds.rename_dims(rename_dims)

    # Variable renames (only those actually present)
    ds = ds.rename_vars(
        {k: v for k, v in _imd_var_mapping.items() if k in ds.variables}
    )
    # Moment renames
    ds = ds.rename_vars(
        {k: v for k, v in imd_mapping.items() if k in ds.variables}
    )
    # Normalize moment attributes. The file carries `units` / `long_name`
    # but no `standard_name`, and `valid_range` / `below_threshold` describe
    # the packed int8 -- stale after mask_and_scale. Drop the stale bits,
    # keep the rest, then overlay canonical CfRadial2 attrs where xradar
    # defines them (e.g. ZDR has a canonical; HCLASS does not). Also set
    # the FM 301 Table 9.iv `coordinates` attribute on each moment.
    _stale_attrs = {"valid_range", "below_threshold", "_FillValue"}
    for mname in set(imd_mapping.values()) & set(ds.variables):
        kept = {k: v for k, v in ds[mname].attrs.items() if k not in _stale_attrs}
        canonical = sweep_vars_mapping.get(mname, {})
        kept.update({k: canonical[k] for k in moment_attrs if k in canonical})
        kept["coordinates"] = "elevation azimuth range"
        ds[mname].attrs = kept

    set_coord = [c for c in ("azimuth", "elevation") if c in ds.variables]
    if set_coord:
        ds = ds.set_coords(set_coord)

    # FM 301 Table 7b: canonical attributes for azimuth/elevation coords.
    if "azimuth" in ds.coords:
        ds["azimuth"].attrs = {
            "units": "degrees",
            "standard_name": "sensor_to_target_azimuth_angle",
            "long_name": "Azimuth angle from true north",
            "axis": "radial_azimuth_coordinate",
        }
    if "elevation" in ds.coords:
        ds["elevation"].attrs = {
            "units": "degrees",
            "standard_name": "sensor_to_target_elevation_angle",
            "long_name": "Elevation angle from horizontal plane",
            "axis": "radial_elevation_coordinate",
        }

    # Build range coordinate from firstGateRange/gateSize
    if "firstGateRange" in ds.variables and "gateSize" in ds.variables:
        first = float(_as_scalar(ds["firstGateRange"]))
        gate = float(_as_scalar(ds["gateSize"]))
        n_gates = ds.sizes["range"]
        ranges = first + np.arange(n_gates) * gate
        ds = ds.assign_coords(
            range=xr.DataArray(
                ranges,
                dims=("range",),
                attrs={
                    # FM 301 Table 6b
                    "units": "meters",
                    "standard_name": "projection_range_coordinate",
                    "long_name": "range_to_measurement_volume",
                    "axis": "radial_range_coordinate",
                    "spacing_is_constant": "true",
                    "meters_to_center_of_first_gate": first,
                    "meters_between_gates": gate,
                },
            )
        )

    # Scan type / sweep_mode
    scan_code = (
        int(_as_scalar(ds["scanType"])) if "scanType" in ds.variables else 4
    )
    sweep_mode, scan_type_name = _scan_type_lookup.get(
        scan_code, ("unknown", "unknown")
    )
    ds["sweep_mode"] = xr.DataArray(sweep_mode)
    ds["scan_type"] = xr.DataArray(scan_type_name)
    # CfRadial2 required sweep metadata -- IMD files don't expose follow_mode.
    if "follow_mode" not in ds.variables:
        ds["follow_mode"] = xr.DataArray("not_set")

    # --- FM 301 Table 8a: PRF-derived time-varying sweep metadata ------------
    # IMD ships these as scalars per sweep; the spec wants (time,) arrays.
    # The ray dim at this point is still "azimuth"; it gets swapped to "time"
    # further down and these vars follow along.
    n_rays = ds.sizes.get("azimuth", 0)
    high_prf = (
        float(_as_scalar(ds["highPRF"])) if "highPRF" in ds.variables else 0.0
    )
    low_prf = (
        float(_as_scalar(ds["lowPRF"])) if "lowPRF" in ds.variables else 0.0
    )
    if high_prf > 0 and n_rays > 0:
        ds["prt"] = xr.DataArray(
            np.full(n_rays, 1.0 / high_prf, dtype="float32"),
            dims=("azimuth",),
            attrs={"units": "seconds", "long_name": "Pulse repetition time"},
        )
        if low_prf > 0 and not np.isclose(low_prf, high_prf):
            ds["prt_ratio"] = xr.DataArray(
                np.full(n_rays, high_prf / low_prf, dtype="float32"),
                dims=("azimuth",),
                attrs={"long_name": "PRT ratio (high/low)"},
            )
            ds["prt_mode"] = xr.DataArray("staggered")
        else:
            ds["prt_mode"] = xr.DataArray("fixed")
    elif "prt_mode" not in ds.variables:
        ds["prt_mode"] = xr.DataArray("not_set")
    ds = ds.drop_vars(
        [v for v in ("highPRF", "lowPRF") if v in ds.variables], errors="ignore"
    )

    # scan_rate from IMD azimuthSpeed (degrees/second, per-sweep scalar)
    if "azimuthSpeed" in ds.variables and n_rays > 0:
        rate = float(_as_scalar(ds["azimuthSpeed"]))
        ds["scan_rate"] = xr.DataArray(
            np.full(n_rays, rate, dtype="float32"),
            dims=("azimuth",),
            attrs={"units": "degrees/s", "long_name": "Antenna scan rate"},
        )
        ds = ds.drop_vars("azimuthSpeed")

    # n_samples from IMD sampleNum
    if "sampleNum" in ds.variables and n_rays > 0:
        ds["n_samples"] = xr.DataArray(
            np.full(n_rays, int(_as_scalar(ds["sampleNum"])), dtype="int32"),
            dims=("azimuth",),
            attrs={
                "long_name": "Maximum number of samples used to compute moments"
            },
        )
        ds = ds.drop_vars("sampleNum")

    # pulse_width: IMD pulseWidth is microseconds and scalar; FM 301 Table 8a
    # wants seconds on the (time,) dim.
    if "pulseWidth" in ds.variables and n_rays > 0:
        pw_seconds = float(_as_scalar(ds["pulseWidth"])) * 1e-6
        ds["pulse_width"] = xr.DataArray(
            np.full(n_rays, pw_seconds, dtype="float32"),
            dims=("azimuth",),
            attrs={"units": "seconds", "long_name": "Pulse width"},
        )
        ds = ds.drop_vars("pulseWidth")

    # frequency coord from IMD waveLength (cm -> Hz). Per FM 301 Table 6a,
    # each sweep has a /frequency coord with frequency dim (usually size 1).
    if "waveLength" in ds.variables:
        wl_cm = float(_as_scalar(ds["waveLength"]))
        if wl_cm > 0:
            freq_hz = 2.99792458e8 / (wl_cm * 1e-2)
            ds = ds.assign_coords(
                frequency=xr.DataArray(
                    np.array([freq_hz], dtype="float32"),
                    dims=("frequency",),
                    attrs={
                        "units": "s-1",
                        "standard_name": "radiation_frequency",
                        "long_name": "Radiation frequency",
                    },
                )
            )
        ds = ds.drop_vars("waveLength")

    # Sweep fixed angle (scalar)
    if "fixed_angle" in ds.variables:
        ds["sweep_fixed_angle"] = xr.DataArray(
            float(_as_scalar(ds["fixed_angle"])),
            attrs={"long_name": "Fixed angle of sweep", "units": "degrees"},
        )
        ds = ds.drop_vars("fixed_angle")

    # Sweep number (scalar int)
    if "elevationNumber" in ds.variables:
        sw_num = int(_as_scalar(ds["elevationNumber"]))
        ds = ds.drop_vars("elevationNumber")
    else:
        sw_num = 0
    ds["sweep_number"] = xr.DataArray(sw_num)

    # Ray angle resolution (scalar)
    if "angleResolution" in ds.variables:
        ds["rays_angle_resolution"] = xr.DataArray(
            float(_as_scalar(ds["angleResolution"])),
            attrs={
                "long_name": "Angular resolution between rays",
                "units": "degrees",
            },
        )
        ds = ds.drop_vars("angleResolution")

    # Time coverage (stash as attrs so the datatree root helper can pick them up)
    if "esStartTime" in ds.variables:
        tcs = ds["esStartTime"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").values.item()
        ds.attrs["time_coverage_start"] = tcs
        ds = ds.drop_vars("esStartTime")
    if "time" in ds.variables:
        tce = ds["time"].max().dt.strftime("%Y-%m-%dT%H:%M:%SZ").values.item()
        ds.attrs["time_coverage_end"] = tce

    # First-dim handling: CfRadial2 default is time; auto swaps to azimuth/elevation
    dim0 = "elevation" if sweep_mode == "elevation_surveillance" or sweep_mode == "manual_rhi" else "azimuth"
    if "azimuth" in ds.dims and "time" in ds.variables and "time" not in ds.dims:
        ds = ds.swap_dims({"azimuth": "time"})
    if first_dim == "auto":
        if "time" in ds.dims and dim0 in ds.variables:
            ds = ds.swap_dims({"time": dim0})
        if dim0 in ds.dims:
            ds = ds.sortby(dim0)
    else:
        if "time" in ds.dims:
            ds = ds.sortby("time")

    # Drop raw IMD leftovers that map to sweep-level scalars already captured
    _drop_if_present = [
        "scanType",
        "firstGateRange",
        "gateSize",
    ]
    ds = ds.drop_vars(
        [v for v in _drop_if_present if v in ds.variables], errors="ignore"
    )

    ds = _apply_site_as_coords(ds, site_as_coords)
    return ds


class IMDBackendEntrypoint(BackendEntrypoint):
    """Xarray BackendEntrypoint for India Meteorological Department (IMD) radar files.

    IMD provides one NetCDF file per sweep. Open a single file with
    ``xr.open_dataset(file, engine="imd")`` to get a CfRadial2-compatible sweep
    :py:class:`xarray.Dataset`. To assemble a multi-sweep volume, use
    :func:`open_imd_datatree` with a list of files.

    Keyword Arguments
    -----------------
    first_dim : str
        Can be ``time`` or ``auto`` (default). ``auto`` selects ``azimuth``
        (PPI) or ``elevation`` (RHI) as the first dimension.
    reindex_angle : bool or dict
        If a dict, kwargs are passed to :func:`xradar.util.reindex_angle`.
        Defaults to ``False``.
    site_as_coords : bool
        If True (default), promote ``latitude``/``longitude``/``altitude`` to
        Dataset coordinates.
    """

    description = "Open India Meteorological Department (IMD) radar NetCDF files in Xarray"
    url = "https://xradar.rtfd.io/en/latest/io.html#imd"

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
        group=None,
        first_dim="auto",
        reindex_angle=False,
        site_as_coords=True,
    ):
        store = NetCDF4DataStore.open(filename_or_obj, format=format, group=group)
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
        ds = _conform_imd_sweep(
            ds, first_dim=first_dim, site_as_coords=site_as_coords
        )

        if decode_coords and reindex_angle is not False:
            ds = ds.pipe(util.remove_duplicate_rays)
            ds = ds.pipe(util.reindex_angle, **reindex_angle)
            ds = ds.pipe(util.ipol_time, **reindex_angle)

        ds._close = store.close
        return ds


def _read_imd_sweep(filename, first_dim="auto", reindex_angle=False, **kwargs):
    """Open one IMD file and return a CfRadial2 sweep Dataset.

    Avoids the xarray entrypoint registry so this works even when the
    ``imd`` engine has not been installed via pip entrypoints.
    """
    ds = xr.open_dataset(
        filename,
        engine="netcdf4",
        decode_timedelta=kwargs.pop("decode_timedelta", False),
        **kwargs,
    )
    ds = _conform_imd_sweep(ds, first_dim=first_dim, site_as_coords=False)
    if reindex_angle is not False:
        ds = ds.pipe(util.remove_duplicate_rays)
        ds = ds.pipe(util.reindex_angle, **reindex_angle)
        ds = ds.pipe(util.ipol_time, **reindex_angle)
    return ds


def _build_imd_root(sweeps):
    """Build a CfRadial2 root Dataset from a list of IMD sweep Datasets."""
    first = sweeps[0]
    last = sweeps[-1]

    root = xr.Dataset()
    root = root.assign(
        {
            "volume_number": 0,
            "platform_type": "fixed",
            "instrument_type": "radar",
            "time_coverage_start": first.attrs.get("time_coverage_start", ""),
            "time_coverage_end": last.attrs.get("time_coverage_end", ""),
        }
    )

    for v in _STATION_VARS:
        if v in first.variables:
            root[v] = first[v]
    promote = _STATION_VARS & set(root.variables)
    if promote:
        root = root.set_coords(list(promote))

    fixed_angles = [
        float(_as_scalar(sw["sweep_fixed_angle"]))
        if "sweep_fixed_angle" in sw.variables
        else float("nan")
        for sw in sweeps
    ]
    root["sweep_fixed_angle"] = xr.DataArray(
        np.asarray(fixed_angles, dtype=float),
        dims=("sweep",),
        attrs={"long_name": "Fixed angle of sweep", "units": "degrees"},
    )
    sweep_names = np.array([f"sweep_{i}" for i in range(len(sweeps))])
    root["sweep_group_name"] = xr.DataArray(sweep_names, dims=("sweep",))
    root.sweep_group_name.encoding["dtype"] = root.sweep_group_name.dtype

    # FM 301 Table 1 + Table 2 -- WMO-mandated global attributes
    root = root.assign_attrs(
        {
            "Conventions": "CF-1.8, WMO CF-1.0, ACDD-1.3",
            "wmo__cf_profile": "FM 301-XX",
            "version": "2.0",
            "title": "IMD radar data",
            "institution": "India Meteorological Department",
            "references": "",
            "source": "IMD NetCDF",
            "history": "",
            "comment": "",
            "instrument_name": first.attrs.get("instrument_name", ""),
            "platform_is_mobile": "false",
        }
    )
    return root


def _open_single_imd_datatree(
    filename,
    first_dim="auto",
    reindex_angle=False,
    site_as_coords=True,
    optional_groups=False,
    **kwargs,
):
    """Build a single-sweep CfRadial2 DataTree from one IMD NetCDF file."""
    sweep_ds = _read_imd_sweep(
        filename, first_dim=first_dim, reindex_angle=reindex_angle, **kwargs
    )
    # position-0 in this volume
    sweep_ds["sweep_number"] = xr.DataArray(0)
    root = _build_imd_root([sweep_ds])

    dtree: dict = {"/": root}
    if optional_groups:
        dtree["/radar_parameters"] = _get_subgroup([sweep_ds], radar_parameters_subgroup)
        dtree["/georeferencing_correction"] = _get_subgroup(
            [sweep_ds], georeferencing_correction_subgroup
        )
        dtree["/radar_calibration"] = _build_imd_calibration(sweep_ds)

    sw = sweep_ds.drop_vars(_STATION_VARS, errors="ignore")
    sw = _apply_site_as_coords(sw, site_as_coords)
    sw.attrs = {}
    dtree["/sweep_0"] = sw
    return DataTree.from_dict(dtree)


def open_imd_datatree(filename_or_obj, **kwargs):
    """Open IMD radar file(s) as a :py:class:`xarray.DataTree`.

    IMD stores one sweep per NetCDF file. A volume is assembled from
    multiple files (typically 2-3 for long-range PPI or 9-10 for
    short-range, high-resolution PPI).

    - Single file path -> single-sweep DataTree.
    - List of file paths -> single volume DataTree. Multi-file stacking
      is delegated to :func:`xradar.util.create_volume`, which sorts
      sweeps by time and supports ``time_coverage_start``,
      ``time_coverage_end``, ``min_angle``, ``max_angle`` filtering.

    To split a directory of mixed-volume files into per-volume groups,
    use :func:`group_imd_files` first::

        for group in xd.io.group_imd_files("/data/imd"):
            dtree = xd.io.open_imd_datatree(group)

    Parameters
    ----------
    filename_or_obj : str, Path, or list/tuple of those
        Single IMD file path, or a list of paths making up one volume.

    Keyword Arguments
    -----------------
    first_dim : str
        ``"auto"`` (default) or ``"time"``.
    reindex_angle : bool or dict
        If a dict, kwargs are passed to :func:`xradar.util.reindex_angle`.
    site_as_coords : bool
        Attach station variables as coordinates on sweep Datasets.
    optional_groups : bool
        Include ``/radar_parameters``, ``/georeferencing_correction`` and
        ``/radar_calibration`` subgroups. Defaults to ``False``.
    time_coverage_start, time_coverage_end, min_angle, max_angle, volume_number
        Forwarded to :func:`xradar.util.create_volume` when multi-file
        input is provided.

    Returns
    -------
    dtree : xarray.DataTree
        CfRadial2-style DataTree with ``/`` root and ``sweep_N`` children.
    """
    # Multi-file kwargs go to create_volume; everything else is per-sweep.
    cv_kwargs = {
        k: kwargs.pop(k)
        for k in (
            "time_coverage_start",
            "time_coverage_end",
            "min_angle",
            "max_angle",
            "volume_number",
        )
        if k in kwargs
    }

    if isinstance(filename_or_obj, (list, tuple)):
        files = list(filename_or_obj)
        if not files:
            raise ValueError("open_imd_datatree requires at least one file.")
        if len(files) == 1:
            return _open_single_imd_datatree(files[0], **kwargs)

        sweep_trees = [_open_single_imd_datatree(f, **kwargs) for f in files]
        volume = util.create_volume(sweep_trees, **cv_kwargs)
        # create_volume copies sweeps verbatim, so each sweep_number stays 0.
        # Renumber to match the final DataTree position.
        for i, key in enumerate(util.get_sweep_keys(volume)):
            sw_ds = volume[key].to_dataset()
            sw_ds["sweep_number"] = xr.DataArray(i)
            volume[key] = DataTree(sw_ds)
        return volume

    if cv_kwargs:
        raise TypeError(
            "time_coverage_start, time_coverage_end, min_angle, max_angle, "
            "volume_number are only valid when opening multiple files."
        )
    return _open_single_imd_datatree(filename_or_obj, **kwargs)


def open_imd_volumes(paths, **kwargs):
    """Open a directory of IMD files as a multi-volume :py:class:`xarray.DataTree`.

    Groups files by filename stem via :func:`group_imd_files`, opens each
    group as a CfRadial2 volume via :func:`open_imd_datatree`, and nests
    them under zero-padded ``vcp_NN`` child nodes of a parent root.
    ``vcp`` stands for *volume coverage pattern*. Padding width is chosen
    so the child names sort lexically (e.g. 121 volumes -> ``vcp_000`` ..
    ``vcp_120``).

    Parameters
    ----------
    paths : str, Path, or iterable of those
        Same as :func:`group_imd_files`: directory, glob, or list.

    Keyword Arguments
    -----------------
    All kwargs are forwarded to :func:`open_imd_datatree` (applied per
    volume). Typical: ``first_dim``, ``reindex_angle``, ``site_as_coords``,
    ``optional_groups``, ``min_angle``, ``max_angle``.

    Returns
    -------
    dtree : xarray.DataTree
        Root with ``vcp_NN`` children, each a full CfRadial2 volume tree::

            /
            ├── vcp_00/
            │   ├── (root: sweep_group_name, sweep_fixed_angle, ...)
            │   ├── sweep_0
            │   ├── sweep_1
            │   └── ...
            ├── vcp_01/
            └── ...

    Examples
    --------
    >>> import xradar as xd
    >>> tree = xd.io.open_imd_volumes("/data/JPR220822IMD-B")   # doctest: +SKIP
    >>> tree["vcp_00/sweep_0"].ds["DBZH"]                       # doctest: +SKIP
    """
    groups = group_imd_files(paths)
    if not groups:
        raise ValueError(f"No IMD files matched at {paths!r}.")

    width = max(2, len(str(len(groups) - 1)))

    flat: dict[str, xr.Dataset] = {
        "/": xr.Dataset(
            attrs={
                "Conventions": "Cf/Radial",
                "institution": "India Meteorological Department",
                "source": "IMD NetCDF",
                "title": f"IMD multi-volume dataset ({len(groups)} volumes)",
            }
        )
    }
    for i, files in enumerate(groups):
        vcp = f"vcp_{i:0{width}d}"
        volume = open_imd_datatree(files, **kwargs)
        # Flatten the volume's subtree under /vcp_NN/...
        for subpath, ds in volume.to_dict().items():
            if subpath == "/":
                flat[f"/{vcp}"] = ds
            else:
                flat[f"/{vcp}{subpath}"] = ds

    return DataTree.from_dict(flat)


# Files matching `<stem>.nc` or `<stem>.nc.<digits>`. The stem -- everything
# up through `.nc` -- identifies the parent volume.
_IMD_STEM_RE = re.compile(r"^(?P<stem>.+\.nc)(?:\.(?P<idx>\d+))?$")


def group_imd_files(paths):
    """Group IMD sweep files into volumes by filename stem.

    IMD distributes one sweep per file. Files of a single volume share a
    common stem ending in ``.nc``; additional sweeps in the same volume
    get numeric suffixes::

        GOA210515003646-IMD-C.nc        <- sweep 0
        GOA210515003646-IMD-C.nc.1      <- sweep 1
        ...
        GOA210515003646-IMD-C.nc.9      <- sweep 9

    A new volume changes the timestamp component of the stem.

    Parameters
    ----------
    paths : str, Path, or iterable of those
        * A directory path -- all IMD files inside are grouped.
        * A glob pattern string -- files matching the glob are grouped.
        * An iterable of file paths -- grouped as-is.

    Returns
    -------
    list[list[str]]
        One list of absolute paths per detected volume, inner lists in
        sweep order (``.nc``, ``.nc.1``, ...).

    Examples
    --------
    >>> import xradar as xd
    >>> for group in xd.io.group_imd_files("/data/goa"):       # doctest: +SKIP
    ...     dtree = xd.io.open_imd_datatree(group)
    """
    if isinstance(paths, (str, os.PathLike)):
        p = os.fspath(paths)
        if os.path.isdir(p):
            candidates = [os.path.join(p, name) for name in os.listdir(p)]
        else:
            candidates = _glob.glob(p)
    else:
        candidates = [os.fspath(x) for x in paths]

    groups: dict[str, list[tuple[int, str]]] = {}
    for path in candidates:
        name = os.path.basename(path)
        m = _IMD_STEM_RE.match(name)
        if not m:
            continue
        stem = m.group("stem")
        idx = int(m.group("idx")) if m.group("idx") is not None else 0
        groups.setdefault(stem, []).append((idx, path))

    return [
        [path for _, path in sorted(groups[stem])] for stem in sorted(groups)
    ]
