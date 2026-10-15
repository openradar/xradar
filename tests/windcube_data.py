#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Synthetic Vaisala WindCube files for the tests.

The files follow the layout of WindCube NetCDF-4 files
(``Conventions = "CF/Radial 2.0 , CF-1.7"``): sweep groups ``Sweep_<id>-<n>``
listed in the root ``sweep_group_name``, ``sweep_mode`` as string variable,
``range`` as dimension or, for DBS, as ``range(time, gate_index)``.
"""

import h5netcdf
import h5py
import numpy as np

GATES = np.arange(50, 550, 50)

#: DBS: 4 inclined beams at 75 deg and a vertical beam (other gate geometry)
DBS_AZ = np.array([0.0, 90.0, 180.0, 270.0, 0.0])
DBS_EL = np.array([75.0, 75.0, 75.0, 75.0, 90.0])
DBS_RNG = np.array([GATES / np.sin(np.deg2rad(75))] * 4 + [GATES]).round()

#: over-the-top RHI at azimuth 90 with an invalid first ray
_EL_UP = np.arange(0, 91, 15.0)
RHI_EL = np.concatenate([[np.nan], _EL_UP, _EL_UP[::-1][1:]])
RHI_AZ = np.concatenate([[np.nan], np.full(_EL_UP.size, 90.0), np.full(6, 270.0)])

PPI_AZ = np.arange(0, 360, 30.0)

SITE = (("latitude", 28.6), ("longitude", 77.2), ("altitude", 227))


def write_sweep(grp, mode, azimuth, elevation, rng, two_d_range=False):
    """Write a sweep group with ``radial_wind_speed`` and ``cnr``."""
    nrays = len(azimuth)
    grp.attrs["res_file_name"] = "25mTP"
    grp.dimensions["time"] = nrays
    gate_dim = "gate_index" if two_d_range else "range"
    ngates = rng.shape[-1]
    grp.dimensions[gate_dim] = ngates
    grp.create_variable(
        "sweep_mode", data=np.array(mode, dtype=object), dtype=h5py.string_dtype()
    )
    grp.create_variable("sweep_index", data=np.int32(1))
    t = grp.create_variable("time", ("time",), data=1.7442e9 + np.arange(nrays))
    t.attrs.update(units="seconds since 1970-01-01T00:00:00Z", calendar="gregorian")
    grp.create_variable("azimuth", ("time",), data=np.asarray(azimuth, "f8"))
    grp.create_variable("elevation", ("time",), data=np.asarray(elevation, "f8"))
    if two_d_range:
        grp.create_variable("gate_index", ("gate_index",), data=np.arange(ngates))
        r = grp.create_variable("range", ("time", "gate_index"), data=rng)
    else:
        r = grp.create_variable("range", ("range",), data=rng)
    r.attrs["units"] = "m"
    dims = ("time", gate_dim)
    rws = grp.create_variable(
        "radial_wind_speed", dims, data=np.arange(nrays * ngates, dtype="f8")
    )
    rws.attrs.update(
        units="m s-1",
        standard_name="radial_velocity_of_scatterers_away_from_instrument",
        ancilliary_variables="radial_wind_speed_ci,radial_wind_speed_status",
        comments="Wind speed vector projected along the line of sights.",
    )
    cnr = grp.create_variable("cnr", dims, data=np.full((nrays, ngates), -20.0))
    cnr.attrs.update(units="dB", standard_name="carrier_to_noise_ratio")


def write_root(f, names, fixed_angles, site=SITE, title="WindCube data", **attrs):
    """Write the root group: sweep list, fixed angles, site and attributes."""
    f.attrs.update(
        title=title,
        Conventions="CF/Radial 2.0 , CF-1.7",
        institution="Vaisala",
        **attrs,
    )
    f.dimensions["sweep"] = len(names)
    f.create_variable(
        "sweep_group_name",
        ("sweep",),
        data=np.array(names, dtype=object),
        dtype=h5py.string_dtype(),
    )
    f.create_variable("sweep_fixed_angle", ("sweep",), data=fixed_angles)
    for name, value in site or ():
        f.create_variable(name, data=np.float64(value))


def write_windcube(path, sweeps, site=SITE, title="WindCube data"):
    """Write a file from ``sweeps``: tuples ``(mode, fixed_angle, az, el, range,
    two_d_range)`` in groups ``Sweep_7-<n>``."""
    names = [f"Sweep_7-{i + 1}" for i in range(len(sweeps))]
    with h5netcdf.File(path, "w") as f:
        write_root(
            f,
            names,
            [sweep[1] for sweep in sweeps],
            site=site,
            title=title,
            instrument_name="WCS000001",
        )
        for name, (mode, _, az, el, rng, two_d) in zip(names, sweeps, strict=True):
            write_sweep(f.create_group(name), mode, az, el, rng, two_d_range=two_d)
    return path


def write_default_file(path):
    """RHI (invalid first ray, over the top), DBS (two gate geometries) and PPI."""
    return write_windcube(
        path,
        [
            ("rhi", 90.0, RHI_AZ, RHI_EL, GATES, False),
            ("dbs", 75.0, DBS_AZ, DBS_EL, DBS_RNG, True),
            ("ppi", 20.0, PPI_AZ, np.full(12, 20.0), GATES, False),
        ],
    )
