#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for the Vaisala WindCube backend.

The files are synthetic but follow the layout of WindCube NetCDF-4 files
(``Conventions = "CF/Radial 2.0 , CF-1.7"``): sweep groups ``Sweep_<id>-<n>``
listed in the root ``sweep_group_name``, ``sweep_mode`` as string variable,
``range`` as dimension or, for DBS, as ``range(time, gate_index)``.
"""

import h5netcdf
import h5py
import numpy as np
import pytest
import xarray as xr

import xradar as xd
from xradar.io.backends.windcube import WindCubeBackendEntrypoint


def _write_sweep(grp, mode, azimuth, elevation, rng, two_d_range=False):
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


@pytest.fixture
def windcube_file(tmp_path):
    path = tmp_path / "WCS000001_2025-04-10_00-00-00_mixed_1_25mTP.nc"
    gates = np.arange(50, 550, 50)
    # over-the-top RHI at azimuth 90 with an invalid first ray
    el_up = np.arange(0, 91, 15.0)
    rhi_el = np.concatenate([[np.nan], el_up, el_up[::-1][1:]])
    rhi_az = np.concatenate([[np.nan], np.full(el_up.size, 90.0), np.full(6, 270.0)])
    # DBS: 4 inclined beams at 75 deg and a vertical beam (other gate geometry)
    dbs_az = np.array([0.0, 90.0, 180.0, 270.0, 0.0])
    dbs_el = np.array([75.0, 75.0, 75.0, 75.0, 90.0])
    dbs_rng = np.array([gates / np.sin(np.deg2rad(75))] * 4 + [gates]).round()
    # PPI
    ppi_az = np.arange(0, 360, 30.0)

    names = ["Sweep_7-1", "Sweep_7-2", "Sweep_7-3"]
    with h5netcdf.File(path, "w") as f:
        f.attrs.update(
            title="WindCube data",
            Conventions="CF/Radial 2.0 , CF-1.7",
            institution="Vaisala",
            instrument_name="WCS000001",
        )
        f.dimensions["sweep"] = 3
        f.create_variable(
            "sweep_group_name",
            ("sweep",),
            data=np.array(names, dtype=object),
            dtype=h5py.string_dtype(),
        )
        f.create_variable("sweep_fixed_angle", ("sweep",), data=[90.0, 75.0, 20.0])
        for name, value in [("latitude", 28.6), ("longitude", 77.2), ("altitude", 227)]:
            f.create_variable(name, data=np.float64(value))
        _write_sweep(f.create_group(names[0]), "rhi", rhi_az, rhi_el, gates)
        _write_sweep(
            f.create_group(names[1]), "dbs", dbs_az, dbs_el, dbs_rng, two_d_range=True
        )
        _write_sweep(f.create_group(names[2]), "ppi", ppi_az, np.full(12, 20.0), gates)
    return path


def test_open_windcube_datatree(windcube_file):
    dtree = xd.io.open_windcube_datatree(windcube_file)
    sweeps = [k for k in dtree.children if k.startswith("sweep_")]
    # RHI, DBS split into vertical and inclined beams, PPI
    assert len(sweeps) == 4
    np.testing.assert_allclose(
        dtree.ds.sweep_fixed_angle.values, [90.0, 90.0, 75.0, 20.0]
    )
    modes = [str(dtree[s].ds.sweep_mode.values) for s in sweeps]
    assert modes == [
        "rhi",
        "doppler_beam_swinging",
        "doppler_beam_swinging",
        "azimuth_surveillance",
    ]
    for s in sweeps:
        for var in ("sweep_number", "sweep_fixed_angle", "follow_mode", "prt_mode"):
            assert var in dtree[s].ds


def test_open_windcube_rhi(windcube_file):
    ds = xd.io.open_windcube_datatree(windcube_file)["sweep_0"].ds
    # invalid first ray dropped, over-the-top RHI unfolded to 0..180 degrees
    assert ds.sizes["elevation"] == 13
    np.testing.assert_allclose(ds.elevation.values, np.arange(0, 181, 15.0))
    assert float(ds.sweep_fixed_angle) == 90.0
    assert ds.sweep_mode.attrs["windcube_sweep_mode"] == "rhi"
    rws = ds.radial_wind_speed
    assert rws.attrs["ancillary_variables"] == (
        "radial_wind_speed_ci,radial_wind_speed_status"
    )
    assert "ancilliary_variables" not in rws.attrs
    assert rws.attrs["long_name"] == "radial_wind_speed"


def test_open_windcube_dbs(windcube_file):
    dtree = xd.io.open_windcube_datatree(windcube_file)
    vertical, inclined = dtree["sweep_1"].ds, dtree["sweep_2"].ds
    assert vertical.sizes["time"] == 1
    assert inclined.sizes["time"] == 4
    np.testing.assert_allclose(vertical.range.values, np.arange(50, 550, 50))
    assert float(vertical.sweep_fixed_angle) == 90.0
    assert float(inclined.sweep_fixed_angle) == 75.0
    assert vertical.sweep_mode.attrs["windcube_sweep_mode"] == "dbs"


def test_open_windcube_engine(windcube_file):
    with xr.open_dataset(windcube_file, engine="windcube", group="sweep_3") as ds:
        assert ds.sizes["azimuth"] == 12
        assert ds.encoding["engine"] == "windcube"
        assert "latitude" in ds.coords
    with xr.open_dataset(
        windcube_file, engine="windcube", group="sweep_3", drop_variables="cnr"
    ) as ds:
        assert "cnr" not in ds
        assert "radial_wind_speed" in ds
    with pytest.raises(ValueError, match="missing"):
        xr.open_dataset(windcube_file, engine="windcube", group="sweep_9")


def test_open_windcube_sweep_selection(windcube_file):
    dtree = xd.io.open_windcube_datatree(windcube_file, sweep=[0, 3])
    assert [k for k in dtree.children if k.startswith("sweep_")] == [
        "sweep_0",
        "sweep_1",
    ]
    assert str(dtree["sweep_1"].ds.sweep_mode.values) == "azimuth_surveillance"


def test_windcube_guess_can_open(windcube_file, tmp_path):
    entrypoint = WindCubeBackendEntrypoint()
    assert entrypoint.guess_can_open(windcube_file)
    other = tmp_path / "other.nc"
    xr.Dataset({"a": 1}).to_netcdf(other, engine="h5netcdf")
    assert not entrypoint.guess_can_open(other)
    text = tmp_path / "other.txt"
    text.write_text("no netcdf")
    assert not entrypoint.guess_can_open(text)


def test_windcube_unfold_rhi_one_sided():
    from xradar.io.backends.windcube import _unfold_rhi

    # RHI that doesn't go over the top keeps its elevations
    ds = xr.Dataset(
        {
            "azimuth": ("time", np.full(4, 90.0)),
            "elevation": ("time", np.array([0.0, 30.0, 60.0, 90.0])),
        }
    )
    xr.testing.assert_identical(_unfold_rhi(ds, 90.0), ds)


@pytest.mark.parametrize(
    ("mode", "azimuth", "elevation", "expected"),
    [
        ("volume", np.arange(0, 360, 10.0), 20.0, "azimuth_surveillance"),
        ("ppi", np.arange(0, 90, 10.0), 20.0, "sector"),
        # full circle with a few dropped rays is no sector
        (
            "volume",
            np.delete(np.arange(1.5, 360, 3.0), [10, 11]),
            20.0,
            "azimuth_surveillance",
        ),
        ("fixed", [180.0], 90.0, "vertical_pointing"),
        ("fixed", [180.0], 45.0, "pointing"),
        ("segment", [0.0], 10.0, "complex_trajectory"),
        ("rhi", [90.0], 10.0, "rhi"),
    ],
)
def test_windcube_sweep_mode_mapping(mode, azimuth, elevation, expected):
    from xradar.io.backends.windcube import _cfradial_sweep_mode

    elevation = np.full(len(azimuth), elevation)
    assert _cfradial_sweep_mode(mode, np.asarray(azimuth), elevation) == expected


def test_open_windcube_no_sweeps(tmp_path):
    path = tmp_path / "empty.nc"
    with h5netcdf.File(path, "w") as f:
        f.attrs["title"] = "WindCube data"
        f.dimensions["sweep"] = 2
        f.create_variable(
            "sweep_group_name",
            ("sweep",),
            data=np.array(["Sweep_1-1", "Sweep_1-2"], dtype=object),
            dtype=h5py.string_dtype(),
        )
        f.create_variable("sweep_fixed_angle", ("sweep",), data=[0.0, 0.0])
        # interrupted scan: group present, but without rays;
        # Sweep_1-2 is listed, but missing
        grp = f.create_group("Sweep_1-1")
        grp.dimensions["time"] = 0
        grp.create_variable("time", ("time",), dtype="f8")
    with pytest.raises(ValueError, match="No WindCube sweeps"):
        xd.io.open_windcube_datatree(path)


@pytest.fixture
def windcube_file_old(tmp_path):
    # layout of older files (e.g. WindCube Lidar server 3.3.3, WLS200s): time
    # in seconds since a ``time_reference`` variable, a root ``sweep``
    # coordinate and the title "Leosphere Windcube data"
    path = tmp_path / "WLS200s-218_2022-10-07_00-51-38_dbs_1823_75m.nc"
    gates = np.arange(50, 550, 50)
    dbs_az = np.array([0.0, 90.0, 180.0, 270.0, 0.0])
    dbs_el = np.array([75.0, 75.0, 75.0, 75.0, 90.0])
    dbs_rng = np.array([gates / np.sin(np.deg2rad(75))] * 4 + [gates]).round()
    with h5netcdf.File(path, "w") as f:
        f.attrs.update(
            title="Leosphere Windcube data",
            Conventions="CF/Radial 2.0 , CF-1.7",
            institution="Leosphere",
        )
        f.dimensions["sweep"] = 1
        f.create_variable("sweep", ("sweep",), data=np.array([1], "i4"))
        f.create_variable(
            "sweep_group_name",
            ("sweep",),
            data=np.array(["Sweep_62961"], dtype=object),
            dtype=h5py.string_dtype(),
        )
        f.create_variable("sweep_fixed_angle", ("sweep",), data=[75.0])
        for name, value in [("latitude", 51.97), ("longitude", 4.93), ("altitude", 0)]:
            f.create_variable(name, data=np.float64(value))
        grp = f.create_group("Sweep_62961")
        _write_sweep(grp, "dbs", dbs_az, dbs_el, dbs_rng, two_d_range=True)
        grp.variables["time"].attrs["units"] = "seconds since time_reference"
        grp.create_variable(
            "time_reference",
            data=np.array("1970-01-01T00:00:00Z", dtype=object),
            dtype=h5py.string_dtype(),
        )
    return path


def test_open_windcube_time_reference(windcube_file_old):
    dtree = xd.io.open_windcube_datatree(windcube_file_old)
    sweeps = [k for k in dtree.children if k.startswith("sweep_")]
    # vertical and inclined DBS beams, no inherited root "sweep" dimension
    assert len(sweeps) == 2
    for s in sweeps:
        assert "sweep" not in dtree[s].ds.dims
    # seconds since the epoch given in ``time_reference``
    expected = np.datetime64(int(1.7442e9 * 1000), "ms")
    assert dtree["sweep_1"].ds.time.values[0] == expected
    assert WindCubeBackendEntrypoint().guess_can_open(windcube_file_old)
