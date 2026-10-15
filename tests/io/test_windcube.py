#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for the Vaisala WindCube backend.

The files are synthetic but follow the layout of WindCube NetCDF-4 files, see
``tests/windcube_data.py``.
"""

import io
import os
import warnings
from pathlib import Path

import h5netcdf
import h5py
import numpy as np
import pytest
import xarray as xr

import xradar as xd
from tests.windcube_data import (
    DBS_AZ,
    DBS_EL,
    DBS_RNG,
    GATES,
    PPI_AZ,
    write_root,
    write_sweep,
    write_windcube,
)
from xradar.io.backends.windcube import (
    WindCubeBackendEntrypoint,
    _cfradial_sweep_mode,
    _gate_geometries,
    _unfold_rhi,
)

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


def _sweeps(dtree):
    return [k for k in dtree.children if k.startswith("sweep_")]


def _open_handles(path):
    """Number of file descriptors of this process which point at ``path``."""
    target = os.path.realpath(path)
    proc = Path("/proc/self/fd")
    if proc.is_dir():
        paths = []
        for fd in proc.iterdir():
            try:
                paths.append(fd.readlink())
            except OSError:
                continue
    elif fcntl is not None and hasattr(fcntl, "F_GETPATH"):  # macOS
        paths = []
        for fd in range(1024):
            try:
                raw = fcntl.fcntl(fd, fcntl.F_GETPATH, b"\0" * 1024)
            except OSError:
                continue
            paths.append(raw.split(b"\0")[0].decode())
    else:
        pytest.skip("cannot list the open files of this process")
    return sum(os.path.realpath(p) == target for p in paths)


def test_open_windcube_datatree(windcube_file):
    with xd.open_datatree(windcube_file, engine="windcube") as dtree:
        sweeps = _sweeps(dtree)
        # RHI, DBS split into inclined and vertical beams, PPI
        assert len(sweeps) == 4
        np.testing.assert_allclose(
            dtree.ds.sweep_fixed_angle.values, [90.0, 75.0, 90.0, 20.0]
        )
        modes = [str(dtree[s].ds.sweep_mode.values) for s in sweeps]
        assert modes == [
            "rhi",
            "doppler_beam_swinging",
            "doppler_beam_swinging",
            "azimuth_surveillance",
        ]
        for i, s in enumerate(sweeps):
            for var in ("sweep_number", "sweep_fixed_angle", "follow_mode", "prt_mode"):
                assert var in dtree[s].ds
            assert int(dtree[s].ds.sweep_number) == i
            # the group of the file and the vendor mode are kept
            attrs = dtree[s].ds.sweep_mode.attrs
            assert attrs["windcube_group_name"] == (
                ["Sweep_7-1", "Sweep_7-2", "Sweep_7-2", "Sweep_7-3"][i]
            )
        assert str(dtree["sweep_0"].ds.prt_mode.values) == "fixed"


def test_open_windcube_rhi(windcube_file):
    with xd.open_datatree(windcube_file, engine="windcube") as dtree:
        ds = dtree["sweep_0"].ds
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


def test_open_windcube_first_dim_time(windcube_file):
    with xd.open_datatree(windcube_file, engine="windcube", first_dim="time") as dtree:
        ds = dtree["sweep_0"].ds
        assert "time" in ds.dims and "elevation" not in ds.dims
        # the elevations are unfolded in any case
        assert float(ds.elevation.max()) == 180.0
        assert "time" in dtree["sweep_3"].ds.dims


def test_open_windcube_dbs(windcube_file):
    with xd.open_datatree(windcube_file, engine="windcube") as dtree:
        # the beams come in the order they were measured
        inclined, vertical = dtree["sweep_1"].ds, dtree["sweep_2"].ds
        assert inclined.sizes["time"] == 4
        assert vertical.sizes["time"] == 1
        np.testing.assert_allclose(vertical.range.values, GATES)
        np.testing.assert_allclose(inclined.range.values, DBS_RNG[0])
        assert float(vertical.sweep_fixed_angle) == 90.0
        assert float(inclined.sweep_fixed_angle) == 75.0
        assert vertical.sweep_mode.attrs["windcube_sweep_mode"] == "dbs"


def test_open_windcube_engine(windcube_file):
    with xr.open_dataset(windcube_file, engine="windcube", group="sweep_3") as ds:
        assert ds.sizes["azimuth"] == 12
        assert ds.encoding["engine"] == "windcube"
        assert "latitude" in ds.coords
        assert int(ds.sweep_number) == 3
    with xr.open_dataset(
        windcube_file, engine="windcube", group="sweep_3", drop_variables="cnr"
    ) as ds:
        assert "cnr" not in ds
        assert "radial_wind_speed" in ds
    with xr.open_dataset(windcube_file, engine="windcube", group="/sweep_1") as ds:
        assert str(ds.sweep_mode.values) == "doppler_beam_swinging"
    with xr.open_dataset(windcube_file, engine="windcube", group=None) as ds:
        assert str(ds.sweep_mode.values) == "rhi"


@pytest.mark.parametrize("group", ["sweep_9", "sweep_-1", "Sweep_7-1", "sweep_x"])
def test_open_windcube_engine_missing_group(windcube_file, group):
    with pytest.raises(ValueError, match="missing"):
        xr.open_dataset(windcube_file, engine="windcube", group=group)


@pytest.mark.parametrize(
    "sweep",
    ["sweep_1", ["sweep_1"], 1, np.int64(1), [np.int64(1)], "/sweep_1"],
)
def test_open_windcube_sweep_selection_forms(windcube_file, sweep):
    with xd.open_datatree(windcube_file, engine="windcube", sweep=sweep) as dtree:
        assert _sweeps(dtree) == ["sweep_0"]
        ds = dtree["sweep_0"].ds
        assert str(ds.sweep_mode.values) == "doppler_beam_swinging"
        assert float(ds.sweep_fixed_angle) == 75.0
        # the sweep is numbered like its group
        assert int(ds.sweep_number) == 0


def test_open_windcube_sweep_selection(windcube_file):
    with xd.open_datatree(windcube_file, engine="windcube", sweep=[0, 3]) as dtree:
        assert _sweeps(dtree) == ["sweep_0", "sweep_1"]
        assert str(dtree["sweep_1"].ds.sweep_mode.values) == "azimuth_surveillance"
        assert int(dtree["sweep_1"].ds.sweep_number) == 1


@pytest.mark.parametrize("sweep", [9, [0, 9], "sweep_9", "other"])
def test_open_windcube_sweep_selection_missing(windcube_file, sweep):
    with pytest.raises(ValueError, match="missing"):
        xd.open_datatree(windcube_file, engine="windcube", sweep=sweep)


def test_open_windcube_sweep_selection_negative(windcube_file):
    # rejected like in the other engines, not the last sweep
    with pytest.raises(TypeError, match="Unsupported sweep -1"):
        xd.open_datatree(windcube_file, engine="windcube", sweep=-1)


def test_open_windcube_is_lazy(windcube_file):
    with xr.open_dataset(windcube_file, engine="windcube", group="sweep_3") as ds:
        for name in ("radial_wind_speed", "cnr"):
            assert not isinstance(ds[name].variable._data, np.ndarray), name
    with xd.open_datatree(windcube_file, engine="windcube") as dtree:
        for sweep in _sweeps(dtree):
            data = dtree[sweep].ds.radial_wind_speed.variable._data
            assert not isinstance(data, np.ndarray), sweep
        # values are right once loaded: the second DBS part has the last ray
        np.testing.assert_array_equal(
            dtree["sweep_2"].ds.radial_wind_speed.values,
            (np.arange(50, dtype="f8").reshape(5, 10)[[4]]),
        )


def test_open_windcube_close_releases_file(windcube_file, tmp_path):
    path = tmp_path / "private" / windcube_file.name
    path.parent.mkdir()
    path.write_bytes(windcube_file.read_bytes())
    assert _open_handles(path) == 0
    dtree = xd.open_datatree(path, engine="windcube")
    assert _open_handles(path) >= 1
    dtree.close()
    assert _open_handles(path) == 0
    ds = xr.open_dataset(path, engine="windcube", group="sweep_2")
    assert _open_handles(path) >= 1
    ds.close()
    assert _open_handles(path) == 0
    # nothing stays open if opening fails
    with pytest.raises(ValueError, match="missing"):
        xd.open_datatree(path, engine="windcube", sweep=9)
    assert _open_handles(path) == 0


def test_open_windcube_filelike(windcube_file):
    data = windcube_file.read_bytes()
    with (
        xd.open_datatree(windcube_file, engine="windcube") as expected,
        xd.open_datatree(io.BytesIO(data), engine="windcube") as actual,
    ):
        xr.testing.assert_equal(actual, expected)


def test_open_windcube_site_coords(windcube_file):
    with xd.open_datatree(windcube_file, engine="windcube", site_coords=False) as dt:
        assert "latitude" not in dt["sweep_0"].ds.coords
    with xd.open_datatree(windcube_file, engine="windcube") as dt:
        assert float(dt.ds.latitude) == pytest.approx(28.6)


def test_windcube_unfold_rhi_one_sided():
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
        ("manual_rhi", [90.0], 10.0, "rhi"),
        ("manual_ppi", [90.0], 10.0, "manual_ppi"),
        ("dbs", [90.0], 75.0, "doppler_beam_swinging"),
        # upper case spelling
        ("PPI", np.arange(0, 360, 10.0), 20.0, "azimuth_surveillance"),
        ("RHI", [90.0], 10.0, "rhi"),
    ],
)
def test_windcube_sweep_mode_mapping(mode, azimuth, elevation, expected):
    elevation = np.full(len(azimuth), elevation)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _cfradial_sweep_mode(mode, np.asarray(azimuth), elevation) == expected


def test_windcube_unknown_sweep_mode_warns():
    azimuth = np.arange(0, 360, 10.0)
    with pytest.warns(UserWarning, match="unknown WindCube sweep mode 'stare_xyz'"):
        mode = _cfradial_sweep_mode("stare_xyz", azimuth, np.full(azimuth.size, 20.0))
    assert mode == "azimuth_surveillance"


def test_open_windcube_unknown_sweep_mode(tmp_path):
    path = write_windcube(
        tmp_path / "unknown.nc",
        [("stare_xyz", 20.0, PPI_AZ, np.full(12, 20.0), GATES, False)],
    )
    with pytest.warns(UserWarning, match="unknown WindCube sweep mode"):
        with xd.open_datatree(path, engine="windcube") as dtree:
            ds = dtree["sweep_0"].ds
    # not a vendor name in sweep_mode, the vendor name is kept in the attribute
    assert str(ds.sweep_mode.values) == "azimuth_surveillance"
    assert ds.sweep_mode.attrs["windcube_sweep_mode"] == "stare_xyz"


def test_windcube_gate_geometries_order_and_masked_gates():
    short = np.arange(50, 300, 50.0)
    long = short * 1.5
    # a ray with a masked last gate stays with its geometry; the beams are
    # returned in the order of their first ray, not in the order of the ranges
    rng = np.array([long, short, long, short, long])
    rng[2, -1] = np.nan
    rng[3, 0] = np.nan
    geometries = _gate_geometries(rng)
    assert [list(rays) for rays, _ in geometries] == [[0, 2, 4], [1, 3]]
    np.testing.assert_allclose(geometries[0][1], long)
    np.testing.assert_allclose(geometries[1][1], short)

    # a gate which is masked in every ray is filled by extrapolation
    rng = np.array([long, long])
    rng[:, -1] = np.nan
    ((rays, row),) = _gate_geometries(rng)
    np.testing.assert_allclose(row, long)


def test_open_windcube_dbs_with_masked_gate(tmp_path):
    # masked last gate in every ray: still one geometry, not one per ray
    rng = np.array([GATES / np.sin(np.deg2rad(75))] * 3)
    rng[:, -1] = np.nan
    path = write_windcube(
        tmp_path / "masked.nc",
        [("dbs", 75.0, DBS_AZ[:3], DBS_EL[:3], rng, True)],
    )
    with xd.open_datatree(path, engine="windcube") as dtree:
        assert _sweeps(dtree) == ["sweep_0"]
        ds = dtree["sweep_0"].ds
        assert ds.sizes["time"] == 3
        assert np.isfinite(ds.range.values).all()
        assert float(ds.sweep_fixed_angle) == 75.0


def test_open_windcube_rhi_with_two_geometries(tmp_path):
    # an RHI keeps its azimuth as fixed angle, also if the gates are split
    el = np.array([10.0, 20.0, 30.0, 40.0])
    rng = np.array([GATES, GATES, GATES * 2, GATES * 2.0])
    path = write_windcube(
        tmp_path / "rhi2.nc",
        [("rhi", 135.0, np.full(4, 135.0), el, rng, True)],
    )
    with xd.open_datatree(path, engine="windcube") as dtree:
        assert _sweeps(dtree) == ["sweep_0", "sweep_1"]
        for sweep in ("sweep_0", "sweep_1"):
            ds = dtree[sweep].ds
            assert float(ds.sweep_fixed_angle) == 135.0
            assert str(ds.sweep_mode.values) == "rhi"
        # no elevation was unfolded
        np.testing.assert_allclose(dtree["sweep_1"].ds.elevation.values, [30.0, 40.0])


def test_open_windcube_all_angles_invalid(tmp_path):
    nan = np.full(4, np.nan)
    path = write_windcube(
        tmp_path / "nan_angles.nc",
        [
            ("ppi", 20.0, nan, nan, GATES, False),
            ("ppi", 20.0, PPI_AZ, np.full(12, 20.0), GATES, False),
        ],
    )
    # the group without valid angles is left out
    with xd.open_datatree(path, engine="windcube") as dtree:
        assert _sweeps(dtree) == ["sweep_0"]
        assert dtree["sweep_0"].ds.sizes["azimuth"] == 12
    only = write_windcube(
        tmp_path / "only_nan.nc", [("fixed", 90.0, nan, nan, GATES, False)]
    )
    with pytest.raises(ValueError, match="No WindCube sweeps"):
        xd.open_datatree(only, engine="windcube")


def test_open_windcube_no_sweeps(tmp_path):
    path = tmp_path / "empty.nc"
    with h5netcdf.File(path, "w") as f:
        write_root(f, ["Sweep_1-1", "Sweep_1-2"], [0.0, 0.0], title="WindCube data")
        # interrupted scan: group present, but without rays;
        # Sweep_1-2 is listed, but missing
        grp = f.create_group("Sweep_1-1")
        grp.dimensions["time"] = 0
        grp.create_variable("time", ("time",), dtype="f8")
    with pytest.warns(UserWarning, match=r"missing: \['Sweep_1-2'\]"):
        with pytest.raises(ValueError, match="No WindCube sweeps"):
            xd.open_datatree(path, engine="windcube")


def test_open_windcube_missing_group_warns(tmp_path):
    path = tmp_path / "missing_group.nc"
    with h5netcdf.File(path, "w") as f:
        write_root(f, ["Sweep_1-1", "Sweep_1-2"], [20.0, 20.0])
        write_sweep(
            f.create_group("Sweep_1-1"),
            "ppi",
            PPI_AZ,
            np.full(12, 20.0),
            GATES,
        )
    with pytest.warns(UserWarning, match=r"missing: \['Sweep_1-2'\]"):
        with xd.open_datatree(path, engine="windcube") as dtree:
            assert _sweeps(dtree) == ["sweep_0"]


def test_open_windcube_fixed_angles_length(tmp_path):
    # two listed groups, one fixed angle
    path = tmp_path / "short_angles.nc"
    with h5netcdf.File(path, "w") as f:
        f.attrs["title"] = "WindCube data"
        f.dimensions["sweep"] = 2
        f.dimensions["angle"] = 1
        f.create_variable(
            "sweep_group_name",
            ("sweep",),
            data=np.array(["Sweep_1-1", "Sweep_1-2"], dtype=object),
            dtype=h5py.string_dtype(),
        )
        f.create_variable("sweep_fixed_angle", ("angle",), data=[20.0])
        for name in ("Sweep_1-1", "Sweep_1-2"):
            write_sweep(f.create_group(name), "ppi", PPI_AZ, np.full(12, 20.0), GATES)
    with pytest.raises(ValueError, match="differ in length"):
        xd.open_datatree(path, engine="windcube")


def test_open_windcube_without_site(tmp_path):
    path = write_windcube(
        tmp_path / "no_site.nc",
        [("ppi", 20.0, PPI_AZ, np.full(12, 20.0), GATES, False)],
        site=None,
    )
    with pytest.warns(UserWarning, match="site coordinates are NaN"):
        with xd.open_datatree(path, engine="windcube") as dtree:
            for name in ("latitude", "longitude", "altitude"):
                assert np.isnan(float(dtree.ds[name]))


@pytest.fixture
def windcube_file_old(tmp_path):
    # layout of older files (e.g. WindCube Lidar server 3.3.3, WLS200s): time
    # in seconds since a ``time_reference`` variable, a root ``sweep``
    # coordinate and the title "Leosphere Windcube data"
    path = tmp_path / "WLS200s-218_2022-10-07_00-51-38_dbs_1823_75m.nc"
    with h5netcdf.File(path, "w") as f:
        write_root(
            f,
            ["Sweep_62961"],
            [75.0],
            site=(("latitude", 51.97), ("longitude", 4.93), ("altitude", 0)),
            title="Leosphere Windcube data",
        )
        f.create_variable("sweep", ("sweep",), data=np.array([1], "i4"))
        grp = f.create_group("Sweep_62961")
        write_sweep(grp, "dbs", DBS_AZ, DBS_EL, DBS_RNG, two_d_range=True)
        grp.variables["time"].attrs["units"] = "seconds since time_reference"
        grp.create_variable(
            "time_reference",
            data=np.array("1970-01-01T00:00:00Z", dtype=object),
            dtype=h5py.string_dtype(),
        )
    return path


def test_open_windcube_time_reference(windcube_file_old):
    with xd.open_datatree(windcube_file_old, engine="windcube") as dtree:
        sweeps = _sweeps(dtree)
        # inclined and vertical DBS beams, no inherited root "sweep" dimension
        assert len(sweeps) == 2
        for s in sweeps:
            assert "sweep" not in dtree[s].ds.dims
        # seconds since the epoch given in ``time_reference``
        expected = np.datetime64(int(1.7442e9 * 1000), "ms")
        assert dtree["sweep_0"].ds.time.values[0] == expected


def test_windcube_entrypoint_has_no_guess_can_open():
    # the engine is chosen explicitly; guessing would have to open every file
    assert WindCubeBackendEntrypoint.guess_can_open is (
        xr.backends.BackendEntrypoint.guess_can_open
    )
