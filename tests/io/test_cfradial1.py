#!/usr/bin/env python
# Copyright (c) 2023-2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.


import numpy as np
import pytest
import xarray as xr
from open_radar_data import DATASETS

import xradar as xd
from xradar.io.export import cfradial1 as cf1_export


def test_cfradial1_open_mfdataset_context_manager(cfradial1_file):
    with xr.open_mfdataset(
        [cfradial1_file],
        engine="cfradial1",
        concat_dim="volume_time",
        combine="nested",
        group="sweep_0",
    ) as ds:
        assert ds is not None
        # closer must exist while inside context
        assert callable(getattr(ds, "_close", None))


def test_cfradial1_dataset_has_close(cfradial1_file):
    ds = xr.open_dataset(cfradial1_file, engine="cfradial1", group="sweep_0")
    assert callable(getattr(ds, "_close", None))
    ds.close()


def test_compare_sweeps(temp_file):
    # Fetch the radar data file
    filename = DATASETS.fetch("cfrad.20080604_002217_000_SPOL_v36_SUR.nc")

    # Open the data tree
    # todo: implement a roundtrip function
    dtree = xd.io.open_cfradial1_datatree(filename)
    # Save the modified data tree to the temporary file
    xd.io.to_cfradial1(dtree.copy(), temp_file, calibs=True)

    # Open the modified data tree
    dtree1 = xd.io.open_cfradial1_datatree(temp_file)
    # todo: check, if we can use xarray machinery for
    #  testing tree equality
    # Compare the values of the DataArrays for all sweeps
    for sweep_num in range(9):  # there are 9 sweeps in this file
        xr.testing.assert_equal(
            dtree[f"sweep_{sweep_num}"].ds, dtree1[f"sweep_{sweep_num}"].ds
        )


def test_cfradial1_export_helper_scalar_normalization():
    assert cf1_export._first_valid_scalar(xr.DataArray(np.array([np.nan, 3.0]))) == 3.0

    masked = xr.DataArray(np.ma.array([1.0, 2.0], mask=[True, False]))
    assert cf1_export._first_valid_scalar(masked) == 2.0

    nat = xr.DataArray(np.array(["NaT", "2025-01-01T00:00:00"], dtype="datetime64[ns]"))
    assert np.datetime64(cf1_export._first_valid_scalar(nat), "ns") == np.datetime64(
        "2025-01-01T00:00:00", "ns"
    )

    text = xr.DataArray(np.array(["azimuth_surveillance"], dtype=object))
    assert cf1_export._first_valid_scalar(text) == "azimuth_surveillance"

    missing = xr.DataArray(np.array([np.nan, np.nan]))
    assert np.isnan(cf1_export._first_valid_scalar(missing))


def test_cfradial1_export_helper_metadata_and_indices():
    sweep = xr.Dataset(
        data_vars={
            "sweep_number": (
                ("azimuth", "range"),
                np.array([[0.0, np.nan], [0.0, np.nan]]),
            ),
            "sweep_mode": (
                ("azimuth", "range"),
                np.array(
                    [
                        ["azimuth_surveillance", None],
                        ["azimuth_surveillance", None],
                    ],
                    dtype=object,
                ),
            ),
            "DBZ": (("azimuth", "range"), np.ones((2, 2), dtype="float32")),
        },
        coords={
            "azimuth": ("azimuth", np.array([0.0, 1.0], dtype="float32")),
            "range": ("range", np.array([100.0, 200.0], dtype="float32")),
            "time": (
                "azimuth",
                np.array(["2025-01-01", "2025-01-01"], dtype="datetime64[ns]"),
            ),
            "elevation": ("azimuth", np.array([0.5, 0.5], dtype="float32")),
        },
    )

    normalized = cf1_export._normalize_sweep_metadata(sweep)
    assert normalized["sweep_number"].dims == ()
    assert normalized["sweep_number"].item() == 0.0
    assert normalized["sweep_mode"].dims == ()
    assert normalized["sweep_mode"].item() == "azimuth_surveillance"

    valid = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(),
            "/sweep_0": xr.Dataset(
                coords={"elevation": ("azimuth", np.array([0.5, 0.5], dtype="float32"))}
            ),
        }
    )
    out = cf1_export.calculate_sweep_indices(valid)
    assert out["sweep_start_ray_index"].dims == ("sweep",)
    assert out["sweep_end_ray_index"].dims == ("sweep",)


def test_cfradial1_export_helper_empty_sweep_info_and_time_fallback():
    empty = xr.DataTree.from_dict({"/": xr.Dataset()})
    sweep_info = cf1_export._collect_sweep_metadata(empty)
    assert "sweep_number" in sweep_info
    assert np.isnan(sweep_info["sweep_number"].values[0])

    sweep = xr.Dataset(
        data_vars={
            "DBZ": (("time", "range"), np.ones((2, 2), dtype="float32")),
            "sweep_mode": ((), "manual"),
            "sweep_number": ((), 0),
            "sweep_fixed_angle": ((), 0.5),
        },
        coords={
            "time": (
                "time",
                np.array(["2025-01-01", "2025-01-01T00:00:01"], dtype="datetime64[ns]"),
            ),
            "range": ("range", np.array([100.0, 200.0], dtype="float32")),
            "azimuth": ("time", np.array([0.0, 1.0], dtype="float32")),
            "elevation": ("time", np.array([0.5, 0.5], dtype="float32")),
        },
    )
    dtree = xr.DataTree.from_dict({"/": xr.Dataset(), "/sweep_0": sweep})
    mapped = cf1_export._combine_sweeps(dtree)
    assert "DBZ" in mapped
    assert mapped["DBZ"].dims == ("time", "range")


def test_cfradial1_export_auto_filename(tmp_path, monkeypatch):
    filename = DATASETS.fetch("cfrad.20080604_002217_000_SPOL_v36_SUR.nc")
    dtree = xd.io.open_cfradial1_datatree(filename)

    # filename=None derives the name from instrument_name + first timestamp
    monkeypatch.chdir(tmp_path)
    xd.io.to_cfradial1(dtree.copy(), filename=None, calibs=True)

    written = list(tmp_path.glob("cfrad1_*.nc"))
    assert len(written) == 1
    assert written[0].name.startswith("cfrad1_")


def test_cfradial1_export_requires_dtree():
    with pytest.raises(ValueError, match="must be a radar"):
        xd.io.to_cfradial1(None)


def test_cfradial1_export_sweep_indices_missing_elevation():
    dtree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(),
            "/sweep_0": xr.Dataset(
                coords={"elevation": ("azimuth", np.array([0.5, 0.5], dtype="float32"))}
            ),
            "/sweep_1": xr.Dataset(),  # no elevation coordinate -> skipped with warning
        }
    )

    with pytest.warns(UserWarning, match="no 'elevation'"):
        out = cf1_export.calculate_sweep_indices(dtree)

    # only the valid sweep contributes a ray-index entry
    assert out["sweep_start_ray_index"].size == 1
    assert out["sweep_end_ray_index"].size == 1


def _ragged_cfradial1_dataset():
    # two sweeps (PPI + RHI) with a variable number of gates per ray,
    # including short rays inside a sweep (#322)
    ngates = np.array([3, 5, 5, 4, 5, 2, 4, 4], dtype="int32")
    nrays = ngates.size
    start = np.r_[0, np.cumsum(ngates)[:-1]].astype("int32")
    npoints = int(ngates.sum())
    dbz = np.arange(npoints, dtype="float32")
    time = np.datetime64("2022-05-25T02:12:16", "ns") + np.arange(
        nrays
    ) * np.timedelta64(1, "s")
    ds = xr.Dataset(
        data_vars=dict(
            DBZ=("n_points", dbz, {"units": "dBZ"}),
            azimuth=("time", np.array([10.0, 20, 30, 40, 90, 90, 90, 90], "float32")),
            elevation=("time", np.array([0.5, 0.5, 0.5, 0.5, 1, 2, 3, 4], "float32")),
            ray_n_gates=("time", ngates),
            ray_start_index=("time", start),
            sweep_number=("sweep", np.array([0, 1], "int32")),
            sweep_mode=("sweep", np.array([b"azimuth_surveillance", b"rhi"])),
            fixed_angle=("sweep", np.array([0.5, 90.0], "float32")),
            sweep_start_ray_index=("sweep", np.array([0, 4], "int32")),
            sweep_end_ray_index=("sweep", np.array([3, 7], "int32")),
            latitude=0.0,
            longitude=0.0,
            altitude=0.0,
        ),
        coords=dict(
            time=("time", time),
            range=("range", np.arange(5, dtype="float32") * 100 + 50),
        ),
        attrs={"Conventions": "CF/Radial"},
    )
    return ds, ngates, start


@pytest.mark.parametrize("chunks", [None, {}])
def test_cfradial1_variable_gates_within_sweep(tmp_path, chunks):
    ds, ngates, start = _ragged_cfradial1_dataset()
    path = tmp_path / "ragged_cfradial1.nc"
    ds.to_netcdf(path)

    trees = [xd.io.open_cfradial1_datatree(path, first_dim="time")]
    with xr.open_dataset(path, chunks=chunks) as src:
        trees.append(src.xradar.to_cfradial2_datatree())

    for dtree in trees:
        for sw, (r0, r1) in {"sweep_0": (0, 4), "sweep_1": (4, 8)}.items():
            swp = dtree[sw].ds
            if "time" not in swp.dims:
                swp = swp.swap_dims({swp.DBZ.dims[0]: "time"}).sortby("time")
            assert swp.DBZ.shape == (r1 - r0, ngates[r0:r1].max())
            for i, ray in enumerate(range(r0, r1)):
                n = ngates[ray]
                np.testing.assert_array_equal(
                    swp.DBZ.values[i, :n], ds.DBZ.values[start[ray] : start[ray] + n]
                )
                assert np.isnan(swp.DBZ.values[i, n:]).all()


def test_cfradial1_rhi_sweep_dimension(tmp_path):
    ds, _, _ = _ragged_cfradial1_dataset()
    path = tmp_path / "ragged_cfradial1.nc"
    ds.to_netcdf(path)
    dtree = xd.io.open_cfradial1_datatree(path)
    assert "azimuth" in dtree["sweep_0"].ds.dims
    assert "elevation" in dtree["sweep_1"].ds.dims
    with xr.open_dataset(path, chunks={}) as src:
        dtree = src.xradar.to_cfradial2_datatree()
    assert dtree["sweep_1"].ds.sweep_mode.item() == "rhi"
    assert "elevation" in dtree["sweep_1"].ds.dims


def test_cfradial1_export_variable_gates_roundtrip(cfradial1n_file, tmp_path):
    # the export is padded to (time, range), so the ragged index variables
    # must not be written (#416)
    dtree = xd.io.open_cfradial1_datatree(cfradial1n_file)
    # source-style global flag, must be flipped as the export is padded
    dtree.attrs["n_gates_vary"] = "true"
    path = tmp_path / "variable_gates.nc"
    xd.io.to_cfradial1(dtree, path)

    with xr.open_dataset(path, decode_timedelta=False) as out:
        assert "ray_n_gates" not in out.variables
        assert "ray_start_index" not in out.variables
        assert "n_points" not in out.dims
        assert out.attrs["n_gates_vary"] == "false"

    back = xd.io.open_cfradial1_datatree(path)
    for sweep in [k for k in dtree.children if k.startswith("sweep_")]:
        nrange = dtree[sweep].ds.sizes["range"]
        np.testing.assert_array_equal(
            back[sweep].ds.DBZ.values[:, :nrange], dtree[sweep].ds.DBZ.values
        )


def test_cfradial1_export_strings_as_char_arrays(odim_file, tmp_path):
    # string variables must be char arrays (not NC_STRING) and "None"
    # placeholders must not be written as global attributes (#417)
    import netCDF4

    dtree = xd.io.open_odim_datatree(odim_file, sweep=[0, 1])
    path = tmp_path / "odim_cf1.nc"
    xd.io.to_cfradial1(dtree, path)

    with netCDF4.Dataset(path) as nc:
        for name in ["time_coverage_start", "instrument_type", "prt_mode"]:
            assert nc[name].dtype == np.dtype("S1"), name
        assert netCDF4.chartostring(nc["time_coverage_start"][:]) == str(
            dtree.ds.time_coverage_start.values
        )
        attrs = {k: nc.getncattr(k) for k in nc.ncattrs()}
    assert "None" not in attrs.values()
    assert not attrs["history"].startswith(("None", ":"))
