#!/usr/bin/env python
# Copyright (c) 2023-2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.


import numpy as np
import pytest
import xarray as xr

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


def test_compare_sweeps(cfradial1_file, temp_file, closing):
    # Open the data tree
    # todo: implement a roundtrip function
    dtree = closing(xd.io.open_cfradial1_datatree(cfradial1_file))
    # Save the modified data tree to the temporary file
    xd.io.to_cfradial1(dtree.copy(), temp_file, calibs=True)

    # Open the modified data tree
    dtree1 = closing(xd.io.open_cfradial1_datatree(temp_file))
    # todo: check, if we can use xarray machinery for
    #  testing tree equality
    # Compare the values of the DataArrays for all sweeps
    for sweep_num in range(9):  # there are 9 sweeps in this file
        xr.testing.assert_equal(
            dtree[f"sweep_{sweep_num}"].dataset, dtree1[f"sweep_{sweep_num}"].dataset
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


def test_cfradial1_export_auto_filename(cfradial1_file, tmp_path, monkeypatch, closing):
    dtree = closing(xd.io.open_cfradial1_datatree(cfradial1_file))

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
def test_cfradial1_variable_gates_within_sweep(tmp_path, chunks, closing):
    ds, ngates, start = _ragged_cfradial1_dataset()
    path = tmp_path / "ragged_cfradial1.nc"
    ds.to_netcdf(path)

    trees = [closing(xd.io.open_cfradial1_datatree(path, first_dim="time"))]
    with xr.open_dataset(path, chunks=chunks) as src:
        trees.append(src.xradar.to_cfradial2_datatree())

    for dtree in trees:
        for sw, (r0, r1) in {"sweep_0": (0, 4), "sweep_1": (4, 8)}.items():
            swp = dtree[sw].dataset
            if "time" not in swp.dims:
                swp = swp.swap_dims({swp.DBZ.dims[0]: "time"}).sortby("time")
            assert swp.DBZ.shape == (r1 - r0, ngates[r0:r1].max())
            for i, ray in enumerate(range(r0, r1)):
                n = ngates[ray]
                np.testing.assert_array_equal(
                    swp.DBZ.values[i, :n], ds.DBZ.values[start[ray] : start[ray] + n]
                )
                assert np.isnan(swp.DBZ.values[i, n:]).all()


def test_cfradial1_rhi_sweep_dimension(tmp_path, closing):
    ds, _, _ = _ragged_cfradial1_dataset()
    path = tmp_path / "ragged_cfradial1.nc"
    ds.to_netcdf(path)
    dtree = closing(xd.io.open_cfradial1_datatree(path))
    assert "azimuth" in dtree["sweep_0"].dataset.dims
    assert "elevation" in dtree["sweep_1"].dataset.dims
    with xr.open_dataset(path, chunks={}) as src:
        dtree = src.xradar.to_cfradial2_datatree()
    assert dtree["sweep_1"].dataset.sweep_mode.item() == "rhi"
    assert "elevation" in dtree["sweep_1"].dataset.dims


def test_cfradial1_export_variable_gates_roundtrip(cfradial1n_file, tmp_path, closing):
    # the export is padded to (time, range), so the ragged index variables
    # must not be written (#416)
    dtree = closing(xd.io.open_cfradial1_datatree(cfradial1n_file))
    # source-style global flag, must be flipped as the export is padded
    dtree.attrs["n_gates_vary"] = "true"
    path = tmp_path / "variable_gates.nc"
    xd.io.to_cfradial1(dtree, path)

    with xr.open_dataset(path, decode_timedelta=False) as out:
        assert "ray_n_gates" not in out.variables
        assert "ray_start_index" not in out.variables
        assert "n_points" not in out.dims
        assert out.attrs["n_gates_vary"] == "false"

    back = closing(xd.io.open_cfradial1_datatree(path))
    for sweep in [k for k in dtree.children if k.startswith("sweep_")]:
        nrange = dtree[sweep].dataset.sizes["range"]
        np.testing.assert_array_equal(
            back[sweep].dataset.DBZ.values[:, :nrange], dtree[sweep].dataset.DBZ.values
        )


def test_cfradial1_export_strings_as_char_arrays(odim_file, tmp_path, closing):
    # string variables must be char arrays (not NC_STRING) and "None"
    # placeholders must not be written as global attributes (#417)
    import netCDF4

    dtree = closing(xd.io.open_odim_datatree(odim_file, sweep=[0, 1]))
    path = tmp_path / "odim_cf1.nc"
    xd.io.to_cfradial1(dtree, path)

    with netCDF4.Dataset(path) as nc:
        for name in ["time_coverage_start", "instrument_type", "prt_mode"]:
            assert nc[name].dtype == np.dtype("S1"), name
        assert netCDF4.chartostring(nc["time_coverage_start"][:]) == str(
            dtree.dataset.time_coverage_start.values
        )
        attrs = {k: nc.getncattr(k) for k in nc.ncattrs()}
    assert "None" not in attrs.values()
    assert not attrs["history"].startswith(("None", ":"))


def test_cfradial1_export_bool_attrs(nexradlevel2_file, tmp_path, closing):
    # netCDF attributes can't be bool; CfRadial uses "true"/"false" (#418)
    import netCDF4

    dtree = xd.io.open_nexradlevel2_datatree(nexradlevel2_file, sweep=[0, 2])
    bool_attrs = [k for k, v in dtree.attrs.items() if isinstance(v, (bool, np.bool_))]
    assert bool_attrs
    path = tmp_path / "nexrad_cf1.nc"
    xd.io.to_cfradial1(dtree, path)
    with netCDF4.Dataset(path) as nc:
        for name in bool_attrs:
            assert nc.getncattr(name) == str(bool(dtree.attrs[name])).lower()
    back = closing(xd.io.open_cfradial1_datatree(path))
    np.testing.assert_array_equal(
        back["sweep_0"].dataset.DBZH.values[
            :, : dtree["sweep_0"].dataset.sizes["range"]
        ],
        dtree["sweep_0"].dataset.DBZH.values,
    )


def test_cfradial1_export_duplicate_ray_times(cfradial1_file, tmp_path):
    # rays sharing a timestamp, also across sweeps, must not be merged on
    # export (#415)
    dtree = xd.io.open_cfradial1_datatree(cfradial1_file, sweep=[0, 1])
    t0 = dtree["sweep_0"].dataset.time.values[0]
    for sweep in ["sweep_0", "sweep_1"]:
        ds = dtree[sweep].to_dataset()
        ds = ds.assign_coords(time=ds.time.copy(data=np.full_like(ds.time, t0)))
        dtree[sweep] = xr.DataTree(ds)
    nrays = [dtree[sweep].dataset.sizes["azimuth"] for sweep in ["sweep_0", "sweep_1"]]

    path = tmp_path / "duplicate_times.nc"
    xd.io.to_cfradial1(dtree, path)
    with xr.open_dataset(path, decode_timedelta=False) as out:
        assert out.sizes["time"] == sum(nrays)
        np.testing.assert_array_equal(out.sweep_start_ray_index, [0, nrays[0]])
        np.testing.assert_array_equal(
            out.sweep_end_ray_index, [nrays[0] - 1, sum(nrays) - 1]
        )

    back = xd.io.open_cfradial1_datatree(path)
    for sweep in ["sweep_0", "sweep_1"]:
        np.testing.assert_array_equal(
            back[sweep].dataset.DBZ.values[:, : dtree[sweep].dataset.sizes["range"]],
            dtree[sweep].dataset.DBZ.values,
        )


def test_cfradial1_export_pads_missing_gates_with_nan(nexradlevel2_file, tmp_path):
    # packed moments (uint8, no _FillValue) must be padded with NaN on export,
    # not with raw 0 that decodes to valid data (-33 dBZ)
    dtree = xd.io.open_nexradlevel2_datatree(nexradlevel2_file, sweep=[0, 1])
    nrange = dtree["sweep_1"].dataset.sizes["range"]
    assert nrange < dtree["sweep_0"].dataset.sizes["range"]
    path = tmp_path / "padded.nc"
    xd.io.to_cfradial1(dtree, path)
    with xd.io.open_cfradial1_datatree(path) as back:
        dbzh = back["sweep_1"].dataset.DBZH.values
    assert np.isnan(dbzh[:, nrange:]).all()
    np.testing.assert_array_equal(
        dbzh[:, :nrange], dtree["sweep_1"].dataset.DBZH.values
    )


def test_cfradial1_export_range_within_tolerance(cfradial1_file, tmp_path):
    # a range that differs by less than the tolerance of the geometry check is
    # snapped, it must not interleave gates in the outer join
    dtree = xd.io.open_cfradial1_datatree(cfradial1_file, sweep=[0, 1])
    nrange = dtree["sweep_0"].dataset.sizes["range"]
    node = dtree["sweep_1"]
    ds = node.to_dataset(inherit=False)
    node.dataset = ds.assign_coords(range=ds.range + 1e-3)
    path = tmp_path / "range_tolerance.nc"
    xd.io.to_cfradial1(dtree, path)
    with xr.open_dataset(path, decode_timedelta=False) as out:
        assert out.sizes["range"] == nrange


def test_cfradial1_export_single_sweep_keeps_scalars(odim_file3, tmp_path):
    # one sweep is not concatenated, its scalars must not get a time dimension
    dtree = xd.io.open_odim_datatree(odim_file3)
    assert dtree["sweep_0"].dataset.nyquist_velocity.dims == ()
    path = tmp_path / "single_sweep.nc"
    xd.io.to_cfradial1(dtree, path)
    with xr.open_dataset(path, decode_timedelta=False) as out:
        assert out.nyquist_velocity.dims == ()


def test_cfradial1_export_sweeps_packed_differently(odim_file, tmp_path):
    # ODIM sweeps have their own scale_factor/add_offset; the packing of the
    # first sweep must not be applied to the others (values were clipped)
    outfile = tmp_path / "odim_cf1.nc"
    with xd.open_datatree(odim_file, engine="odim") as dtree:
        packings = {
            (
                dtree[sweep]["KDP"].encoding["scale_factor"],
                dtree[sweep]["KDP"].encoding["add_offset"],
            )
            for sweep in dtree.match("sweep_*")
        }
        assert len(packings) > 1
        xd.io.to_cfradial1(dtree, outfile)

        with xd.open_datatree(outfile, engine="cfradial1") as back:
            for sweep in dtree.match("sweep_*"):
                for name, var in dtree[sweep].data_vars.items():
                    if var.ndim != 2 or var.dtype.kind != "f":
                        continue
                    expected = var.values
                    result = back[sweep][name].values[:, : expected.shape[1]]
                    np.testing.assert_allclose(
                        result, expected, rtol=1e-6, equal_nan=True, err_msg=name
                    )


def test_cfradial1_export_sweep_attrs_differ(uf_file_1, tmp_path):
    # variable attributes that differ between sweeps (here the time units of
    # the UF reader) don't make the export fail
    outfile = tmp_path / "uf_cf1.nc"
    with xd.open_datatree(uf_file_1, engine="uf") as dtree:
        units = {dtree[s]["time"].attrs.get("units") for s in dtree.match("sweep_*")}
        assert len(units) > 1
        nsweeps = len(list(dtree.match("sweep_*")))
        xd.io.to_cfradial1(dtree, outfile)
    with xd.open_datatree(outfile, engine="cfradial1") as back:
        assert len(list(back.match("sweep_*"))) == nsweeps


def test_cfradial1_export_drops_sweep_group_name(hpl_file1, tmp_path):
    # the HPL reader has sweep_group_name in every sweep; it's no CfRadial1
    # variable and made the written file unreadable
    import netCDF4

    outfile = tmp_path / "hpl_cf1.nc"
    with xd.open_datatree(hpl_file1, engine="hpl") as dtree:
        assert "sweep_group_name" in dtree["sweep_0"].dataset
        nsweeps = len(list(dtree.match("sweep_*")))
        xd.io.to_cfradial1(dtree, outfile)
    with netCDF4.Dataset(outfile) as nc:
        assert "sweep_group_name" not in nc.variables
    with xd.open_datatree(outfile, engine="cfradial1") as back:
        assert len(list(back.match("sweep_*"))) == nsweeps


@pytest.mark.parametrize(
    "engine, file_fixture", [("cfradial1", "cfradial1_file"), ("odim", "odim_file")]
)
def test_cfradial1_export_repeated_with_optional_groups(
    engine, file_fixture, request, tmp_path
):
    # data read with optional_groups=True from a file written by xradar can be
    # exported again, and the groups survive any number of round trips
    filename = request.getfixturevalue(file_fixture)
    groups = ("radar_parameters", "radar_calibration", "georeferencing_correction")
    with xd.open_datatree(filename, engine=engine, optional_groups=True) as source:
        current = source
        trees = []
        for i in range(3):
            outfile = tmp_path / f"optional_{i}.nc"
            xd.io.to_cfradial1(current, outfile)
            current = xd.open_datatree(
                outfile, engine="cfradial1", optional_groups=True
            )
            trees.append(current)
    try:
        for tree in trees:
            assert all(group in tree.children for group in groups)
        for first, second in zip(trees[:-1], trees[1:]):
            assert list(first.subtree_with_keys) and [
                p for p, _ in first.subtree_with_keys
            ] == [p for p, _ in second.subtree_with_keys]
            for (_, node1), (_, node2) in zip(
                first.subtree_with_keys, second.subtree_with_keys
            ):
                ds1 = node1.to_dataset(inherit=False).load()
                ds2 = node2.to_dataset(inherit=False).load()
                for ds in (ds1, ds2):
                    ds.attrs.pop("history", None)
                xr.testing.assert_identical(ds1, ds2)
    finally:
        for tree in trees:
            tree.close()


def test_cfradial1_export_version_and_sub_conventions(cfradial1_file, tmp_path):
    # CfRadial 1.4: "Cf/Radial" plus the sub-conventions used, whose variables
    # carry a meta_group attribute (#388)
    import netCDF4

    path = tmp_path / "cf14.nc"
    with xd.io.open_cfradial1_datatree(cfradial1_file, optional_groups=True) as dtree:
        xd.io.to_cfradial1(dtree, path)
    with netCDF4.Dataset(path) as nc:
        assert nc.version == "1.4"
        conventions = nc.Conventions.split()
        assert conventions[0] == "Cf/Radial"
        groups = {
            nc[v].meta_group for v in nc.variables if "meta_group" in nc[v].ncattrs()
        }
        assert set(conventions[1:]) == groups
        assert {
            "instrument_parameters",
            "radar_parameters",
            "radar_calibration",
            "geometry_correction",
        } <= groups
        assert nc["prt_mode"].meta_group == "instrument_parameters"
        assert nc["r_calib_index"].meta_group == "radar_calibration"
        assert "meta_group" not in nc["DBZ"].ncattrs()


@pytest.mark.parametrize(
    "optional_groups, calibs",
    [(False, True), (True, False)],
    ids=["default", "no-calibs"],
)
def test_cfradial1_export_without_calibration(cfradial1_file, optional_groups, calibs):
    # without r_calib variables there is no r_calib dimension, so r_calib_index
    # is kept untagged and radar_calibration is not listed (#388)
    with xd.io.open_cfradial1_datatree(
        cfradial1_file, optional_groups=optional_groups
    ) as dtree:
        ds = cf1_export._build_cfradial1_dataset(dtree, calibs=calibs)
    assert "meta_group" not in ds["r_calib_index"].attrs
    assert "radar_calibration" not in ds.attrs["Conventions"].split()


def test_cfradial1_meta_groups():
    # tags follow CfRadial 1.4 Section 5, not the variable name prefix, and
    # only sub-conventions of the specification go into Conventions (#388)
    ds = xr.Dataset(
        {
            "radar_measured_sky_noise": ((), 1.0),
            "radar_beam_width_h": ((), 1.0),
            "radar_echo_class": ((), 1.0),
            "custom": ((), 1.0, {"meta_group": "not_a_sub_convention"}),
        },
        coords={"frequency": ("frequency", [2.8e9])},
    )
    groups = cf1_export._add_meta_groups(ds)
    assert groups == ["instrument_parameters", "radar_parameters"]
    assert ds["radar_measured_sky_noise"].attrs["meta_group"] == "instrument_parameters"
    assert ds["frequency"].attrs["meta_group"] == "instrument_parameters"
    assert ds["radar_beam_width_h"].attrs["meta_group"] == "radar_parameters"
    assert "meta_group" not in ds["radar_echo_class"].attrs


def test_cfradial1_export_metadata_group_names(cfradial1_file, tmp_path):
    # radar_parameters, radar_calibration and georeferencing_correction are
    # written with the names of the CfRadial 1.4 tables (Sections 5.2, 5.4 and
    # 5.7), not with the aliases other software uses (Radx: radar_rx_bandwidth,
    # r_calib_base_dbz_1km_hc, altitude_correction, eastward_velocity_correction;
    # the reader accepts both) (#419)
    import netCDF4

    path = tmp_path / "metadata_groups.nc"
    with xd.io.open_cfradial1_datatree(cfradial1_file, optional_groups=True) as dtree:
        xd.io.to_cfradial1(dtree, path)

    aliases = {
        "radar_receiver_bandwidth": "radar_rx_bandwidth",
        "r_calib_base_1km_hc": "r_calib_base_dbz_1km_hc",
        "radar_altitude_correction": "altitude_correction",
        "eastward_ground_speed_correction": "eastward_velocity_correction",
    }
    with netCDF4.Dataset(cfradial1_file) as src, netCDF4.Dataset(path) as out:
        for table_name, alias in aliases.items():
            assert alias in src.variables  # the file uses the aliases
            assert table_name in out.variables
            assert alias not in out.variables
            np.testing.assert_allclose(out[table_name][:], src[alias][:])

    # the written file reads back to the same names
    with xd.io.open_cfradial1_datatree(path, optional_groups=True) as back:
        assert "radar_receiver_bandwidth" in back["radar_parameters"].dataset
        assert "base_1km_hc" in back["radar_calibration"].dataset
        assert "radar_altitude_correction" in back["georeferencing_correction"].dataset


def test_cfradial1_unknown_calibration_names(file_store, tmp_path):
    # variables on the r_calib dimension with non-standard names must not
    # break reading the radar_calibration group and keep their names on
    # export (#419)
    import netCDF4

    filename = file_store.fetch("20220628072500_savevol_COSMO_LOOKUP_TEMP.nc")
    path = tmp_path / "unknown_calibration.nc"
    with xd.io.open_cfradial1_datatree(filename, optional_groups=True) as dtree:
        assert "calibration_constant_hh" in dtree["radar_calibration"].dataset
        xd.io.to_cfradial1(dtree, path)
    with netCDF4.Dataset(path) as out:
        assert "calibration_constant_hh" in out.variables
        assert "r_calib_calibration_constant_hh" not in out.variables


@pytest.mark.parametrize(
    "names, expected",
    [
        (
            ["r_calib_receiver_mismatch_loss_h", "r_calib_receiver_mismatch_loss_v"],
            ["receiver_mismatch_loss_h", "receiver_mismatch_loss_v"],
        ),
        (["r_calib_noise_hc", "noise_hc_offset"], ["noise_hc", "noise_hc_offset"]),
    ],
)
def test_cfradial1_calibration_names_exact_match(names, expected):
    # calibration names are matched exactly, not as substrings (#419)
    from xradar.io.backends.cfradial1 import _get_radar_calibration

    ds = xr.Dataset({name: ("r_calib", [1.0]) for name in names})
    calib = _get_radar_calibration(ds)
    assert sorted(calib.data_vars) == expected


def test_cfradial1_export_split_cuts(nexradlevel2_file, tmp_path):
    # NEXRAD split cuts: same elevation, different moments; combining the
    # per-sweep metadata conflicted on sweep_number (#418)
    dtree = xd.io.open_nexradlevel2_datatree(nexradlevel2_file, sweep=[0, 1])
    # bool attributes are handled separately (#423)
    dtree.attrs = {
        k: v for k, v in dtree.attrs.items() if not isinstance(v, (bool, np.bool_))
    }
    assert (
        "VRADH" not in dtree["sweep_0"].dataset and "VRADH" in dtree["sweep_1"].dataset
    )
    path = tmp_path / "split_cuts.nc"
    xd.io.to_cfradial1(dtree, path)
    back = xd.io.open_cfradial1_datatree(path, first_dim="time")
    for i, sweep in enumerate(["sweep_0", "sweep_1"]):
        src = dtree[sweep].dataset.sortby("time")
        out = back[sweep].dataset
        assert int(out.sweep_number) == i
        assert float(out.sweep_fixed_angle) == float(src.sweep_fixed_angle)
        np.testing.assert_array_equal(
            out.DBZH.values[:, : src.sizes["range"]], src.DBZH.values
        )
    np.testing.assert_array_equal(
        back["sweep_1"].dataset.VRADH.values[
            :, : dtree["sweep_1"].dataset.sizes["range"]
        ],
        dtree["sweep_1"].dataset.sortby("time").VRADH.values,
    )


def test_cfradial1_export_range_geometry(gamic_file, tmp_path):
    # sweeps with different gate spacing can't share one range coordinate,
    # CfRadial 1.4 section 2.5 asks for one geometry, the message points to
    # reindexing and says what that does
    dtree = xd.io.open_gamic_datatree(gamic_file)
    with pytest.raises(ValueError, match="gate geometry.*nearest gate") as err:
        xd.io.to_cfradial1(dtree, tmp_path / "gamic.nc")
    assert "reindex_range" in str(err.value)

    dtree = xd.io.open_gamic_datatree(
        gamic_file,
        reindex_coord=dict(
            range=dict(start_range=75.0, stop_range=150000.0, range_res=150.0)
        ),
    )
    xd.io.to_cfradial1(dtree, tmp_path / "gamic.nc")
    back = xd.io.open_cfradial1_datatree(tmp_path / "gamic.nc")
    np.testing.assert_allclose(back["sweep_9"].dataset.range.diff("range"), 150.0)


def test_cfradial1_per_sweep_range_helper():
    ranges = [np.arange(50.0, 350.0, 100.0), np.arange(75.0, 675.0, 150.0)]
    rng = cf1_export._per_sweep_range(ranges, {"units": "meters"})
    assert rng.dims == ("sweep", "range")
    assert rng.shape == (2, 4)
    np.testing.assert_allclose(rng[0], [50.0, 150.0, 250.0, 350.0])  # continued
    np.testing.assert_allclose(rng[1], ranges[1])
    assert rng.attrs["units"] == "meters"
    np.testing.assert_allclose(rng.attrs["meters_to_center_of_first_gate"], [50, 75])
    np.testing.assert_allclose(rng.attrs["meters_between_gates"], [100, 150])
    # a sweep with a single gate has no spacing, its padding is NaN
    rng = cf1_export._per_sweep_range([np.array([50.0]), ranges[1]], {})
    assert np.isnan(rng[0, 1:]).all()


def test_cfradial1_export_range_geometry_option(gamic_file, tmp_path):
    with xd.open_datatree(gamic_file, engine="gamic", sweep=[0, 9]) as dtree:
        with pytest.raises(ValueError, match="range_geometry"):
            xd.io.to_cfradial1(dtree, tmp_path / "x.nc", range_geometry="other")
        # the default still raises, and names the option
        with pytest.raises(ValueError, match="range_geometry=.per_sweep.") as err:
            xd.io.to_cfradial1(dtree, tmp_path / "x.nc")
        assert "reindex_range" in str(err.value)


def test_cfradial1_export_per_sweep_range(gamic_file, tmp_path):
    # GAMIC: 100 m gates in the lowest and 150 m gates in the highest sweep,
    # different numbers of gates, different first gates (50 m and 75 m)
    path = tmp_path / "per_sweep.nc"
    with xd.open_datatree(gamic_file, engine="gamic") as dtree:
        sweeps = [k for k in dtree.children if k.startswith("sweep_")]
        xd.io.to_cfradial1(dtree, path, range_geometry="per_sweep")

        with xr.open_dataset(path, decode_timedelta=False) as ds:
            assert ds["range"].dims == ("sweep", "range")
            assert ds.sizes["sweep"] == len(sweeps)
            attrs = ds["range"].attrs
            assert set(np.unique(attrs["meters_to_center_of_first_gate"])) == {50, 75}
            assert set(np.unique(attrs["meters_between_gates"])) == {100, 150}
            # increasing in every gate, also in the padding
            assert (np.diff(ds["range"].values, axis=1) > 0).all()

        # every sweep comes back as it was written: its own range, all gates,
        # the same values, NaN in the padding
        with xd.open_datatree(path, engine="cfradial1") as back:
            for sweep in sweeps:
                source, result = dtree[sweep].dataset, back[sweep].dataset
                ngates = source.sizes["range"]
                np.testing.assert_allclose(result.range.values[:ngates], source.range)
                for name in ("meters_to_center_of_first_gate", "meters_between_gates"):
                    assert result.range.attrs[name] == source.range.attrs[name]
                moments = [
                    name
                    for name, var in source.data_vars.items()
                    if "range" in var.dims and var.dtype.kind == "f" and name in result
                ]
                assert moments
                for name in moments:
                    np.testing.assert_array_equal(
                        result[name].values[..., :ngates], source[name].values
                    )
                    assert np.isnan(result[name].values[..., ngates:]).all()


def test_cfradial1_export_per_sweep_range_with_common_geometry(
    cfradial1_file, tmp_path
):
    # one geometry in all sweeps: the usual 1-D range, also when asked per sweep
    path = tmp_path / "common.nc"
    with xd.open_datatree(cfradial1_file, engine="cfradial1") as dtree:
        xd.io.to_cfradial1(dtree, path, range_geometry="per_sweep")
    with xr.open_dataset(path, decode_timedelta=False) as ds:
        assert ds["range"].dims == ("range",)


def test_cfradial1_per_sweep_range_dataset(gamic_file):
    # the in-memory dataset and the accessor take the option as well
    with xd.open_datatree(gamic_file, engine="gamic", sweep=[0, 9]) as dtree:
        ds = xd.transform.to_cfradial1(dtree, range_geometry="per_sweep")
        assert ds["range"].dims == ("sweep", "range")
        assert dtree.xradar.to_cfradial1(range_geometry="per_sweep").identical(ds)
        assert dtree.xradar.to_cfradial1_dataset(range_geometry="per_sweep").identical(
            ds
        )
        with pytest.raises(ValueError, match="range_geometry"):
            xd.transform.to_cfradial1(dtree)


def test_cfradial1_sweeps_not_in_time_order(gamic_file, tmp_path, closing):
    # the sweep ray indices refer to the order of the file, not to the time
    # sorted rays, so a later sweep written first must keep its own rays (#550)
    dtree = closing(xd.io.open_gamic_datatree(gamic_file, sweep=[5, 0]))
    assert dtree["sweep_0"].dataset.time[0] > dtree["sweep_1"].dataset.time[0]
    xd.io.to_cfradial1(dtree, tmp_path / "unsorted.nc")
    back = closing(xd.io.open_cfradial1_datatree(tmp_path / "unsorted.nc"))
    for sweep in ["sweep_0", "sweep_1"]:
        src, res = dtree[sweep].dataset, back[sweep].dataset
        assert res.sweep_fixed_angle.item() == src.sweep_fixed_angle.item()
        np.testing.assert_array_equal(res.time.values, src.time.values)
        np.testing.assert_allclose(res.azimuth.values, src.azimuth.values)
        nrange = min(src.sizes["range"], res.sizes["range"])
        np.testing.assert_array_equal(
            res.DBZH.values[:, :nrange], src.DBZH.values[:, :nrange]
        )
