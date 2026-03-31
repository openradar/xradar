#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

import numpy as np
import pytest
import xarray as xr

import xradar as xd
from xradar.io.backends import cfradial2 as cf2


def _write_institutional_cfradial2(outfile):
    root = xr.Dataset(
        data_vars={
            "sweep_group_name": ("nsweep", np.array(["sweep_0000"], dtype=str)),
        },
        attrs={
            "Conventions": "CF-1.7, Cf/Radial",
            "version": "2.0",
            "title": "Radar CfRadial2 Dataset",
            "history": "created for test",
            "RadarName": "test-radar",
            "Latitude": 50.0,
            "Longitude": 8.0,
            "Height": 100.0,
        },
    )

    sweep = xr.Dataset(
        data_vars={
            "time_us": (
                "ntime",
                np.array(
                    ["2025-05-01T00:00:00", "2025-05-01T00:00:01"],
                    dtype="datetime64[ns]",
                ),
            ),
            "range_m": ("nrange", np.array([50.0, 150.0], dtype="float32")),
            "azimuth_deg": ("ntime", np.array([0.0, 1.0], dtype="float32")),
            "elevation_deg": ("ntime", np.array([2.0, 2.0], dtype="float32")),
            "DBZ": (
                ("ntime", "nrange"),
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"),
            ),
            "carrier_frequency_hz": (
                "ntime",
                np.array([2.8e9, 2.8e9], dtype="float64"),
            ),
        },
        attrs={"ScanType": 2},
    )

    tree = xr.DataTree.from_dict({"/": root, "sweep_0000": sweep})
    tree.to_netcdf(outfile, engine="netcdf4")


def _make_selection_tree():
    return xr.DataTree.from_dict(
        {
            "/": xr.Dataset(),
            "sweep_0": xr.Dataset(),
            "sweep_0002": xr.Dataset(),
            "sweep_10": xr.Dataset(),
        }
    )


def test_open_cfradial2_roundtrip(cfradial1_file, temp_file):
    dtree = xd.io.open_cfradial1_datatree(cfradial1_file, first_dim="time")
    outfile = temp_file.with_suffix(".nc")
    xd.io.to_cfradial2(dtree.copy(), outfile, engine="netcdf4")

    dtree2 = xd.io.open_cfradial2_datatree(outfile, engine="netcdf4")

    assert isinstance(dtree2, xr.DataTree)
    assert "sweep_0" in dtree2.children
    assert "DBZ" in dtree2["sweep_0"].data_vars
    xr.testing.assert_equal(dtree["sweep_0"].ds["DBZ"], dtree2["sweep_0"].ds["DBZ"])
    assert "latitude" in dtree2.ds.coords
    assert dtree2.ds["latitude"].attrs["standard_name"] == "latitude"
    assert (
        dtree2["sweep_0"].ds["range"].attrs["standard_name"]
        == "projection_range_coordinate"
    )
    assert "coordinates" in dtree2["sweep_0"].ds["DBZ"].attrs
    dtree2.close()


def test_open_cfradial2_sweep_selection_and_first_dim(cfradial1_file, temp_file):
    dtree = xd.io.open_cfradial1_datatree(cfradial1_file, first_dim="time")
    outfile = temp_file.with_suffix(".nc")
    xd.io.to_cfradial2(dtree.copy(), outfile, engine="netcdf4")

    dtree2 = xd.io.open_cfradial2_datatree(
        outfile, engine="netcdf4", sweep=1, first_dim="auto"
    )

    assert list(dtree2.children) == ["sweep_0"]
    assert "time" not in dtree2["sweep_0"].dims
    assert ("azimuth" in dtree2["sweep_0"].dims) or (
        "elevation" in dtree2["sweep_0"].dims
    )
    assert dtree2.ds["sweep_group_name"].item() == "sweep_0"
    dtree2.close()


def test_open_cfradial2_normalizes_common_aliases(temp_file):
    outfile = temp_file.with_suffix(".nc")
    _write_institutional_cfradial2(outfile)

    with pytest.warns(
        UserWarning, match="renumbered into sequential `sweep_<n>` order"
    ):
        dtree = xd.io.open_cfradial2_datatree(outfile, engine="netcdf4")

    sweep = dtree["sweep_0"].ds
    assert "time" in sweep.coords
    assert "range" in sweep.coords
    assert "azimuth" in sweep.coords
    assert "elevation" in sweep.coords
    assert "frequency" in sweep.coords
    assert sweep["sweep_number"].item() == 0
    assert sweep["sweep_mode"].item() == "azimuth_surveillance"
    assert sweep["follow_mode"].item() == "none"
    assert sweep["prt_mode"].item() == "fixed"
    assert dtree.ds["sweep_group_name"].item() == "sweep_0"
    assert dtree.ds.coords["latitude"].item() == 50.0
    assert dtree.attrs["instrument_name"] == "test-radar"
    assert sweep["time"].attrs["standard_name"] == "time"
    assert sweep["azimuth"].attrs["axis"] == "radial_azimuth_coordinate"
    assert sweep["DBZ"].attrs["coordinates"] == "elevation azimuth range"
    dtree.close()


def test_open_cfradial2_invalid_path():
    with pytest.raises(FileNotFoundError):
        xd.io.open_cfradial2_datatree("missing-cfradial2-file.nc")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("sweep_0002", "sweep_2"),
        ("sweep_x", "sweep_x"),
        ("root", "root"),
    ],
)
def test_cfradial2_helper_normalize_sweep_name(name, expected):
    assert cf2._normalize_sweep_name(name) == expected


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        (None, ["sweep_0", "sweep_2", "sweep_10"]),
        ("sweep_0002", ["sweep_2"]),
        (10, ["sweep_10"]),
        ([0, "sweep_0002"], ["sweep_0", "sweep_2"]),
    ],
)
def test_cfradial2_helper_selection(selection, expected):
    tree = _make_selection_tree()
    assert cf2._iter_selected_sweeps(tree, selection) == expected


def test_cfradial2_helper_selection_error():
    tree = _make_selection_tree()

    with pytest.raises(TypeError):
        cf2._iter_selected_sweeps(tree, 1.5)


def test_cfradial2_helper_root_and_subgroup_normalization():
    root = xr.Dataset(
        data_vars={
            "status_xml": ((), "ok"),
            "sweep_group_name": ("sweep", np.array(["legacy"], dtype=str)),
        },
        coords={
            "sweep_fixed_angle": ("sweep", np.array([0.5], dtype="float32")),
        },
        attrs={
            "RadarName": "root-radar",
            "Latitude": 1.0,
            "Longitude": 2.0,
            "Height": 3.0,
            "comment": "keep",
            "foo": "drop",
        },
    )
    sweep = xr.Dataset(
        data_vars={
            "fixed_angle": ((), 1.5),
            "time": ("time", np.array(["2025-01-01"], dtype="datetime64[ns]")),
        }
    )

    out = cf2._normalize_root_dataset(root, ["sweep_0"], [sweep], optional=False)

    assert out.attrs["instrument_name"] == "root-radar"
    assert "foo" not in out.attrs
    assert "status_xml" not in out.data_vars
    assert out["sweep_group_name"].item() == "sweep_0"
    assert out["sweep_fixed_angle"].item() == pytest.approx(1.5)
    assert "latitude" in out.coords

    sub = xr.DataTree(
        xr.Dataset(
            data_vars={"radar_rx_bandwidth": ((), 42.0), "other": ((), 1.0)},
            attrs={"drop": "me"},
        ),
        name="radar_parameters",
    )
    normalized_sub = cf2._normalize_subgroup(sub, cf2._SUBGROUPS["radar_parameters"])
    assert "radar_receiver_bandwidth" in normalized_sub.data_vars
    assert normalized_sub.attrs == {}


def test_cfradial2_helper_root_fixed_angle_precedence_warning():
    root = xr.Dataset()
    sweep = xr.Dataset(
        data_vars={
            "sweep_fixed_angle": ((), 1.0),
            "fixed_angle": ((), 2.0),
            "elevation": ("time", np.array([3.0], dtype="float32")),
            "azimuth": ("time", np.array([4.0], dtype="float32")),
            "time": ("time", np.array(["2025-01-01"], dtype="datetime64[ns]")),
        }
    )

    with pytest.warns(UserWarning, match="multiple fixed-angle candidates"):
        out = cf2._normalize_root_dataset(root, ["sweep_0"], [sweep], optional=True)

    assert out["sweep_fixed_angle"].item() == pytest.approx(1.0)


def test_cfradial2_helper_sweep_normalization_branches():
    ds = xr.Dataset(
        data_vars={
            "azimuth": ("azimuth", np.array([0.0, 1.0], dtype="float32")),
            "elevation": ("azimuth", np.array([2.0, 2.0], dtype="float32")),
            "fixed_angle": ((), 2.0),
            "DBZ": (("azimuth", "range"), np.ones((2, 2), dtype="float32")),
            "scalar_junk": ((), 99),
        },
        coords={"range": ("range", np.array([100.0, 200.0], dtype="float32"))},
        attrs={"sweep_mode": "rhi"},
    )

    out = cf2._normalize_sweep_dataset(ds, "sweep_3", first_dim="time", optional=False)
    assert "time" in out.dims
    assert out["sweep_mode"].item() == "rhi"
    assert out["sweep_fixed_angle"].item() == pytest.approx(2.0)
    assert "frequency" in out.coords
    assert "scalar_junk" not in out.data_vars

    no_range = xr.Dataset({"foo": ("x", np.array([1, 2]))})
    assert cf2._derive_range_attrs(no_range).identical(no_range)

    assert (
        cf2._infer_sweep_mode(xr.Dataset(attrs={"sweep_mode": "manual_ppi"}))
        == "manual_ppi"
    )
    assert cf2._infer_sweep_mode(xr.Dataset(attrs={"ScanType": 1})) == "rhi"


def test_open_cfradial2_optional_groups_and_missing_root_warning(temp_file):
    outfile = temp_file.with_suffix(".nc")

    root = xr.Dataset(attrs={"RadarName": "warn-radar"})
    sweep = xr.Dataset(
        data_vars={
            "time_us": (
                "ntime",
                np.array(
                    ["2025-05-01T00:00:00", "2025-05-01T00:00:01"],
                    dtype="datetime64[ns]",
                ),
            ),
            "range_m": ("nrange", np.array([50.0, 150.0], dtype="float32")),
            "azimuth_deg": ("ntime", np.array([0.0, 1.0], dtype="float32")),
            "elevation_deg": ("ntime", np.array([2.0, 2.0], dtype="float32")),
            "DBZ": (("ntime", "nrange"), np.ones((2, 2), dtype="float32")),
        }
    )
    radar_parameters = xr.Dataset(data_vars={"radar_rx_bandwidth": ((), 7.0)})
    tree = xr.DataTree.from_dict(
        {"/": root, "sweep_0": sweep, "radar_parameters": radar_parameters}
    )
    tree.to_netcdf(outfile, engine="netcdf4")

    with pytest.warns(
        UserWarning, match="could not fully normalize FM301 root variables"
    ):
        dtree = xd.io.open_cfradial2_datatree(
            outfile, engine="netcdf4", optional_groups=True
        )

    assert "radar_parameters" in dtree.children
    assert "radar_receiver_bandwidth" in dtree["radar_parameters"].data_vars

    with pytest.raises(ValueError, match="missing from file"):
        xd.io.open_cfradial2_datatree(outfile, engine="netcdf4", sweep="sweep_9")
