#!/usr/bin/env python
# Copyright (c) 2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for `io.backends.odim` module.

ported from wradlib
"""

from contextlib import nullcontext

import h5netcdf
import numpy as np
import pytest
from xarray import DataTree, open_dataset, open_mfdataset

from xradar.io.backends import odim, open_odim_datatree
from xradar.io.backends.common import _maybe_recover_surrogate


def create_startazA(nrays=360):
    arr = np.linspace(0, 360, 360, endpoint=False, dtype=np.float32)
    if nrays == 361:
        arr = np.insert(arr, 10, (arr[10] + arr[9]) / 2, axis=0)
    return arr


def create_stopazA(nrays=360):
    arr = np.linspace(1, 361, 360, endpoint=False, dtype=np.float32)
    # arr = np.arange(1, 361, 1, dtype=np.float32)
    arr[arr >= 360] -= 360
    if nrays == 361:
        arr = np.insert(arr, 10, (arr[10] + arr[9]) / 2, axis=0)
    return arr


def create_how(nrays=360, stopaz=False):
    how = dict(startazA=create_startazA(nrays))
    if stopaz:
        how.update(stopazA=create_stopazA(nrays))
    return how


@pytest.mark.parametrize("stopaz", [False, True])
def test_get_azimuth_how(stopaz):
    how = create_how(stopaz=stopaz)
    actual = odim._get_azimuth_how(how)
    wanted = np.arange(0.5, 360, 1.0)
    np.testing.assert_equal(actual, wanted)


@pytest.mark.parametrize("nrays", [180, 240, 360, 720])
def test_get_azimuth_where(nrays):
    where = dict(nrays=nrays)
    actual = odim._get_azimuth_where(where)
    udiff = np.unique(np.diff(actual))
    assert len(actual) == nrays
    assert len(udiff) == 1
    assert udiff[0] == 360.0 / nrays


@pytest.mark.parametrize(
    "ang",
    [("az_angle", "elevation"), ("az_angle", "elevation"), ("elangle", "azimuth")],
)
def test_get_fixed_dim_and_angle(ang):
    where = {ang[0]: 1.0}
    dim, angle = odim._get_fixed_dim_and_angle(where)
    assert dim == ang[1]
    assert angle == 1.0


def create_el_how(rhi):
    if rhi:
        return dict(startelA=1.0, stopelA=2.0)
    else:
        return dict(elangles=1.5)


@pytest.mark.parametrize("rhi", [True, False])
def test_get_elevation_how(rhi):
    how = create_el_how(rhi)
    el = odim._get_elevation_how(how)
    assert el == 1.5


def test_get_elevation_where():
    where = dict(nrays=360, elangle=0.5)
    actual = odim._get_elevation_where(where)
    udiff = np.unique(actual)
    assert len(actual) == 360
    assert len(udiff) == 1
    assert udiff[0] == 0.5
    assert actual.dtype == np.float32


def test_get_time_how():
    how = dict(startazT=np.array([10, 20, 30]), stopazT=np.array([20, 30, 40]))
    time = odim._get_time_how(how)
    np.testing.assert_array_equal(time, np.array([15.0, 25.0, 35.0]))


@pytest.mark.parametrize("a1gate", [(0, 946684800.0416666), (10, 946684829.2083472)])
@pytest.mark.parametrize("enddate", [True, False])
def test_get_time_what(a1gate, enddate):
    what = dict(
        startdate="20000101",
        starttime="000000",
    )
    if enddate:
        what.update(enddate="20000101", endtime="000030")
        a1g = a1gate[1]
    else:
        a1g = 946684800.0
    where = dict(nrays=360, a1gate=a1gate[0])
    if not enddate:
        check = pytest.warns(
            UserWarning, match="Equal ODIM `starttime` and `endtime` values"
        )
    else:
        check = nullcontext()
    with check:
        time = odim._get_time_what(what, where)
    assert time[0] == a1g
    assert len(time) == 360


@pytest.mark.parametrize("rscale", [100, 150, 300, 1000])
def test_get_range(rscale):
    where = dict(nbins=10, rstart=0, rscale=rscale)
    rng, cent_first, bin_range = odim._get_range(where)
    assert np.unique(np.diff(rng))[0] == rscale
    assert cent_first == rscale / 2
    assert bin_range == rscale


@pytest.mark.parametrize(
    "point",
    [
        ("start", np.datetime64("2000-01-01T00:00:00", "s")),
        ("end", np.datetime64("2000-01-01T00:00:30", "s")),
    ],
)
def test_get_time(point):
    what = dict(
        startdate="20000101", starttime="000000", enddate="20000101", endtime="000030"
    )
    time = odim._get_time(what, point=point[0])
    assert time == point[1]


def test_get_a1gate():
    where = dict(a1gate=20)
    assert odim._get_a1gate(where) == 20


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (None, {}),
        ("", {}),
        ("WMO:26232", {"WMO": "26232"}),
        (
            " WIGOS:0-233-2-26232 , WMO:26232 , NOD:eesur ",
            {"WIGOS": "0-233-2-26232", "WMO": "26232", "NOD": "eesur"},
        ),
        (
            "wmo:26232,plc:Surgavere",
            {"WMO": "26232", "PLC": "Surgavere"},
        ),
        (
            b"WIGOS:0-233-2-26232,WMO:26232,NOD:eesur",
            {"WIGOS": "0-233-2-26232", "WMO": "26232", "NOD": "eesur"},
        ),
        (
            "RAD:EE41:SUBSYSTEM",
            {"RAD": "EE41:SUBSYSTEM"},
        ),
        (
            "WMO:26232,invalid,missing_colon,PLC:Surgavere",
            {"WMO": "26232", "PLC": "Surgavere"},
        ),
        (
            "WMO:26232,PLC:,NOD:eesur, :ignored",
            {"WMO": "26232", "NOD": "eesur"},
        ),
        (
            "WMO:11111,WMO:26232",
            {"WMO": "26232"},
        ),
        (
            "WIGOS:0-233-2-26232,WMO:26232,RAD:EE41,PLC:S\udcc3\udcbcrgavere,NOD:eesur",
            {
                "WIGOS": "0-233-2-26232",
                "WMO": "26232",
                "RAD": "EE41",
                "PLC": "Sürgavere",
                "NOD": "eesur",
            },
        ),
    ],
)
def test_parse_odim_source_extensive(source, expected):
    assert odim._parse_odim_source(source) == expected


def test_parse_odim_source_handles_non_string_input():
    class DummySource:
        def __str__(self):
            return "WMO:26232,NOD:eesur"

    parsed = odim._parse_odim_source(DummySource())
    assert parsed == {"WMO": "26232", "NOD": "eesur"}


def test_parse_odim_source_surrogate_repair_unicodeerror_fallback():
    # U+DC80 maps to raw byte 0x80 with surrogateescape, which is invalid
    # as standalone UTF-8 and forces the repair decode to raise UnicodeError.
    source = "PLC:\udc80_station,WMO:26232"

    parsed = odim._parse_odim_source(source)

    # Parser should not crash and should keep original value when repair fails.
    assert parsed["PLC"] == "\udc80_station"
    assert parsed["WMO"] == "26232"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Surgavere", "Surgavere"),
        ("S\udcc3\udcbcrgavere", "Sürgavere"),
        ("\udc80_station", "\udc80_station"),
        (123, 123),
    ],
)
def test_maybe_recover_surrogate(value, expected):
    assert _maybe_recover_surrogate(value) == expected


def test_OdimH5NetCDFMetadata(odim_file):
    store = odim.OdimStore.open(odim_file, group="sweep_0")
    with pytest.warns(DeprecationWarning):
        assert store.substore[0].root.first_dim == "azimuth"


def test_odim_open_mfdataset_context_manager(odim_file):
    with open_mfdataset(
        [odim_file],
        engine="odim",
        concat_dim="volume_time",
        combine="nested",
        group="sweep_0",
    ) as ds:
        assert ds is not None
        # closer must exist while inside context
        assert callable(getattr(ds, "_close", None))


def test_odim_dataset_has_close(odim_file):
    ds = open_dataset(odim_file, engine="odim", group="sweep_0")
    assert callable(getattr(ds, "_close", None))
    ds.close()


@pytest.mark.parametrize(
    "fixture_name",
    ["odim_file", "odim_file2", "odim_file3", "odim_file4", "odim_file5"],
)
def test_odim_source_global_attributes(request, fixture_name):
    filename = request.getfixturevalue(fixture_name)
    print(f"Testing file: {filename}")

    # Build expected global attrs from raw /what/source.
    expected = {}
    with h5netcdf.File(filename, mode="r") as root:
        if "what" in root:
            source = root["what"].attrs.get("source", None)
            print(f"Raw ODIM source attribute: {source}")
            parsed = odim._parse_odim_source(source)
            for odim_key, global_attr in odim._ODIM_SOURCE_TO_GLOBAL_ATTRS.items():
                value = parsed.get(odim_key)
                if value is not None:
                    expected[global_attr] = value
            print(f"Parsed ODIM source attributes: {expected}")

    with open_dataset(filename, engine="odim", group="sweep_0") as ds:
        print(f"Dataset global attributes: {ds.attrs}")
        for global_attr, value in expected.items():
            assert ds.attrs.get(global_attr) == value


def test_open_odim_datatree(odim_file):
    # Define kwargs to pass into the function
    kwargs = {
        "sweep": [0, 1, 2],  # Specify sweeps to extract
        "first_dim": "auto",
        "reindex_angle": False,
        "fix_second_angle": False,
        "site_as_coords": True,
    }

    # Call the function with an ODIM file
    dtree = open_odim_datatree(odim_file, **kwargs)

    # Assertions to check DataTree structure
    assert isinstance(dtree, DataTree), "Expected a DataTree instance"
    subtree_paths = [n.path for n in dtree.subtree]
    assert "/" in subtree_paths, "Root group should be present in the DataTree"
    # optional_groups=False by default: metadata subgroups should NOT be present
    assert "radar_parameters" not in dtree.children
    assert "georeferencing_correction" not in dtree.children
    assert "radar_calibration" not in dtree.children

    # Check if the correct sweep groups are attached
    sweep_groups = [key for key in dtree.match("sweep_*")]
    assert len(sweep_groups) == 3, "Expected three sweep groups in the DataTree"
    sample_sweep = sweep_groups[0]

    # Check data variables in the sweep group
    assert (
        "DBZH" in dtree[sample_sweep].variables.keys()
    ), "Expected 'DBZH' variable in the sweep group"
    assert (
        "ZDR" in dtree[sample_sweep].variables.keys()
    ), "Expected 'ZDR' variable in the sweep group"
    assert dtree[sample_sweep]["DBZH"].shape == (
        360,
        1200,
    ), "Shape mismatch for 'DBZH' variable"
    # Station coords should be on root as coordinates, NOT on sweeps
    assert "latitude" in dtree.ds.coords
    assert "longitude" in dtree.ds.coords
    assert "altitude" in dtree.ds.coords
    assert "latitude" not in dtree.ds.data_vars

    # Validate attributes
    assert len(dtree.attrs) == 10
    assert (
        dtree.attrs["Conventions"] == "ODIM_H5/V2_2"
    ), "Instrument name should match expected value"


def test_open_odim_datatree_optional_groups(odim_file):
    """Test that optional_groups=True includes metadata subgroups."""
    dtree = open_odim_datatree(odim_file, optional_groups=True)
    assert "radar_parameters" in dtree.children
    assert "georeferencing_correction" in dtree.children
    assert "radar_calibration" in dtree.children


@pytest.mark.parametrize("layout", ["fmi", "odim"])
def test_open_odim_quality_legend(odim_file, tmp_path, layout):
    # quality groups with compound-dtype legend tables must not be merged as
    # ray data (#395). "odim": ODIM_H5 2.3/2.4 Section 6.2
    # {char[64] key; char[32] value}, "fmi": {int64 code; string class}
    import shutil

    import h5py

    path = tmp_path / "odim_quality_legend.h5"
    shutil.copy(odim_file, path)
    entries = [(60, "NONMET.BIOL.INSECT"), (72, "NONMET.CLUTTER.CCOR"), (246, "NOISE")]
    if layout == "fmi":
        legend_dtype = np.dtype([("code", ">i8"), ("class", h5py.string_dtype())])
        legend = np.array(entries, dtype=legend_dtype)
    else:
        legend_dtype = np.dtype([("key", "S64"), ("value", "S32")])
        legend = np.array(
            [(name.encode(), str(code).encode()) for code, name in entries],
            dtype=legend_dtype,
        )
    with h5py.File(path, "a") as f:
        shape = f["dataset1/data1/data"].shape
        n = 1
        while f"quality{n}" in f["dataset1"]:
            n += 1
        qual = f.create_group(f"dataset1/quality{n}")
        what = qual.create_group("what")
        what.attrs.update(
            quantity=np.bytes_("ECHO_CLASS"),
            gain=1.0,
            offset=0.0,
            nodata=255.0,
            undetect=0.0,
            legend=np.bytes_("72:NONMET.CLUTTER.CCOR,60:NONMET.BIOL.INSECT,246:NOISE"),
        )
        qual.create_dataset("data", data=np.full(shape, 72, dtype="uint8"))
        qual.create_dataset("legend", data=legend)

    dtree = open_odim_datatree(path, sweep=0)
    ds = dtree["sweep_0"].ds
    assert "legend" not in ds.variables
    assert ds.ECHO_CLASS.dims == ds.DBZH.dims
    flag_values = ds.ECHO_CLASS.attrs["flag_values"]
    np.testing.assert_array_equal(flag_values, [60, 72, 246])
    assert flag_values.dtype == np.dtype("int64")
    assert ds.ECHO_CLASS.attrs["flag_meanings"] == (
        "NONMET.BIOL.INSECT NONMET.CLUTTER.CCOR NOISE"
    )
    # the producer's what/legend string is kept as is
    assert ds.ECHO_CLASS.attrs["legend"] == (
        "72:NONMET.CLUTTER.CCOR,60:NONMET.BIOL.INSECT,246:NOISE"
    )
    assert "flag_values" not in ds.DBZH.attrs
    assert "flag_values" not in ds.CLASS.attrs
