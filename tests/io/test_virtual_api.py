#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""The public surface store builders rely on.

raw2zarr deletes its own ``parsers/`` package once xradar ships these, so
every name below is a contract: renaming or removing one breaks a
downstream builder. Requires the ``xradar[virtual]`` extra.
"""

import importlib

import numpy as np
import pytest

pytest.importorskip("zarr", minversion="3.1.6")  # zarr v3 only
pytest.importorskip("virtualizarr")

#: module -> public names a store builder imports from it
PUBLIC = {
    "xradar.io.virtual": (
        "IrisParser",
        "IrisSweepCodec",
        "azimuth_sort_order",
    ),
    "xradar.io.virtual.manifest": (
        "MOMENT_COORDINATES",
        "FM301_STRING_DEFAULTS",
        "moment_attrs",
        "variable_to_inline_manifest_array",
        "inline_variable",
        "inline_scalar",
        "inline_root",
        "fetch_bytes",
    ),
    "xradar.io.virtual.iris.format": (
        "NO_DATA_ZERO_TYPES",
        "RECORD_SIZE",
        "IngestHeader",
        "parse_ingest_header",
        "index_sweeps",
        "read_volume",
        "range_centers",
        "azimuth_midpoints",
        "sweep_words",
        "walk_sweep",
        "decode_sweep_moment",
    ),
    "xradar.io.backends.iris": ("iris_mapping",),
}


@pytest.mark.parametrize("module", sorted(PUBLIC))
def test_public_names_importable(module):
    mod = importlib.import_module(module)
    missing = [name for name in PUBLIC[module] if not hasattr(mod, name)]
    assert not missing, f"{module} lost public names {missing}"
    exported = getattr(mod, "__all__", None)
    if exported is not None and module != "xradar.io.backends.iris":
        unlisted = [n for n in PUBLIC[module] if n not in exported]
        assert not unlisted, f"{module}.__all__ misses {unlisted}"


def test_one_sort_rule_everywhere():
    """Builders and both codecs share ONE permutation function."""
    from xradar.io.virtual import azimuth_sort_order
    from xradar.io.virtual.iris import format as iris_format

    assert iris_format.azimuth_sort_order is azimuth_sort_order
    assert azimuth_sort_order([1.0, 0.0, 1.0, np.nan, 0.0]).tolist() == [1, 4, 0, 2, 3]


def _inline_values(arr):
    (data,) = arr.manifest._inlined.values()
    return data


def test_inline_root_is_the_eager_root():
    from xradar.io.virtual.manifest import inline_root

    times = [
        np.array([1_000.0, np.nan, 3_000.0]),  # NaN = a padded missing ray
        np.array([2_000.0, 61_500.0]),
    ]
    arrays, attrs = inline_root(
        times, np.float64(10.5), np.float64(-75.0), np.float64(120.0), {"x": 1}
    )
    assert sorted(arrays) == sorted(
        [
            "volume_number",
            "platform_type",
            "instrument_type",
            "time_coverage_start",
            "time_coverage_end",
            "latitude",
            "longitude",
            "altitude",
        ]
    )
    assert b"1970-01-01T00:00:01Z" in _inline_values(arrays["time_coverage_start"])
    assert b"1970-01-01T00:01:01Z" in _inline_values(arrays["time_coverage_end"])
    assert attrs["coordinates"] == "latitude longitude altitude"
    assert attrs["comment"] == "im/exported using xradar"
    assert attrs["x"] == 1
    lat = arrays["latitude"].metadata.attributes
    assert lat["standard_name"] == "latitude" and lat["units"] == "degrees_north"


def test_inline_root_needs_ray_times():
    from xradar.io.virtual.manifest import inline_root

    with pytest.raises(ValueError, match="no ray times"):
        inline_root([np.array([np.nan])], 0.0, 0.0, 0.0, {})


def test_moment_attrs_order_is_fixed():
    """Attr key order must not depend on the process (``model.moment_attrs``
    used to be a set)."""
    from xradar.io.virtual.manifest import moment_attrs

    assert list(moment_attrs("DBZH")) == ["units", "standard_name", "long_name"]
    assert moment_attrs("NOT_A_MOMENT") == {}


def test_json_safe_and_endianness_helpers():
    """Attrs reach zarr.json as plain JSON; inline chunks are little-endian."""
    from xradar.io.virtual.manifest import native_endian_bytes, to_json_safe

    value = {"a": np.float32(1.5), "b": [np.int64(2), (np.bool_(True),)]}
    assert to_json_safe(value) == {"a": 1.5, "b": [2, [True]]}
    assert to_json_safe(np.arange(3, dtype="uint8")) == [0, 1, 2]
    one_byte, endian = native_endian_bytes(np.arange(3, dtype="uint8"))
    assert endian is None and one_byte.dtype == np.uint8
    big = np.arange(3, dtype=">u2")
    little, endian = native_endian_bytes(big)
    assert endian == "little" and little.dtype.byteorder in ("<", "=")
    np.testing.assert_array_equal(little, big)


def test_inline_variables_round_trip_through_zarr():
    """Inline uint8 and datetime64 variables read back exactly; datetimes
    carry CF units so xarray decodes them."""
    import xarray as xr
    from virtualizarr.manifests import ManifestGroup, ManifestStore

    from xradar.io.virtual.manifest import inline_variable

    small = np.array([1, 2, 3], dtype="uint8")
    times = np.array(["2026-01-15T00:01:09", "2026-01-15T00:01:10"], "datetime64[ms]")
    group = ManifestGroup(
        arrays={
            "small": inline_variable(("x",), small, {}),
            "times": inline_variable(("t",), times, {}),
        },
        attributes={},
    )
    times_attrs = group.arrays["times"].metadata.attributes
    assert times_attrs["units"] == "ms since 1970-01-01T00:00:00"
    assert times_attrs["calendar"] == "proleptic_gregorian"
    assert group.arrays["small"].metadata.to_dict()["codecs"][0] == {"name": "bytes"}
    ds = xr.open_dataset(
        ManifestStore(group=group), engine="zarr", consolidated=False, zarr_format=3
    )
    np.testing.assert_array_equal(ds["small"].values, small)
    np.testing.assert_array_equal(ds["times"].values, times.astype("datetime64[ns]"))
