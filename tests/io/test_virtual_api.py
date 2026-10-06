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
