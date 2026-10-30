#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the FM301-2022 attributes of the root variables
``time_coverage_start`` and ``time_coverage_end``.

FM301 requires both root variables to be strings carrying the string
attributes ``units`` ("seconds since YYYY-MM-DDThh:mm:ssZ"), ``calendar``
(a CF calendar name) and ``standard_name`` (``"time"``).
"""

import re
from datetime import datetime

import numpy as np
import pytest

import xradar as xd

UNITS_PREFIX = "seconds since "
TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
CALENDARS = {
    "gregorian",
    "standard",
    "proleptic_gregorian",
    "noleap",
    "365_day",
    "all_leap",
    "366_day",
    "360_day",
    "julian",
    "none",
}

# One backend per way the root group is built:
# - shared path (_get_required_root_dataset -> _assign_root): odim, gamic,
#   furuno, rainbow, iris
# - direct _assign_root callers: nexradlevel2, uf
# - own root implementation: cfradial1, cfradial2
ENGINES = [
    pytest.param(("odim", "odim_file"), id="odim"),
    pytest.param(("gamic", "gamic_file"), id="gamic"),
    pytest.param(("furuno", "furuno_scn_file"), id="furuno"),
    pytest.param(("rainbow", "rainbow_file"), id="rainbow"),
    pytest.param(("iris", "iris0_file"), id="iris"),
    pytest.param(("nexradlevel2", "nexradlevel2_file"), id="nexradlevel2"),
    pytest.param(("uf", "uf_file_1"), id="uf"),
    pytest.param(("cfradial1", "cfradial1_file"), id="cfradial1"),
    pytest.param(("cfradial2", "cfradial2_file"), id="cfradial2"),
]

# module-wide expected-failure marker
# Known gap: the attributes are lost / never set on the root variables.
# strict=True turns these into failures as soon as the gap is fixed.
pytestmark = pytest.mark.xfail(
    reason="FM301 attributes of time_coverage_* missing on root variables",
    strict=True,
)


# Return the root dataset and close its owning DataTree after each test.
@pytest.fixture(params=ENGINES)
@pytest.mark.file
def root_ds(request):
    engine, fixture_name = request.param
    filepath = request.getfixturevalue(fixture_name)
    dtree = xd.open_datatree(filepath, engine=engine)
    try:
        yield dtree.to_dataset(inherit=False)
    finally:
        dtree.close()


# check the FM301 time coverage attributes on an opened root dataset
@pytest.mark.parametrize("name", ["time_coverage_start", "time_coverage_end"])
def test_time_coverage_fm301_attrs(root_ds, name):
    var = root_ds[name]

    # the variable must be scalar (0-dimensional)
    assert var.ndim == 0

    # the variable must be a double (float64): seconds since the reference time
    assert var.dtype == np.float64, f"{name} must be double, got {var.dtype}"

    # check if the required FM301 attributes are present and correctly typed
    attrs = var.attrs
    for key in ("units", "calendar", "standard_name"):
        assert key in attrs, f"{name} is missing FM301 attribute {key!r}"
        assert isinstance(attrs[key], str), f"{name}:{key} must be a string"

    # check the format of the units attribute, which should start with the UNITS_PREFIX and contain a valid timestamp
    assert attrs["units"].startswith(UNITS_PREFIX), attrs["units"]
    timestamp = attrs["units"][len(UNITS_PREFIX) :]
    assert TIMESTAMP_PATTERN.fullmatch(timestamp), attrs["units"]
    try:
        datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        pytest.fail(f"{name} units contain an invalid timestamp: {timestamp!r}")
    # check the calendar and standard_name attributes
    assert attrs["calendar"] in CALENDARS
    assert attrs["standard_name"] == "time"
