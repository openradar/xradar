#!/usr/bin/env python
# Copyright (c) 2023-2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for `io.backends.rainbow` module.

Ported from wradlib.
"""

import datetime
import warnings
import zlib
from io import BytesIO

import numpy as np
import pytest
import xmltodict
from xarray import DataTree, open_dataset, open_mfdataset

import xradar as xd
from xradar.io.backends import open_rainbow_datatree, rainbow


def test_find_key():
    indict = {
        "A": {
            "AA": {"AAA": 0, "X": 1},
            "AB": {"ABA": 2, "X": 3},
            "AC": {"ACA": 4, "X": 5},
            "AD": [{"ADA": 4, "X": 2}],
        }
    }
    outdict = [
        {"X": 1, "AAA": 0},
        {"ABA": 2, "X": 3},
        {"X": 5, "ACA": 4},
        {"ADA": 4, "X": 2},
    ]

    assert list(rainbow.find_key("X", indict)) == outdict
    assert list(rainbow.find_key("Y", indict)) == []


def test_decompress():
    dstring = b"very special compressed string"
    cstring = zlib.compress(dstring)
    assert rainbow.decompress(cstring) == dstring


def test_get_rb_data_layout():
    assert rainbow.get_rb_data_layout(8) == (1, ">u1")
    assert rainbow.get_rb_data_layout(16) == (2, ">u2")
    assert rainbow.get_rb_data_layout(32) == (4, ">u4")
    with pytest.raises(ValueError):
        rainbow.get_rb_data_layout(128)


def test_get_rb_data_layout_big(monkeypatch):
    monkeypatch.setattr("sys.byteorder", "big")
    assert rainbow.get_rb_data_layout(8) == (1, "<u1")
    assert rainbow.get_rb_data_layout(16) == (2, "<u2")
    assert rainbow.get_rb_data_layout(32) == (4, "<u4")


def test_get_rb_data_attribute():
    data = xmltodict.parse(
        '<slicedata time="13:30:05" date="2013-04-26">'
        '#<rayinfo refid="startangle" blobid="0" '
        'rays="361" depth="16"/> '
        '#<rawdata blobid="1" rays="361" type="dBuZ" '
        'bins="400" min="-31.5" max="95.5" '
        'depth="8"/> #</slicedata>'
    )
    data = list(rainbow.find_key("@blobid", data))
    assert rainbow.get_rb_data_attribute(data[0], "blobid") == 0
    assert rainbow.get_rb_data_attribute(data[1], "blobid") == 1
    assert rainbow.get_rb_data_attribute(data[0], "rays") == 361
    assert rainbow.get_rb_data_attribute(data[1], "rays") == 361
    assert rainbow.get_rb_data_attribute(data[1], "bins") == 400
    with pytest.raises(KeyError):
        rainbow.get_rb_data_attribute(data[0], "Nonsense")
    assert rainbow.get_rb_data_attribute(data[0], "depth") == 16


def test_get_rb_blob_attribute():
    xmldict = xmltodict.parse('<BLOB blobid="0" size="737" compression="qt"></BLOB>')
    assert rainbow.get_rb_blob_attribute(xmldict, "compression") == "qt"
    assert rainbow.get_rb_blob_attribute(xmldict, "size") == "737"
    assert rainbow.get_rb_blob_attribute(xmldict, "blobid") == "0"
    with pytest.raises(KeyError):
        rainbow.get_rb_blob_attribute(xmldict, "Nonsense")


def test_get_rb_data_shape():
    data = xmltodict.parse(
        '<slicedata time="13:30:05" date="2013-04-26">'
        '#<rayinfo refid="startangle" blobid="0" '
        'rays="361" depth="16"/> #<rawdata blobid="1" '
        'rays="361" type="dBuZ" bins="400" '
        'min="-31.5" max="95.5" depth="8"/> #<flagmap '
        'blobid="2" rows="800" type="dBuZ" '
        'columns="400" min="-31.5" max="95.5" '
        'depth="6"/> #<defect blobid="3" type="dBuZ" '
        'columns="400" min="-31.5" max="95.5" '
        'depth="6"/> #<rawdata2 '
        'blobid="4" rows="800" type="dBuZ" '
        'columns="400" min="-31.5" max="95.5" '
        'depth="8"/> #</slicedata>'
    )
    data = list(rainbow.find_key("@blobid", data))
    assert rainbow.get_rb_data_shape(data[0]) == 361
    assert rainbow.get_rb_data_shape(data[1]) == (361, 400)
    assert rainbow.get_rb_data_shape(data[2]) == (800, 400, 6)
    assert rainbow.get_rb_data_shape(data[4]) == (800, 400)
    with pytest.raises(KeyError):
        rainbow.get_rb_data_shape(data[3])


def test_map_rb_data():
    indata = b"0123456789"
    outdata8 = np.array([48, 49, 50, 51, 52, 53, 54, 55, 56, 57], dtype=np.uint8)
    outdata16 = np.array([12337, 12851, 13365, 13879, 14393], dtype=np.uint16)
    outdata32 = np.array([808530483, 875902519], dtype=np.uint32)
    np.testing.assert_allclose(rainbow.map_rb_data(indata, 8), outdata8)
    np.testing.assert_allclose(rainbow.map_rb_data(indata, 16), outdata16)
    np.testing.assert_allclose(rainbow.map_rb_data(indata, 32), outdata32)
    flagdata = b"1"
    np.testing.assert_allclose(
        rainbow.map_rb_data(flagdata, 1, 8), [0, 0, 1, 1, 0, 0, 0, 1]
    )
    # added test for truncation
    np.testing.assert_allclose(rainbow.map_rb_data(flagdata, 1, 6), [0, 0, 1, 1, 0, 0])


def test_get_rb_blob_data():
    datastring = b'<BLOB blobid="0" size="737" compression="qt"></BLOB>'
    with pytest.raises(EOFError):
        rainbow.get_rb_blob_data(datastring, 1)


def test_get_rb_header():
    rb_header = (
        b'<volume version="5.34.16" '
        b'datetime="2013-07-03T08:33:55"'
        b' type="azi" owner="RainAnalyzer"> '
        b'<scan name="analyzer.azi" time="08:34:00" '
        b'date="2013-07-03">'
    )

    buf = BytesIO(rb_header)
    with pytest.raises(IOError):
        rainbow.get_rb_header(buf)


def test_get_rb_header_from_file(rainbow_file):
    with open(rainbow_file, "rb") as rb_fh:
        rb_header = rainbow.get_rb_header(rb_fh)
        assert rb_header["volume"]["@version"] == "5.36.5"


def test_rainbow_file_meta(rainbow_file):
    with rainbow.RainbowFile(rainbow_file, loaddata=False) as rbdict:
        assert rbdict.version == "5.36.5"
        assert rbdict.header
        assert rbdict.site_coords == (6.379967, 50.856633, 116.7)
        assert rbdict.first_dimension == "azimuth"
        assert rbdict.history is None
        assert rbdict.pargroup
        assert rbdict.sensorinfo == dict(
            [
                ("@id", "143DEX"),
                ("@name", "Gematronik"),
                ("@type", "gdrx"),
                ("alt", "116.700000"),
                ("beamwidth", "1.326"),
                ("lat", "50.856633"),
                ("lon", "6.379967"),
                ("wavelen", "0.0319"),
            ]
        )
        assert rbdict.datetime == datetime.datetime(2013, 5, 10, 0, 3, 17)
        assert rbdict.type == "vol"


def test_rainbow_file_data(rainbow_file):
    with rainbow.RainbowFile(rainbow_file, loaddata=True) as rbdict:
        sdata = rbdict.slices[0]["slicedata"]
        print(sdata["rayinfo"]["data"][0])
        np.testing.assert_equal(sdata["rayinfo"]["data"][0], np.array(47.0159912109375))
        np.testing.assert_equal(sdata["rawdata"]["data"][0, 0], np.array(113))
        assert sdata["rawdata"]["data"].shape == (361, 400)


def test_rainbow_open_mfdataset_context_manager(rainbow_file):
    with open_mfdataset(
        [rainbow_file],
        engine="rainbow",
        concat_dim="volume_time",
        combine="nested",
        group="sweep_0",
    ) as ds:
        assert ds is not None
        # closer must exist while inside context
        assert callable(getattr(ds, "_close", None))


def test_rainbow_dataset_has_close(rainbow_file):
    ds = open_dataset(rainbow_file, engine="rainbow", group="sweep_0")
    assert callable(getattr(ds, "_close", None))
    ds.close()


def test_open_rainbow_datatree(rainbow_file):
    # Define kwargs to pass into the function
    kwargs = {
        "sweep": [0, 1],  # Specify sweeps to extract
        "first_dim": "auto",
        "reindex_angle": False,
        "site_as_coords": True,
    }

    # Call the function with the Rainbow file
    dtree = open_rainbow_datatree(rainbow_file, **kwargs)

    # Assertions to check DataTree structure
    assert isinstance(dtree, DataTree), "Expected a DataTree instance"
    subtree_paths = [n.path for n in dtree.subtree]
    assert "/" in subtree_paths, "Root group should be present in the DataTree"
    # optional_groups=False by default: metadata subgroups should NOT be present
    assert "radar_parameters" not in dtree.children
    assert "georeferencing_correction" not in dtree.children
    assert "radar_calibration" not in dtree.children

    # Check if the specified sweep groups are attached
    sweep_groups = [key for key in dtree.match("sweep_*")]
    assert len(sweep_groups) == 2, "Expected two sweep groups in the DataTree"
    sample_sweep = sweep_groups[0]

    # Verify the presence of specific data variables
    assert (
        "DBZH" in dtree[sample_sweep].variables.keys()
    ), "Expected 'DBZH' variable in the sweep group"
    assert dtree[sample_sweep]["DBZH"].shape == (
        361,
        400,
    ), "Shape mismatch for 'DBZH' variable"

    # Station coords should be on root as coordinates, NOT on sweeps
    assert "latitude" in dtree.ds.coords
    assert "longitude" in dtree.ds.coords
    assert "altitude" in dtree.ds.coords
    assert "latitude" not in dtree.ds.data_vars

    # Validate attributes
    assert len(dtree.attrs) == 9


def test_open_rainbow_datatree_optional_groups(rainbow_file):
    """Test that optional_groups=True includes metadata subgroups."""
    dtree = open_rainbow_datatree(rainbow_file, optional_groups=True)
    assert "radar_parameters" in dtree.children
    assert "georeferencing_correction" in dtree.children
    assert "radar_calibration" in dtree.children


@pytest.mark.parametrize(
    ("tag", "start"),
    [
        (None, 0.0),
        (b"<start_range>2</start_range>", 2000.0),
        # older name, padded to keep the length of the header
        (b"<startrange>2  </startrange>", 2000.0),
    ],
)
def test_rainbow_start_range(rainbow_file, tmp_path, tag, start):
    # range is read from `start_range` (start of the first bin, km), #272
    path = tmp_path / "start_range.vol"
    content = open(rainbow_file, "rb").read()
    if tag is not None:
        content = content.replace(b"<start_range>0</start_range>", tag)
    path.write_bytes(content)
    with open_dataset(path, engine="rainbow", group="sweep_0") as ds:
        # rangestep 0.25 km, bin centers
        expected = start + 250.0 * (np.arange(ds.sizes["range"]) + 0.5)
        np.testing.assert_allclose(ds.range.values, expected)
        assert ds.range.attrs["meters_to_center_of_first_gate"] == start + 125.0
        assert ds.sizes["range"] == 400


def test_rainbow_pathlib(rainbow_file):
    import pathlib

    with open_dataset(
        pathlib.Path(rainbow_file), engine="rainbow", group="sweep_0"
    ) as ds:
        assert ds.sizes["range"] == 400


@pytest.mark.parametrize(
    "fixture, sweep, expected",
    [
        # Rainbow 5.36: beam width from sensorinfo, radar constants per pulse
        # width ("77.20 74.09 71.03" with pw_index 1), nothing else
        (
            "rainbow_file",
            0,
            dict(
                radar_beam_width_h=1.326,
                radar_beam_width_v=1.326,
                radar_constant_h=74.09,
                radar_constant_v=74.26,
            ),
        ),
        # Rainbow 5.59, slices 0-5 (PRF 500/600 Hz)
        (
            "rainbow_file2",
            0,
            dict(
                radar_beam_width_h=0.97,
                radar_beam_width_v=0.96,
                radar_antenna_gain_h=45.31,
                radar_antenna_gain_v=45.19,
                radar_receiver_bandwidth=600e3,
                radar_constant_h=66.467,
                radar_constant_v=66.686,
                xmit_power_h=83.35920,  # 216.7286 kW in dBm
                xmit_power_v=83.41547,  # 219.5568 kW
            ),
        ),
        # slices 6-10 (PRF 1200 Hz): +3.01 dB and twice the bandwidth, as the
        # pulse width is halved
        (
            "rainbow_file2",
            6,
            dict(
                radar_beam_width_h=0.97,
                radar_beam_width_v=0.96,
                radar_antenna_gain_h=45.31,
                radar_antenna_gain_v=45.19,
                radar_receiver_bandwidth=1.2e6,
                radar_constant_h=69.477,
                radar_constant_v=69.696,
                xmit_power_h=83.35920,
                xmit_power_v=83.41547,
            ),
        ),
    ],
    ids=["5.36", "5.59-slice0", "5.59-slice6"],
)
def test_rainbow_slice_metadata(fixture, sweep, expected, request, closing):
    filename = request.getfixturevalue(fixture)
    dtree = closing(
        xd.open_datatree(filename, engine="rainbow", sweep=sweep, optional_groups=True)
    )
    found = {
        **{k: float(v) for k, v in dtree["radar_parameters"].ds.data_vars.items()},
        **{k: float(v) for k, v in dtree["radar_calibration"].ds.data_vars.items()},
    }
    assert set(found) == set(expected)
    for name, value in expected.items():
        np.testing.assert_allclose(found[name], value, rtol=1e-6, err_msg=name)
    # the values are no attributes of the root
    assert not set(expected) & set(dtree.attrs)


def test_rainbow_slice_metadata_differs_between_slices(rainbow_file2, closing):
    # the calibration changes with the pulse width: values which differ between
    # the selected slices are left out of the groups, with a warning which lists
    # them
    with pytest.warns(UserWarning, match="values differ between the sweeps") as record:
        dtree = closing(
            xd.open_datatree(
                rainbow_file2, engine="rainbow", sweep=[6, 0], optional_groups=True
            )
        )
    messages = " ".join(str(w.message) for w in record)
    assert "radar_receiver_bandwidth" in messages
    assert "radar_constant_h, radar_constant_v" in messages
    assert "radar_constant_h" not in dtree["radar_calibration"].ds
    assert "radar_receiver_bandwidth" not in dtree["radar_parameters"].ds
    # values which agree are there
    np.testing.assert_allclose(dtree["radar_parameters"].ds.radar_beam_width_h, 0.97)
    # slices which agree do not warn
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        agree = closing(
            xd.open_datatree(
                rainbow_file2, engine="rainbow", sweep=[0, 1], optional_groups=True
            )
        )
    assert "radar_constant_h" in agree["radar_calibration"].ds
    # the values of one slice
    one = closing(
        xd.open_datatree(rainbow_file2, engine="rainbow", sweep=6, optional_groups=True)
    )
    np.testing.assert_allclose(one["radar_calibration"].ds.radar_constant_h, 69.477)
    np.testing.assert_allclose(
        one["radar_parameters"].ds.radar_receiver_bandwidth, 1.2e6
    )


@pytest.mark.parametrize(
    ("value", "index", "expected"),
    [
        ("66.467", 0, 66.467),
        ("77.20 74.09 71.03", 2, 71.03),
        # no value for that pulse width: none, not the last one
        ("66.467", 1, None),
        ("77.20 74.09 71.03", 5, None),
        (None, 0, None),
        ("n/a", 0, None),
        ("", 0, None),
    ],
)
def test_rainbow_get_float(value, index, expected):
    assert rainbow._get_float(value, index) == expected


class _StubStore(rainbow.RainbowStore):
    """Store on a given header, without a file."""

    def __init__(self, slc, pargroup=None, sensorinfo=None):
        self._group = 0
        scan = {"slice": [slc]}
        if pargroup is not None:
            scan["pargroup"] = pargroup
        root = object.__new__(rainbow.RainbowFile)
        root._fp = None  # looked up when the file is closed
        root._header = {"scan": scan}
        if sensorinfo is not None:
            root._header["sensorinfo"] = sensorinfo
        self._stub = root

    @property
    def root(self):
        return self._stub


@pytest.mark.parametrize(
    ("slc", "pargroup", "sensorinfo", "expected"),
    [
        # no pargroup, the slice has the values; 0 is "not set"
        (
            {"spbhorbeam": "0.9", "spbverbeam": "0", "rxbandwidth": "0"},
            None,
            None,
            {"radar_beam_width_h": 0.9},
        ),
        # a slice key without value falls back on the pargroup
        (
            {"spbantgain": None},
            {"spbantgain": "45.3"},
            None,
            {"radar_antenna_gain_h": 45.3},
        ),
        # a beam width which is missing or 0 falls back on the sensorinfo
        (
            {},
            None,
            {"beamwidth": "1.3"},
            {"radar_beam_width_h": 1.3, "radar_beam_width_v": 1.3},
        ),
        (
            {"spbhorbeam": "0.9", "spbverbeam": "0"},
            None,
            {"@beamwidth": "1.3"},
            {"radar_beam_width_h": 0.9, "radar_beam_width_v": 1.3},
        ),
        # inf and nan are no values
        (
            {"spbhorbeam": "inf", "rxbandwidth": "nan", "spbantgain": "inf"},
            None,
            None,
            {},
        ),
    ],
    ids=["slice", "pargroup", "sensorinfo", "sensorinfo-at", "not-finite"],
)
def test_rainbow_store_radar_parameters_edge_cases(slc, pargroup, sensorinfo, expected):
    store = _StubStore(slc, pargroup=pargroup, sensorinfo=sensorinfo)
    assert dict(store.get_radar_parameters()) == expected


@pytest.mark.parametrize(
    ("slc", "expected"),
    [
        # not positive or not finite transmitter power: no value
        ({"gdrx5txpowkwh": "-5", "gdrx5txpowkwv": "0"}, {}),
        ({"gdrx5txpowkwh": "inf", "gdrx5txpowkwv": "nan"}, {}),
        # kW as dBm
        ({"gdrx5txpowkwh": "1"}, {"xmit_power_h": 60.0}),
        # the radar constant of the pulse width, none if there is no such value
        (
            {"rspdphradconst": "77.2 74.1", "pw_index": "1"},
            {"radar_constant_h": 74.1},
        ),
        ({"rspdphradconst": "77.2", "pw_index": "1"}, {}),
        # no pulse width index: only a constant with one value is used
        ({"rspdphradconst": "77.2"}, {"radar_constant_h": 77.2}),
        ({"rspdphradconst": "77.2 74.1"}, {}),
        # an index which is no non-negative whole number: no radar constant,
        # not the one of another pulse width (-1 is the last one of a list)
        ({"rspdphradconst": "77.2 74.1", "pw_index": "-1"}, {}),
        ({"rspdphradconst": "77.2 74.1", "pw_index": "nan"}, {}),
        ({"rspdphradconst": "77.2 74.1", "pw_index": "inf"}, {}),
        ({"rspdphradconst": "77.2 74.1", "pw_index": "0.5"}, {}),
        ({"rspdphradconst": "77.2 74.1", "pw_index": "x"}, {}),
    ],
    ids=[
        "tx-not-positive",
        "tx-not-finite",
        "tx-dbm",
        "pulse-width",
        "no-such-pulse-width",
        "no-index-one-value",
        "no-index-several-values",
        "negative-index",
        "nan-index",
        "inf-index",
        "fraction-index",
        "text-index",
    ],
)
def test_rainbow_store_calibration_parameters_edge_cases(slc, expected):
    store = _StubStore(slc)
    found = dict(store.get_calibration_parameters())
    assert found.keys() == expected.keys()
    for name, value in expected.items():
        np.testing.assert_allclose(found[name], value, rtol=1e-6)


def test_rainbow_store_empty_pargroup():
    # xmltodict gives None for an empty <pargroup/>
    store = _StubStore({"spbhorbeam": "0.9"})
    store.root._header["scan"]["pargroup"] = None
    assert dict(store.get_radar_parameters()) == {"radar_beam_width_h": 0.9}


def test_rainbow_store_slice_value_before_pargroup_value():
    store = _StubStore({"spbhorbeam": "0.9"}, pargroup={"spbhorbeam": "1.4"})
    assert dict(store.get_radar_parameters()) == {"radar_beam_width_h": 0.9}
    store = _StubStore({}, pargroup={"spbhorbeam": "1.4"})
    assert dict(store.get_radar_parameters()) == {"radar_beam_width_h": 1.4}
