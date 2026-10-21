#!/usr/bin/env python
# Copyright (c) 2023-2025, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for `io.backends.iris` module.

Ported from wradlib.
"""

import copy
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from xarray import DataTree, open_dataset, open_mfdataset

import xradar as xd
from xradar.io.backends import iris, open_iris_datatree
from xradar.util import _get_data_file


def test_open_iris(iris0_file, file_or_filelike):
    with _get_data_file(iris0_file, file_or_filelike) as sigmetfile:
        data = iris.IrisRawFile(sigmetfile, loaddata=False)
    assert isinstance(data.rh, iris.IrisRecord)
    assert isinstance(data.fh, (np.memmap, np.ndarray))
    with _get_data_file(iris0_file, file_or_filelike) as sigmetfile:
        data = iris.IrisRawFile(sigmetfile, loaddata=True)
    assert data._record_number == 511
    assert data.filepos == 3139584


def test_IrisRecord(iris0_file, file_or_filelike):
    with _get_data_file(iris0_file, file_or_filelike) as sigmetfile:
        data = iris.IrisRecordFile(sigmetfile, loaddata=False)
    # reset record after init
    data.init_record(1)
    assert isinstance(data.rh, iris.IrisRecord)
    assert data.rh.pos == 0
    assert data.rh.recpos == 0
    assert data.rh.recnum == 1
    rlist = [23, 0, 4, 0, 20, 19, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    np.testing.assert_array_equal(data.rh.read(10, 2), rlist)
    assert data.rh.pos == 20
    assert data.rh.recpos == 10
    data.rh.pos -= 20
    np.testing.assert_array_equal(data.rh.read(20, 1), rlist)
    data.rh.recpos -= 10
    np.testing.assert_array_equal(data.rh.read(5, 4), rlist)


def test_decode_bin_angle():
    assert iris.decode_bin_angle(20000, 2) == 109.86328125
    assert iris.decode_bin_angle(2000000000, 4) == 167.63806343078613


def test_decode_array():
    data = np.arange(0, 11)
    np.testing.assert_array_equal(
        iris.decode_array(data),
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    np.testing.assert_array_equal(
        iris.decode_array(data, offset=1.0),
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
    )
    np.testing.assert_array_equal(
        iris.decode_array(data, scale=0.5),
        [0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
    )
    np.testing.assert_array_equal(
        iris.decode_array(data, offset=1.0, scale=0.5),
        [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0],
    )
    np.testing.assert_array_equal(
        iris.decode_array(data, offset=1.0, scale=0.5, offset2=-2.0),
        [0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
    )
    data = np.array(
        [0, 1, 255, 1000, 9096, 22634, 34922, 50000, 65534], dtype=np.uint16
    )
    np.testing.assert_array_equal(
        iris.decode_array(data, scale=1000, tofloat=True),
        [0.0, 0.001, 0.255, 1.0, 10.0, 100.0, 800.0, 10125.312, 134184.96],
    )


def test_decode_velc():
    """4.4.42: raw 128 is zero velocity, one step of 75 / 127 m/s on both
    sides, raw 0 is no data (#477)."""
    decoded = _decode_raw(_ENTRIES["DB_VELC"], [0, 1, 2, 128, 129, 254, 255])
    np.testing.assert_allclose(
        decoded.filled(),
        [np.nan, -75.0, -74.409449, 0.0, 0.590551, 74.409449, 75.0],
        atol=1e-6,
    )
    assert decoded[3] == 0.0  # exactly zero, not just within atol


def test_decode_kdp():
    np.testing.assert_array_almost_equal(
        iris.decode_kdp(
            np.array(
                [
                    0,
                    1,
                    2,
                    127,
                    -128,  # 128 uint8
                    -127,  # 129 uint8
                    -126,  # 130 uint8
                    -2,  # 254 uint8
                    -1,  # 255 uint8
                ],
                dtype="int8",
            ),
            wavelength=10.0,
        ),
        [np.nan, -15.0, -14.257469, -0.025, 0.0, 0.025, 0.026302, 14.257469, np.nan],
    )


def test_decode_phidp():
    np.testing.assert_array_almost_equal(
        iris.decode_phidp(np.arange(0, 10, dtype="uint8"), scale=254.0, offset=-1),
        [
            -0.70866142,
            0.0,
            0.70866142,
            1.41732283,
            2.12598425,
            2.83464567,
            3.54330709,
            4.2519685,
            4.96062992,
            5.66929134,
        ],
    )


def test_decode_phidp2():
    np.testing.assert_array_almost_equal(
        iris.decode_phidp2(np.arange(0, 10, dtype="uint16"), scale=65534.0, offset=-1),
        [
            -0.00549333,
            0.0,
            0.00549333,
            0.01098666,
            0.01648,
            0.02197333,
            0.02746666,
            0.03295999,
            0.03845332,
            0.04394665,
        ],
    )


def test_decode_sqi():
    np.testing.assert_array_almost_equal(
        iris.decode_sqi(np.arange(0, 10, dtype="uint8"), scale=253.0, offset=-1),
        [
            np.nan,
            0.0,
            0.06286946,
            0.08891084,
            0.1088931,
            0.12573892,
            0.14058039,
            0.1539981,
            0.16633696,
            0.17782169,
        ],
    )


def test_decode_rainrate2():
    vals = np.array(
        [0, 1, 2, 255, 1000, 9096, 22634, 34922, 50000, 65534, 65535],
        dtype="uint16",
    )
    prod = iris.SIGMET_DATA_TYPES[13]
    np.testing.assert_array_almost_equal(
        iris.decode_array(vals.copy(), **prod["fkw"]),
        [
            -1.00000000e-04,
            0.00000000e00,
            1.00000000e-04,
            2.54000000e-02,
            9.99000000e-02,
            9.99900000e-01,
            9.99990000e00,
            7.99999000e01,
            1.01253110e03,
            1.34184959e04,
            1.34201343e04,
        ],
    )


def test_decode_time():
    timestring = b"\xd1\x9a\x00\x000\t\xdd\x07\x0b\x00\x19\x00"
    assert (
        iris.decode_time(timestring).isoformat() == "2013-11-25T11:00:33.304000+00:00"
    )


def test_decode_string():
    assert iris.decode_string(b"EEST\x00\x00\x00\x00") == "EEST"


def test__get_fmt_string():
    fmt = "<12sHHi12s12s12s6s12s12sHiiiiiiiiii2sH12sHB1shhiihh80s16s12s48s"
    assert iris._get_fmt_string(iris.PRODUCT_CONFIGURATION) == fmt


def test_read_from_record(iris0_file, file_or_filelike):
    """Test reading a specified number of words from a record."""
    with _get_data_file(iris0_file, file_or_filelike) as sigmetfile:
        data = iris.IrisRecordFile(sigmetfile, loaddata=True)
        data.init_record(0)  # Start from the first record
        record_data = data.read_from_record(10, dtype="int16")
        assert len(record_data) == 10
        assert isinstance(record_data, np.ndarray)


def test_decode_data(iris0_file, file_or_filelike):
    """Test decoding of data with provided product function."""

    # Sample data to decode
    data = np.array([0, 2, 3, 128, 255], dtype="int16")
    # Sample product dict with decoding function and parameters
    prod = {
        "func": iris.decode_vel,
        "dtype": "int16",
        "fkw": {"scale": 0.5, "offset": -1},
    }

    # Open the file as per the testing framework
    with _get_data_file(iris0_file, file_or_filelike) as sigmetfile:
        iris_file = iris.IrisRawFile(sigmetfile, loaddata=False)

        # Decode data using the provided product function
        decoded_data = iris_file.decode_data(data, prod)

    # Check that the decoded data is as expected
    assert isinstance(decoded_data, np.ndarray), "Decoded data should be a numpy array"
    assert decoded_data.dtype in [
        np.float32,
        np.float64,
    ], "Decoded data should have float32 or float64 type"

    # Expected decoded values
    expected_data = [-13.325, 13.325, 26.65, 1692.275, 3384.55]
    np.testing.assert_array_almost_equal(decoded_data, expected_data, decimal=2)


def test_get_sweep(iris0_file, file_or_filelike):
    """Test retrieval of sweep data for specified moments."""

    # Select the sweep number and moments to retrieve
    sweep_number = 1
    moments = ["DB_DBZ", "DB_VEL"]

    # Open the file and load data
    with _get_data_file(iris0_file, file_or_filelike) as sigmetfile:
        iris_file = iris.IrisRawFile(sigmetfile, loaddata=True)

        # Use get_sweep to retrieve data for the selected sweep and moments
        iris_file.get_sweep(sweep_number, moments)
        sweep_data = iris_file.data[sweep_number]["sweep_data"]

    # Verify that sweep_data structure is populated with the selected moments
    for moment in moments:
        assert moment in sweep_data, f"{moment} should be in sweep_data"
        moment_data = sweep_data[moment]
        assert moment_data.shape == (360, 664), f"{moment} data shape mismatch"

        # raw 0 is "no data" for both moments: masked, never -32 dBZ (#465)
        assert isinstance(
            moment_data, np.ma.MaskedArray
        ), f"{moment} should be a masked array"
        assert moment_data.mask.sum() > 0, f"{moment} should have masked values"


def test_array_from_file(iris0_file, file_or_filelike):
    """Test retrieving an array from a file."""
    with _get_data_file(iris0_file, file_or_filelike) as sigmetfile:
        data = iris.IrisRawFile(sigmetfile, loaddata=True)
        array_data = data.read_from_file(5)  # Adjusted to read_from_file

        # Assertions for the read array
        assert len(array_data) == 5
        assert isinstance(array_data, np.ndarray)


def test_iris_open_mfdataset_context_manager(iris0_file):
    with open_mfdataset(
        [iris0_file],
        engine="iris",
        concat_dim="volume_time",
        combine="nested",
        group="sweep_0",
    ) as ds:
        assert ds is not None
        # closer must exist while inside context
        assert callable(getattr(ds, "_close", None))


def test_iris_dataset_has_close(iris0_file):
    ds = open_dataset(iris0_file, engine="iris", group="sweep_0")
    assert callable(getattr(ds, "_close", None))
    ds.close()


def test_open_iris_datatree(iris0_file):
    # Define kwargs to pass into the function
    kwargs = {
        "sweep": [0, 1, 2, 4],  # Test with specific sweeps
        "first_dim": "auto",
        "reindex_coord": {
            "angle": {
                "start_angle": 0.0,
                "stop_angle": 360.0,
                "angle_res": 1.0,
                "direction": 1,
            }
        },
        "fix_second_angle": True,
        "site_as_coords": True,
    }

    # Call the function with an actual Iris/Sigmet file
    dtree = open_iris_datatree(iris0_file, **kwargs)

    # Assertions
    assert isinstance(dtree, DataTree), "Expected a DataTree instance"
    subtree_paths = [n.path for n in dtree.subtree]
    assert "/" in subtree_paths, "Root group should be present in the DataTree"
    # optional_groups=False by default: metadata subgroups should NOT be present
    assert "radar_parameters" not in dtree.children
    assert "georeferencing_correction" not in dtree.children
    assert "radar_calibration" not in dtree.children

    # Check if at least one sweep group is attached (e.g., "/sweep_0")
    sweep_groups = [key for key in dtree.match("sweep_*")]
    assert len(sweep_groups) == 4, "Expected four sweep groups in the DataTree"

    # Verify a sample variable in one of the sweep groups (adjust based on expected variables)
    sample_sweep = sweep_groups[0]
    assert (
        len(dtree[sample_sweep].data_vars) == 12
    ), f"Expected data variables in {sample_sweep}"
    assert dtree[sample_sweep]["DBZH"].shape == (360, 664)
    assert (
        "DBZH" in dtree[sample_sweep].data_vars
    ), f"DBZH should be a data variable in {sample_sweep}"
    assert (
        "VRADH" in dtree[sample_sweep].data_vars
    ), f"VRADH should be a data variable in {sample_sweep}"

    # Station coords should be on root as coordinates, NOT on sweeps
    assert "latitude" in dtree.ds.coords
    assert "longitude" in dtree.ds.coords
    assert "altitude" in dtree.ds.coords
    assert "latitude" not in dtree.ds.data_vars

    # Validate attributes
    assert len(dtree.attrs) == 10
    assert (
        dtree.attrs["instrument_name"] == "Corozal, Radar"
    ), "Instrument name should match expected value"
    assert dtree.attrs["source"] == "Sigmet", "Source should match expected value"


def test_open_iris_datatree_optional_groups(iris0_file):
    """Test that optional_groups=True includes metadata subgroups."""
    dtree = open_iris_datatree(iris0_file, optional_groups=True)
    assert "radar_parameters" in dtree.children
    assert "georeferencing_correction" in dtree.children
    assert "radar_calibration" in dtree.children


def test_first_loaded_moment_aligned_with_others(iris0_file):
    """Regression for openradar/xradar#357.

    The first moment loaded in a sweep takes the on-the-fly path in
    ``IrisRawFile._get_ray_record_offsets_and_data``. An off-by-one in
    that path (``j = -1``) caused its first matching ray to be written
    to ``raw_data[-1]`` (the last row), rotating the whole moment by 1
    ray. Subsequent moments use a separate cache-hit path and were fine.

    ``iris0_file`` (cor-main) has DB_DBZ as ``data_types[0]`` and no
    DB_XHDR, so DBZH is the first-loaded user-visible moment. Per-row
    correlation between DBZH and other moments derived from the same
    dwell must peak at k=0 (no offset).
    """
    dtree = open_iris_datatree(iris0_file)
    sw = dtree["sweep_0"].to_dataset()

    def best_shift(a, b):
        m = ~(np.isnan(a) | np.isnan(b))
        a = np.where(m, a, 0.0)
        b = np.where(m, b, 0.0)
        a = a - a.mean()
        b = b - b.mean()
        scores = [(((a * np.roll(b, k, axis=0)).sum()), k) for k in range(-3, 4)]
        return max(scores)[1]

    # Moments physically related to DBZH (reflectivity-like statistics).
    # VRADH is excluded — velocity correlates poorly with reflectivity and
    # the noise-driven peak is not informative for alignment.
    related = [m for m in ("ZDR", "KDP", "PHIDP", "RHOHV") if m in sw.data_vars]
    assert related, "test fixture must expose at least one DBZH-related moment"

    for moment in related:
        assert (
            best_shift(sw["DBZH"].values, sw[moment].values) == 0
        ), f"DBZH is row-rotated relative to {moment}"


@pytest.mark.parametrize("fixture", ["iris0_file", "iris1_file"])
def test_range_attrs_match_range(fixture, request):
    # the header range fields are in cm, the attrs are in m (#525)
    path = request.getfixturevalue(fixture)
    with xd.open_datatree(path, engine="iris") as dtree:
        rng = dtree["sweep_0"].ds["range"]
        # also what the CfRadial1 export writes
        exported = xd.transform.to_cfradial1(dtree)["range"].attrs
        for attrs in (rng.attrs, exported):
            np.testing.assert_allclose(
                attrs["meters_between_gates"], np.diff(rng.values)[0]
            )
            np.testing.assert_allclose(
                attrs["meters_to_center_of_first_gate"], rng.values[0]
            )
            assert attrs["spacing_is_constant"] == "true"


def _patch_range_info(monkeypatch, **fields):
    """Serve a deep copy of the ingest header with ``task_range_info`` updated."""

    def header(self):
        hdr = copy.deepcopy(self._ingest_header)
        hdr["task_configuration"]["task_range_info"].update(fields)
        return hdr

    monkeypatch.setattr(iris.IrisIngestHeader, "ingest_header", property(header))


@pytest.mark.parametrize(
    "first_bin, step, spacing, first_centre",
    [
        # an odd centimetre step gives the exact header metres (125.01 m);
        # float32 range values misread such steps beyond about 168 km
        (12345, 12501, 125.01, 123.45),
        # a first bin of 0 means the first gate is centred at half a step
        (0, 30000, 300.0, 150.0),
    ],
)
def test_range_attrs_follow_the_header_exactly(
    iris0_file, monkeypatch, first_bin, step, spacing, first_centre
):
    _patch_range_info(monkeypatch, range_first_bin=first_bin, step_output_bins=step)
    with open_dataset(iris0_file, engine="iris", group="sweep_0") as ds:
        attrs = ds["range"].attrs
        first = ds["range"].values[0]
    assert attrs["meters_between_gates"] == spacing
    assert attrs["meters_to_center_of_first_gate"] == first_centre
    assert attrs["spacing_is_constant"] == "true"
    np.testing.assert_allclose(first, first_centre, rtol=1e-6)


def test_range_attrs_variable_spacing_flag(iris0_file, monkeypatch):
    # with variable bin spacing in the header there is no single gate spacing
    _patch_range_info(monkeypatch, variable_range_bin_spacing_flag=1)
    with open_dataset(iris0_file, engine="iris", group="sweep_0") as ds:
        attrs = ds["range"].attrs
        first = ds["range"].values[0]
    assert attrs["spacing_is_constant"] == "false"
    assert "meters_between_gates" not in attrs
    np.testing.assert_allclose(attrs["meters_to_center_of_first_gate"], first)


def _task_range_info(first, step, n, variable=0):
    # range_last_bin as the IRIS headers give it: the last bin for a non-zero
    # first bin, the end of the last gate for a first bin of 0 (SUR fixture)
    return {
        "range_first_bin": first,
        "range_last_bin": first + step * (n - 1) if first else step * n - 1,
        "step_output_bins": step,
        "number_output_bins": n,
        "variable_range_bin_spacing_flag": variable,
    }


@pytest.mark.parametrize(
    "first, step, n",
    [
        # the #525 case: float32 centimetres were off beyond 2**24 cm
        pytest.param(12345, 12501, 2400, id="odd-cm-step-to-300km"),
        # SUR and IDEAM-like headers
        pytest.param(0, 30000, 833, id="first-bin-zero-sur-header"),
        pytest.param(100000, 30000, 994, id="ideam-like"),
        pytest.param(0, 30000, 1, id="one-gate"),
        pytest.param(0, 30000, 0, id="no-gates"),
    ],
)
def test_range_centers_round_once(first, step, n):
    # every gate is the float32 nearest the exact header distance (#525)
    rng = iris._range_centers(_task_range_info(first, step, n))
    start = first if first else step / 2
    exact = (start + step * np.arange(n, dtype="int64")) / 1e2
    assert rng.dtype == np.float32 and rng.size == n
    np.testing.assert_array_equal(rng, exact.astype("float32"))


def test_range_centers_far_gates():
    # hard-coded far gates of the odd-cm case; float32 centimetres gave
    # 167886.88 m for gate 1342 before (#525)
    rng = iris._range_centers(_task_range_info(12345, 12501, 2400))
    assert rng[1342] == np.float32(167886.87)
    assert rng[-1] == np.float32(300022.44)


def test_range_centers_of_the_sur_header():
    # the SUR fixture's task_range_info, with hard-coded gate centers
    task = {
        "range_first_bin": 0,
        "range_last_bin": 24989999,
        "step_output_bins": 30000,
        "number_output_bins": 833,
    }
    rng = iris._range_centers(task)
    assert rng.size == 833
    assert (rng[0], rng[1], rng[-1]) == (150.0, 450.0, 249750.0)


def test_range_centers_reject_a_zero_step():
    # a corrupt header must not open with a made-up range
    with pytest.raises(ZeroDivisionError):
        iris._range_centers(_task_range_info(100000, 0, 994))


@pytest.mark.parametrize(
    "lon, lat, expected",
    [
        (10.5, 52.3, (10.5, 52.3)),
        (288.5, 4.6, (-71.5, 4.6)),
        (151.2, 326.1, (151.2, -33.9)),
        (300.0, 340.0, (-60.0, -20.0)),
    ],
)
def test_site_coords_fold(lon, lat, expected):
    # BIN4 angles are decoded to [0, 360), southern latitudes must fold
    # by latitude, not longitude (#391)
    obj = SimpleNamespace(
        ingest_header={
            "ingest_configuration": {
                "longitude_radar": lon,
                "latitude_radar": lat,
                "altitude_radar": 12300,
            }
        }
    )
    lon_out, lat_out, alt = iris.IrisRawFile.site_coords.fget(obj)
    np.testing.assert_allclose((lon_out, lat_out), expected)
    assert alt == 123.0


def test_iris_8bit_without_decoding(iris0_file):
    # DB_HCLASS (type 55) holds two 8-bit classes per 16-bit word, which were
    # returned as packed int16 words (#390)
    with open_dataset(
        iris0_file, engine="iris", group="sweep_0", first_dim="time"
    ) as ds:
        hclass = ds.DB_HCLASS
        assert hclass.dtype == np.uint8
        assert hclass.shape == ds.DBZH.shape
        values = hclass.values

    raw = iris.IrisRawFile(iris0_file, rawdata=True)
    expected = _raw_words(raw, 1, "DB_HCLASS", values.shape[1])
    np.testing.assert_array_equal(
        np.sort(values, axis=None), np.sort(expected, axis=None)
    )
    # classes, no packed words
    assert values.max() < 256
    assert {0, 9, 17}.issubset(np.unique(values))


@pytest.mark.parametrize(
    "identifiers, nbytes, n_flags, last",
    [
        ([1, 2, 3, 0, 0, 0], 1, 17, ("cell_convection", 192, 64)),
        ([2, 0, 0, 0, 0, 0], 1, 8, ("precip_heavy_precipitation", 7, 7)),
        ([3, 1, 0, 0, 0, 0], 1, 9, ("meteo_hail", 56, 48)),
        ([0, 0, 0, 1, 0, 0], 2, 7, ("meteo_hail", 7 << 8, 6 << 8)),
        # classes beyond the 2-bit segment are not representable
        ([0, 0, 2, 0, 0, 0], 1, 4, ("precip_precipitation", 192, 192)),
        ([0, 0, 0, 0, 0, 0], 1, 0, None),
        ([9, 255, 0, 0, 0, 0], 1, 0, None),
    ],
)
def test_hclass_flag_attrs(identifiers, nbytes, n_flags, last):
    # HydroClass bit segments, IRIS Programming Guide 4.4.14, Tables 10-12
    attrs = iris.hclass_flag_attrs(identifiers, nbytes=nbytes)
    if last is None:
        assert attrs == {}
        return
    meanings = attrs["flag_meanings"].split()
    assert len(meanings) == len(attrs["flag_masks"]) == len(attrs["flag_values"])
    assert len(meanings) == n_flags
    assert (meanings[-1], attrs["flag_masks"][-1], attrs["flag_values"][-1]) == last
    assert attrs["flag_masks"].dtype == (np.uint8 if nbytes == 1 else np.uint16)
    # values lie within their masks
    assert np.all(attrs["flag_values"] & ~attrs["flag_masks"] == 0)


def test_iris_hclass_flag_attrs(iris0_file, monkeypatch):
    # the sample stores no classifier identifiers, so there are no flags
    with open_dataset(iris0_file, engine="iris", group="sweep_0") as ds:
        assert "flag_meanings" not in ds.DB_HCLASS.attrs

    init = iris.IrisRawFile.__init__

    def init_with_identifiers(self, *args, **kwargs):
        init(self, *args, **kwargs)
        task_end_info = self.ingest_header["task_configuration"]["task_end_info"]
        task_end_info["echo_class_identifiers"] = bytes([1, 2, 3, 0, 0, 0])

    monkeypatch.setattr(iris.IrisRawFile, "__init__", init_with_identifiers)
    with open_dataset(iris0_file, engine="iris", group="sweep_0") as ds:
        hclass = ds.DB_HCLASS
        attrs = hclass.attrs
        assert hclass.dtype == np.uint8
        value = 106  # 0b01_101_010
        assert value in np.unique(hclass.values)
        meanings = [
            meaning
            for meaning, mask, flag in zip(
                attrs["flag_meanings"].split(),
                attrs["flag_masks"],
                attrs["flag_values"],
                strict=True,
            )
            if value & mask == flag
        ]
        assert meanings == [
            "meteo_rain",
            "precip_light_precipitation",
            "cell_convection",
        ]
    # the IRIS names are kept with the classes
    assert iris.HCLASS_CLASSIFIERS[1][2][2] == ("MET_CLASS_RAIN", "rain")


def test_iris_velocity_no_data_is_nan(iris0_file):
    # DB_VEL raw 0 is "velocity data not available" (IRIS Programming Guide
    # 4.4.44), raw 128 is zero velocity; no-data bins must be NaN, not 0 (#462)
    with open_dataset(
        iris0_file, engine="iris", group="sweep_0", first_dim="time"
    ) as ds:
        vel = ds.VRADH.values
    raw = iris.IrisRawFile(iris0_file, loaddata=False, rawdata=True)
    words = _raw_words(raw, 1, "DB_VEL", vel.shape[1])
    # rays are sorted in the Dataset, so compare counts
    assert np.isnan(vel).sum() == (words == 0).sum() > 0
    # true zero velocities stay zero
    assert (vel == 0).sum() == (words == 128).sum() > 0


def test_nyquist_matches_the_programming_guide_example():
    """Guide 5.1.2 productx example: 10.63 cm, 840/560 Hz, 2:3 dual PRF gives
    Nyquist 22.32 m/s for width and 44.65 m/s for velocity."""
    nyquist = iris._nyquist(1063, 840)
    one = np.array([1], dtype="uint16")
    kw = {"scale": 1.0, "nyquist": nyquist, "multi_prf_mode_flag": 1}
    assert iris.decode_width(one, **kw)[0] == pytest.approx(22.32, abs=0.005)
    assert iris.decode_vel(one, **kw)[0] == pytest.approx(44.65, abs=0.005)


@pytest.mark.parametrize("multi_prf_mode_flag", [0, 1, 2, 3])
def test_dual_prf_scales_velocity_not_width(multi_prf_mode_flag):
    """Dual-PRF modes 1:1, 2:3, 3:4, 4:5 (task_dsp_info byte 144; the 4.4.44
    text calls the third "4:3") multiply the velocity Nyquist by 1-4; the
    width keeps the single-PRF one (4.4.48)."""
    data = np.array([2, 3], dtype="uint16")
    kw = {"scale": 1.0, "nyquist": 10.0, "multi_prf_mode_flag": multi_prf_mode_flag}
    vel = iris.decode_vel(data, **kw)
    np.testing.assert_allclose(vel, data * 10.0 * (multi_prf_mode_flag + 1))
    np.testing.assert_allclose(iris.decode_width(data, **kw), data * 10.0)


def _header_stub():
    """The headers ``decode_data`` reads for the Nyquist, with a different
    wavelength and PRF in ``product_end`` than in the task configuration
    (dual-PRF flag 1 in both cases)."""
    dsp_info = {"prf": 7777, "multi_prf_mode_flag": 1}
    return SimpleNamespace(
        _rawdata=False,
        product_hdr={"product_end": {"wavelength": 533, "prf": 1000}},
        ingest_header={
            "task_configuration": {
                "task_dsp_info": dsp_info,
                "task_misc_info": {"wavelength": 999},
            }
        },
    )


def test_moment_names_keep_colliding_types():
    assert iris._moment_names(["DB_DBZ", "DB_VEL", "DB_DBZ2"]) == [
        "DBZH",
        "VRADH",
        "DB_DBZ2",
    ]
    assert iris._moment_names(["DB_DBZ2", "DB_DBZ"]) == ["DBZH", "DB_DBZ"]
    with pytest.raises(ValueError, match="appears twice"):
        iris._moment_names(["DB_DBZ", "DB_DBZ2", "DB_DBZ2"])


def test_eager_reader_keeps_both_colliding_moments(iris0_file, monkeypatch):
    """Two types mapping to one CfRadial name used to keep only the later
    one; now the second keeps its Sigmet name. (The collision is staged by
    mapping DB_VEL onto DBZH: no fixture has DB_DBZ next to DB_DBZ2.)"""
    with open_dataset(iris0_file, engine="iris", group="sweep_0") as before:
        dbzh, vradh = before.DBZH.values, before.VRADH.values
    monkeypatch.setitem(iris.iris_mapping, "DB_VEL", "DBZH")
    with open_dataset(iris0_file, engine="iris", group="sweep_0") as ds:
        assert {"DBZH", "DB_VEL"} <= set(ds.data_vars)
        np.testing.assert_array_equal(ds.DBZH.values, dbzh)
        np.testing.assert_array_equal(ds.DB_VEL.values, vradh)
        # CF attrs follow the mapped name, also under the Sigmet name
        assert ds.DB_VEL.attrs["units"] == ds.DBZH.attrs["units"]


#: each decode path with the wavelength and PRF it reads from ``_header_stub``
_DECODE_PATHS = pytest.mark.parametrize(
    "cls, wavelength_prf",
    [(iris.IrisRawFile, (533, 1000)), (iris.IrisIngestDataFile, (999, 7777))],
)


@_DECODE_PATHS
def test_decode_paths_read_their_own_headers(cls, wavelength_prf):
    """RAW products take wavelength and PRF from ``product_end``, ingest
    data files from the task configuration. (2-byte words, viewed the same
    in both paths; the ingest path's 1-byte view has no file test, no
    ingest data file fixture exists.)"""
    data = np.array([2, 3], dtype="uint16")
    nyquist = iris._nyquist(*wavelength_prf)
    for func, factor in ((iris.decode_vel, 2), (iris.decode_width, 1)):
        prod = {"func": func, "dtype": "uint16", "fkw": {"scale": 1.0}}
        decoded = cls.decode_data(_header_stub(), data, prod)
        np.testing.assert_allclose(decoded, data * nyquist * factor, err_msg=func)


@_DECODE_PATHS
@pytest.mark.parametrize("flag", [4, 65535])
def test_undefined_multi_prf_flag_warns_and_uses_single_prf(cls, wavelength_prf, flag):
    """task_dsp_info defines flags 0-3 only; any other value must not scale
    the velocity (#480). The width ignores the flag, so it does not warn."""
    stub = _header_stub()
    stub.ingest_header["task_configuration"]["task_dsp_info"][
        "multi_prf_mode_flag"
    ] = flag
    data = np.array([2, 3], dtype="uint16")
    nyquist = iris._nyquist(*wavelength_prf)
    vel = {"func": iris.decode_vel, "dtype": "uint16", "fkw": {"scale": 1.0}}
    with pytest.warns(UserWarning, match=f"multi_prf_mode_flag={flag} is not defined"):
        decoded = cls.decode_data(stub, data, vel)
    np.testing.assert_allclose(decoded, data * nyquist)
    width = {**vel, "func": iris.decode_width}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        decoded = cls.decode_data(stub, data, width)
    np.testing.assert_allclose(decoded, data * nyquist)


@pytest.mark.parametrize("cls", [iris.IrisRawFile, iris.IrisIngestDataFile])
def test_kdp_needs_no_prf(cls):
    """``DB_KDP`` returns before the PRF lookup: it only needs the
    wavelength."""
    stub = _header_stub()
    del stub.ingest_header["task_configuration"]["task_dsp_info"]["prf"]
    del stub.product_hdr["product_end"]["prf"]
    words = np.zeros((1, 2), dtype="int16")  # one ray of raw words (KDP 0: no data)
    kdp = iris.SIGMET_DATA_TYPES[14]
    assert kdp["name"] == "DB_KDP"
    assert np.isnan(cls.decode_data(stub, words, kdp)).all()


#: Data types whose raw word 0 is "no data" (IRIS Programming Guide
#: M212927EN-B, 4.4). The corrected and V/E reflectivity types have no section
#: there and follow their base types (#467).
MANUAL_ZERO_IS_NO_DATA = frozenset(
    {
        # reflectivity, 4.4.31-32
        "DB_DBT", "DB_DBZ", "DB_DBT2", "DB_DBZ2", "DB_DBZC", "DB_DBZC2",
        "DB_DBTV8", "DB_DBTV16", "DB_DBZV8", "DB_DBZV16",
        "DB_DBTE8", "DB_DBTE16", "DB_DBZE8", "DB_DBZE16",
        # velocity and width, 4.4.42-45, 4.4.48-49
        "DB_VEL", "DB_VEL2", "DB_VELC", "DB_VELC2", "DB_WIDTH", "DB_WIDTH2",
        # ZDR, 4.4.53-54
        "DB_ZDR", "DB_ZDR2", "DB_ZDRC", "DB_ZDRC2",
        # phase, 4.4.24-27
        "DB_PHIDP", "DB_PHIDP2", "DB_PHIH", "DB_PHIV", "DB_PHIH2", "DB_PHIV2",
        # correlation and quality, 4.4.28-29, 4.4.33-36, 4.4.39-40, 4.4.51-52
        "DB_RHOHV", "DB_RHOHV2", "DB_RHOH", "DB_RHOV", "DB_RHOH2", "DB_RHOV2",
        "DB_SQI", "DB_SQI2", "DB_PMI8", "DB_PMI16", "DB_CCOR8", "DB_CCOR16",
        # KDP and LDR, 4.4.19-21
        "DB_KDP2", "DB_LDRH", "DB_LDRV", "DB_LDRH2", "DB_LDRV2",
        # derived, 4.4.11, 4.4.30, 4.4.41, 4.4.47, 4.4.50
        "DB_HEIGHT", "DB_VIL2", "DB_RAINRATE2", "DB_TIME2", "DB_SHEAR",
        # SNR family, below the -31.5 dB floor of 4.4.37 (#475)
        "DB_SNR8", "DB_SNR16", "DB_LOG8", "DB_LOG16", "DB_CSP8", "DB_CSP16",
        "DB_AH8", "DB_AH16", "DB_AV8", "DB_AV16", "DB_AZDR8", "DB_AZDR16",
    }
)  # fmt: skip

#: Raw 0 is a real value (4.4.2, 4.4.9, 4.4.10, 4.4.12, 4.4.13, 4.4.46): no
#: mask, raw 0 decodes to 0.
MANUAL_ZERO_IS_DATA = frozenset(
    {"DB_AXDIL2", "DB_DEFORM2", "DB_DIVERGE2", "DB_FLIQUID2", "DB_HDIR2", "DB_VVEL2"}
)

#: The guide gives raw 0 no meaning (DB_HVEL2 has no section): no mask,
#: nothing pinned.
MANUAL_ZERO_UNDEFINED = frozenset({"DB_HVEL2"})

#: ``decode_kdp`` NaNs raw 0 (and 255) itself
NAN_AT_ZERO = MANUAL_ZERO_IS_NO_DATA | {"DB_KDP"}

#: Sigmet type name -> its SIGMET_DATA_TYPES entry
_ENTRIES = {e["name"]: e for e in iris.SIGMET_DATA_TYPES.values()}


def _decode_raw(entry, words):
    """Decode raw words through a table entry, with the per-file arguments
    ``IrisRawFile.decode_data`` would add."""
    kwargs = dict(entry.get("fkw") or {})
    if entry["func"] in (iris.decode_vel, iris.decode_width):
        kwargs["nyquist"] = 10.0
    if entry["func"] is iris.decode_kdp:
        kwargs["wavelength"] = 5.33
    return entry["func"](np.asarray(words).astype(entry["dtype"]), **kwargs)


def _raw_words(raw, sweep, name, nbins):
    """One raw word per range bin of ``name`` from an ``IrisRawFile`` opened
    with ``rawdata=True``."""
    raw.get_moment(sweep, name)
    words = raw.data[sweep]["sweep_data"][name]
    one_byte = iris.get_dtype_size(_ENTRIES[name]["dtype"]) == 1
    words = words.view("(2,)uint8" if one_byte else "uint16")
    return words.reshape(words.shape[0], -1)[:, :nbins]


def test_hvel2_decodes_like_vvel2():
    """The guide has no DB_HVEL2 section: signed 0.01 m/s like DB_VVEL2
    (4.4.46) (#479); the sign is checked here, raw 0 by the generic test."""
    hvel2 = {k: v for k, v in _ENTRIES["DB_HVEL2"].items() if k != "name"}
    vvel2 = {k: v for k, v in _ENTRIES["DB_VVEL2"].items() if k != "name"}
    assert hvel2 == vvel2
    np.testing.assert_allclose(
        _decode_raw(_ENTRIES["DB_HVEL2"], [-100, 0, 100]), [-1.0, 0.0, 1.0]
    )


def test_no_data_masks_follow_the_manual():
    """The table masks raw 0 for exactly the no-data types above, and every
    type with a decoder is classified once."""
    masked = {n for n, e in _ENTRIES.items() if "mask" in (e.get("fkw") or {})}
    assert masked == MANUAL_ZERO_IS_NO_DATA
    assert all(_ENTRIES[n]["fkw"]["mask"] == 0 for n in masked)
    decoded = {n for n, e in _ENTRIES.items() if e["func"] is not None}
    groups = (MANUAL_ZERO_IS_NO_DATA, MANUAL_ZERO_IS_DATA, MANUAL_ZERO_UNDEFINED)
    assert decoded == set().union(*groups) | {"DB_KDP"}
    assert sum(map(len, groups)) == len(set().union(*groups))


@pytest.mark.parametrize(
    "entry",
    [e for e in _ENTRIES.values() if e["func"] is not None],
    ids=lambda e: e["name"],
)
def test_raw_zero_decodes_as_the_manual_says(entry):
    with np.errstate(invalid="raise"):  # no sqrt of a negative "no data"
        out = _decode_raw(entry, [0, 2])
    if entry["name"] in MANUAL_ZERO_IS_NO_DATA:  # masked, NaN under the mask
        assert isinstance(out, np.ma.MaskedArray)
        assert out.mask.tolist() == [True, False]
        assert np.isnan(out.fill_value)
    decoded = np.asarray(out, dtype="float64")  # NaN shows without filled()
    assert np.isnan(decoded[0]) == (entry["name"] in NAN_AT_ZERO)
    if entry["name"] in MANUAL_ZERO_IS_DATA:
        assert decoded[0] == 0.0
    assert np.isfinite(decoded[1])  # a data word is never masked


@pytest.mark.parametrize("name", ["DB_PHIH", "DB_PHIV"])
def test_phih_phiv_decode_like_phidp(name):
    """4.4.24: same format as DB_PHIDP (#474)."""
    assert _ENTRIES[name]["fkw"] == _ENTRIES["DB_PHIDP"]["fkw"]
    np.testing.assert_allclose(
        _decode_raw(_ENTRIES[name], [1, 128, 254]),
        [0.0, 90.0, 179.2913],
        atol=1e-4,
    )


@pytest.mark.parametrize(
    "name", ["DB_SQI2", "DB_RHOHV2", "DB_RHOH2", "DB_RHOV2", "DB_PMI16", "DB_CCOR16"]
)
def test_sqi2_format_scale(name):
    """4.4.40: (N - 1) / 65533, 65534 = 1.00000 (#478)."""
    decoded = _decode_raw(_ENTRIES[name], [1, 128, 65533, 65534])
    np.testing.assert_allclose(decoded, [0.0, 0.00194, 0.99998, 1.0], atol=5e-6)
    assert decoded[-1] == 1.0  # exactly one, not just within atol


@pytest.mark.filterwarnings("error::RuntimeWarning")  # RHOHV no-data before sqrt
@pytest.mark.parametrize("fixture", ["iris0_file", "iris1_file"])
def test_no_data_bins_are_nan(fixture, request):
    """The decoded NaNs are exactly the no-data words (#465): in the table
    decoders and in ``IrisRawFile`` (masked arrays with NaN under the mask,
    #467), and as counts in the Dataset (rays reordered)."""
    path = request.getfixturevalue(fixture)
    raw = iris.IrisRawFile(path, loaddata=False, rawdata=True)
    decoded = iris.IrisRawFile(path, loaddata=False)
    sweep = 1  # IRIS numbers sweeps from 1: the file's first sweep is sweep_0
    names = [n for n in raw.data[sweep]["ingest_data_hdrs"] if n in NAN_AT_ZERO]
    with open_dataset(path, engine="iris", group="sweep_0") as ds:
        for name in names:
            # (iris_mapping holds for these files: no two types share a name)
            values = ds[iris.iris_mapping.get(name, name)].values
            words = _raw_words(raw, sweep, name, values.shape[1])
            no_data = np.isin(words, (0, 255) if name == "DB_KDP" else (0,))
            nan = np.isnan(np.asarray(_decode_raw(_ENTRIES[name], words)))
            np.testing.assert_array_equal(nan, no_data, err_msg=name)
            assert np.isnan(values).sum() == no_data.sum(), name
            decoded.get_moment(sweep, name)
            moment = decoded.data[sweep]["sweep_data"][name][:, : values.shape[1]]
            np.testing.assert_array_equal(np.isnan(np.asarray(moment)), no_data, name)
            if name in MANUAL_ZERO_IS_NO_DATA:  # (DB_KDP is a plain array)
                np.testing.assert_array_equal(moment.mask, no_data, err_msg=name)
                np.testing.assert_array_equal(
                    np.isnan(moment.filled()), no_data, err_msg=name
                )
    assert len(names) >= 6


def test_ingest_data_no_data_reads_as_nan():
    """The ingest-data decode path gives the same NaN-under-mask arrays."""
    stub = SimpleNamespace(_rawdata=False)
    decoded = iris.IrisIngestDataFile.decode_data(
        stub, np.array([[0, 2, 128]], dtype="uint8"), _ENTRIES["DB_DBZ"]
    )
    assert isinstance(decoded, np.ma.MaskedArray)
    np.testing.assert_array_equal(np.asarray(decoded), [[np.nan, -31.0, 32.0]])
    np.testing.assert_array_equal(decoded.mask, [[True, False, False]])


def test_shear_is_unsigned():
    """4.4.50: subtract 128 and multiply by 0.2; 1 = -25.4, 254 = +25.2
    (#476)."""
    np.testing.assert_allclose(
        _decode_raw(_ENTRIES["DB_SHEAR"], [1, 128, 200, 254]), [-25.4, 0.0, 14.4, 25.2]
    )


def test_decoders_keep_a_caller_mask_and_take_scalars():
    """A mask the caller passes in is merged with the no-data words, and a
    single raw word decodes too."""
    dbz = _ENTRIES["DB_DBZ"]
    words = np.ma.MaskedArray([0, 2, 128], mask=[False, True, False], dtype="uint8")
    out = dbz["func"](words, **dbz["fkw"])
    assert out.mask.tolist() == [True, True, False]
    assert np.isnan(np.asarray(out)[:2]).all() and np.asarray(out)[2] == 32.0
    assert np.isnan(np.asarray(dbz["func"](np.uint8(0), **dbz["fkw"])))
    assert dbz["func"](np.uint8(128), **dbz["fkw"]) == 32.0


def _decode_entry(name, words):
    """``_decode_raw`` by type name, as plain floats (NaN at masked bins)."""
    return np.asarray(_decode_raw(_ENTRIES[name], words), dtype="float64")


@pytest.mark.parametrize("family", ["SNR", "LOG", "CSP", "AH", "AV", "AZDR"])
def test_snr_family_scaling(family):
    """1-byte: -31.5 ... +95.5 dB in 0.5 dB steps (4.4.37); 2-byte:
    (N - 32768) / 100 like the other 2-byte dB types (#475)."""
    np.testing.assert_allclose(
        _decode_entry(f"DB_{family}8", [1, 64, 255]), [-31.5, 0.0, 95.5]
    )
    np.testing.assert_allclose(
        _decode_entry(f"DB_{family}16", [27133, 32768, 39853]), [-56.35, 0.0, 70.85]
    )


def test_snr16_on_the_test_file(iris1_file):
    """SUR's SNRH was -31.5 ... 19895 dB with the 1-byte scaling; it now
    gives -56 ... +71 dB, and raw 0 is masked where DBZH has no data
    (#475)."""
    with open_dataset(iris1_file, engine="iris", group="sweep_0") as ds:
        snr, dbz = ds.SNRH.values, ds.DBZH.values
        assert ds.SNRH.attrs["units"] == "dB"
    assert -60 < np.nanmin(snr) < np.nanmax(snr) < 75
    assert np.isnan(snr).any()
    assert not (np.isnan(snr) & ~np.isnan(dbz)).any()


def test_snr8_matches_reflectivity_on_arouca(iris_arouca_file):
    """IRIS computes DBZ = C + SNR + 20 log10(r) + gas * r, so the 1-byte
    SNR scale (N - 64) / 2 must close that equation to within the rounding
    of two 0.5 dB values for almost all bins (#475)."""
    with iris.IrisRawFile(iris_arouca_file, loaddata=False) as raw:
        task = raw.ingest_header["task_configuration"]
    cal = task["task_calib_info"]["calibration_reflectivity"] / 16
    gas = task["task_dsp_info"]["gas_attenuation"] / 1e5  # dB/km
    with open_dataset(iris_arouca_file, engine="iris", group="sweep_0") as ds:
        snr, dbz = ds.SNRH.values, ds.DBZH.values
        r = ds.range.values / 1000.0
    ok = ~np.isnan(snr) & ~np.isnan(dbz)
    resid = (dbz - snr - 20 * np.log10(r) - gas * r - cal)[ok]
    # 7882 bins, median -0.01, 97 % within 0.5 dB; main's (N - 63) / 2
    # gives median -0.51 and 46 %
    assert resid.size > 5000
    assert np.abs(np.median(resid)) < 0.05
    assert np.mean(np.abs(resid) <= 0.5) > 0.95
