#!/usr/bin/env python
# Copyright (c) 2023-2025, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for `io.backends.iris` module.

Ported from wradlib.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from xarray import DataTree, open_dataset, open_mfdataset

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


def decode_array():
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
    data = [0, 1, 2, 128, 129, 254, 255]
    np.testing.assert_array_almost_equal(
        iris.decode_array(data, scale=127 / 75.0, offset=-1, offset2=-75, mask=0.0),
        [np.inf, -75.0, -74.409449, 0.0, 0.590551, 74.409449, 75.0],
    )


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

        # Check data types for moments, including masked arrays for velocity
        if moment == "DB_VEL":
            assert isinstance(
                moment_data, np.ma.MaskedArray
            ), "DB_VEL should be a masked array"
        else:
            assert isinstance(
                moment_data, np.ndarray
            ), f"{moment} should be a numpy array"

        # Optional: check for expected placeholder/masked values
        if moment == "DB_DBZ":
            assert (
                moment_data == -32
            ).sum() > 0, "DB_DBZ should contain placeholder values (-32)"
        if moment == "DB_VEL":
            assert moment_data.mask.sum() > 0, "DB_VEL should have masked values"


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
    raw.get_moment(1, "DB_HCLASS")
    words = raw.data[1]["sweep_data"]["DB_HCLASS"]
    expected = words.view("(2,)uint8").reshape(words.shape[0], -1)[:, : values.shape[1]]
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
    raw = iris.IrisRawFile(iris0_file, loaddata=False, rawdata=True)
    raw.get_moment(1, "DB_VEL")
    words = raw.data[1]["sweep_data"]["DB_VEL"]
    # one value per range bin, like the decoded moment
    words = words.view("(2,)uint8").reshape(words.shape[0], -1)
    with open_dataset(
        iris0_file, engine="iris", group="sweep_0", first_dim="time"
    ) as ds:
        vel = ds.VRADH.values
    # rays are sorted in the Dataset, so compare counts
    words = words[:, : vel.shape[1]]
    assert np.isnan(vel).sum() == (words == 0).sum() > 0
    # true zero velocities stay zero
    assert (vel == 0).sum() == (words == 128).sum() > 0


def test_nyquist_matches_the_programming_guide_example():
    """The guide's productx example (5.1.2): 10.63 cm, PRF 840/560 Hz in 2:3
    dual-PRF mode gives Nyquist 44.65 m/s for velocity and 22.32 m/s for
    width, computed from the higher PRF (4.4.44, 4.4.48)."""
    assert iris._nyquist(1063, 840) == pytest.approx(22.32, abs=0.005)
    assert iris._nyquist(1063, 840, 1) == pytest.approx(44.65, abs=0.005)


#: (wavelength 1/100 cm, PRF Hz) each decode path must read, and a decoy in
#: the header it must not read
_PRODUCT_END, _TASK = (533, 1000), (999, 7777)


def _header_stub(cls, multi_prf_mode_flag):
    """The headers ``decode_data`` reads for the Nyquist: ``IrisRawFile``
    takes wavelength and PRF from ``product_end``, ``IrisIngestDataFile``
    from the task configuration; the other header holds a decoy. (The
    unbound method runs on this stub, so a new ``self`` attribute in
    ``decode_data`` fails loudly here.)"""
    raw = cls is iris.IrisRawFile
    (end_wl, end_prf), (task_wl, task_prf) = (
        (_PRODUCT_END, _TASK) if raw else (_TASK, _PRODUCT_END)
    )
    dsp_info = {"prf": task_prf, "multi_prf_mode_flag": multi_prf_mode_flag}
    return SimpleNamespace(
        _rawdata=False,
        product_hdr={"product_end": {"wavelength": end_wl, "prf": end_prf}},
        ingest_header={
            "task_configuration": {
                "task_dsp_info": dsp_info,
                "task_misc_info": {"wavelength": task_wl},
            }
        },
    )


@pytest.mark.parametrize("multi_prf_mode_flag", [0, 1, 2, 3])
@pytest.mark.parametrize("cls", [iris.IrisRawFile, iris.IrisIngestDataFile])
def test_dual_prf_scales_velocity_not_width(cls, multi_prf_mode_flag):
    """Dual-PRF modes 1:1, 2:3, 3:4, 4:5 (task_dsp_info byte 144) multiply
    the velocity Nyquist by 1, 2, 3, 4; the width Nyquist stays the
    single-PRF one. Each path reads its own header (2-byte words here; the
    1-byte views differ between the paths and are covered by the file tests)."""
    stub = _header_stub(cls, multi_prf_mode_flag=multi_prf_mode_flag)
    data = np.array([1, 2, 3], dtype="uint16")
    vel = {"func": iris.decode_vel, "dtype": "uint16", "fkw": {"scale": 1.0}}
    width = {"func": iris.decode_width, "dtype": "uint16", "fkw": {"scale": 1.0}}
    nyquist = iris._nyquist(*_PRODUCT_END)
    np.testing.assert_allclose(
        cls.decode_data(stub, data, vel), data * nyquist * (multi_prf_mode_flag + 1)
    )
    np.testing.assert_allclose(cls.decode_data(stub, data, width), data * nyquist)


@pytest.mark.parametrize("cls", [iris.IrisRawFile, iris.IrisIngestDataFile])
def test_kdp_needs_no_prf(cls):
    """``DB_KDP`` returns before the PRF lookup: it only needs the
    wavelength."""
    stub = _header_stub(cls, multi_prf_mode_flag=0)
    del stub.ingest_header["task_configuration"]["task_dsp_info"]["prf"]
    del stub.product_hdr["product_end"]["prf"]
    kdp = {"func": iris.decode_kdp, "dtype": "int8"}
    words = np.zeros((1, 2), dtype="int16")  # one ray of raw words (KDP 0: no data)
    assert np.isnan(cls.decode_data(stub, words, kdp)).all()
