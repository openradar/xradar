#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for `io.backends.nexrad_level3` module.

Unit tests run against synthetic Level 3 files built byte-by-byte with
known ground truth. Integration tests against real files from
open-radar-data live in the ``TestRealFiles`` class and skip when the
fixtures are unavailable.
"""

import bz2
import io
import struct
from datetime import datetime

import numpy as np
import pytest
import xarray as xr
from xarray import DataTree

from xradar.io.backends import open_nexradlevel3_datatree
from xradar.io.backends.nexrad_level3 import (
    IN_TO_MM,
    KT_TO_MS,
    NEXRADLevel3File,
)

KLOT_LAT = 41604  # 41.604 deg N in 0.001 deg units
KLOT_LON = -88085  # 88.085 deg W
KLOT_HEIGHT_FT = 663

VOL_SCAN = datetime(2026, 7, 17, 19, 11, 15)
PRODUCT_TIME = datetime(2026, 7, 17, 19, 12, 42)


def _mdate_mtime(dt):
    """Days since 1970-01-01 (day 1 = the epoch) and seconds of day."""
    epoch = datetime(1970, 1, 1)
    days = (dt.date() - epoch.date()).days + 1
    seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
    return days, seconds


def _radial_packet16(nradials, nbins, data, first_bin=0, range_scale=1000):
    packet = struct.pack(">7h", 16, first_bin, nbins, 256, 280, range_scale, nradials)
    for i in range(nradials):
        angle_start = int(i * 3600 / nradials)
        packet += struct.pack(">3h", nbins, angle_start, 5)
        packet += data[i].astype("u1").tobytes()
    return packet


def _radial_packet_af1f(nradials, nbins, rle_radials, first_bin=0, range_scale=1000):
    packet = struct.pack(
        ">7h", -20705, first_bin, nbins, 256, 280, range_scale, nradials
    )
    for i in range(nradials):
        rle = rle_radials[i]
        if len(rle) % 2:
            rle = rle + b"\x00"  # pad to a full halfword
        angle_start = int(i * 3600 / nradials)
        packet += struct.pack(">3h", len(rle) // 2, angle_start, 10)
        packet += rle
    return packet


def _xdr_string(value):
    raw = value.encode("ascii")
    padded = (len(raw) + 3) & ~3
    return struct.pack(">I", len(raw)) + raw.ljust(padded, b"\x00")


def _generic_packet28(
    nradials,
    nbins,
    data,
    first_gate=125.0,
    gate_width=250.0,
    component_code=1,
    parameters=None,
    ncomponents=1,
):
    xdr = _xdr_string("DPR")
    xdr += _xdr_string("Digital Inst Precip Rate")
    xdr += struct.pack(">i", 176)
    xdr += struct.pack(">i", 0)
    xdr += struct.pack(">I", 1200000000)
    xdr += _xdr_string("KLOT")
    xdr += struct.pack(">f", 41.604)
    xdr += struct.pack(">f", -88.085)
    xdr += struct.pack(">f", 202.0)
    xdr += struct.pack(">I", 1200000000)
    xdr += struct.pack(">I", 1200000000)
    xdr += struct.pack(">f", 0.0)
    xdr += struct.pack(">i", 1)
    xdr += struct.pack(">i", 2)
    xdr += struct.pack(">i", 212)
    xdr += struct.pack(">i", 0)
    xdr += struct.pack(">i", 0)
    xdr += struct.pack(">i", 0)
    if parameters:
        xdr += struct.pack(">2i", len(parameters), 0)
        for i, (key, value) in enumerate(parameters):
            xdr += _xdr_string(key) + _xdr_string(value)
            if i < len(parameters) - 1:
                xdr += struct.pack(">i", 0)
    else:
        # parameters: count 0 + garbage pointer
        xdr += struct.pack(">2i", 0, 0)
    # components: count + garbage pointer + first component code
    xdr += struct.pack(">3i", ncomponents, 0, component_code)
    xdr += _xdr_string("Radial")
    xdr += struct.pack(">2f", gate_width, first_gate)
    xdr += struct.pack(">2i", 0, 0)  # component parameters: none
    xdr += struct.pack(">i", nradials)
    for i in range(nradials):
        azimuth = i * 360.0 / nradials
        xdr += struct.pack(">3f", azimuth, 0.0, 1.0)
        xdr += struct.pack(">i", nbins)
        xdr += _xdr_string("")
        xdr += struct.pack(">I", nbins)
        xdr += struct.pack(f">{nbins}i", *[int(v) for v in data[i]])
    for _ in range(ncomponents - 1):
        xdr += struct.pack(">2i", 0, 1)  # list pointer + next component code
        xdr += _xdr_string("Radial")
        xdr += struct.pack(">2f", gate_width, first_gate)
        xdr += struct.pack(">2i", 0, 0)
        xdr += struct.pack(">i", 0)  # zero radials in the extra component
    return struct.pack(">2hi", 28, 0, len(xdr)) + xdr


def _flag_threshold(scale, offset, max_val=255, leading=2, trailing=0):
    """Threshold block for float-scaled products incl. flag counts
    (halfwords 31-34 scale/offset floats, 36 max level, 37/38 flags)."""
    return (
        struct.pack(">2f", scale, offset)
        + b"\x00\x00"
        + struct.pack(">H2h", max_val, leading, trailing)
    ).ljust(32, b"\x00")


def build_level3_file(
    msg_code=153,
    packet=None,
    threshold=None,
    elevation_tenths=5,
    elevation_num=1,
    vcp=212,
    operational_mode=2,
    compress=False,
    site="LOT",
    version=0,
):
    """Assemble a complete synthetic NEXRAD Level 3 file."""
    if threshold is None:
        threshold = struct.pack(">2h", -320, 5).ljust(32, b"\x00")
    assert len(threshold) == 32

    if packet is None:
        data = np.tile(np.arange(8, dtype="u1") * 10, (4, 1))
        packet = _radial_packet16(4, 8, data)

    symbology = (
        struct.pack(">2hi2hi", -1, 1, 16 + len(packet), 1, -1, len(packet)) + packet
    )
    if compress:
        symbology_out = bz2.compress(symbology)
        compression_flag = 1
        uncompressed_size = len(symbology)
    else:
        symbology_out = symbology
        compression_flag = 0
        uncompressed_size = 0

    hw47_53 = struct.pack(">4h", 0, 0, 0, 0) + struct.pack(
        ">hI", compression_flag, uncompressed_size
    )
    vol_days, vol_secs = _mdate_mtime(VOL_SCAN)
    prod_days, prod_secs = _mdate_mtime(PRODUCT_TIME)

    pdb = struct.pack(
        ">hiihhhhhhhihi4sh2s32s14sBBiii",
        -1,
        KLOT_LAT,
        KLOT_LON,
        KLOT_HEIGHT_FT,
        msg_code,
        operational_mode,
        vcp,
        0,
        1,
        vol_days,
        vol_secs,
        prod_days,
        prod_secs,
        b"\x00" * 4,
        elevation_num,
        struct.pack(">h", elevation_tenths),
        threshold,
        hw47_53,
        version,
        0,
        60,  # symbology offset in halfwords from the message header
        0,
        0,
    )
    assert len(pdb) == 102

    length = 18 + 102 + len(symbology_out)
    msg_date, msg_secs = _mdate_mtime(PRODUCT_TIME)
    mhb = struct.pack(">2h2i3h", msg_code, msg_date, msg_secs, length, 1, 0, 3)

    text_header = f"SDUS53 K{site} 171911\r\r\nN0B{site}\r\r\n".encode("ascii")
    assert len(text_header) == 30

    return text_header + mhb + pdb + symbology_out


class TestParser:
    def test_linear_hw_decode(self):
        raw = np.tile(np.array([0, 1, 2, 100, 255], dtype="u1"), (4, 1))
        buf = build_level3_file(
            msg_code=153,
            packet=_radial_packet16(4, 5, raw),
            threshold=struct.pack(">2h", -320, 5).ljust(32, b"\x00"),
        )
        f = NEXRADLevel3File(io.BytesIO(buf))
        data = f.get_data()
        # (raw - 2) * 0.5 - 32.0
        assert data[0, 3] == pytest.approx(17.0)
        assert data[0, 4] == pytest.approx(94.5)
        assert np.isnan(data[0, 0]) and np.isnan(data[0, 1])
        rf = f.get_range_folded()
        assert rf[0, 1] and not rf[0, 0] and not rf[0, 3]

    def test_linear_hw_scale_offset_attrs(self):
        buf = build_level3_file(msg_code=153)
        f = NEXRADLevel3File(io.BytesIO(buf))
        scale, offset = f.get_scale_offset()
        raw = 100
        assert raw * scale + offset == pytest.approx(17.0)

    def test_float_scale_decode(self):
        raw = np.tile(np.array([0, 1, 130, 200], dtype="u1"), (4, 1))
        buf = build_level3_file(
            msg_code=159,
            packet=_radial_packet16(4, 4, raw),
            threshold=struct.pack(">2f", 16.0, 128.0).ljust(32, b"\x00"),
        )
        f = NEXRADLevel3File(io.BytesIO(buf))
        data = f.get_data()
        assert data[0, 2] == pytest.approx((130 - 128.0) / 16.0)
        assert data[0, 3] == pytest.approx(4.5)
        assert np.isnan(data[0, 0]) and np.isnan(data[0, 1])

    def test_precip_decode_mm(self):
        raw = np.tile(np.array([0, 1, 50], dtype="u1"), (4, 1))
        buf = build_level3_file(
            msg_code=170,
            packet=_radial_packet16(4, 3, raw),
            threshold=struct.pack(">2f", 2.0, 0.0).ljust(32, b"\x00"),
        )
        f = NEXRADLevel3File(io.BytesIO(buf))
        data = f.get_data()
        # (raw / 2.0) hundredths of inches -> mm
        assert data[0, 2] == pytest.approx(25.0 * 0.01 * IN_TO_MM)
        assert data[0, 1] == pytest.approx(0.5 * 0.01 * IN_TO_MM)
        assert np.isnan(data[0, 0])

    def test_legacy16_decode_per_flag_scale(self):
        # level 0: bad (0x80); level 1: +5 scaled 1/10 (0x10);
        # level 2: -10 (sign 0x01); level 3: +30 scaled 1/20 (0x20)
        pairs = [(0x80, 0), (0x10, 5), (0x01, 10), (0x20, 30)] + [(0x80, 0)] * 12
        threshold = b"".join(struct.pack(">2B", f, v) for f, v in pairs)
        rle = [bytes([0x10, 0x21, 0x12, 0x13]) for _ in range(4)]
        buf = build_level3_file(
            msg_code=19,
            packet=_radial_packet_af1f(4, 5, rle),
            threshold=threshold,
        )
        f = NEXRADLevel3File(io.BytesIO(buf))
        data = f.get_data()
        # RLE: 1x level0, 2x level1, 1x level2, 1x level3
        assert np.isnan(data[0, 0])
        assert data[0, 1] == pytest.approx(0.5)
        assert data[0, 2] == pytest.approx(0.5)
        assert data[0, 3] == pytest.approx(-10.0)
        assert data[0, 4] == pytest.approx(1.5)

    def test_legacy16_velocity_knots_to_ms(self):
        pairs = [(0x80, 0), (0x01, 10), (0x00, 10)] + [(0x80, 0)] * 13
        threshold = b"".join(struct.pack(">2B", f, v) for f, v in pairs)
        rle = [bytes([0x11, 0x22]) for _ in range(4)]
        buf = build_level3_file(
            msg_code=27,
            packet=_radial_packet_af1f(4, 3, rle),
            threshold=threshold,
        )
        f = NEXRADLevel3File(io.BytesIO(buf))
        data = f.get_data()
        assert data[0, 0] == pytest.approx(-10.0 * KT_TO_MS)
        assert data[0, 1] == pytest.approx(10.0 * KT_TO_MS)
        assert data[0, 2] == pytest.approx(10.0 * KT_TO_MS)

    def test_af1f_expansion_halfword_sizing(self):
        # 3-byte run list would misalign if nbytes were counted in bytes
        rle = [bytes([0x31, 0x22, 0x10]) for _ in range(2)]
        buf = build_level3_file(
            msg_code=19,
            packet=_radial_packet_af1f(2, 6, rle),
            threshold=b"".join(struct.pack(">2B", 0x00, v) for v in range(16)),
        )
        f = NEXRADLevel3File(io.BytesIO(buf))
        raw = f.get_data_raw()
        np.testing.assert_array_equal(raw[0], [1, 1, 1, 2, 2, 0])

    def test_class_int_decode(self):
        raw = np.tile(np.array([0, 10, 60, 150], dtype="u1"), (4, 1))
        buf = build_level3_file(
            msg_code=165,
            packet=_radial_packet16(4, 4, raw),
        )
        f = NEXRADLevel3File(io.BytesIO(buf))
        data = f.get_data()
        assert np.isnan(data[0, 0])
        assert data[0, 1] == 10.0
        assert data[0, 2] == 60.0
        assert data[0, 3] == 150.0

    def test_packet28_xdr_rate(self):
        data = np.tile(np.array([0, 100, 400], dtype="u2"), (4, 1))
        buf = build_level3_file(
            msg_code=176,
            packet=_generic_packet28(4, 3, data),
            threshold=_flag_threshold(1000.0, 0.0, max_val=65535, leading=0),
        )
        f = NEXRADLevel3File(io.BytesIO(buf))
        decoded = f.get_data()
        # raw/1000 inches/hour -> mm/hour; DPR has no flag levels, so a
        # raw 0 is a genuine zero rain rate, not missing data
        assert decoded[0, 0] == pytest.approx(0.0)
        assert decoded[0, 1] == pytest.approx(0.1 * IN_TO_MM)
        assert decoded[0, 2] == pytest.approx(0.4 * IN_TO_MM)
        assert f.get_range_folded() is None
        assert f.raw_data.shape == (4, 3)
        np.testing.assert_allclose(f.get_azimuth(), [0.0, 90.0, 180.0, 270.0])
        # generic packets carry gate geometry in meters with first_gate
        # already at the first bin center
        rng = f.get_range()
        assert rng[0] == pytest.approx(125.0)
        assert rng[1] - rng[0] == pytest.approx(250.0)

    def test_metadata(self):
        buf = build_level3_file()
        f = NEXRADLevel3File(io.BytesIO(buf))
        latitude, longitude, altitude = f.get_location()
        assert latitude == pytest.approx(41.604)
        assert longitude == pytest.approx(-88.085)
        assert altitude == pytest.approx(KLOT_HEIGHT_FT * 0.3048)
        assert f.get_elevation() == pytest.approx(0.5)
        assert f.get_volume_start_datetime() == VOL_SCAN
        assert f.get_product_datetime() == PRODUCT_TIME
        assert f.prod_descr["vcp"] == 212
        np.testing.assert_allclose(f.get_azimuth(), [0.0, 90.0, 180.0, 270.0])

    def test_range_centers(self):
        buf = build_level3_file(msg_code=153)
        f = NEXRADLevel3File(io.BytesIO(buf))
        rng = f.get_range()
        # range_scale 1000 * mult 0.25 -> 250 m spacing, centered
        assert rng[0] == pytest.approx(125.0)
        assert rng[1] - rng[0] == pytest.approx(250.0)

    def test_flag_counts_mask_and_valid_max(self):
        raw = np.tile(np.array([0, 1, 130, 254, 255], dtype="u1"), (4, 1))
        buf = build_level3_file(
            msg_code=159,
            packet=_radial_packet16(4, 5, raw),
            threshold=_flag_threshold(16.0, 128.0, max_val=255, leading=2, trailing=1),
        )
        f = NEXRADLevel3File(io.BytesIO(buf))
        data = f.get_data()
        assert np.isnan(data[0, 0]) and np.isnan(data[0, 1])  # flag levels
        assert data[0, 2] == pytest.approx(0.125)
        assert data[0, 3] == pytest.approx((254 - 128.0) / 16.0)
        assert np.isnan(data[0, 4])  # above max_val - trailing
        rf = f.get_range_folded()
        assert rf[0, 1] and not rf[0, 2]

    def test_legacy16_hundredths_scale_flag(self):
        # flag 0x40 scales the level value by 1/100 (ICD bit 14)
        pairs = [(0x80, 0), (0x40, 25)] + [(0x80, 0)] * 14
        threshold = b"".join(struct.pack(">2B", f, v) for f, v in pairs)
        rle = [bytes([0x11, 0x21]) for _ in range(4)]
        buf = build_level3_file(
            msg_code=78,
            packet=_radial_packet_af1f(4, 3, rle),
            threshold=threshold,
        )
        f = NEXRADLevel3File(io.BytesIO(buf))
        data = f.get_data()
        # 0.25 inches -> mm
        assert data[0, 1] == pytest.approx(0.25 * IN_TO_MM)

    def test_first_bin_is_an_index(self):
        raw = np.tile(np.arange(4, dtype="u1"), (4, 1))
        buf = build_level3_file(
            msg_code=153,
            packet=_radial_packet16(4, 4, raw, first_bin=2),
        )
        f = NEXRADLevel3File(io.BytesIO(buf))
        rng = f.get_range()
        # first_bin counts bins, not meters
        assert rng[0] == pytest.approx((2 + 0.5) * 250.0)

    def test_version_warning(self):
        buf = build_level3_file(version=3)
        with pytest.warns(UserWarning, match="newer than the latest"):
            NEXRADLevel3File(io.BytesIO(buf))

    def test_zero_scale_warns_all_nan(self):
        raw = np.tile(np.array([5, 6], dtype="u1"), (4, 1))
        buf = build_level3_file(
            msg_code=159,
            packet=_radial_packet16(4, 2, raw),
            threshold=_flag_threshold(0.0, 0.0),
        )
        f = NEXRADLevel3File(io.BytesIO(buf))
        with pytest.warns(UserWarning, match="Zero threshold scale"):
            data = f.get_data()
        assert np.isnan(data).all()

    def test_truncated_file_raises(self):
        buf = build_level3_file()
        with pytest.raises(ValueError, match="[Tt]runcated"):
            NEXRADLevel3File(io.BytesIO(buf[:-20]))

    def test_corrupt_bz2_raises(self):
        buf = bytearray(build_level3_file(compress=True))
        buf[-8:] = b"\x00" * 8
        with pytest.raises(ValueError, match="decompress"):
            NEXRADLevel3File(io.BytesIO(bytes(buf)))

    def test_zero_radials_raises(self):
        packet = struct.pack(">7h", 16, 0, 8, 256, 280, 1000, 0)
        buf = build_level3_file(packet=packet)
        with pytest.raises(ValueError, match="corrupt or empty"):
            NEXRADLevel3File(io.BytesIO(buf))

    def test_failed_open_no_del_noise(self):
        with pytest.raises(FileNotFoundError):
            NEXRADLevel3File("/nonexistent/level3/file")

    def test_bz2_roundtrip(self):
        plain = NEXRADLevel3File(io.BytesIO(build_level3_file(compress=False)))
        packed = NEXRADLevel3File(io.BytesIO(build_level3_file(compress=True)))
        np.testing.assert_array_equal(plain.get_data_raw(), packed.get_data_raw())
        np.testing.assert_allclose(plain.get_azimuth(), packed.get_azimuth())

    def test_deferred_product_raises(self):
        buf = build_level3_file(msg_code=134)
        with pytest.raises(NotImplementedError, match="High Resolution VIL"):
            NEXRADLevel3File(io.BytesIO(buf))

    def test_unknown_product_raises(self):
        buf = build_level3_file(msg_code=57)
        with pytest.raises(NotImplementedError, match="57"):
            NEXRADLevel3File(io.BytesIO(buf))

    def test_not_a_level3_file(self):
        with pytest.raises(ValueError, match="Not a valid NEXRAD Level 3"):
            NEXRADLevel3File(io.BytesIO(b"\x00" * 200))

    def test_bytes_input(self):
        f = NEXRADLevel3File(build_level3_file())
        assert f.raw_data.shape == (4, 8)


class TestDatasetAPI:
    @pytest.fixture
    def l3_file(self, tmp_path):
        path = tmp_path / "LOT_N0B_2026_07_17_19_11_15"
        path.write_bytes(build_level3_file())
        return str(path)

    def test_open_dataset(self, l3_file):
        ds = xr.open_dataset(l3_file, engine="nexradlevel3")
        assert "DBZH" in ds.data_vars
        assert ds["DBZH"].dims == ("azimuth", "range")
        for name in (
            "sweep_number",
            "sweep_mode",
            "sweep_fixed_angle",
            "prt_mode",
            "follow_mode",
        ):
            assert name in ds.variables
        assert ds["sweep_mode"].values.item() == "azimuth_surveillance"
        assert ds["range"].attrs["units"] == "meters"
        assert ds["range"].attrs["meters_between_gates"] == pytest.approx(250.0)
        for coord in ("latitude", "longitude", "altitude"):
            assert coord in ds.coords
        assert ds.attrs["vcp"] == 212
        assert ds.attrs["operational_mode"] == "precipitation"
        assert ds.attrs["instrument_name"] == "KLOT"
        assert callable(ds._close)
        # azimuth coord holds ray CENTERS: start + delta/2 (delta = 0.5 deg)
        np.testing.assert_allclose(ds["azimuth"].values, [0.25, 90.25, 180.25, 270.25])

    def test_time_coord(self, l3_file):
        ds = xr.open_dataset(l3_file, engine="nexradlevel3")
        expected = np.datetime64(VOL_SCAN, "ns")
        assert (ds["time"].values == expected).all()

    def test_decode_times_false(self, l3_file):
        ds = xr.open_dataset(l3_file, engine="nexradlevel3", decode_times=False)
        seconds = VOL_SCAN.hour * 3600 + VOL_SCAN.minute * 60 + VOL_SCAN.second
        assert (ds["time"].values == seconds).all()
        assert "since" in ds["time"].attrs["units"]

    def test_first_dim_time(self, l3_file):
        ds = xr.open_dataset(l3_file, engine="nexradlevel3", first_dim="time")
        assert ds["DBZH"].dims == ("time", "range")

    def test_site_as_coords_false(self, l3_file):
        ds = xr.open_dataset(l3_file, engine="nexradlevel3", site_as_coords=False)
        assert "latitude" in ds.data_vars

    def test_mask_and_scale_false(self, l3_file):
        ds = xr.open_dataset(l3_file, engine="nexradlevel3", mask_and_scale=False)
        assert ds["DBZH"].dtype == np.uint8
        attrs = ds["DBZH"].attrs
        assert attrs["scale_factor"] == pytest.approx(0.5)
        assert attrs["add_offset"] == pytest.approx(-33.0)
        assert attrs["_FillValue"] == 0

    def test_moment_attributes(self, l3_file):
        ds = xr.open_dataset(l3_file, engine="nexradlevel3")
        attrs = ds["DBZH"].attrs
        assert attrs["standard_name"] == "radar_equivalent_reflectivity_factor_h"
        assert "coordinates" in attrs

    def test_file_like(self, l3_file):
        with open(l3_file, "rb") as fh:
            ds = xr.open_dataset(fh, engine="nexradlevel3")
        assert "DBZH" in ds.data_vars


class TestDatatree:
    def _tilt_file(self, tmp_path, name, elevation_tenths, elevation_num, msg_code=153):
        path = tmp_path / name
        path.write_bytes(
            build_level3_file(
                msg_code=msg_code,
                elevation_tenths=elevation_tenths,
                elevation_num=elevation_num,
            )
        )
        return str(path)

    def test_single_file(self, tmp_path):
        f = self._tilt_file(tmp_path, "N0B", 5, 1)
        dtree = open_nexradlevel3_datatree(f)
        assert isinstance(dtree, DataTree)
        assert "sweep_0" in dtree.children
        root = dtree.to_dataset()
        assert root["sweep_fixed_angle"].values[0] == pytest.approx(0.5)
        for coord in ("latitude", "longitude", "altitude"):
            assert coord in root.coords

    def test_multi_tilt_sorted(self, tmp_path):
        files = [
            self._tilt_file(tmp_path, "N1B", 9, 3),
            self._tilt_file(tmp_path, "N0B", 5, 1),
        ]
        dtree = open_nexradlevel3_datatree(files)
        root = dtree.to_dataset()
        np.testing.assert_allclose(root["sweep_fixed_angle"].values, [0.5, 0.9])
        assert dtree["sweep_0"]["sweep_number"].values.item() == 0
        assert dtree["sweep_1"]["sweep_number"].values.item() == 1
        assert dtree["sweep_0"]["sweep_fixed_angle"].values.item() == pytest.approx(0.5)

    def test_mixed_products_rejected(self, tmp_path):
        files = [
            self._tilt_file(tmp_path, "N0B", 5, 1, msg_code=153),
            self._tilt_file(tmp_path, "N0X", 5, 1, msg_code=159),
        ]
        with pytest.raises(ValueError, match="different Level 3 products"):
            open_nexradlevel3_datatree(files)

    def test_empty_list_rejected(self):
        with pytest.raises(ValueError, match="at least one file"):
            open_nexradlevel3_datatree([])


def _mutate(buf, offset, data):
    """Overwrite bytes at a fixed offset (header fields have fixed positions
    when the builder's text header carries no padding)."""
    return buf[:offset] + data + buf[offset + len(data) :]


class TestMalformedAndEdges:
    # absolute offsets in a pad-free synthetic file:
    # PDB divider 48, latitude 50, product_code 60, uncompressed size 132,
    # symbology offset 138, symbology divider 150, packet code 166,
    # packet nbins 170

    def test_pdb_divider_invalid(self):
        buf = _mutate(build_level3_file(), 48, struct.pack(">h", 0))
        with pytest.raises(ValueError, match="divider"):
            NEXRADLevel3File(io.BytesIO(buf))

    def test_product_code_mismatch_warns(self):
        buf = _mutate(build_level3_file(), 60, struct.pack(">h", 154))
        with pytest.warns(UserWarning, match="disagree"):
            NEXRADLevel3File(io.BytesIO(buf))

    def test_site_out_of_bounds_warns(self):
        buf = _mutate(build_level3_file(), 50, struct.pack(">i", 95000000))
        with pytest.warns(UserWarning, match="out of bounds"):
            NEXRADLevel3File(io.BytesIO(buf))

    def test_bad_symbology_header_warns(self):
        buf = _mutate(build_level3_file(), 150, struct.pack(">h", 0))
        with pytest.warns(UserWarning, match="symbology block header"):
            NEXRADLevel3File(io.BytesIO(buf))

    def test_unsupported_packet_code_raises(self):
        buf = _mutate(build_level3_file(), 166, struct.pack(">h", 6))
        with pytest.raises(NotImplementedError, match="packet code 6"):
            NEXRADLevel3File(io.BytesIO(buf))

    def test_uncompressed_size_mismatch_warns(self):
        buf = _mutate(build_level3_file(compress=True), 132, struct.pack(">I", 7))
        with pytest.warns(UserWarning, match="does not match"):
            NEXRADLevel3File(io.BytesIO(buf))

    def test_symbology_offset_override(self):
        # move the symbology block 4 bytes later and point the PDB at it
        buf = build_level3_file()
        moved = buf[:150] + b"\x00" * 4 + buf[150:]
        moved = _mutate(moved, 138, struct.pack(">i", 62))
        f = NEXRADLevel3File(io.BytesIO(moved))
        assert f.raw_data.shape == (4, 8)

    def test_packet16_nbins_from_first_radial_nbytes(self):
        # header claims 10 bins; the per-radial byte count (8) wins
        buf = _mutate(build_level3_file(), 170, struct.pack(">h", 10))
        f = NEXRADLevel3File(io.BytesIO(buf))
        assert f.raw_data.shape == (4, 8)

    def test_packet16_ragged_radials(self):
        packet = struct.pack(">7h", 16, 0, 8, 256, 280, 1000, 2)
        packet += struct.pack(">3h", 8, 0, 5) + bytes(range(8))
        packet += struct.pack(">3h", 6, 900, 5) + bytes(range(6))
        f = NEXRADLevel3File(io.BytesIO(build_level3_file(packet=packet)))
        assert f.raw_data.shape == (2, 8)
        np.testing.assert_array_equal(f.raw_data[1], [0, 1, 2, 3, 4, 5, 0, 0])
        assert f.get_azimuth()[1] == pytest.approx(90.0)

    def test_af1f_bad_run_sum_raises(self):
        rle = [bytes([0x11, 0x11])] * 2  # expands to 2 bins, header says 6
        buf = build_level3_file(msg_code=19, packet=_radial_packet_af1f(2, 6, rle))
        with pytest.raises(ValueError, match="expand"):
            NEXRADLevel3File(io.BytesIO(buf))

    def test_truncated_xdr_raises(self):
        data = np.tile(np.array([1, 2], dtype="u2"), (2, 1))
        packet = _generic_packet28(2, 2, data)
        buf = build_level3_file(msg_code=176, packet=packet[: len(packet) - 12])
        with pytest.raises(ValueError, match="XDR"):
            NEXRADLevel3File(io.BytesIO(buf))

    def test_xdr_unknown_component_raises(self):
        data = np.tile(np.array([1, 2], dtype="u2"), (2, 1))
        packet = _generic_packet28(2, 2, data, component_code=4)
        buf = build_level3_file(msg_code=176, packet=packet)
        with pytest.raises(NotImplementedError, match="XDR component"):
            NEXRADLevel3File(io.BytesIO(buf))

    def test_xdr_parameters_parsed(self):
        data = np.tile(np.array([1, 2], dtype="u2"), (2, 1))
        packet = _generic_packet28(
            2, 2, data, parameters=[("lag", "1"), ("mode", "precip")]
        )
        buf = build_level3_file(msg_code=176, packet=packet)
        f = NEXRADLevel3File(io.BytesIO(buf))
        assert f.gen_data_pack["parameters"] == [("lag", "1"), ("mode", "precip")]

    def test_xdr_single_parameter_collapses(self):
        data = np.tile(np.array([1, 2], dtype="u2"), (2, 1))
        packet = _generic_packet28(2, 2, data, parameters=[("lag", "1")])
        buf = build_level3_file(msg_code=176, packet=packet)
        f = NEXRADLevel3File(io.BytesIO(buf))
        assert f.gen_data_pack["parameters"] == ("lag", "1")

    def test_xdr_multiple_components(self):
        from xradar.io.backends.nexrad_level3 import _Level3XDRParser

        data = np.tile(np.array([1, 2], dtype="u2"), (2, 1))
        packet = _generic_packet28(2, 2, data, ncomponents=2)
        xdr = _Level3XDRParser(packet[8:])()
        assert len(xdr["components"]) == 2
        assert len(xdr["components"][0].radials) == 2

    def test_context_manager(self):
        with NEXRADLevel3File(build_level3_file()) as f:
            assert f.raw_data.shape == (4, 8)

    def test_surface_product_range_and_elevation(self):
        raw = np.tile(np.array([0, 1, 50], dtype="u1"), (4, 1))
        buf = build_level3_file(
            msg_code=170,
            packet=_radial_packet16(4, 3, raw),
            threshold=_flag_threshold(2.0, 0.0, leading=1),
        )
        f = NEXRADLevel3File(io.BytesIO(buf))
        assert f.get_elevation() == 0.0
        rng = f.get_range()
        # bin_size is None for surface products: packet range scale is exact
        assert rng[1] - rng[0] == pytest.approx(1000.0)

    def test_class_product_has_no_scale_offset(self):
        raw = np.tile(np.array([0, 10], dtype="u1"), (4, 1))
        buf = build_level3_file(msg_code=165, packet=_radial_packet16(4, 2, raw))
        f = NEXRADLevel3File(io.BytesIO(buf))
        assert f.get_scale_offset() is None

    def test_instrument_name_fallback(self, tmp_path):
        buf = _mutate(build_level3_file(), 7, b"K LT ")
        path = tmp_path / "weird_header"
        path.write_bytes(buf)
        ds = xr.open_dataset(str(path), engine="nexradlevel3")
        assert "instrument_name" not in ds.attrs

    def test_range_folded_mask_variable(self, tmp_path):
        raw = np.tile(np.array([0, 1, 100], dtype="u1"), (4, 1))
        path = tmp_path / "rf_file"
        path.write_bytes(
            build_level3_file(msg_code=153, packet=_radial_packet16(4, 3, raw))
        )
        ds = xr.open_dataset(str(path), engine="nexradlevel3")
        assert "DBZH_range_folded" in ds.data_vars
        assert ds["DBZH_range_folded"].values[:, 1].all()
        assert "range-folded" in ds["DBZH"].attrs["comment"]

    def test_hclass_flag_attrs(self, tmp_path):
        raw = np.tile(np.array([0, 60], dtype="u1"), (4, 1))
        path = tmp_path / "hclass_file"
        path.write_bytes(
            build_level3_file(msg_code=165, packet=_radial_packet16(4, 2, raw))
        )
        ds = xr.open_dataset(str(path), engine="nexradlevel3")
        assert ds["HCLASS"].attrs["flag_values"][0] == 10
        assert "biological" in ds["HCLASS"].attrs["flag_meanings"]

    def test_mask_and_scale_false_float_product(self, tmp_path):
        raw = np.tile(np.array([0, 1, 130], dtype="u1"), (4, 1))
        path = tmp_path / "zdr_raw"
        path.write_bytes(
            build_level3_file(
                msg_code=159,
                packet=_radial_packet16(4, 3, raw),
                threshold=_flag_threshold(
                    16.0, 128.0, max_val=255, leading=2, trailing=1
                ),
            )
        )
        ds = xr.open_dataset(str(path), engine="nexradlevel3", mask_and_scale=False)
        attrs = ds["ZDR"].attrs
        assert attrs["valid_min"] == 2
        assert attrs["valid_max"] == 254
        assert attrs["range_folded_raw_value"] == 1

    def test_reindex_angle_synthetic(self, tmp_path):
        path = tmp_path / "reindex_file"
        path.write_bytes(build_level3_file())
        ds = xr.open_dataset(
            str(path),
            engine="nexradlevel3",
            reindex_angle=dict(
                start_angle=0, stop_angle=360, angle_res=90.0, direction=1
            ),
        )
        assert ds.sizes["azimuth"] == 4
        assert not np.isnat(ds["time"].values).any()

    def test_drop_variables(self, tmp_path):
        path = tmp_path / "drop_file"
        path.write_bytes(build_level3_file())
        ds = xr.open_dataset(str(path), engine="nexradlevel3", drop_variables=["DBZH"])
        assert "DBZH" not in ds.data_vars


class TestRealFiles:
    """Integration tests against real LOT files from open-radar-data.

    These skip until the Level 3 samples land in an open-radar-data
    release; locally, point XRADAR_L3_TEST_DATA at a directory holding
    the raw files to run them.
    """

    @pytest.fixture
    def real_file(self, request):
        import os

        name = request.param
        local_dir = os.environ.get("XRADAR_L3_TEST_DATA")
        if local_dir and os.path.exists(os.path.join(local_dir, name)):
            return os.path.join(local_dir, name)
        from open_radar_data import DATASETS

        try:
            return DATASETS.fetch(name)
        except ValueError as err:
            if "not in the registry" in str(err) or "doesn't exist" in str(err):
                pytest.skip(f"Level 3 sample {name} not in open-radar-data yet")
            raise  # hash mismatch / registry typo must FAIL, not skip

    # Pinned from the 2026-07-17 19:30 LOT scene; decoded values
    # cross-checked against Py-ART 2.2.5 / MetPy 1.7.1 (bit-identical
    # after the documented unit conversions).
    @pytest.mark.parametrize(
        "real_file, moment, shape, spacing, rng0, vmin, vmax, nvalid",
        [
            (
                "LOT_N0B_2026_07_17_19_30_15",
                "DBZH",
                (720, 1840),
                250.0,
                125.0,
                -26.5,
                62.0,
                250509,
            ),
            (
                "LOT_N1B_2026_07_17_19_30_15",
                "DBZH",
                (720, 1688),
                250.0,
                125.0,
                -29.0,
                61.5,
                194118,
            ),
            (
                "LOT_N0G_2026_07_17_19_30_15",
                "VRADH",
                (720, 1200),
                250.0,
                125.0,
                -42.0,
                56.5,
                183462,
            ),
            (
                "LOT_N0S_2026_07_17_19_30_15",
                "SRMV",
                (360, 230),
                1000.0,
                500.0,
                -28.29,
                28.29,
                27273,
            ),
            (
                "LOT_N0X_2026_07_17_19_30_15",
                "ZDR",
                (360, 1200),
                250.0,
                125.0,
                -7.88,
                7.94,
                135392,
            ),
            (
                "LOT_N0C_2026_07_17_19_30_15",
                "RHOHV",
                (360, 1200),
                250.0,
                125.0,
                0.21,
                1.05,
                135392,
            ),
            (
                "LOT_N0H_2026_07_17_19_30_15",
                "HCLASS",
                (360, 1200),
                250.0,
                125.0,
                10.0,
                140.0,
                111201,
            ),
            (
                "LOT_HHC_2026_07_17_19_30_15",
                "HCLASS",
                (360, 920),
                250.0,
                125.0,
                10.0,
                100.0,
                97476,
            ),
            (
                "LOT_DAA_2026_07_17_19_30_15",
                "ACCUM",
                (360, 920),
                250.0,
                125.0,
                0.03,
                51.41,
                73628,
            ),
            (
                "LOT_DPR_2026_07_17_19_30_15",
                "RATE",
                (360, 920),
                250.0,
                125.0,
                0.0,
                200.0,
                331200,
            ),
        ],
        indirect=["real_file"],
    )
    def test_open_real_product(
        self, real_file, moment, shape, spacing, rng0, vmin, vmax, nvalid
    ):
        ds = xr.open_dataset(real_file, engine="nexradlevel3")
        assert moment in ds.data_vars
        assert ds["latitude"].values.item() == pytest.approx(41.604, abs=0.001)
        assert ds["longitude"].values.item() == pytest.approx(-88.085, abs=0.001)
        values = ds[moment].values
        assert values.shape == shape
        rng = ds["range"].values
        assert rng[1] - rng[0] == pytest.approx(spacing)
        assert rng[0] == pytest.approx(rng0)
        assert np.nanmin(values) == pytest.approx(vmin, abs=0.01)
        assert np.nanmax(values) == pytest.approx(vmax, abs=0.01)
        assert int(np.isfinite(values).sum()) == nvalid

    @pytest.mark.parametrize(
        "real_file", ["LOT_N0B_2026_07_17_19_30_15"], indirect=["real_file"]
    )
    def test_real_multi_tilt_datatree(self, real_file):
        import os

        base = os.path.dirname(real_file)
        tilts = [os.path.join(base, f"LOT_N{i}B_2026_07_17_19_30_15") for i in range(4)]
        if not all(os.path.exists(t) for t in tilts):
            pytest.skip("full N0B-N3B tilt set not available")
        dtree = open_nexradlevel3_datatree(tilts)
        assert list(dtree.children) == [f"sweep_{i}" for i in range(4)]
        np.testing.assert_allclose(
            dtree.to_dataset()["sweep_fixed_angle"].values, [0.5, 1.3, 2.4, 3.1]
        )
