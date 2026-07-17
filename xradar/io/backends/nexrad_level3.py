#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
NEXRAD Level 3
==============

This sub-module contains the xarray backend for reading NEXRAD Level 3 (NIDS)
radial products. Each Level 3 file holds a single product on a single
elevation cut; a multi-tilt volume of the same product (e.g. N0B/N1B/N2B/N3B
super-resolution reflectivity) can be assembled from a list of files.

Real-time and archived Level 3 files are available on AWS at
``s3://unidata-nexrad-level3/`` with keys like
``LOT_N0B_2026_07_17_19_11_15`` (site without leading K, product code,
timestamp).

Supported are the radial packet formats: digital radial data array
(packet code 16), run-length encoded radials (packet code AF1F) and the
generic data packet (packet code 28, e.g. DPR/HHC). Raster, graphic and
tabular products are out of scope.

Example::

    import xarray as xr
    import xradar as xd

    # Single product/sweep via xarray engine
    ds = xr.open_dataset("LOT_N0B_2026_07_17_19_11_15", engine="nexradlevel3")

    # Single-sweep DataTree
    dtree = xd.io.open_nexradlevel3_datatree("LOT_N0B_2026_07_17_19_11_15")

    # Multi-tilt volume DataTree (same product, different cuts)
    dtree = xd.io.open_nexradlevel3_datatree(
        ["LOT_N0B_...", "LOT_N1B_...", "LOT_N2B_...", "LOT_N3B_..."]
    )

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = [
    "NexradLevel3BackendEntrypoint",
    "open_nexradlevel3_datatree",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import bz2
import io
import struct
import warnings
from collections import namedtuple
from datetime import datetime, timedelta, timezone

import numpy as np
import xarray as xr
from xarray import DataTree
from xarray.backends.common import BackendEntrypoint

from ... import util
from ...model import (
    get_altitude_attrs,
    get_azimuth_attrs,
    get_elevation_attrs,
    get_latitude_attrs,
    get_longitude_attrs,
    get_range_attrs,
    get_time_attrs,
    moment_attrs,
    sweep_vars_mapping,
)
from .common import _STATION_VARS, _apply_site_as_coords

#: 1 international knot in m/s. Legacy 16-level velocity products (25/27,
#: 28/30, 56) carry data levels in knots; they are converted so VRADH /
#: WRADH / SRMV are uniformly m/s (deliberate divergence from Py-ART,
#: which returns the raw knot values).
KT_TO_MS = 0.514444

#: 1 inch in mm. Accumulation products are converted so ACCUM is mm.
IN_TO_MM = 25.4

#: Structure of a Level 3 product specification.
#:
#: - ``moment``: CfRadial2 moment name the product maps to
#: - ``decode``: decode scheme (``linear_hw``, ``float_scale``, ``precip``,
#:   ``legacy16``, ``class_int``)
#: - ``bin_size``: range bin size in meters (the ICD product resolution).
#:   ``None`` uses the radial-packet range scale field directly; for
#:   elevation-bearing products that field is ``floor(1000*cos(elevation))``
#:   (a truncated ground-projection value, verified on live files), so those
#:   products carry the exact ICD resolution here instead.
#: - ``has_elevation``: whether PDB halfword 30 holds the elevation angle
#: - ``post_scale``: factor applied after decoding (unit conversions)
ProductSpec = namedtuple(
    "ProductSpec", ["moment", "decode", "bin_size", "has_elevation", "post_scale"]
)

#: Message code -> product specification for the radial products supported
#: by this backend. Range multipliers follow the ICD product resolutions
#: (Table III / "Products with Version Numbers", ICD 2620001).
PRODUCT_TABLE = {
    # legacy 8/16-level run-length encoded products
    19: ProductSpec("DBZH", "legacy16", 1000.0, True, 1.0),
    20: ProductSpec("DBZH", "legacy16", 2000.0, True, 1.0),
    25: ProductSpec("VRADH", "legacy16", 250.0, True, KT_TO_MS),
    27: ProductSpec("VRADH", "legacy16", 1000.0, True, KT_TO_MS),
    28: ProductSpec("WRADH", "legacy16", 250.0, True, KT_TO_MS),
    30: ProductSpec("WRADH", "legacy16", 1000.0, True, KT_TO_MS),
    56: ProductSpec("SRMV", "legacy16", 1000.0, True, KT_TO_MS),
    78: ProductSpec("ACCUM", "legacy16", None, False, IN_TO_MM),
    79: ProductSpec("ACCUM", "legacy16", None, False, IN_TO_MM),
    80: ProductSpec("ACCUM", "legacy16", None, False, IN_TO_MM),
    # digital products, linear halfword scale/offset
    94: ProductSpec("DBZH", "linear_hw", 1000.0, True, 1.0),
    99: ProductSpec("VRADH", "linear_hw", 250.0, True, 1.0),
    153: ProductSpec("DBZH", "linear_hw", 250.0, True, 1.0),
    154: ProductSpec("VRADH", "linear_hw", 250.0, True, 1.0),
    155: ProductSpec("WRADH", "linear_hw", 250.0, True, 1.0),
    # digital dual-pol products, float scale/offset
    159: ProductSpec("ZDR", "float_scale", 250.0, True, 1.0),
    161: ProductSpec("RHOHV", "float_scale", 250.0, True, 1.0),
    163: ProductSpec("KDP", "float_scale", 250.0, True, 1.0),
    165: ProductSpec("HCLASS", "class_int", 250.0, True, 1.0),
    # digital precipitation products (surface: packet range scale is exact)
    170: ProductSpec("ACCUM", "precip", None, False, 1.0),
    172: ProductSpec("ACCUM", "precip", None, False, 1.0),
    173: ProductSpec("ACCUM", "precip", None, False, 1.0),
    174: ProductSpec("ACCUM", "precip", None, False, 1.0),
    175: ProductSpec("ACCUM", "precip", None, False, 1.0),
    # generic data packet products (XDR; gate geometry from the component)
    176: ProductSpec("RATE", "rate", None, False, 1.0),
    177: ProductSpec("HCLASS", "class_int", 250.0, False, 1.0),
}

#: Radial products not yet implemented; mapped to a descriptive name so the
#: error message can point users at the tracking issue.
DEFERRED_PRODUCTS = {
    32: "Digital Hybrid Scan Reflectivity",
    34: "Clutter Filter Control",
    134: "High Resolution VIL",
    135: "Enhanced Echo Tops",
    138: "Digital Storm Total Precipitation",
    169: "One Hour Accumulation",
    171: "Storm Total Accumulation",
    180: "TDWR Base Reflectivity",
    181: "TDWR Base Reflectivity",
    182: "TDWR Base Velocity",
    186: "TDWR Base Reflectivity",
}

#: Maximum product version implemented, per the ICD "Products with Version
#: Numbers" table. Newer versions trigger a warning, not an error.
SUPPORTED_VERSION_NUMBERS = {
    19: 0,
    20: 0,
    25: 0,
    27: 0,
    28: 0,
    30: 0,
    56: 0,
    78: 1,
    79: 1,
    80: 1,
    94: 0,
    99: 0,
    153: 0,
    154: 0,
    155: 0,
    159: 1,
    161: 1,
    163: 0,
    165: 1,
    170: 0,
    172: 1,
    173: 0,
    174: 0,
    175: 0,
    176: 0,
    177: 0,
}

#: Hydrometeor classification data levels (ICD Figure 3-6, page 3-37).
HCLASS_FLAG_VALUES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 140, 150]
HCLASS_FLAG_MEANINGS = (
    "biological anomalous_propagation_ground_clutter ice_crystals dry_snow "
    "wet_snow light_moderate_rain heavy_rain big_drops graupel small_hail "
    "large_hail giant_hail unknown range_folded"
)

#: Message Header Block, 18 bytes (ICD Figure 3-3).
MESSAGE_HEADER = (
    ("code", "h"),
    ("date", "h"),
    ("time", "i"),
    ("length", "i"),
    ("source", "h"),
    ("dest", "h"),
    ("nblocks", "h"),
)

#: Product Description Block, 102 bytes (ICD Figure 3-6).
PRODUCT_DESCRIPTION = (
    ("divider", "h"),
    ("latitude", "i"),
    ("longitude", "i"),
    ("height", "h"),
    ("product_code", "h"),
    ("operational_mode", "h"),
    ("vcp", "h"),
    ("sequence_num", "h"),
    ("vol_scan_num", "h"),
    ("vol_scan_date", "h"),
    ("vol_scan_time", "i"),
    ("product_date", "h"),
    ("product_time", "i"),
    ("halfwords_27_28", "4s"),
    ("elevation_num", "h"),
    ("halfword_30", "2s"),
    ("threshold_data", "32s"),
    ("halfwords_47_53", "14s"),
    ("version", "B"),
    ("spot_blank", "B"),
    ("offset_symbology", "i"),
    ("offset_graphic", "i"),
    ("offset_tabular", "i"),
)

#: Product Symbology Block header, 16 bytes (ICD Figure 3-6 Sheet 8).
SYMBOLOGY_HEADER = (
    ("divider", "h"),
    ("id", "h"),
    ("block_length", "i"),
    ("layers", "h"),
    ("layer_divider", "h"),
    ("layer_length", "i"),
)

#: Radial packet header for packet codes 16 and AF1F, 14 bytes
#: (ICD Figures 3-10 and 3-11c).
RADIAL_PACKET_HEADER = (
    ("packet_code", "h"),
    ("first_bin", "h"),
    ("nbins", "h"),
    ("i_sweep_center", "h"),
    ("j_sweep_center", "h"),
    ("range_scale", "h"),
    ("nradials", "h"),
)

#: Generic data packet header, 8 bytes (ICD Figure 3-15c).
GEN_DATA_PACK_HEADER = (
    ("packet_code", "h"),
    ("reserved", "h"),
    ("num_bytes", "i"),
)

AF1F = -20705  # 0xAF1F as signed big-endian int16
SUPPORTED_PACKET_CODES = [16, AF1F, 28]

OPERATIONAL_MODES = {0: "maintenance", 1: "clear-air", 2: "precipitation"}


def _unpack_from_buf(buf, pos, structure):
    """Unpack a big-endian structure from a buffer at the given position."""
    fmt = ">" + "".join([f for _, f in structure])
    values = struct.unpack_from(fmt, buf, pos)
    return dict(zip([name for name, _ in structure], values))


def _datetime_from_mdate_mtime(mdate, mtime):
    """Return a datetime for a modified Julian date / seconds pair.

    ``mdate`` counts days since 1970-01-01 with day 1 being the epoch
    itself, hence the ``mdate - 1``.
    """
    epoch = datetime.fromtimestamp(0, tz=timezone.utc).replace(tzinfo=None)
    return epoch + timedelta(days=int(mdate) - 1, seconds=int(mtime))


class _XDRUnpacker:
    """Minimal XDR unpacker covering the subset used by packet code 28.

    Replaces the ``mda_xdrlib`` dependency used by Py-ART/MetPy.
    """

    def __init__(self, data):
        self._data = data
        self._pos = 0

    def unpack_int(self):
        (value,) = struct.unpack_from(">i", self._data, self._pos)
        self._pos += 4
        return value

    def unpack_uint(self):
        (value,) = struct.unpack_from(">I", self._data, self._pos)
        self._pos += 4
        return value

    def unpack_float(self):
        (value,) = struct.unpack_from(">f", self._data, self._pos)
        self._pos += 4
        return value

    def unpack_string(self):
        nbytes = self.unpack_uint()
        padded = (nbytes + 3) & ~3
        value = self._data[self._pos : self._pos + nbytes]
        self._pos += padded
        return value.decode("ascii", errors="replace")

    def unpack_int_array(self):
        """Bulk-decode an XDR int array (vectorized fast path)."""
        nitems = self.unpack_uint()
        values = np.frombuffer(self._data, ">i4", count=nitems, offset=self._pos)
        self._pos += 4 * nitems
        return values


_RadialComponent = namedtuple(
    "_RadialComponent",
    ["description", "gate_width", "first_gate", "parameters", "radials"],
)
_RadialData = namedtuple(
    "_RadialData",
    ["azimuth", "elevation", "width", "num_bins", "attributes", "data"],
)


class _Level3XDRParser(_XDRUnpacker):
    """Parse the XDR-encoded product description of packet code 28.

    Field order follows MetPy/Py-ART; the ICD misdocuments several fields
    (int2 vs int4, garbage list "pointers") which this layout accounts for.
    """

    def __call__(self):
        xdr = {}
        xdr["name"] = self.unpack_string()
        xdr["description"] = self.unpack_string()
        xdr["code"] = self.unpack_int()
        xdr["type"] = self.unpack_int()
        xdr["prod_time"] = self.unpack_uint()
        xdr["radar_name"] = self.unpack_string()
        xdr["latitude"] = self.unpack_float()
        xdr["longitude"] = self.unpack_float()
        xdr["height"] = self.unpack_float()
        xdr["vol_time"] = self.unpack_uint()
        xdr["el_time"] = self.unpack_uint()
        xdr["el_angle"] = self.unpack_float()
        xdr["vol_num"] = self.unpack_int()
        xdr["op_mode"] = self.unpack_int()
        xdr["vcp_num"] = self.unpack_int()
        xdr["el_num"] = self.unpack_int()
        xdr["compression"] = self.unpack_int()
        xdr["uncompressed_size"] = self.unpack_int()
        xdr["parameters"] = self._unpack_parameters()
        xdr["components"] = self._unpack_components()
        return xdr

    def _unpack_parameters(self):
        num = self.unpack_int()
        self.unpack_int()  # documented "pointer", unused garbage
        if num == 0:
            return None
        ret = []
        for i in range(num):
            ret.append((self.unpack_string(), self.unpack_string()))
            if i < num - 1:
                self.unpack_int()  # list "pointer"
        if num == 1:
            ret = ret[0]
        return ret

    def _unpack_components(self):
        num = self.unpack_int()
        self.unpack_int()  # documented "pointer", unused garbage
        ret = []
        for i in range(num):
            code = self.unpack_int()
            if code == 1:
                ret.append(self._unpack_radial())
            else:
                raise NotImplementedError(f"Unknown XDR component: {code}")
            if i < num - 1:
                self.unpack_int()  # list "pointer"
        if num == 1:
            ret = ret[0]
        return ret

    def _unpack_radial(self):
        component = _RadialComponent(
            description=self.unpack_string(),
            gate_width=self.unpack_float(),
            first_gate=self.unpack_float(),
            parameters=self._unpack_parameters(),
            radials=None,
        )
        num_rads = self.unpack_int()
        rads = []
        for _ in range(num_rads):
            rads.append(
                _RadialData(
                    azimuth=self.unpack_float(),
                    elevation=self.unpack_float(),
                    width=self.unpack_float(),
                    num_bins=self.unpack_int(),
                    attributes=self.unpack_string(),
                    data=self.unpack_int_array(),
                )
            )
        return component._replace(radials=rads)


class NEXRADLevel3File:
    """Read a NEXRAD Level 3 (NIDS) radial product file.

    Parameters
    ----------
    filename_or_obj : str, Path, file-like or bytes
        Level 3 file to read. bz2-compressed symbology blocks are
        decompressed transparently.

    Attributes
    ----------
    text_header : bytes
        WMO/AWIPS text header.
    msg_header : dict
        Message Header Block.
    prod_descr : dict
        Product Description Block.
    packet_header : dict
        Radial packet header.
    raw_data : :class:`numpy:numpy.ndarray`
        Raw unscaled data levels, shape (nradials, nbins).
    """

    def __init__(self, filename_or_obj):
        self._fh = None
        if isinstance(filename_or_obj, bytes):
            filename_or_obj = io.BytesIO(filename_or_obj)
        if hasattr(filename_or_obj, "read"):
            fh = filename_or_obj
        else:
            fh = open(filename_or_obj, "rb")
        self._fh = fh
        buf = fh.read()

        pad = buf.find(b"SDUS", 0, 80)
        if pad == -1:
            raise ValueError("Not a valid NEXRAD Level 3 file.")
        self.text_header = buf[: pad + 30]
        bpos = pad + 30

        self.msg_header = _unpack_from_buf(buf, bpos, MESSAGE_HEADER)
        code = self.msg_header["code"]
        if code in DEFERRED_PRODUCTS:
            raise NotImplementedError(
                f"NEXRAD Level 3 product {code} "
                f"({DEFERRED_PRODUCTS[code]}) is not implemented yet "
                "(see openradar/xradar tracking issue for Level 3 products)."
            )
        if code not in PRODUCT_TABLE:
            raise NotImplementedError(
                f"NEXRAD Level 3 message code {code} is not a supported "
                "radial product."
            )
        self.product_spec = PRODUCT_TABLE[code]

        self.prod_descr = _unpack_from_buf(buf, bpos + 18, PRODUCT_DESCRIPTION)
        self._validate_headers()

        version = self.prod_descr["version"]
        max_version = SUPPORTED_VERSION_NUMBERS.get(code, 0)
        if version > max_version:
            warnings.warn(
                f"Product version {version} is newer than the latest "
                f"implemented version {max_version} for message code "
                f"{code}; decoded values may be wrong.",
                UserWarning,
                stacklevel=2,
            )

        # The symbology block starts right after the PDB; cross-check with
        # the PDB's halfword offset (counted from the message header).
        bpos = pad + 150
        offset_symbology = self.prod_descr["offset_symbology"]
        if offset_symbology > 0 and pad + 30 + offset_symbology * 2 != bpos:
            bpos = pad + 30 + offset_symbology * 2

        if buf[bpos : bpos + 2] == b"BZ":
            try:
                buf2 = bz2.decompress(buf[bpos:])
            except OSError as err:
                raise ValueError(
                    "Failed to decompress the symbology block; file is "
                    "corrupt or truncated."
                ) from err
            self._check_uncompressed_size(len(buf2))
        else:
            buf2 = buf[bpos:]

        self.symbology_header = _unpack_from_buf(buf2, 0, SYMBOLOGY_HEADER)
        if self.symbology_header["divider"] != -1 or self.symbology_header["id"] != 1:
            warnings.warn(
                "Unexpected symbology block header; file may be malformed.",
                UserWarning,
                stacklevel=2,
            )

        (packet_code,) = struct.unpack_from(">h", buf2, 16)
        if packet_code not in SUPPORTED_PACKET_CODES:
            raise NotImplementedError(
                f"Unsupported symbology packet code {packet_code}."
            )
        if packet_code == 28:
            self._read_generic_packet(buf2)
        else:
            self._read_radial_packet(buf2, packet_code)

    def close(self):
        """Close the underlying file object."""
        if self._fh is not None:
            self._fh.close()

    __del__ = close

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def _validate_headers(self):
        pd = self.prod_descr
        if pd["divider"] != -1:
            raise ValueError(
                "Invalid product description block (divider != -1); "
                "not a NEXRAD Level 3 file?"
            )
        if pd["product_code"] != self.msg_header["code"]:
            warnings.warn(
                f"Message code {self.msg_header['code']} and product code "
                f"{pd['product_code']} disagree; using message code.",
                UserWarning,
                stacklevel=3,
            )
        latitude = pd["latitude"] * 0.001
        longitude = pd["longitude"] * 0.001
        if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
            warnings.warn(
                f"Radar site coordinates ({latitude}, {longitude}) out of "
                "bounds; file may be malformed.",
                UserWarning,
                stacklevel=3,
            )

    def _check_uncompressed_size(self, actual):
        # halfwords 47-53: hw51 flags compression, hw52-53 hold the
        # uncompressed size of the symbology block and beyond.
        hw = self.prod_descr["halfwords_47_53"]
        (compression_flag,) = struct.unpack_from(">h", hw, 8)
        (uncompressed_size,) = struct.unpack_from(">I", hw, 10)
        if compression_flag == 1 and uncompressed_size not in (0, actual):
            warnings.warn(
                f"Uncompressed symbology size {actual} does not match the "
                f"product description block value {uncompressed_size}.",
                UserWarning,
                stacklevel=3,
            )

    def _read_radial_packet(self, buf2, packet_code):
        self.packet_header = _unpack_from_buf(buf2, 16, RADIAL_PACKET_HEADER)
        nradials = self.packet_header["nradials"]
        nbins = self.packet_header["nbins"]
        if nradials < 1 or nbins < 1:
            raise ValueError(
                f"Radial packet reports {nradials} radials x {nbins} bins; "
                "file is corrupt or empty."
            )

        try:
            if packet_code == 16:
                (nbytes0,) = struct.unpack_from(">h", buf2, 30)
                if nbytes0 != nbins:
                    # occasionally the header nbins disagrees with the
                    # per-radial byte count; the byte count wins
                    nbins = nbytes0
                self._read_packet16(buf2, nradials, nbins)
            else:
                self._read_packet_af1f(buf2, nradials, nbins)
        except (struct.error, ValueError) as err:
            if isinstance(err, ValueError) and "expand" in str(err):
                raise
            raise ValueError("Truncated or corrupt radial packet data.") from err

    def _read_packet16(self, buf2, nradials, nbins):
        radial_dtype = np.dtype(
            [
                ("nbytes", ">i2"),
                ("angle_start", ">i2"),
                ("angle_delta", ">i2"),
                ("data", "u1", (nbins,)),
            ]
        )
        end = 30 + nradials * radial_dtype.itemsize
        if len(buf2) >= end:
            radials = np.frombuffer(buf2, radial_dtype, count=nradials, offset=30)
            if (radials["nbytes"] == nbins).all():
                self.raw_data = radials["data"].copy()
                self._angle_start = radials["angle_start"] * 0.1
                self._angle_delta = radials["angle_delta"] * 0.1
                return
        # ragged per-radial byte counts: fall back to a per-radial loop
        raw = np.zeros((nradials, nbins), dtype="u1")
        angle_start = np.empty(nradials)
        angle_delta = np.empty(nradials)
        pos = 30
        for i in range(nradials):
            nbytes, start, delta = struct.unpack_from(">3h", buf2, pos)
            pos += 6
            count = min(nbytes, nbins)
            raw[i, :count] = np.frombuffer(buf2, "u1", count=count, offset=pos)
            pos += nbytes
            angle_start[i] = start * 0.1
            angle_delta[i] = delta * 0.1
        self.raw_data = raw
        self._angle_start = angle_start
        self._angle_delta = angle_delta

    def _read_packet_af1f(self, buf2, nradials, nbins):
        angle_start = np.empty(nradials)
        angle_delta = np.empty(nradials)
        slices = []
        pos = 30
        for i in range(nradials):
            nbytes, start, delta = struct.unpack_from(">3h", buf2, pos)
            pos += 6
            rle_size = nbytes * 2  # nbytes counts halfwords
            slices.append((pos, rle_size))
            pos += rle_size
            angle_start[i] = start * 0.1
            angle_delta[i] = delta * 0.1

        rle = np.concatenate(
            [
                np.frombuffer(buf2, "u1", count=size, offset=start)
                for start, size in slices
            ]
        )
        colors = rle & 0x0F
        runs = rle >> 4
        # per-radial run sums must each equal nbins for a clean reshape
        run_lengths = np.array([size for _, size in slices])
        ends = np.cumsum(run_lengths)
        totals = np.add.reduceat(runs, np.r_[0, ends[:-1]])
        if not (totals == nbins).all():
            raise ValueError(
                "Run-length encoded radial does not expand to the expected "
                f"number of bins (expected {nbins})."
            )
        self.raw_data = np.repeat(colors, runs).reshape(nradials, nbins)
        self._angle_start = angle_start
        self._angle_delta = angle_delta

    def _read_generic_packet(self, buf2):
        self.packet_header = _unpack_from_buf(buf2, 16, GEN_DATA_PACK_HEADER)
        num_bytes = self.packet_header["num_bytes"]
        parser = _Level3XDRParser(buf2[24 : 24 + num_bytes])
        try:
            self.gen_data_pack = parser()
        except struct.error as err:
            raise ValueError("Truncated or corrupt generic data packet (XDR).") from err
        component = self.gen_data_pack["components"]

        radials = component.radials
        nradials = len(radials)
        nbins = radials[0].num_bins
        self.raw_data = np.empty((nradials, nbins), dtype="u2")
        for i, radial in enumerate(radials):
            self.raw_data[i, :] = radial.data
        self._angle_start = np.array([radial.azimuth for radial in radials])
        self._angle_delta = np.array([radial.width for radial in radials])
        self.packet_header["nradials"] = nradials
        self.packet_header["nbins"] = nbins
        # generic radial components carry gate geometry in meters, with
        # first_gate already pointing at the first bin CENTER (verified
        # against live DPR products: gate_width=250.0, first_gate=125.0)
        self.packet_header["first_bin"] = component.first_gate
        self.packet_header["range_scale"] = component.gate_width
        self._range_from_centers = True

    def get_azimuth(self):
        """Return ray start azimuth angles in degrees."""
        return np.asarray(self._angle_start) % 360.0

    def get_azimuth_delta(self):
        """Return per-ray azimuth widths in degrees."""
        return np.asarray(self._angle_delta)

    def get_range(self):
        """Return range bin center distances in meters."""
        nbins = self.raw_data.shape[1]
        if getattr(self, "_range_from_centers", False):
            spacing = self.packet_header["range_scale"]
            first_center = self.packet_header["first_bin"]
            return first_center + np.arange(nbins, dtype="float64") * spacing
        spacing = self.product_spec.bin_size
        if spacing is None:
            spacing = float(self.packet_header["range_scale"])
        # first_bin is the ICD "index of first range bin" (0 in all files
        # observed so far), not a distance
        first_bin = self.packet_header["first_bin"]
        return (first_bin + np.arange(nbins, dtype="float64") + 0.5) * spacing

    def get_location(self):
        """Return radar site (latitude deg, longitude deg, height m)."""
        latitude = self.prod_descr["latitude"] * 0.001
        longitude = self.prod_descr["longitude"] * 0.001
        height = self.prod_descr["height"] * 0.3048  # feet MSL -> meters
        return latitude, longitude, height

    def get_elevation(self):
        """Return the sweep elevation angle in degrees.

        Surface and volume products (accumulations, rates, hybrid
        classifications) have no elevation angle and return 0.0.
        """
        if not self.product_spec.has_elevation:
            return 0.0
        (hw30,) = struct.unpack(">h", self.prod_descr["halfword_30"])
        return hw30 * 0.1

    def get_volume_start_datetime(self):
        """Return the volume scan start datetime."""
        return _datetime_from_mdate_mtime(
            self.prod_descr["vol_scan_date"], self.prod_descr["vol_scan_time"]
        )

    def get_product_datetime(self):
        """Return the product generation datetime."""
        return _datetime_from_mdate_mtime(
            self.prod_descr["product_date"], self.prod_descr["product_time"]
        )

    def get_data_raw(self):
        """Return the raw unscaled data levels."""
        return self.raw_data

    def get_scale_offset(self):
        """Return (scale_factor, add_offset) mapping raw levels to
        physical values, or ``None`` for non-linear decode schemes."""
        spec = self.product_spec
        threshold = self.prod_descr["threshold_data"]
        if spec.decode == "linear_hw":
            hw31, hw32 = struct.unpack(">2h", threshold[:4])
            scale = hw32 / 10.0 * spec.post_scale
            offset = (hw31 / 10.0 - 2.0 * hw32 / 10.0) * spec.post_scale
            return scale, offset
        if spec.decode in ("float_scale", "precip", "rate"):
            scale, offset = struct.unpack(">2f", threshold[:8])
            if scale == 0.0:
                return None
            factor = spec.post_scale
            if spec.decode == "precip":
                factor = 0.01 * IN_TO_MM
            elif spec.decode == "rate":
                factor = IN_TO_MM
            return factor / scale, -offset * factor / scale
        return None

    def get_flag_counts(self):
        """Return (leading_flags, valid_max) for the float-scaled decode
        schemes, from PDB halfwords 36-38 (max data value, number of
        leading/trailing flag levels)."""
        threshold = self.prod_descr["threshold_data"]
        (max_val,) = struct.unpack(">H", threshold[10:12])
        leading, trailing = struct.unpack(">2h", threshold[12:16])
        if not 0 <= leading <= 8 or not 0 <= trailing <= 8 or max_val == 0:
            # implausible flag counts: fall back to the fixed ICD defaults
            leading = 1 if self.product_spec.decode in ("precip", "rate") else 2
            return leading, None
        return leading, max_val - trailing

    def get_data(self):
        """Return decoded physical values as float32 with NaN for
        below-threshold/missing bins."""
        spec = self.product_spec
        raw = self.raw_data

        if spec.decode == "legacy16":
            return self._decode_legacy16()

        if spec.decode == "class_int":
            data = raw.astype("float32")
            data[raw == 0] = np.nan
            return data

        scale_offset = self.get_scale_offset()
        if scale_offset is None:
            warnings.warn(
                "Zero threshold scale; returning all-NaN data.",
                UserWarning,
                stacklevel=2,
            )
            return np.full(raw.shape, np.nan, dtype="float32")
        scale, offset = scale_offset
        data = (raw * scale + offset).astype("float32")
        leading, valid_max = self.get_flag_counts()
        data[raw < leading] = np.nan
        if valid_max is not None:
            data[raw > valid_max] = np.nan
        return data

    def get_range_folded(self):
        """Return a boolean mask of range-folded bins, or ``None`` when
        the product does not encode range folding."""
        if self.product_spec.decode == "linear_hw":
            return self.raw_data == 1
        if self.product_spec.decode in ("float_scale", "precip", "rate"):
            leading, _ = self.get_flag_counts()
            if leading > 1:
                return self.raw_data == 1
        return None

    def _decode_legacy16(self):
        threshold = np.frombuffer(self.prod_descr["threshold_data"], ">B")
        flags = threshold[::2]
        values = threshold[1::2].astype("float64")

        sign = np.where(flags & 0x01, -1.0, 1.0)
        scale = np.ones(16)
        scale[(flags & 0x20) != 0] = 1 / 20.0
        scale[(flags & 0x10) != 0] = 1 / 10.0
        scale[(flags & 0x40) != 0] = 1 / 100.0
        levels = values * sign * scale * self.product_spec.post_scale
        levels[(flags & 0x80) != 0] = np.nan

        return levels[self.raw_data].astype("float32")


def _moment_attributes(moment, mask_and_scale):
    mapping = sweep_vars_mapping.get(moment, {})
    attrs = {key: mapping[key] for key in moment_attrs if key in mapping}
    attrs["coordinates"] = "elevation azimuth range latitude longitude altitude time"
    if moment == "HCLASS" and mask_and_scale:
        attrs["flag_values"] = HCLASS_FLAG_VALUES
        attrs["flag_meanings"] = HCLASS_FLAG_MEANINGS
    return attrs


def _build_l3_sweep(
    fdata,
    mask_and_scale=True,
    decode_times=True,
    first_dim="auto",
    site_as_coords=True,
):
    """Build a CfRadial2 sweep Dataset from a parsed Level 3 file."""
    spec = fdata.product_spec
    code = fdata.msg_header["code"]

    azimuth = (fdata.get_azimuth() + fdata.get_azimuth_delta() / 2.0) % 360.0
    rng = fdata.get_range()
    nradials = azimuth.shape[0]
    fixed_angle = fdata.get_elevation()

    volume_start = fdata.get_volume_start_datetime()
    if decode_times:
        time_values = np.full(
            nradials, np.datetime64(volume_start, "ns"), dtype="datetime64[ns]"
        )
        time_attrs = {"standard_name": "time"}
    else:
        midnight = volume_start.replace(hour=0, minute=0, second=0)
        seconds = (volume_start - midnight).total_seconds()
        time_values = np.full(nradials, seconds)
        time_attrs = get_time_attrs(f"{midnight.isoformat()}Z")

    if mask_and_scale:
        moment_data = fdata.get_data()
        data_attrs = _moment_attributes(spec.moment, mask_and_scale)
        range_folded = fdata.get_range_folded()
        if range_folded is not None and range_folded.any():
            data_attrs["comment"] = (
                "range-folded bins are set to NaN; see the "
                f"{spec.moment}_range_folded mask variable"
            )
    else:
        moment_data = fdata.get_data_raw()
        data_attrs = _moment_attributes(spec.moment, mask_and_scale)
        data_attrs["_FillValue"] = 0
        scale_offset = fdata.get_scale_offset()
        if scale_offset is not None:
            data_attrs["scale_factor"], data_attrs["add_offset"] = scale_offset
            if spec.decode != "linear_hw":
                leading, valid_max = fdata.get_flag_counts()
            else:
                leading, valid_max = 2, None
            # valid_min/max are in packed (raw) units so CF-aware decoders
            # mask flag levels (below-threshold, range-folded) too
            data_attrs["valid_min"] = leading
            if valid_max is not None:
                data_attrs["valid_max"] = valid_max
        if fdata.get_range_folded() is not None:
            data_attrs["range_folded_raw_value"] = 1
        range_folded = None

    latitude, longitude, altitude = fdata.get_location()

    ds = xr.Dataset(
        data_vars={
            spec.moment: (("azimuth", "range"), moment_data, data_attrs),
        },
        coords={
            "azimuth": ("azimuth", azimuth, get_azimuth_attrs(azimuth)),
            "range": ("range", rng, get_range_attrs(rng)),
            "elevation": (
                "azimuth",
                np.full(nradials, fixed_angle),
                get_elevation_attrs(),
            ),
            "time": ("azimuth", time_values, time_attrs),
        },
    )
    if range_folded is not None and mask_and_scale:
        ds[f"{spec.moment}_range_folded"] = xr.DataArray(
            range_folded,
            dims=("azimuth", "range"),
            attrs={
                "long_name": "Range-folded (purple haze) bin mask",
                "standard_name": "range_folded_mask",
            },
        )

    ds["sweep_mode"] = xr.DataArray("azimuth_surveillance")
    ds["sweep_number"] = xr.DataArray(max(fdata.prod_descr["elevation_num"] - 1, 0))
    ds["prt_mode"] = xr.DataArray("not_set")
    ds["follow_mode"] = xr.DataArray("not_set")
    ds["sweep_fixed_angle"] = xr.DataArray(
        fixed_angle, attrs={"long_name": "Fixed angle of sweep", "units": "degrees"}
    )
    ds["latitude"] = xr.DataArray(latitude, attrs=get_latitude_attrs())
    ds["longitude"] = xr.DataArray(longitude, attrs=get_longitude_attrs())
    ds["altitude"] = xr.DataArray(altitude, attrs=get_altitude_attrs())

    volume_iso = volume_start.strftime("%Y-%m-%dT%H:%M:%SZ")
    ds.attrs["time_coverage_start"] = volume_iso
    ds.attrs["time_coverage_end"] = volume_iso
    ds.attrs["nexrad_level3_message_code"] = code
    ds.attrs["nexrad_level3_product_time"] = fdata.get_product_datetime().strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    ds.attrs["vcp"] = fdata.prod_descr["vcp"]
    ds.attrs["operational_mode"] = OPERATIONAL_MODES.get(
        fdata.prod_descr["operational_mode"], "unknown"
    )
    instrument = _instrument_name(fdata.text_header)
    if instrument:
        ds.attrs["instrument_name"] = instrument

    if first_dim == "auto":
        ds = ds.sortby("azimuth")
    else:
        ds = ds.swap_dims({"azimuth": "time"})

    ds = _apply_site_as_coords(ds, site_as_coords)
    return ds


def _instrument_name(text_header):
    """Extract the radar id (e.g. KLOT) from the WMO text header."""
    try:
        parts = text_header.decode("ascii", errors="replace").split()
        if len(parts) >= 2 and len(parts[1]) == 4:
            return parts[1]
    except (IndexError, UnicodeDecodeError):  # pragma: no cover
        pass
    return None


class NexradLevel3BackendEntrypoint(BackendEntrypoint):
    """Xarray BackendEntrypoint for NEXRAD Level 3 (NIDS) radial products.

    Each Level 3 file holds one product on one elevation cut. Open a single
    file with ``xr.open_dataset(file, engine="nexradlevel3")`` to get a
    CfRadial2-compatible sweep :py:class:`xarray.Dataset`. To assemble a
    multi-tilt volume of the same product, use
    :func:`open_nexradlevel3_datatree` with a list of files.

    Keyword Arguments
    -----------------
    mask_and_scale : bool
        If True (default), decode raw data levels into physical values with
        NaN for below-threshold bins. If False, return raw integer levels
        with ``scale_factor``/``add_offset`` attributes where the product
        encoding is linear.
    first_dim : str
        Can be ``time`` or ``auto`` (default). ``auto`` selects ``azimuth``
        as the first dimension, ``time`` swaps the ray dimension to time.
    reindex_angle : bool or dict
        If a dict, kwargs are passed to :func:`xradar.util.reindex_angle`.
        Defaults to ``False``.
    site_as_coords : bool
        If True (default), promote ``latitude``/``longitude``/``altitude``
        to Dataset coordinates.
    """

    description = "Open NEXRAD Level 3 (NIDS) radial products in Xarray"
    url = "https://xradar.rtfd.io/en/latest/io.html#nexrad-level3"

    def open_dataset(
        self,
        filename_or_obj,
        *,
        mask_and_scale=True,
        decode_times=True,
        concat_characters=True,
        decode_coords=True,
        drop_variables=None,
        use_cftime=None,
        decode_timedelta=None,
        group=None,
        first_dim="auto",
        reindex_angle=False,
        site_as_coords=True,
    ):
        fdata = NEXRADLevel3File(filename_or_obj)
        ds = _build_l3_sweep(
            fdata,
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            first_dim=first_dim,
            site_as_coords=site_as_coords,
        )

        if decode_coords and reindex_angle is not False:
            ds = ds.pipe(util.remove_duplicate_rays)
            ds = ds.pipe(util.reindex_angle, **reindex_angle)
            ds = ds.pipe(util.ipol_time, **reindex_angle)

        if drop_variables is not None:
            ds = ds.drop_vars(drop_variables, errors="ignore")

        ds._close = fdata.close
        return ds


def _build_l3_root(sweeps):
    """Build a CfRadial2 root Dataset from a list of Level 3 sweep Datasets."""
    first = sweeps[0]
    last = sweeps[-1]

    root = xr.Dataset()
    root = root.assign(
        {
            "volume_number": 0,
            "platform_type": "fixed",
            "instrument_type": "radar",
            "time_coverage_start": first.attrs.get("time_coverage_start", ""),
            "time_coverage_end": last.attrs.get("time_coverage_end", ""),
        }
    )

    for name in _STATION_VARS:
        if name in first.variables:
            root[name] = first[name]
    promote = _STATION_VARS & set(root.variables)
    if promote:
        root = root.set_coords(list(promote))

    fixed_angles = [float(sw["sweep_fixed_angle"].values.item()) for sw in sweeps]
    root["sweep_fixed_angle"] = xr.DataArray(
        np.asarray(fixed_angles, dtype=float),
        dims=("sweep",),
        attrs={"long_name": "Fixed angle of sweep", "units": "degrees"},
    )
    sweep_names = np.array([f"sweep_{i}" for i in range(len(sweeps))])
    root["sweep_group_name"] = xr.DataArray(sweep_names, dims=("sweep",))
    root.sweep_group_name.encoding["dtype"] = root.sweep_group_name.dtype

    root = root.assign_attrs(
        {
            "Conventions": "CF-1.8, WMO CF-1.0, ACDD-1.3",
            "wmo__cf_profile": "FM 301-XX",
            "version": "2.0",
            "title": "NEXRAD Level 3 radar data",
            "institution": "NOAA National Weather Service",
            "references": "ICD for the RPG to Class 1 User (2620001)",
            "source": "NEXRAD Level 3 (NIDS)",
            "history": "",
            "comment": "",
            "instrument_name": first.attrs.get("instrument_name", ""),
            "platform_is_mobile": "false",
        }
    )
    return root


def _sweep_for_datatree(ds, site_as_coords, sweep_number):
    sw = ds.drop_vars(_STATION_VARS, errors="ignore")
    sw = _apply_site_as_coords(sw, site_as_coords)
    sw["sweep_number"] = xr.DataArray(sweep_number)
    sw.attrs = {}
    return sw


def open_nexradlevel3_datatree(filename_or_obj, **kwargs):
    """Open NEXRAD Level 3 file(s) as a :py:class:`xarray.DataTree`.

    Each Level 3 file holds one product on one elevation cut.

    - Single file path -> single-sweep DataTree.
    - List of files of the *same product* at different cuts (e.g.
      N0B/N1B/N2B/N3B super-resolution reflectivity) -> multi-sweep volume
      DataTree ordered by fixed angle.

    Files of *different* products cannot be combined into one volume: their
    azimuth and range grids differ (e.g. 720x1840 super-resolution
    reflectivity vs 360x1200 dual-pol), so each product must be opened as
    its own DataTree.

    Level 3 files carry no per-ray times, so every ray holds the volume
    scan start time. Multi-tilt volumes therefore export cleanly to
    CfRadial2 but not to CfRadial1 (which requires distinct ray times to
    stack sweeps along a time dimension).

    Parameters
    ----------
    filename_or_obj : str, Path, file-like, bytes, or list of those
        Single Level 3 file, or a list of same-product files making up one
        volume.

    Keyword Arguments
    -----------------
    mask_and_scale : bool
        Decode raw levels into physical values. Defaults to True.
    first_dim : str
        ``"auto"`` (default) or ``"time"``.
    reindex_angle : bool or dict
        If a dict, kwargs are passed to :func:`xradar.util.reindex_angle`.
    site_as_coords : bool
        Attach station variables as coordinates on sweep Datasets.

    Returns
    -------
    dtree : xarray.DataTree
        CfRadial2-style DataTree with ``/`` root and ``sweep_N`` children.
    """
    if isinstance(filename_or_obj, (list, tuple)):
        files = list(filename_or_obj)
        if not files:
            raise ValueError("open_nexradlevel3_datatree requires at least one file.")
    else:
        files = [filename_or_obj]

    site_as_coords = kwargs.pop("site_as_coords", True)
    backend = NexradLevel3BackendEntrypoint()
    sweeps = [backend.open_dataset(f, site_as_coords=False, **kwargs) for f in files]

    codes = {ds.attrs["nexrad_level3_message_code"] for ds in sweeps}
    if len(codes) > 1:
        raise ValueError(
            f"Cannot combine different Level 3 products {sorted(codes)} into "
            "one volume: their azimuth/range grids differ. Open each product "
            "as its own DataTree."
        )

    sweeps.sort(key=lambda ds: float(ds["sweep_fixed_angle"].values.item()))

    root = _build_l3_root(sweeps)
    dtree: dict = {"/": root}
    for i, ds in enumerate(sweeps):
        dtree[f"/sweep_{i}"] = _sweep_for_datatree(ds, site_as_coords, i)
    return DataTree.from_dict(dtree)
