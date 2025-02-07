#!/usr/bin/env python
# Copyright (c) 2024-2025, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
NEXRAD Level2 Data I/O
^^^^^^^^^^^^^^^^^^^^^^

Reads data from NEXRAD Level2 data format

See https://www.roc.noaa.gov/WSR88D/BuildInfo/Files.aspx

Documents:
 - ICD 2620002 M ICD FOR RDA/RPG - Build RDA 11.5/RPG 13.0 (PDF)
 - ICD 2620010 E ICD FOR ARCHIVE II/USER - Build 12.0 (PDF)

To read from NEXRAD Level2 files :class:`numpy:numpy.memmap` is used for
uncompressed files (pre 2016-06-01) and :class:`bz2:BZ2Decompressor` for BZ2
compressed data. The NEXRAD header (`VOLUME_HEADER`, `MSG_HEADER`) are read in
any case into dedicated OrderedDict's. Reading sweep data can be skipped by
setting `loaddata=False`. By default, the data is decoded on the fly.
Using `rawdata=True` the data will be kept undecoded.

Code adapted from Py-ART.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = [
    "NexradLevel2BackendEntrypoint",
    "open_nexradlevel2_datatree",
]
__doc__ = __doc__.format("\n   ".join(__all__))

import bz2
import struct
from collections import OrderedDict, defaultdict

import numpy as np
import xarray as xr
from xarray import DataTree
from xarray.backends.common import AbstractDataStore, BackendArray, BackendEntrypoint
from xarray.backends.file_manager import CachingFileManager
from xarray.backends.locks import SerializableLock, ensure_lock
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.utils import FrozenDict, close_on_error
from xarray.core.variable import Variable

from xradar import util
from xradar.io.backends.common import (
    _assign_root,
    _get_radar_calibration,
    _get_subgroup,
)
from xradar.model import (
    georeferencing_correction_subgroup,
    get_altitude_attrs,
    get_azimuth_attrs,
    get_elevation_attrs,
    get_latitude_attrs,
    get_longitude_attrs,
    get_range_attrs,
    get_time_attrs,
    moment_attrs,
    radar_calibration_subgroup,
    radar_parameters_subgroup,
    sweep_vars_mapping,
)

from .iris import (
    BIN2,
    IrisRecord,
    _get_fmt_string,
    _unpack_dictionary,
    string_dict,
)

NEXRADL2_LOCK = SerializableLock()


#: mapping from NEXRAD names to CfRadial2/ODIM
nexrad_mapping = {
    "REF": "DBZH",
    "VEL": "VRADH",
    "SW ": "WRADH",
    "ZDR": "ZDR",
    "PHI": "PHIDP",
    "RHO": "RHOHV",
    "CFP": "CCORH",
}

# NEXRAD Level II file structures and sizes
# The deails on these structures are documented in:
# "Interface Control Document for the Achive II/User" RPG Build 12.0
# Document Number 2620010E
# and
# "Interface Control Document for the RDA/RPG" Open Build 13.0
# Document Number 2620002M
# Tables and page number refer to those in the second document unless
# otherwise noted.
RECORD_BYTES = 2432
COMPRESSION_RECORD_SIZE = 12
CONTROL_WORD_SIZE = 4

# format of structure elements
# section 3.2.1, page 3-2
UINT1 = {"fmt": "B", "dtype": "unit8"}
UINT2 = {"fmt": "H", "dtype": "uint16"}
UINT4 = {"fmt": "I", "dtype": "uint32"}
SINT1 = {"fmt": "b", "dtype": "int8"}
SINT2 = {"fmt": "h", "dtype": "int16"}
SINT4 = {"fmt": "i", "dtype": "int32"}
FLT4 = {"fmt": "f", "dtype": "float32"}
CODE1 = {"fmt": "B"}
CODE2 = {"fmt": "H"}


class NEXRADFile:
    """
    Class for accessing data in a NEXRAD (WSR-88D) Level II file.

    NEXRAD Level II files [1]_, also know as NEXRAD Archive Level II or
    WSR-88D Archive level 2, are available from the NOAA National Climate Data
    Center [2]_ as well as on the UCAR THREDDS Data Server [3]_. Files with
    uncompressed messages and compressed messages are supported. This class
    supports reading both "message 31" and "message 1" type files.

    Parameters
    ----------
    filename : str
        Filename of Archive II file to read.

    References
    ----------
    .. [1] http://www.roc.noaa.gov/WSR88D/Level_II/Level2Info.aspx
    .. [2] http://www.ncdc.noaa.gov/
    .. [3] http://thredds.ucar.edu/thredds/catalog.html

    """

    def __init__(self, filename, mode="r", loaddata=False):
        """initalize the object."""
        self._fp = None
        self._filename = filename
        # read in the volume header and compression_record
        if hasattr(filename, "read"):
            self._fh = filename
        else:
            self._fp = open(filename, "rb")
            self._fh = np.memmap(self._fp, mode=mode)
        self._filepos = 0
        self._rawdata = False
        self._loaddata = loaddata
        self._bz2_indices = None
        self.volume_header = self.get_header(VOLUME_HEADER)
        return

    @property
    def filename(self):
        return self._filename

    @property
    def fh(self):
        return self._fh

    @property
    def filepos(self):
        return self._filepos

    @filepos.setter
    def filepos(self, pos):
        self._filepos = pos

    def read_from_file(self, size):
        """Read from file.

        Parameters
        ----------
        size : int
            Number of data words to read.

        Returns
        -------
        data : array-like
            numpy array of data
        """
        start = self._filepos
        self._filepos += size
        return self._fh[start : self._filepos]

    def get_header(self, header):
        len = struct.calcsize(_get_fmt_string(header))
        head = _unpack_dictionary(self.read_from_file(len), header, self._rawdata)
        return head

    @property
    def bz2_record_indices(self):
        """Get file offsets of bz2 records."""
        if self._bz2_indices is None:
            # magic number inside BZ2
            # BZhX1AY&SY (where X is any number between 0..9)
            seq = np.array([66, 90, 104, 0, 49, 65, 89, 38, 83, 89], dtype=np.uint8)
            rd = util.rolling_dim(self._fh, len(seq))
            self._bz2_indices = np.nonzero((rd == seq).sum(1) >= 9)[0] - 4
        return self._bz2_indices

    @property
    def is_compressed(self):
        """File contains bz2 compressed data."""
        size = self._fh[24:28].view(dtype=">u4")[0]
        return size > 0

    def close(self):
        if self._fp is not None:
            self._fp.close()

    __del__ = close

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


class NEXRADRecordFile(NEXRADFile):
    """NEXRADRecordFile Record File class"""

    def __init__(self, filename, **kwargs):
        super().__init__(filename=filename, **kwargs)
        self._rh = None
        self._rc = None
        self._ldm = dict()
        self._record_number = None

    @property
    def rh(self):
        """Returns current record object."""
        return self._rh

    @rh.setter
    def rh(self, value):
        """Sets current record object."""
        self._rh = value

    @property
    def record_number(self):
        """Returns current record number."""
        return self._record_number

    @record_number.setter
    def record_number(self, value):
        """Sets current record number."""
        self._record_number = value

    @property
    def record_size(self):
        """Returns current record number."""
        return self._record_size

    @record_size.setter
    def record_size(self, value):
        """Sets current record number."""
        self._record_size = value

    def _check_record(self):
        """Checks record for correct size.

        Need to be implemented in the derived classes
        """
        return True

    def get_end(self, buf):
        if len(buf) < 16:
            return 0
        header = _unpack_dictionary(buf, MSG_HEADER, self._rawdata, byte_order=">")
        size = header["size"] * 2 + 12
        if header["type"] != 31:
            size = size if size >= RECORD_BYTES else RECORD_BYTES
        return size

    def init_record(self, recnum):
        """Initialize record using given number."""

        # map record numbers to ldm compressed records
        def get_ldm(recnum):
            if recnum < 134:
                return 0
            mod = ((recnum - 134) // 120) + 1
            return mod

        if self.is_compressed:
            ldm = get_ldm(recnum)
            # get uncompressed ldm record
            if self._ldm.get(ldm, None) is None:
                # otherwise extract wanted ldm compressed record
                if ldm >= len(self.bz2_record_indices):
                    return False
                start = self.bz2_record_indices[ldm]
                size = self._fh[start : start + 4].view(dtype=">u4")[0]
                self._fp.seek(start + 4)
                dec = bz2.BZ2Decompressor()
                self._ldm[ldm] = np.frombuffer(
                    dec.decompress(self._fp.read(size)), dtype=np.uint8
                )

        # rectrieve wanted record and put into self.rh
        if recnum < 134:
            start = recnum * RECORD_BYTES
            if not self.is_compressed:
                start += 24
            stop = start + RECORD_BYTES
        else:
            if self.is_compressed:
                # get index into current compressed ldm record
                rnum = (recnum - 134) % 120
                start = self.record_size + self.filepos if rnum else 0
                buf = self._ldm[ldm][start + 12 : start + 12 + LEN_MSG_HEADER]
                size = self.get_end(buf)
                if not size:
                    return False
                stop = start + size
            else:
                start = self.record_size + self.filepos
                buf = self.fh[start + 12 : start + 12 + LEN_MSG_HEADER]
                size = self.get_end(buf)
                if not size:
                    return False
                stop = start + size
        self.record_number = recnum
        self.record_size = stop - start
        if self.is_compressed:
            self.rh = IrisRecord(self._ldm[ldm][start:stop], recnum)
        else:
            self.rh = IrisRecord(self.fh[start:stop], recnum)
        self.filepos = start
        return self._check_record()

    def init_record_by_filepos(self, recnum, filepos):
        """Initialize record using given record number and position."""
        start = filepos
        buf = self.fh[start + 12 : start + 12 + LEN_MSG_HEADER]
        size = self.get_end(buf)
        if not size:
            return False
        stop = start + size
        self.record_number = recnum
        self.record_size = stop - start
        self.rh = IrisRecord(self.fh[start:stop], recnum)
        self.filepos = start
        return self._check_record()

    def init_next_record(self):
        """Get next record from file.

        This increases record_number count and initialises a new IrisRecord
        with the calculated start and stop file offsets.

        Returns
        -------
        chk : bool
            True, if record is truncated.
        """
        return self.init_record(self.record_number + 1)

    def array_from_record(self, words, width, dtype):
        """Retrieve array from current record.

        Parameters
        ----------
        words : int
            Number of data words to read.
        width : int
            Size of the data word to read in bytes.
        dtype : str
            dtype string specifying data format.

        Returns
        -------
        data : array-like
            numpy array of data
        """
        return self.rh.read(words, width=width).view(dtype=dtype)

    def bytes_from_record(self, words, width):
        """Retrieve bytes from current record.

        Parameters
        ----------
        words : int
            Number of data words to read.
        width : int
            Size of the data word to read in bytes.

        Returns
        -------
        data : array-like
            numpy array of data
        """
        return self.rh.read(words, width=width)


class NEXRADLevel2File(NEXRADRecordFile):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)

        # check compression

        self._data_header = None
        # get all metadata headers
        # message 15, 13, 18, 3, 5 and 2
        self._meta_header = None

        # message 15
        # RDA Clutter Map Data
        # message 13
        # RDA Clutter Filter Bypass Map
        # message 18
        # RDA Adaptable Parameters
        # message 3
        # RDA Performance/Maintenance Data
        # message 5
        # RDA Volume Coverage Data
        self._msg_5_data = None
        # message 2
        # RDA Status Data

        # message 31 headers
        # Digital Radar Data Generic Format
        self._data_header = None
        self._msg_31_header = None
        self._msg_31_data_header = None
        self._data = OrderedDict()

    @property
    def data_header(self):
        """Retrieve data header."""
        if self._data_header is None:
            (
                self._data_header,
                self._msg_31_header,
                self._msg_31_data_header,
            ) = self.get_data_header()
        return self._data_header

    @property
    def msg_31_header(self):
        """Retrieve MSG31 Header."""
        if self._msg_31_header is None:
            (
                self._data_header,
                self._msg_31_header,
                self._msg_31_data_header,
            ) = self.get_data_header()
        return self._msg_31_header

    @property
    def msg_31_data_header(self):
        """Retrieve MSG31 data header."""
        if self._msg_31_data_header is None:
            (
                self._data_header,
                self._msg_31_header,
                self._msg_31_data_header,
            ) = self.get_data_header()
        return self._msg_31_data_header

    @property
    def meta_header(self):
        """Retrieve metadat header."""
        if self._meta_header is None:
            self._meta_header = self.get_metadata_header()
        return self._meta_header

    @property
    def msg_5(self):
        """Retrieve MSG5 data."""
        if self._msg_5_data is None:
            self._msg_5_data = self.get_msg_5_data()
        return self._msg_5_data

    @property
    def data(self):
        """Retrieve data."""
        return self._data

    def get_sweep(self, sweep_number, moments=None):
        """Retrieve sweep according sweep_number."""
        if moments is None:
            moments = self.msg_31_data_header[sweep_number]["msg_31_data_header"].keys()
        # get selected coordinate type names
        selected = [k for k in DATA_BLOCK_CONSTANT_IDENTIFIERS.keys() if k in moments]
        for moment in selected:
            self.get_moment(sweep_number, moment, "sweep_constant_data")

        # get selected data type names
        selected = [k for k in DATA_BLOCK_VARIABLE_IDENTIFIERS.keys() if k in moments]
        for moment in selected:
            self.get_moment(sweep_number, moment, "sweep_data")

    def get_moment(self, sweep_number, moment, mtype):
        """Retrieve Moment according sweep_number and moment."""
        sweep = self.data[sweep_number]
        # create new sweep_data OrderedDict
        if not sweep.get(mtype, False):
            sweep[mtype] = OrderedDict()
        # check if moment exist and return early
        data = sweep[mtype].get(moment, False)
        if data is not False:
            return

        data = OrderedDict()
        data[moment] = self.msg_31_data_header[sweep_number]["msg_31_data_header"].pop(
            moment
        )
        sweep.update(self.msg_31_data_header[sweep_number])
        sweep[mtype].update(data)

    def get_metadata_header(self):
        """Get metadaata header"""
        # data offsets
        # ICD 2620010J
        # 7.3.5 Metadata Record
        # the above document will evolve over time
        # please revisit and adopt accordingly
        meta_headers = defaultdict(list)
        rec = 0
        # iterate over alle messages until type outside [2, 3, 5, 13, 15, 18, 32]
        while True:
            self.init_record(rec)
            rec += 1
            filepos = self.filepos
            message_header = self.get_message_header()
            # do not read zero blocks of data
            if (mtype := message_header["type"]) == 0:
                continue
            # stop if first non meta header is found
            if mtype not in [2, 3, 5, 13, 15, 18, 32]:
                break
            message_header["record_number"] = self.record_number
            message_header["filepos"] = filepos
            meta_headers[f"msg_{mtype}"].append(message_header)
        return meta_headers

    def get_msg_5_data(self):
        """Get MSG5 data."""
        # get the record number from the meta header
        recnum = self.meta_header["msg_5"][0]["record_number"]
        self.init_record(recnum)
        # skip header
        self.rh.pos += LEN_MSG_HEADER + 12
        # unpack msg header
        msg_5 = _unpack_dictionary(
            self._rh.read(LEN_MSG_5, width=1),
            MSG_5,
            self._rawdata,
            byte_order=">",
        )
        msg_5["elevation_data"] = []
        # unpack elevation cuts
        for i in range(msg_5["number_elevation_cuts"]):
            msg_5_elev = _unpack_dictionary(
                self._rh.read(LEN_MSG_5_ELEV, width=1),
                MSG_5_ELEV,
                self._rawdata,
                byte_order=">",
            )
            msg_5["elevation_data"].append(msg_5_elev)
        return msg_5

    def get_message_header(self):
        """Read and unpack message header."""
        self._rh.pos += 12
        return _unpack_dictionary(
            self._rh.read(LEN_MSG_HEADER, width=1),
            MSG_HEADER,
            self._rawdata,
            byte_order=">",
        )

    def get_data(self, sweep_number, moment=None):
        """Load sweep data from file."""
        sweep = self.data[sweep_number]
        start = sweep["record_number"]
        stop = sweep["record_end"]
        intermediate_records = [
            rec["record_number"] for rec in sweep["intermediate_records"]
        ]
        filepos = sweep["filepos"]
        moments = sweep["sweep_data"]
        if moment is None:
            moment = moments
        elif isinstance(moment, str):
            moment = [moment]
        for name in moment:
            if self.is_compressed:
                self.init_record(start)
            else:
                self.init_record_by_filepos(start, filepos)
            ngates = moments[name]["ngates"]
            word_size = moments[name]["word_size"]
            data_offset = moments[name]["data_offset"]
            width = {8: 1, 16: 2}[word_size]
            data = []
            self.rh.pos += data_offset
            data.append(self._rh.read(ngates, width=width).view(f">u{width}"))
            while self.init_next_record() and self.record_number <= stop:
                if self.record_number in intermediate_records:
                    continue
                self.rh.pos += data_offset
                data.append(self._rh.read(ngates, width=width).view(f">u{width}"))
            moments[name].update(data=data)

    def get_data_header(self):
        """Load all data header from file."""
        # get the record number from the meta header
        # message 2 is the last meta header, after which the
        # data records are located
        recnum = self.meta_header["msg_2"][0]["record_number"]
        self.init_record(recnum)
        current_sweep = -1
        current_header = -1
        sweep_msg_31_header = []
        sweep_intermediate_records = []
        sweep = OrderedDict()

        data_header = []
        _msg_31_header = []
        _msg_31_data_header = []

        while self.init_next_record():
            current_header += 1
            # get message headers
            msg_header = self.get_message_header()
            # append to data headers list
            msg_header["record_number"] = self.record_number
            msg_header["filepos"] = self.filepos

            # keep all data headers
            data_header.append(msg_header)
            # check which message type we have (1 or 31)
            if (msg_type := msg_header["type"]) in [1, 31]:
                msg_len, msg = {1: (LEN_MSG_1, MSG_1), 31: (LEN_MSG_31, MSG_31)}[
                    msg_type
                ]
                # get msg_31_header
                msg_31_header = _unpack_dictionary(
                    self._rh.read(msg_len, width=1),
                    msg,
                    self._rawdata,
                    byte_order=">",
                )
                # add record_number/filepos
                msg_31_header["record_number"] = self.record_number
                msg_31_header["filepos"] = self.filepos
                # retrieve data/const headers from msg 31
                # check if this is a new sweep
                status = msg_31_header["radial_status"]
                if status == 1:
                    # 1 - intermediate radial
                    pass
                if status in [2, 4]:
                    # 2 - end of elevation
                    # 4 - end of volume
                    sweep["record_end"] = self.rh.recnum
                    sweep["intermediate_records"] = sweep_intermediate_records
                    self._data[current_sweep] = sweep
                    _msg_31_header.append(sweep_msg_31_header)
                if status in [0, 3, 5]:
                    # 0 - start of new elevation
                    # 3 - start of new volume
                    # 5 - start of new elevation, last elevation in VCP
                    current_sweep += 1
                    # create new sweep object
                    sweep = OrderedDict()
                    sweep_msg_31_header = []
                    sweep_intermediate_records = []
                    sweep["record_number"] = self.rh.recnum
                    sweep["filepos"] = self.filepos
                    sweep["msg_31_header"] = msg_31_header
                    # add msg_type for later reconstruction
                    sweep["msg_type"] = msg_type
                    # new message 31 data
                    if msg_type == 31:
                        block_pointers = [
                            v
                            for k, v in msg_31_header.items()
                            if k.startswith("block_pointer") and v > 0
                        ]
                        block_header = {}
                        for i, block_pointer in enumerate(
                            block_pointers[: msg_31_header["block_count"]]
                        ):
                            self.rh.pos = block_pointer + 12 + LEN_MSG_HEADER
                            buf = self._rh.read(4, width=1)
                            dheader = _unpack_dictionary(
                                buf,
                                DATA_BLOCK_HEADER,
                                self._rawdata,
                                byte_order=">",
                            )
                            block = DATA_BLOCK_TYPE_IDENTIFIER[dheader["block_type"]][
                                dheader["data_name"]
                            ]
                            LEN_BLOCK = struct.calcsize(
                                _get_fmt_string(block, byte_order=">")
                            )
                            block_header[dheader["data_name"]] = _unpack_dictionary(
                                self._rh.read(LEN_BLOCK, width=1),
                                block,
                                self._rawdata,
                                byte_order=">",
                            )
                            if dheader["block_type"] == "D":
                                block_header[dheader["data_name"]][
                                    "data_offset"
                                ] = self.rh.pos

                        sweep["msg_31_data_header"] = block_header
                        _msg_31_data_header.append(sweep)
                    # former message 1 data
                    elif msg_type == 1:
                        # creating block_pointers
                        # data_offset is from start of message so need to
                        # add LEN_MSG_HEADER + 12
                        block_pointers = {
                            "REF": dict(
                                ngates=msg_31_header["sur_nbins"],
                                gate_spacing=msg_31_header["sur_range_step"],
                                first_gate=msg_31_header["sur_range_first"],
                                word_size=8,
                                scale=2.0,
                                offset=66.0,
                                data_offset=(
                                    msg_31_header["sur_pointer"] + LEN_MSG_HEADER + 12
                                ),
                            ),
                            "VEL": dict(
                                ngates=msg_31_header["doppler_nbins"],
                                gate_spacing=msg_31_header["doppler_range_step"],
                                first_gate=msg_31_header["doppler_range_first"],
                                word_size=8,
                                scale={2: 2.0, 4: 1.0, 0: 0.0}[
                                    msg_31_header["doppler_resolution"]
                                ],
                                offset=129.0,
                                data_offset=(
                                    msg_31_header["vel_pointer"] + LEN_MSG_HEADER + 12
                                ),
                            ),
                            "SW": dict(
                                ngates=msg_31_header["doppler_nbins"],
                                gate_spacing=msg_31_header["doppler_range_step"],
                                first_gate=msg_31_header["doppler_range_first"],
                                word_size=8,
                                scale=2.0,
                                offset=192.0,
                                data_offset=(
                                    msg_31_header["width_pointer"] + LEN_MSG_HEADER + 12
                                ),
                            ),
                        }
                        block_header = {}
                        block_header["VOL"] = dict(
                            lat=0,
                            lon=0,
                            height=0,
                            feedhorn_height=0,
                            vcp=msg_31_header.pop("vcp"),
                        )
                        for k, v in block_pointers.items():
                            if v["ngates"]:
                                block_header[k] = v
                        sweep["msg_31_data_header"] = block_header
                        _msg_31_data_header.append(sweep)
                sweep_msg_31_header.append(msg_31_header)
            else:
                sweep_intermediate_records.append(msg_header)

        return data_header, _msg_31_header, _msg_31_data_header

    def _check_record(self):
        """Checks record for correct size.

        Returns
        -------
        chk : bool
            False, if record is truncated.
        """
        chk = self._rh.record.shape[0] == self.record_size
        if not chk:
            raise EOFError(f"Unexpected file end detected at record {self.rh.recnum}")
        return chk


# Figure 1 in Interface Control Document for the Archive II/User
# page 7-2
VOLUME_HEADER = OrderedDict(
    [
        ("tape", {"fmt": "9s"}),
        ("extension", {"fmt": "3s"}),
        ("date", UINT4),
        ("time", UINT4),
        ("icao", {"fmt": "4s"}),
    ]
)

# Table II Message Header Data
# page 3-7
MSG_HEADER = OrderedDict(
    [
        ("size", UINT2),  # size of data, no including header
        ("channels", UINT1),
        ("type", UINT1),
        ("seq_id", UINT2),
        ("date", UINT2),
        ("ms", UINT4),
        ("segments", UINT2),
        ("seg_num", UINT2),
    ]
)
LEN_MSG_HEADER = struct.calcsize(_get_fmt_string(MSG_HEADER, byte_order=">"))

# Table XVII Digital Radar Generic Format Blocks (Message Type 31)
# pages 3-87 to 3-89
MSG_31 = OrderedDict(
    [
        ("id", {"fmt": "4s"}),  # 0-3
        ("collect_ms", UINT4),  # 4-7
        ("collect_date", UINT2),  # 8-9
        ("azimuth_number", UINT2),  # 10-11
        ("azimuth_angle", FLT4),  # 12-15
        ("compress_flag", CODE1),  # 16
        ("spare_0", UINT1),  # 17
        ("radial_length", UINT2),  # 18-19
        ("azimuth_resolution", CODE1),  # 20
        ("radial_status", CODE1),  # 21
        ("elevation_number", UINT1),  # 22
        ("cut_sector", UINT1),  # 23
        ("elevation_angle", FLT4),  # 24-27
        ("radial_blanking", CODE1),  # 28
        ("azimuth_mode", SINT1),  # 29
        ("block_count", UINT2),  # 30-31
        ("block_pointer_1", UINT4),  # 32-35  Volume Data Constant XVII-E
        ("block_pointer_2", UINT4),  # 36-39  Elevation Data Constant XVII-F
        ("block_pointer_3", UINT4),  # 40-43  Radial Data Constant XVII-H
        ("block_pointer_4", UINT4),  # 44-47  Moment "REF" XVII-{B/I}
        ("block_pointer_5", UINT4),  # 48-51  Moment "VEL"
        ("block_pointer_6", UINT4),  # 52-55  Moment "SW"
        ("block_pointer_7", UINT4),  # 56-59  Moment "ZDR"
        ("block_pointer_8", UINT4),  # 60-63  Moment "PHI"
        ("block_pointer_9", UINT4),  # 64-67  Moment "RHO"
        ("block_pointer_10", UINT4),  # Moment "CFP"
    ]
)
LEN_MSG_31 = struct.calcsize(_get_fmt_string(MSG_31, byte_order=">"))


# Table III Digital Radar Data (Message Type 1)
# pages 3-7 to
MSG_1 = OrderedDict(
    [
        ("collect_ms", UINT4),  # 0-3
        ("collect_date", UINT2),  # 4-5
        ("unambig_range", SINT2),  # 6-7
        ("azimuth_angle", CODE2),  # 8-9
        ("azimuth_number", UINT2),  # 10-11
        ("radial_status", CODE2),  # 12-13
        ("elevation_angle", UINT2),  # 14-15
        ("elevation_number", UINT2),  # 16-17
        ("sur_range_first", CODE2),  # 18-19
        ("doppler_range_first", CODE2),  # 20-21
        ("sur_range_step", CODE2),  # 22-23
        ("doppler_range_step", CODE2),  # 24-25
        ("sur_nbins", UINT2),  # 26-27
        ("doppler_nbins", UINT2),  # 28-29
        ("cut_sector_num", UINT2),  # 30-31
        ("calib_const", FLT4),  # 32-35
        ("sur_pointer", UINT2),  # 36-37
        ("vel_pointer", UINT2),  # 38-39
        ("width_pointer", UINT2),  # 40-41
        ("doppler_resolution", CODE2),  # 42-43
        ("vcp", UINT2),  # 44-45
        ("spare_1", {"fmt": "8s"}),  # 46-53
        ("spare_2", {"fmt": "2s"}),  # 54-55
        ("spare_3", {"fmt": "2s"}),  # 56-57
        ("spare_4", {"fmt": "2s"}),  # 58-59
        ("nyquist_vel", SINT2),  # 60-61
        ("atmos_attenuation", SINT2),  # 62-63
        ("threshold", SINT2),  # 64-65
        ("spot_blank_status", UINT2),  # 66-67
        ("spare_5", {"fmt": "32s"}),  # 68-99
        # 100+  reflectivity, velocity and/or spectral width data, CODE1
    ]
)
LEN_MSG_1 = struct.calcsize(_get_fmt_string(MSG_1, byte_order=">"))

# Table XI Volume Coverage Pattern Data (Message Type 5 & 7)
# pages 3-51 to 3-54
MSG_5 = OrderedDict(
    [
        ("message_size", UINT2),
        ("pattern_type", CODE2),
        ("pattern_number", UINT2),
        ("number_elevation_cuts", UINT2),
        ("clutter_map_group_number", UINT2),
        ("doppler_velocity_resolution", CODE1),  # 2: 0.5 degrees, 4: 1.0 degrees
        ("pulse_width", CODE1),  # 2: short, 4: long
        ("spare", {"fmt": "10s"}),  # halfwords 7-11 (10 bytes, 5 halfwords)
    ]
)
LEN_MSG_5 = struct.calcsize(_get_fmt_string(MSG_5, byte_order=">"))

MSG_5_ELEV = OrderedDict(
    [
        ("elevation_angle", BIN2),  # scaled by 360/65536 for value in degrees.
        ("channel_config", CODE1),
        ("waveform_type", CODE1),
        ("super_resolution", CODE1),
        ("prf_number", UINT1),
        ("prf_pulse_count", UINT2),
        ("azimuth_rate", CODE2),
        ("ref_thresh", SINT2),
        ("vel_thresh", SINT2),
        ("sw_thresh", SINT2),
        ("zdr_thres", SINT2),
        ("phi_thres", SINT2),
        ("rho_thres", SINT2),
        ("edge_angle_1", CODE2),
        ("dop_prf_num_1", UINT2),
        ("dop_prf_pulse_count_1", UINT2),
        ("spare_1", {"fmt": "2s"}),
        ("edge_angle_2", CODE2),
        ("dop_prf_num_2", UINT2),
        ("dop_prf_pulse_count_2", UINT2),
        ("spare_2", {"fmt": "2s"}),
        ("edge_angle_3", CODE2),
        ("dop_prf_num_3", UINT2),
        ("dop_prf_pulse_count_3", UINT2),
        ("spare_3", {"fmt": "2s"}),
    ]
)
LEN_MSG_5_ELEV = struct.calcsize(_get_fmt_string(MSG_5_ELEV, byte_order=">"))

MSG_18 = OrderedDict(
    [
        ("adap_file_name", {"fmt": "12s"}),
        ("adap_format", {"fmt": "4s"}),
        ("adap_revision", {"fmt": "4s"}),
        ("adap_date", {"fmt": "12s"}),
        ("adap_time", {"fmt": "12s"}),
        ("k1", FLT4),
        ("az_lat", FLT4),
        ("k3", FLT4),
        ("el_lat", FLT4),
        ("park_az", FLT4),
        ("park_el", FLT4),
        ("a_fuel_conv0", FLT4),
        ("a_fuel_conv1", FLT4),
        ("a_fuel_conv2", FLT4),
        ("a_fuel_conv3", FLT4),
        ("a_fuel_conv4", FLT4),
        ("a_fuel_conv5", FLT4),
        ("a_fuel_conv6", FLT4),
        ("a_fuel_conv7", FLT4),
        ("a_fuel_conv8", FLT4),
        ("a_fuel_conv9", FLT4),
        ("a_fuel_conv10", FLT4),
        ("a_min_shelter_temp", FLT4),
        ("a_max_shelter_temp", FLT4),
        ("a_min_shelter_ac_temp_diff", FLT4),
        ("a_max_xmtr_air_temp", FLT4),
        ("a_max_rad_temp", FLT4),
        ("a_max_rad_temp_rise", FLT4),
        ("ped_28v_reg_lim", FLT4),
        ("ped_5v_reg_lim", FLT4),
        ("ped_15v_reg_lim", FLT4),
        ("a_min_gen_room_temp", FLT4),
        ("a_max_gen_room_temp", FLT4),
        ("dau_5v_reg_lim", FLT4),
        ("dau_15v_reg_lim", FLT4),
        ("dau_28v_reg_lim", FLT4),
        ("en_5v_reg_lim", FLT4),
        ("en_5v_nom_volts", FLT4),
        ("rpg_co_located", {"fmt": "4s"}),
        ("spec_filter_installed", {"fmt": "4s"}),
        ("tps_installed", {"fmt": "4s"}),
        ("rms_installed", {"fmt": "4s"}),
        ("a_hvdl_tst_int", UINT4),
        ("a_rpg_lt_int", UINT4),
        ("a_min_stab_util_pwr_time", UINT4),
        ("a_gen_auto_exer_interval", UINT4),
        ("a_util_pwr_sw_req_interval", UINT4),
        ("a_low_fuel_level", FLT4),
        ("config_chan_number", UINT4),
        ("a_rpg_link_type", UINT4),
        ("redundant_chan_config", UINT4),
    ]
)
for i in np.arange(104):
    MSG_18[f"atten_table_{i:03d}"] = FLT4
MSG_18.update(
    [
        ("spare_01", {"fmt": "24s"}),
        ("path_losses_07", FLT4),
        ("spare_02", {"fmt": "12s"}),
        ("path_losses_11", FLT4),
        ("spare_03", {"fmt": "4s"}),
        ("path_losses_13", FLT4),
        ("path_losses_14", FLT4),
        ("spare_04", {"fmt": "16s"}),
        ("path_losses_19", FLT4),
        ("spare_05", {"fmt": "32s"}),
    ]
)
for i in np.arange(28, 48):
    MSG_18[f"path_losses_{i:02d}"] = FLT4
MSG_18.update(
    [
        ("h_coupler_cw_loss", FLT4),
        ("v_coupler_xmt_loss", FLT4),
        ("path_losses_50", FLT4),
        ("spare_06", {"fmt": "4s"}),
        ("path_losses_52", FLT4),
        ("v_coupler_cw_loss", FLT4),
        ("path_losses_54", FLT4),
        ("spare_07", {"fmt": "12s"}),
        ("path_losses_58", FLT4),
        ("path_losses_59", FLT4),
        ("path_losses_60", FLT4),
        ("path_losses_61", FLT4),
        ("spare_08", {"fmt": "4s"}),
        ("path_losses_63", FLT4),
        ("path_losses_64", FLT4),
        ("path_losses_65", FLT4),
        ("path_losses_66", FLT4),
        ("path_losses_67", FLT4),
        ("path_losses_68", FLT4),
        ("path_losses_69", FLT4),
        ("chan_cal_diff", FLT4),
        ("spare_09", {"fmt": "4s"}),
        ("log_amp_factor_1", FLT4),
        ("log_amp_factor_2", FLT4),
        ("v_ts_cw", FLT4),
    ]
)
for i in np.arange(13):
    MSG_18[f"rnscale_{i:02d}"] = FLT4
for i in np.arange(13):
    MSG_18[f"atmos_{i:02d}"] = FLT4
for i in np.arange(12):
    MSG_18[f"el_index_{i:02d}"] = FLT4
MSG_18.update(
    [
        ("tfreq_mhz", UINT4),
        ("base_data_tcn", FLT4),
        ("refl_data_tover", FLT4),
        ("tar_h_dbz0_lp", FLT4),
        ("tar_v_dbz0_lp", FLT4),
        ("init_phi_dp", UINT4),
        ("norm_init_phi_dp", UINT4),
        ("lx_lp", FLT4),
        ("lx_sp", FLT4),
        ("meteor_param", FLT4),
        ("beamwidth", FLT4),
        ("antenna_gain", FLT4),
        ("spare_10", {"fmt": "4s"}),
        ("vel_maint_limit", FLT4),
        ("wth_maint_limit", FLT4),
        ("vel_degrad_limit", FLT4),
        ("wth_degrad_limit", FLT4),
        ("h_noisetemp_degrad_limit", FLT4),
        ("h_noisetemp_maint_limit", FLT4),
        ("v_noisetemp_degrad_limit", FLT4),
        ("v_noisetemp_maint_limit", FLT4),
        ("kly_degrade_limit", FLT4),
        ("ts_coho", FLT4),
        ("h_ts_cw", FLT4),
        ("ts_rf_sp", FLT4),
        ("ts_rf_lp", FLT4),
        ("ts_stalo", FLT4),
        ("ame_h_noise_enr", FLT4),
        ("xmtr_peak_pwr_high_limit", FLT4),
        ("xmtr_peak_pwr_low_limit", FLT4),
        ("h_dbz0_delta_limit", FLT4),
        ("threshold1", FLT4),
        ("threshold2", FLT4),
        ("clut_supp_dgrad_lim", FLT4),
        ("clut_supp_maint_lim", FLT4),
        ("range0_value", FLT4),
        ("xmtr_pwr_mtr_scale", FLT4),
        ("v_dbz0_delta_limit", FLT4),
        ("tar_h_dbz0_sp", FLT4),
        ("tar_v_dbz0_sp", FLT4),
        ("deltaprf", UINT4),
        ("spare_11", {"fmt": "8s"}),
        ("tau_sp", UINT4),
        ("tau_lp", UINT4),
        ("nc_dead_value", UINT4),
        ("tau_rf_sp", UINT4),
        ("tau_rf_lp", UINT4),
        ("seg1lim", FLT4),
        ("slatsec", FLT4),
        ("slonsec", FLT4),
        ("spare_12", {"fmt": "4s"}),
        ("slatdeg", UINT4),
        ("slatmin", UINT4),
        ("slondeg", UINT4),
        ("slonmin", UINT4),
        ("slatdir", {"fmt": "4s"}),
        ("slondir", {"fmt": "4s"}),
        ("spare_13", {"fmt": "4s"}),
        ("vcpat_11", {"fmt": "1172s"}),
        ("vcpat_21", {"fmt": "1172s"}),
        ("vcpat_31", {"fmt": "1172s"}),
        ("vcpat_32", {"fmt": "1172s"}),
        ("vcpat_300", {"fmt": "1172s"}),
        ("vcpat_301", {"fmt": "1172s"}),
        ("az_correction_factor", FLT4),
        ("el_correction_factor", FLT4),
        ("site_name", {"fmt": "4s"}),
        ("ant_manual_setup_ielmin", SINT4),
        ("ant_manual_setup_ielmax", SINT4),
        ("ant_manual_setup_fazvelmax", SINT4),
        ("ant_manual_setup_felvelmax", SINT4),
        ("ant_manual_setup_ignd_hgt", SINT4),
        ("ant_manual_setup_irad_hgt", SINT4),
        ("spare_14", {"fmt": "300s"}),
        ("rvp8nv_iwaveguide_lenght", UINT4),
        ("spare_15", {"fmt": "44s"}),
        ("vel_data_tover", FLT4),
        ("width_data_tover", FLT4),
        ("spare_16", {"fmt": "12s"}),
        ("doppler_range_start", FLT4),
        ("max_el_index", UINT4),
        ("seg2lim", FLT4),
        ("seg3lim", FLT4),
        ("seg4lim", FLT4),
        ("nbr_el_segments", UINT4),
        ("h_noise_long", FLT4),
        ("ant_noise_temp", FLT4),
        ("h_noise_short", FLT4),
        ("h_noise_tolerance", FLT4),
        ("min_h_dyn_range", FLT4),
        ("gen_installed", {"fmt": "4s"}),
        ("gen_exercise", {"fmt": "4s"}),
        ("v_noise_tolerance", FLT4),
        ("min_v_dyn_range", FLT4),
        ("zdr_bias_dgrad_lim", FLT4),
        ("spare_17", {"fmt": "16s"}),
        ("v_noise_long", FLT4),
        ("v_noise_short", FLT4),
        ("zdr_data_tover", FLT4),
        ("phi_data_tover", FLT4),
        ("rho_data_tover", FLT4),
        ("stalo_power_dgrad_limit", FLT4),
        ("stalo_power_maint_limit", FLT4),
        ("min_h_pwr_sense", FLT4),
        ("min_v_pwr_sense", FLT4),
        ("h_pwr_sense_offset", FLT4),
        ("v_pwr_sense_offset", FLT4),
        ("ps_gain_ref", FLT4),
        ("rf_pallet_broad_loss", FLT4),
        ("zdr_check_threshold", FLT4),
        ("phi_check_threshold", FLT4),
        ("rho_check_threshold", FLT4),
        ("spare_18", {"fmt": "52s"}),
        ("ame_ps_tolerance", FLT4),
        ("ame_max_temp", FLT4),
        ("ame_min_temp", FLT4),
        ("rcvr_mod_max_temp", FLT4),
        ("rcvr_mod_min_temp", FLT4),
        ("bite_mod_max_temp", FLT4),
        ("bite_mod_min_temp", FLT4),
        ("default_polarization", UINT4),
        ("tr_limit_dgrad_limit", FLT4),
        ("tr_limit_fail_limit", FLT4),
        ("spare_19", {"fmt": "8s"}),
        ("ame_current_tolerance", FLT4),
        ("h_only_polarization", UINT4),
        ("v_only_polarization", UINT4),
        ("spare_20", {"fmt": "8s"}),
        ("reflector_bias", FLT4),
        ("a_min_shelter_temp_warn", FLT4),
        ("spare_21", {"fmt": "432s"}),
    ]
)
LEN_MSG_18 = struct.calcsize(_get_fmt_string(MSG_18, byte_order=">"))

DATA_BLOCK_HEADER = OrderedDict(
    [("block_type", string_dict(1)), ("data_name", string_dict(3))]
)
LEN_DATA_BLOCK_HEADER = struct.calcsize(
    _get_fmt_string(DATA_BLOCK_HEADER, byte_order=">")
)


# Table XVII-B Data Block (Descriptor of Generic Data Moment Type)
# pages 3-90 and 3-91
GENERIC_DATA_BLOCK = OrderedDict(
    [
        ("reserved", UINT4),
        ("ngates", UINT2),
        ("first_gate", SINT2),
        ("gate_spacing", SINT2),
        ("thresh", SINT2),
        ("snr_thres", SINT2),
        ("flags", CODE1),
        ("word_size", UINT1),
        ("scale", FLT4),
        ("offset", FLT4),
        # then data
    ]
)
LEN_GENERIC_DATA_BLOCK = struct.calcsize(
    _get_fmt_string(GENERIC_DATA_BLOCK, byte_order=">")
)

# Table XVII-E Data Block (Volume Data Constant Type)
# page 3-92
VOLUME_DATA_BLOCK = OrderedDict(
    [
        ("lrtup", UINT2),
        ("version_major", UINT1),
        ("version_minor", UINT1),
        ("lat", FLT4),
        ("lon", FLT4),
        ("height", SINT2),
        ("feedhorn_height", UINT2),
        ("refl_calib", FLT4),
        ("power_h", FLT4),
        ("power_v", FLT4),
        ("diff_refl_calib", FLT4),
        ("init_phase", FLT4),
        ("vcp", UINT2),
        ("spare", {"fmt": "2s"}),
    ]
)
LEN_VOLUME_DATA_BLOCK = struct.calcsize(
    _get_fmt_string(VOLUME_DATA_BLOCK, byte_order=">")
)

# Table XVII-F Data Block (Elevation Data Constant Type)
# page 3-93
ELEVATION_DATA_BLOCK = OrderedDict(
    [
        ("lrtup", UINT2),
        ("atmos", SINT2),
        ("refl_calib", FLT4),
    ]
)
LEN_ELEVATION_DATA_BLOCK = struct.calcsize(
    _get_fmt_string(ELEVATION_DATA_BLOCK, byte_order=">")
)

# Table XVII-H Data Block (Radial Data Constant Type)
# pages 3-93
RADIAL_DATA_BLOCK = OrderedDict(
    [
        ("lrtup", UINT2),
        ("unambig_range", SINT2),
        ("noise_h", FLT4),
        ("noise_v", FLT4),
        ("nyquist_vel", SINT2),
        ("spare", {"fmt": "2s"}),
    ]
)
LEN_RADIAL_DATA_BLOCK = struct.calcsize(
    _get_fmt_string(RADIAL_DATA_BLOCK, byte_order=">")
)

DATA_BLOCK_CONSTANT_IDENTIFIERS = OrderedDict(
    [
        ("VOL", VOLUME_DATA_BLOCK),
        ("ELV", ELEVATION_DATA_BLOCK),
        ("RAD", RADIAL_DATA_BLOCK),
    ]
)

DATA_BLOCK_VARIABLE_IDENTIFIERS = OrderedDict(
    [
        ("REF", GENERIC_DATA_BLOCK),
        ("VEL", GENERIC_DATA_BLOCK),
        ("SW ", GENERIC_DATA_BLOCK),
        ("ZDR", GENERIC_DATA_BLOCK),
        ("PHI", GENERIC_DATA_BLOCK),
        ("RHO", GENERIC_DATA_BLOCK),
        ("CFP", GENERIC_DATA_BLOCK),
    ]
)

DATA_BLOCK_TYPE_IDENTIFIER = OrderedDict(
    [
        ("R", DATA_BLOCK_CONSTANT_IDENTIFIERS),
        ("D", DATA_BLOCK_VARIABLE_IDENTIFIERS),
    ]
)


class NexradLevel2ArrayWrapper(BackendArray):
    """Wraps array of NexradLevel2 data."""

    def __init__(self, datastore, name, var):
        self.datastore = datastore
        self.group = datastore._group
        self.name = name
        # get rays and bins
        # retrieve number of intermediate records between record_number and record_end
        intermediate = [
            0
            for irec in datastore.ds["intermediate_records"]
            if irec["record_number"] <= datastore.ds["record_end"]
        ]
        nrays = (
            datastore.ds["record_end"]
            - datastore.ds["record_number"]
            + 1
            - len(intermediate)
        )
        nbins = max([v["ngates"] for k, v in datastore.ds["sweep_data"].items()])
        word_size = datastore.ds["sweep_data"][name]["word_size"]
        width = {8: 1, 16: 2}[word_size]
        self.dtype = np.dtype(f">u{width}")
        self.shape = (nrays, nbins)

    def _getitem(self, key):
        with self.datastore.lock:
            # read the data if not available
            try:
                data = self.datastore.ds["sweep_data"][self.name]["data"]
            except KeyError:
                self.datastore.root.get_data(self.group, self.name)
                data = self.datastore.ds["sweep_data"][self.name]["data"]
            # see 3.2.4.17.6 Table XVII-I Data Moment Characteristics and Conversion for Data Names
            word_size = self.datastore.ds["sweep_data"][self.name]["word_size"]
            if self.name == "PHI" and word_size == 16:
                # 10 bit mask, but only for 2 byte data
                x = np.uint16(0x3FF)
            elif self.name == "ZDR" and word_size == 16:
                # 11 bit mask, but only for 2 byte data
                x = np.uint16(0x7FF)
            else:
                x = np.uint8(0xFF)
            if len(data[0]) < self.shape[1]:
                return np.pad(
                    np.vstack(data) & x,
                    ((0, 0), (0, self.shape[1] - len(data[0]))),
                    mode="constant",
                    constant_values=0,
                )[key]
            else:
                return (np.vstack(data) & x)[key]

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._getitem,
        )


class NexradLevel2Store(AbstractDataStore):
    def __init__(self, manager, group=None, lock=NEXRADL2_LOCK):
        self._manager = manager
        self._group = int(group[6:])
        self._filename = self.filename
        self.lock = ensure_lock(lock)

    @classmethod
    def open(cls, filename, mode="r", group=None, lock=None, **kwargs):
        if lock is None:
            lock = NEXRADL2_LOCK
        manager = CachingFileManager(
            NEXRADLevel2File, filename, mode=mode, kwargs=kwargs
        )
        return cls(manager, group=group, lock=lock)

    @classmethod
    def open_groups(cls, filename, groups, mode="r", lock=None, **kwargs):
        if lock is None:
            lock = NEXRADL2_LOCK
        manager = CachingFileManager(
            NEXRADLevel2File, filename, mode=mode, kwargs=kwargs
        )
        return {group: cls(manager, group=group, lock=lock) for group in groups}

    @property
    def filename(self):
        with self._manager.acquire_context(False) as root:
            return root.filename

    @property
    def root(self):
        with self._manager.acquire_context(False) as root:
            return root

    def _acquire(self, needs_lock=True):
        with self._manager.acquire_context(needs_lock) as root:
            root.get_sweep(self._group)
            ds = root.data[self._group]
        return ds

    @property
    def ds(self):
        return self._acquire()

    def open_store_variable(self, name, var):
        dim = "azimuth"

        data = indexing.LazilyOuterIndexedArray(
            NexradLevel2ArrayWrapper(self, name, var)
        )
        encoding = {"group": self._group, "source": self._filename}

        mname = nexrad_mapping.get(name, name)
        mapping = sweep_vars_mapping.get(mname, {})
        attrs = {key: mapping[key] for key in moment_attrs if key in mapping}
        attrs["scale_factor"] = 1.0 / var["scale"]
        attrs["add_offset"] = -var["offset"] / var["scale"]
        attrs["coordinates"] = (
            "elevation azimuth range latitude longitude altitude time"
        )
        return mname, Variable((dim, "range"), data, attrs, encoding)

    def open_store_coordinates(self):
        msg_31_header = self.root.msg_31_header[self._group]
        msg_31_data_header = self.root.msg_31_data_header[self._group]
        # check message type
        msg_type = msg_31_data_header["msg_type"]
        if msg_type == 1:
            angle_scale = 180 / (4096 * 8.0)
        else:
            angle_scale = 1.0
        # azimuth/elevation
        azimuth = np.array([ms["azimuth_angle"] for ms in msg_31_header]) * angle_scale
        elevation = (
            np.array([ms["elevation_angle"] for ms in msg_31_header]) * angle_scale
        )

        # time
        # date is 1-based (1 -> 1970-01-01T00:00:00), so we need to subtract 1
        date = (np.array([ms["collect_date"] for ms in msg_31_header]) - 1) * 86400e3
        milliseconds = np.array([ms["collect_ms"] for ms in msg_31_header])
        rtime = date + milliseconds
        time_prefix = "milli"
        rtime_attrs = get_time_attrs(date_unit=f"{time_prefix}seconds")

        # site coords
        vol = self.ds["sweep_constant_data"]["VOL"]
        lon, lat, alt = vol["lon"], vol["lat"], vol["height"] + vol["feedhorn_height"]

        # range
        sweep_data = list(self.ds["sweep_data"].values())[0]
        first_gate = sweep_data["first_gate"]
        gate_spacing = sweep_data["gate_spacing"]
        ngates = max([v["ngates"] for k, v in self.ds["sweep_data"].items()])
        last_gate = first_gate + gate_spacing * (ngates - 0.5)
        rng = np.arange(first_gate, last_gate, gate_spacing, "float32")
        range_attrs = get_range_attrs(rng)

        encoding = {"group": self._group}
        dim = "azimuth"
        sweep_mode = "azimuth_surveillance"
        sweep_number = self._group
        prt_mode = "not_set"
        follow_mode = "not_set"

        elev_data = self.root.msg_5["elevation_data"]
        # in some cases msg_5 doesn't contain meaningful data
        # we extract the needed values from the data header instead
        if elev_data:
            fixed_angle = self.root.msg_5["elevation_data"][self._group][
                "elevation_angle"
            ]
        else:
            fixed_angle = self.root.msg_31_header[self._group][0]["elevation_angle"]
        fixed_angle *= angle_scale

        coords = {
            "azimuth": Variable((dim,), azimuth, get_azimuth_attrs(), encoding),
            "elevation": Variable((dim,), elevation, get_elevation_attrs(), encoding),
            "time": Variable((dim,), rtime, rtime_attrs, encoding),
            "range": Variable(("range",), rng, range_attrs),
            "longitude": Variable((), lon, get_longitude_attrs()),
            "latitude": Variable((), lat, get_latitude_attrs()),
            "altitude": Variable((), alt, get_altitude_attrs()),
            "sweep_mode": Variable((), sweep_mode),
            "sweep_number": Variable((), sweep_number),
            "prt_mode": Variable((), prt_mode),
            "follow_mode": Variable((), follow_mode),
            "sweep_fixed_angle": Variable((), fixed_angle),
        }

        return coords

    def get_variables(self):
        return FrozenDict(
            (k1, v1)
            for k1, v1 in {
                **dict(
                    self.open_store_variable(k, v)
                    for k, v in self.ds["sweep_data"].items()
                ),
                **self.open_store_coordinates(),
            }.items()
        )

    def get_attrs(self):
        _attributes = [
            ("instrument_name", self.root.volume_header["icao"].decode()),
            ("scan_name", f"VCP-{self.root.msg_5['pattern_number']}"),
        ]

        return FrozenDict(_attributes)


class NexradLevel2BackendEntrypoint(BackendEntrypoint):
    """
    Xarray BackendEntrypoint for NEXRAD Level2 Data
    """

    description = "Open NEXRAD Level2 files in Xarray"
    url = "tbd"

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
        lock=None,
        first_dim="auto",
        reindex_angle=False,
        fix_second_angle=False,
        site_coords=True,
        optional=True,
    ):
        store = NexradLevel2Store.open(
            filename_or_obj,
            group=group,
            lock=lock,
            loaddata=False,
        )

        store_entrypoint = StoreBackendEntrypoint()

        ds = store_entrypoint.open_dataset(
            store,
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            concat_characters=concat_characters,
            decode_coords=decode_coords,
            drop_variables=drop_variables,
            use_cftime=use_cftime,
            decode_timedelta=decode_timedelta,
        )

        # reassign azimuth/elevation/time coordinates
        ds = ds.assign_coords({"azimuth": ds.azimuth})
        ds = ds.assign_coords({"elevation": ds.elevation})
        ds = ds.assign_coords({"time": ds.time})

        ds.encoding["engine"] = "nexradlevel2"

        # handle duplicates and reindex
        if decode_coords and reindex_angle is not False:
            ds = ds.pipe(util.remove_duplicate_rays)
            ds = ds.pipe(util.reindex_angle, **reindex_angle)
            ds = ds.pipe(util.ipol_time, **reindex_angle)

        # handling first dimension
        dim0 = "elevation" if ds.sweep_mode.load() == "rhi" else "azimuth"

        # todo: could be optimized
        if first_dim == "time":
            ds = ds.swap_dims({dim0: "time"})
            ds = ds.sortby("time")
        else:
            ds = ds.sortby(dim0)

        # assign geo-coords
        if site_coords:
            ds = ds.assign_coords(
                {
                    "latitude": ds.latitude,
                    "longitude": ds.longitude,
                    "altitude": ds.altitude,
                }
            )

        return ds


def open_nexradlevel2_datatree(
    filename_or_obj,
    mask_and_scale=True,
    decode_times=True,
    concat_characters=True,
    decode_coords=True,
    drop_variables=None,
    use_cftime=None,
    decode_timedelta=None,
    sweep=None,
    first_dim="auto",
    reindex_angle=False,
    fix_second_angle=False,
    site_coords=True,
    optional=True,
    lock=None,
    **kwargs,
):
    """Open a NEXRAD Level2 dataset as an `xarray.DataTree`.

    This function loads NEXRAD Level2 radar data into a DataTree structure, which
    organizes radar sweeps as separate nodes. Provides options for decoding time
    and applying various transformations to the data.

    Parameters
    ----------
    filename_or_obj : str, Path, file-like, or DataStore
        The path or file-like object representing the radar file.
        Path-like objects are interpreted as local or remote paths.

    mask_and_scale : bool, optional
        If True, replaces values in the dataset that match `_FillValue` with NaN
        and applies scale and offset adjustments. Default is True.

    decode_times : bool, optional
        If True, decodes time variables according to CF conventions. Default is True.

    concat_characters : bool, optional
        If True, concatenates character arrays along the last dimension, forming
        string arrays. Default is True.

    decode_coords : bool, optional
        If True, decodes the "coordinates" attribute to identify coordinates in the
        resulting dataset. Default is True.

    drop_variables : str or list of str, optional
        Specifies variables to exclude from the dataset. Useful for removing problematic
        or inconsistent variables. Default is None.

    use_cftime : bool, optional
        If True, uses cftime objects to represent time variables; if False, uses
        `np.datetime64` objects. If None, chooses the best format automatically.
        Default is None.

    decode_timedelta : bool, optional
        If True, decodes variables with units of time (e.g., seconds, minutes) into
        timedelta objects. If False, leaves them as numeric values. Default is None.

    sweep : int or list of int, optional
        Sweep numbers to extract from the dataset. If None, extracts all sweeps into
        a list. Default is the first sweep.

    first_dim : {"time", "auto"}, optional
        Defines the first dimension for each sweep. If "time," uses time as the
        first dimension. If "auto," determines the first dimension based on the sweep
        type (azimuth or elevation). Default is "auto."

    reindex_angle : bool or dict, optional
        Controls angle reindexing. If True or a dictionary, applies reindexing with
        specified settings (if given). Only used if `decode_coords=True`. Default is False.

    fix_second_angle : bool, optional
        If True, corrects errors in the second angle data, such as misaligned
        elevation or azimuth values. Default is False.

    site_coords : bool, optional
        Attaches radar site coordinates to the dataset if True. Default is True.

    optional : bool, optional
        If True, suppresses errors for optional dataset attributes, making them
        optional instead of required. Default is True.

    kwargs : dict
        Additional keyword arguments passed to `xarray.open_dataset`.

    Returns
    -------
    dtree : xarray.DataTree
        An `xarray.DataTree` representing the radar data organized by sweeps.
    """
    from xarray.core.treenode import NodePath

    comment = None

    if isinstance(sweep, str):
        sweep = NodePath(sweep).name
        sweeps = [sweep]
    elif isinstance(sweep, int):
        sweeps = [f"sweep_{sweep}"]
    elif isinstance(sweep, list):
        if isinstance(sweep[0], int):
            sweeps = [f"sweep_{i}" for i in sweep]
        elif isinstance(sweep[0], str):
            sweeps = [NodePath(i).name for i in sweep]
        else:
            raise ValueError(
                "Invalid type in 'sweep' list. Expected integers (e.g., [0, 1, 2]) or strings (e.g. [/sweep_0, sweep_1])."
            )
    else:
        with NEXRADLevel2File(filename_or_obj, loaddata=False) as nex:
            nsweeps = nex.msg_5["number_elevation_cuts"]
            n_sweeps = len(nex.msg_31_data_header)
            # check for zero (old files)
            if nsweeps == 0:
                nsweeps = n_sweeps
                comment = "No message 5 information available"
            # Check if duplicated sweeps ("split cut mode")
            elif nsweeps > n_sweeps:
                nsweeps = n_sweeps
                comment = "Split Cut Mode scanning strategy"

        sweeps = [f"sweep_{i}" for i in range(nsweeps)]

    sweep_dict = open_sweeps_as_dict(
        filename_or_obj=filename_or_obj,
        mask_and_scale=mask_and_scale,
        decode_times=decode_times,
        concat_characters=concat_characters,
        decode_coords=decode_coords,
        drop_variables=drop_variables,
        use_cftime=use_cftime,
        decode_timedelta=decode_timedelta,
        sweeps=sweeps,
        first_dim=first_dim,
        reindex_angle=reindex_angle,
        fix_second_angle=fix_second_angle,
        site_coords=site_coords,
        optional=optional,
        lock=lock,
        **kwargs,
    )
    ls_ds: list[xr.Dataset] = [sweep_dict[sweep] for sweep in sweep_dict.keys()]
    ls_ds.insert(0, xr.Dataset())
    if comment is not None:
        ls_ds[0].attrs["comment"] = comment
    dtree: dict = {
        "/": _assign_root(ls_ds),
        "/radar_parameters": _get_subgroup(ls_ds, radar_parameters_subgroup),
        "/georeferencing_correction": _get_subgroup(
            ls_ds, georeferencing_correction_subgroup
        ),
        "/radar_calibration": _get_radar_calibration(ls_ds, radar_calibration_subgroup),
    }
    # todo: refactor _assign_root and _get_subgroup to recieve dict instead of list of datasets.
    # avoiding remove the attributes in the following line
    sweep_dict = {
        sweep_path: sweep_dict[sweep_path].drop_attrs(deep=False)
        for sweep_path in sweep_dict.keys()
    }
    dtree = dtree | sweep_dict
    return DataTree.from_dict(dtree)


def open_sweeps_as_dict(
    filename_or_obj,
    mask_and_scale=True,
    decode_times=True,
    concat_characters=True,
    decode_coords=True,
    drop_variables=None,
    use_cftime=None,
    decode_timedelta=None,
    sweeps=None,
    first_dim="auto",
    reindex_angle=False,
    fix_second_angle=False,
    site_coords=True,
    optional=True,
    lock=None,
    **kwargs,
):
    stores = NexradLevel2Store.open_groups(
        filename=filename_or_obj,
        lock=lock,
        groups=sweeps,
    )
    groups_dict = {}
    for path_group, store in stores.items():
        store_entrypoint = StoreBackendEntrypoint()
        with close_on_error(store):
            group_ds = store_entrypoint.open_dataset(
                store,
                mask_and_scale=mask_and_scale,
                decode_times=decode_times,
                concat_characters=concat_characters,
                decode_coords=decode_coords,
                drop_variables=drop_variables,
                use_cftime=use_cftime,
                decode_timedelta=decode_timedelta,
            )
            # reassign azimuth/elevation/time coordinates
            group_ds = group_ds.assign_coords({"azimuth": group_ds.azimuth})
            group_ds = group_ds.assign_coords({"elevation": group_ds.elevation})
            group_ds = group_ds.assign_coords({"time": group_ds.time})

            group_ds.encoding["engine"] = "nexradlevel2"

            # handle duplicates and reindex
            if decode_coords and reindex_angle is not False:
                group_ds = group_ds.pipe(util.remove_duplicate_rays)
                group_ds = group_ds.pipe(util.reindex_angle, **reindex_angle)
                group_ds = group_ds.pipe(util.ipol_time, **reindex_angle)

            # handling first dimension
            dim0 = "elevation" if group_ds.sweep_mode.load() == "rhi" else "azimuth"

            # todo: could be optimized
            if first_dim == "time":
                group_ds = group_ds.swap_dims({dim0: "time"})
                group_ds = group_ds.sortby("time")
            else:
                group_ds = group_ds.sortby(dim0)

            # assign geo-coords
            if site_coords:
                group_ds = group_ds.assign_coords(
                    {
                        "latitude": group_ds.latitude,
                        "longitude": group_ds.longitude,
                        "altitude": group_ds.altitude,
                    }
                )

            groups_dict[path_group] = group_ds
    return groups_dict
