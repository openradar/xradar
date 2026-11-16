#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for the ``xradar-iris-decode`` codec.

The ramp test compares the codec, read through zarr, with the eager decoder
called the way the eager reader calls it (words viewed through the table's
dtype, its ``fkw``, the wavelength for KDP). Real files cover KDP, RHOHV and
SQI end to end, in the parity tests of ``test_iris_virtual_parser.py``.
"""

import numpy as np
import pytest

zarr = pytest.importorskip("zarr", minversion="3.1.6")

from zarr.core.buffer import default_buffer_prototype  # noqa: E402

from xradar.io.backends import iris  # noqa: E402
from xradar.io.virtual.iris.decode import (  # noqa: E402
    _DECODED,
    DECODE_CODEC_NAME,
    IrisDecodeCodec,
    decode_codec_config,
)

WAVELENGTH_CM = 5.33
NYQUIST, FLAG = 8.5, 1  # 2:3 multi-PRF: the velocity's nyquist is doubled
VOLUME = {"wavelength": WAVELENGTH_CM, "nyquist": NYQUIST, "multi_prf_mode_flag": FLAG}

#: name -> type code
_CODE = {entry["name"]: code for code, entry in iris.SIGMET_DATA_TYPES.items()}

#: every type with a decoder but no CF scaling
_DECODED_NAMES = sorted(_DECODED, key=_CODE.get)


def _through_zarr(words, codec):
    """Raw words as the stored chunk of a float32 array decoded by ``codec``."""
    store_dict = {}
    store = zarr.storage.MemoryStore(store_dict=store_dict)
    zarr.create_array(
        store,
        shape=words.shape,
        chunks=words.shape,
        dtype="float32",
        fill_value=np.nan,
        filters=[codec],
        compressors=None,
    )
    store_dict["c/0/0"] = default_buffer_prototype().buffer.from_bytes(words.tobytes())
    return zarr.open_array(store, mode="r")[:]


def _config(type_name):
    return decode_codec_config(_CODE[type_name], VOLUME)


def test_decoded_types():
    # the 6 signed linear types, the 6 square-root types, KDP, the two
    # IRIS-float types, and the 1-byte velocity and width
    assert len(_DECODED_NAMES) == 17
    assert {"DB_KDP", "DB_RHOHV", "DB_VVEL2", "DB_RAINRATE2", "DB_VEL"} <= set(_DECODED)


@pytest.mark.parametrize("type_name", _DECODED_NAMES)
def test_decode_matches_eager_on_every_word(type_name):
    entry = _DECODED[type_name]
    dtype = np.dtype(entry.get("dtype", "uint8"))
    words = np.arange(2 ** (8 * dtype.itemsize), dtype=f"u{dtype.itemsize}")
    codec = IrisDecodeCodec.from_dict(
        {"name": DECODE_CODEC_NAME, "configuration": _config(type_name)}
    )
    ours = _through_zarr(words.reshape(1, -1), codec)
    # the eager reader's call: the wavelength for KDP, the single-PRF nyquist
    # and the multi-PRF flag for velocity and width
    kwargs = dict(entry.get("fkw") or {})
    if entry["func"] is iris.decode_kdp:
        kwargs["wavelength"] = WAVELENGTH_CM
    if entry["func"] in (iris.decode_vel, iris.decode_width):
        kwargs.update(nyquist=NYQUIST, multi_prf_mode_flag=FLAG)
    eager = np.ma.filled(entry["func"](words.view(dtype), **kwargs), np.nan)
    assert ours.dtype == np.float32
    np.testing.assert_array_equal(ours[0], eager.astype("float32"))


def test_decode_codec_config():
    assert _config("DB_RHOHV") == {"sigmet_data_type": "DB_RHOHV"}
    assert _config("DB_KDP") == {"sigmet_data_type": "DB_KDP", "wavelength": 5.33}
    # what the decoders take: the single-PRF nyquist, the flag for velocity
    assert _config("DB_VEL") == {
        "sigmet_data_type": "DB_VEL",
        "nyquist": 8.5,
        "multi_prf_mode_flag": 1,
    }
    assert _config("DB_WIDTH") == {"sigmet_data_type": "DB_WIDTH", "nyquist": 8.5}
    # CF-scaled types and HydroClass (no decoder) take no decode codec
    for name in ("DB_DBZ", "DB_VEL2", "DB_HCLASS"):
        assert _config(name) is None
    assert decode_codec_config(9999, VOLUME) is None


@pytest.mark.parametrize(
    "config",
    [
        {"sigmet_data_type": "DB_SQI"},
        {"sigmet_data_type": "DB_KDP", "wavelength": WAVELENGTH_CM},
        {"sigmet_data_type": "DB_VEL", "nyquist": 8.5, "multi_prf_mode_flag": 1},
    ],
)
def test_codec_to_dict_round_trip(config):
    data = {"name": DECODE_CODEC_NAME, "configuration": config}
    codec = IrisDecodeCodec.from_dict(data)
    assert codec.to_dict() == data
    assert IrisDecodeCodec.from_dict(codec.to_dict()) == codec


def test_codec_accepts_an_integral_wavelength():
    # a JSON writer may write 10.0 as 10 (S band)
    data = {
        "name": DECODE_CODEC_NAME,
        "configuration": {"sigmet_data_type": "DB_KDP", "wavelength": 10},
    }
    assert IrisDecodeCodec.from_dict(data).wavelength == 10.0


@pytest.mark.parametrize(
    "config, match",
    [
        ({"sigmet_data_type": "DB_NOPE"}, "cannot decode"),
        ({"sigmet_data_type": "DB_DBZ"}, "cannot decode"),
        ({"sigmet_data_type": "DB_KDP"}, r"takes \['wavelength'\]"),
        ({"sigmet_data_type": "DB_VEL", "nyquist": 8.5}, "multi_prf_mode_flag"),
        ({"sigmet_data_type": "DB_WIDTH", "wavelength": 5.33}, r"takes \['nyquist'\]"),
        ({"sigmet_data_type": "DB_SQI", "nyquist": 8.5}, "no volume values"),
        (
            {"sigmet_data_type": "DB_VEL", "nyquist": -1.0, "multi_prf_mode_flag": 0},
            "positive",
        ),
        (
            {"sigmet_data_type": "DB_VEL", "nyquist": 8.5, "multi_prf_mode_flag": 1.0},
            "must be int",
        ),
        ({"sigmet_data_type": "DB_KDP", "wavelength": 0.0}, "positive"),
        ({"sigmet_data_type": "DB_KDP", "wavelength": -5.33}, "positive"),
        ({"sigmet_data_type": "DB_KDP", "wavelength": "5.33"}, "must be float"),
        ({"sigmet_data_type": "DB_KDP", "wavelength": True}, "must be float"),
        ({"sigmet_data_type": "DB_SQI", "scale": 2.0}, "unknown configuration"),
        ({}, "requires"),
    ],
)
def test_codec_from_dict_strict(config, match):
    with pytest.raises(ValueError, match=match):
        IrisDecodeCodec.from_dict({"name": DECODE_CODEC_NAME, "configuration": config})
