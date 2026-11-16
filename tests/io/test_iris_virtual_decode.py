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


def test_decoded_types():
    # the 6 signed linear types, the 6 square-root types, KDP and the two
    # IRIS-float types
    assert len(_DECODED_NAMES) == 15
    assert {"DB_KDP", "DB_RHOHV", "DB_VVEL2", "DB_RAINRATE2"} <= set(_DECODED)


@pytest.mark.parametrize("type_name", _DECODED_NAMES)
def test_decode_matches_eager_on_every_word(type_name):
    entry = _DECODED[type_name]
    dtype = np.dtype(entry.get("dtype", "uint8"))
    words = np.arange(2 ** (8 * dtype.itemsize), dtype=f"u{dtype.itemsize}")
    config = decode_codec_config(_CODE[type_name], WAVELENGTH_CM)
    codec = IrisDecodeCodec.from_dict(
        {"name": DECODE_CODEC_NAME, "configuration": config}
    )
    ours = _through_zarr(words.reshape(1, -1), codec)
    kwargs = dict(entry.get("fkw") or {})
    if entry["func"] is iris.decode_kdp:
        kwargs["wavelength"] = WAVELENGTH_CM
    eager = np.ma.filled(entry["func"](words.view(dtype), **kwargs), np.nan)
    assert ours.dtype == np.float32
    np.testing.assert_array_equal(ours[0], eager.astype("float32"))


def test_decode_codec_config():
    assert decode_codec_config(_CODE["DB_RHOHV"], WAVELENGTH_CM) == {
        "sigmet_data_type": "DB_RHOHV"
    }
    assert decode_codec_config(_CODE["DB_KDP"], WAVELENGTH_CM) == {
        "sigmet_data_type": "DB_KDP",
        "wavelength": WAVELENGTH_CM,
    }
    # CF-scaled types and HydroClass (no decoder) take no decode codec
    for name in ("DB_VEL", "DB_DBZ", "DB_HCLASS"):
        assert decode_codec_config(_CODE[name], WAVELENGTH_CM) is None
    assert decode_codec_config(9999, WAVELENGTH_CM) is None


@pytest.mark.parametrize(
    "config",
    [
        {"sigmet_data_type": "DB_SQI"},
        {"sigmet_data_type": "DB_KDP", "wavelength": WAVELENGTH_CM},
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
        ({"sigmet_data_type": "DB_VEL"}, "cannot decode"),
        ({"sigmet_data_type": "DB_DBZ"}, "cannot decode"),
        ({"sigmet_data_type": "DB_KDP"}, "required for DB_KDP"),
        ({"sigmet_data_type": "DB_SQI", "wavelength": WAVELENGTH_CM}, "only for it"),
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
