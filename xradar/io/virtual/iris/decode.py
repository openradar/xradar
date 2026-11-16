#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""The ``xradar-iris-decode`` zarr codec: raw Sigmet words to physical values.

Most IRIS data types are linear and decode through CF ``scale_factor`` /
``add_offset``. The others cannot: signed words (the sweep codec gives
unsigned words), square roots (``decode_sqi``: RHOHV, SQI), the logarithmic
KDP, the 16-bit IRIS floats (``tofloat``: RAINRATE2, FLIQUID2), and the
1-byte velocity and width, which scale by the volume's nyquist. This
array-array codec decodes them with the eager reader's own decoder from
``SIGMET_DATA_TYPES``, so both readers give the same values. It runs after
``xradar-iris-sweep``, which demuxes the words::

    "codecs": [
        {"name": "xradar-iris-decode",
         "configuration": {"sigmet_data_type": "DB_KDP", "wavelength": 5.33}},
        {"name": "xradar-iris-sweep",
         "configuration": {"moment_index": 2, "ndatatypes": 12,
                           "sort_rays": false, "pad_missing_rays": false}},
    ]

The volume values a decoder takes (the eager ``_VOLUME_KWARGS``) are part of
the configuration, so arrays of volumes that differ in them do not
concatenate: ``wavelength`` (cm) for ``DB_KDP``, the single-PRF ``nyquist``
(m/s) for ``DB_VEL`` and ``DB_WIDTH``, and ``multi_prf_mode_flag`` for
``DB_VEL``. They are passed to the decoder as they are, so an undefined
multi-PRF flag warns on read, like the eager reader. No-data words decode to
NaN. Rows ``pad_missing_rays`` adds hold word 0, which is no data for every
type but the 6 signed ones and ``DB_FLIQUID2``: there it is the value 0.0, and
the slot's NaN azimuth and NaT time mark the padding. In ``codecs`` it is listed before
``xradar-iris-sweep``, so on read it runs after it. Decode-only, registered
like ``xradar-iris-sweep``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from zarr.abc.codec import ArrayArrayCodec

from xradar.io.backends.iris import (
    _VOLUME_KWARGS,
    SIGMET_DATA_TYPES,
    _apply_decoder,
    decode_array,
    decode_phidp,
    decode_phidp2,
)
from xradar.io.virtual._codec import (
    _DecodeOnly,
    normalize_fields,
    parse_config,
    register_codec_name,
)

if TYPE_CHECKING:
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import NDBuffer

DECODE_CODEC_NAME = "xradar-iris-decode"

__all__ = ["IrisDecodeCodec", "DECODE_CODEC_NAME", "decode_codec_config"]

#: Frozen configuration contract (key -> type); order is the JSON order.
_FIELDS = {
    "sigmet_data_type": str,
    "wavelength": float,
    "nyquist": float,
    "multi_prf_mode_flag": int,
}

#: the volume values; wavelength and nyquist must be positive
_VOLUME = ("wavelength", "nyquist", "multi_prf_mode_flag")
_POSITIVE = ("wavelength", "nyquist")


def _cf_scaling(type_code: int) -> tuple[float, float, float | None] | None:
    """(scale_factor, add_offset, fill_or_None) turning raw words into
    physical values, derived from the eager backend's ``SIGMET_DATA_TYPES``
    entry for this type.

    The eager decoders are ``decode_array = (raw + offset)/scale + offset2``
    plus ``decode_phidp``/``_phidp2``, which multiply by 180/360 degrees; a
    ``mask`` value becomes ``_FillValue`` so CF mask_and_scale reproduces the
    eager masking. ``None`` for the types a fixed linear transform cannot
    decode: SIGNED types (the arrays hold the unsigned wire words, words at or
    above half-range would wrap), nonlinear decoders (sqrt SQI/RHOHV, log KDP),
    float-word types (``tofloat``) and the velocity and width scaled by the
    volume's nyquist decode in the ``xradar-iris-decode`` codec instead; the
    categorical types (HCLASS) stay raw.
    """
    entry = SIGMET_DATA_TYPES.get(type_code) or {}
    func = entry.get("func")
    fkw = entry.get("fkw") or {}
    scale = fkw.get("scale")
    offset = fkw.get("offset", 0.0)
    offset2 = fkw.get("offset2", 0.0)
    if not scale:
        return None
    if np.dtype(entry.get("dtype", "uint8")).kind != "u":
        return None  # signed wire dtype: linear CF on our unsigned words wraps
    if func is decode_array and "tofloat" not in fkw:
        return 1.0 / scale, offset / scale + offset2, fkw.get("mask")
    if func is decode_phidp:
        return 180.0 / scale, offset * 180.0 / scale, fkw.get("mask")
    if func is decode_phidp2:
        return 360.0 / scale, offset * 360.0 / scale, fkw.get("mask")
    return None


#: name -> entry of the types this codec decodes: a decoder, but no CF scaling
_DECODED = {
    entry["name"]: entry
    for code, entry in SIGMET_DATA_TYPES.items()
    if entry.get("func") is not None and _cf_scaling(code) is None
}


def decode_codec_config(type_code: int, volume: dict) -> dict[str, Any] | None:
    """Configuration of the decode codec for the data type ``type_code``,
    ``None`` when the type has CF scaling or no decoder. ``volume`` holds the
    volume values (``wavelength`` in cm, the single-PRF ``nyquist``,
    ``multi_prf_mode_flag``); the configuration takes those its decoder
    takes."""
    entry = SIGMET_DATA_TYPES.get(type_code)
    if entry is None or entry["name"] not in _DECODED:
        return None
    config: dict[str, Any] = {"sigmet_data_type": entry["name"]}
    for key in _VOLUME_KWARGS.get(entry["func"], ()):
        config[key] = _FIELDS[key](volume[key])
    return config


@dataclass(frozen=True)
class IrisDecodeCodec(_DecodeOnly, ArrayArrayCodec):
    """Decode raw Sigmet words with the eager reader's decoder."""

    sigmet_data_type: str = ""
    wavelength: float | None = None
    nyquist: float | None = None
    multi_prf_mode_flag: int | None = None
    _entry: dict = field(default=None, init=False, repr=False, compare=False)

    def _volume(self) -> dict[str, Any]:
        """The volume values set in this configuration."""
        return {k: getattr(self, k) for k in _VOLUME if getattr(self, k) is not None}

    def __post_init__(self) -> None:
        fields = {"sigmet_data_type": str} | {k: _FIELDS[k] for k in self._volume()}
        normalize_fields(self, DECODE_CODEC_NAME, fields)
        entry = _DECODED.get(self.sigmet_data_type)
        if entry is None:
            raise ValueError(
                f"{DECODE_CODEC_NAME} cannot decode {self.sigmet_data_type!r}"
            )
        object.__setattr__(self, "_entry", entry)
        takes = set(_VOLUME_KWARGS.get(entry["func"], ()))
        if set(self._volume()) != takes:
            raise ValueError(
                f"{DECODE_CODEC_NAME}: {self.sigmet_data_type!r} takes "
                f"{sorted(takes) or 'no volume values'}, got {self._volume()}"
            )
        for key in _POSITIVE:
            value = getattr(self, key)
            if value is not None and not (np.isfinite(value) and value > 0):
                raise ValueError(
                    f"{DECODE_CODEC_NAME}: {key!r} must be a positive number, "
                    f"got {value!r}"
                )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IrisDecodeCodec:
        config = parse_config(
            data, DECODE_CODEC_NAME, _FIELDS, required=("sigmet_data_type",)
        )
        return cls(**config)

    def to_dict(self) -> dict[str, Any]:
        config = {"sigmet_data_type": self.sigmet_data_type} | self._volume()
        return {"name": DECODE_CODEC_NAME, "configuration": config}

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        """The unsigned words ``xradar-iris-sweep`` gives for this type."""
        from zarr.core.array_spec import ArraySpec
        from zarr.core.dtype import get_data_type_from_native_dtype

        size = np.dtype(self._entry.get("dtype", "uint8")).itemsize
        words = np.dtype(f"u{size}")
        return ArraySpec(
            shape=chunk_spec.shape,
            dtype=get_data_type_from_native_dtype(words),
            fill_value=words.type(0),
            config=chunk_spec.config,
            prototype=chunk_spec.prototype,
        )

    def _decode_sync(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        out_dtype = np.dtype(chunk_spec.dtype.to_native_dtype())
        if out_dtype.kind != "f":
            raise ValueError(
                f"{DECODE_CODEC_NAME} decodes into a float array, not {out_dtype}"
            )
        # the eager reader views the words through the table's dtype (signed
        # types) before decoding; the decoders put NaN under their mask
        words = chunk_array.as_numpy_array().view(self._entry.get("dtype", "uint8"))
        values = _apply_decoder(self._entry, words, **self._volume())
        values = np.ma.getdata(values).astype(out_dtype, copy=False)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(values)


register_codec_name(IrisDecodeCodec, DECODE_CODEC_NAME)
