#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""The ``xradar-iris-decode`` zarr codec: raw Sigmet words to physical values.

Most IRIS data types are linear and decode through CF ``scale_factor`` /
``add_offset``. The others cannot: signed words (the sweep codec gives
unsigned words), square roots (``decode_sqi``: RHOHV, SQI), the logarithmic
KDP, and the 16-bit IRIS floats (``tofloat``: RAINRATE2, FLIQUID2). This
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

``wavelength`` (cm, a positive number) is required for ``DB_KDP`` and only for
it, so a KDP array of a volume with another wavelength does not concatenate
with this one. No-data words decode to NaN. Rows ``pad_missing_rays`` adds
hold word 0, which is no data for every type but the 6 signed ones and
``DB_FLIQUID2``: there it is the value 0.0, and the slot's NaN azimuth and
NaT time mark the padding. In ``codecs`` it is listed before
``xradar-iris-sweep``, so on read it runs after it. Decode-only, registered
like ``xradar-iris-sweep``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from xradar.io.backends.iris import (
    SIGMET_DATA_TYPES,
    decode_kdp,
    decode_vel,
    decode_width,
)
from xradar.io.virtual._codec import (
    DecodeOnlyArrayCodec,
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
_FIELDS = {"sigmet_data_type": str, "wavelength": float}

_BY_NAME = {
    entry["name"]: entry
    for entry in SIGMET_DATA_TYPES.values()
    if isinstance(entry, dict)
}


def _decodable(entry: dict[str, Any] | None) -> bool:
    """Whether this codec decodes the type: it has a decoder, and it is not
    velocity or width, which need the sweep's nyquist and stay CF-scaled."""
    return entry is not None and entry.get("func") not in (
        None,
        decode_vel,
        decode_width,
    )


def _entry(type_name: str) -> dict[str, Any]:
    """The ``SIGMET_DATA_TYPES`` entry this codec decodes, or ``ValueError``."""
    entry = _BY_NAME.get(type_name)
    if not _decodable(entry):
        raise ValueError(f"{DECODE_CODEC_NAME} cannot decode {type_name!r}")
    return entry


def decode_codec_config(
    type_name: str, wavelength_cm: float
) -> dict[str, str | float] | None:
    """Configuration of the decode codec for ``type_name``, ``None`` when the
    type has no decoder this codec runs."""
    entry = _BY_NAME.get(type_name)
    if not _decodable(entry):
        return None
    config: dict[str, str | float] = {"sigmet_data_type": type_name}
    if entry["func"] is decode_kdp:
        config["wavelength"] = float(wavelength_cm)
    return config


@dataclass(frozen=True)
class IrisDecodeCodec(DecodeOnlyArrayCodec):
    """Decode raw Sigmet words with the eager reader's decoder."""

    sigmet_data_type: str = ""
    wavelength: float | None = None

    def _config(self) -> dict[str, str | float]:
        """The set configuration keys, in the contract's order."""
        return {
            key: getattr(self, key) for key in _FIELDS if getattr(self, key) is not None
        }

    def __post_init__(self) -> None:
        normalize_fields(
            self, DECODE_CODEC_NAME, {key: _FIELDS[key] for key in self._config()}
        )
        is_kdp = _entry(self.sigmet_data_type)["func"] is decode_kdp
        if is_kdp != (self.wavelength is not None):
            raise ValueError(
                f"{DECODE_CODEC_NAME}: 'wavelength' (cm) is required for "
                f"DB_KDP and only for it, got {self.sigmet_data_type!r} with "
                f"wavelength={self.wavelength!r}"
            )
        if is_kdp and not (np.isfinite(self.wavelength) and self.wavelength > 0):
            raise ValueError(
                f"{DECODE_CODEC_NAME}: 'wavelength' must be a positive number "
                f"of cm, got {self.wavelength!r}"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IrisDecodeCodec:
        config = dict(
            (data.get("configuration") or {}) if isinstance(data, dict) else {}
        )
        wavelength = config.get("wavelength")
        if isinstance(wavelength, int) and not isinstance(wavelength, bool):
            # a JSON writer may drop the ".0" of an integral wavelength
            config["wavelength"] = float(wavelength)
            data = {**data, "configuration": config}
        config = parse_config(
            data, DECODE_CODEC_NAME, _FIELDS, required=("sigmet_data_type",)
        )
        return cls(**config)

    def to_dict(self) -> dict[str, Any]:
        return {"name": DECODE_CODEC_NAME, "configuration": self._config()}

    @property
    def _word_dtype(self) -> np.dtype:
        """The unsigned words ``xradar-iris-sweep`` gives for this type."""
        entry_dtype = np.dtype(_entry(self.sigmet_data_type).get("dtype", "uint8"))
        return np.dtype(f"u{entry_dtype.itemsize}")

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        from zarr.core.array_spec import ArraySpec
        from zarr.core.dtype import get_data_type_from_native_dtype

        words = self._word_dtype
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
        entry = _entry(self.sigmet_data_type)
        # the eager reader views the words through the table's dtype (signed
        # types) before decoding
        words = chunk_array.as_numpy_array().view(entry.get("dtype", "uint8"))
        kwargs = dict(entry.get("fkw") or {})
        if self.wavelength is not None:
            kwargs["wavelength"] = self.wavelength
        values = np.ma.filled(entry["func"](words, **kwargs), np.nan)
        values = np.asarray(values, dtype=out_dtype)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(values)


register_codec_name(IrisDecodeCodec, DECODE_CODEC_NAME)
