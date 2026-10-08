#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""The ``xradar-iris-sweep`` zarr codec: per-moment demux of an IRIS sweep span.

One virtual chunk = the contiguous byte range of ALL records of one sweep
(``raw_prod_bhdr`` record headers and the ``ingest_data_header`` prologue
included). Every data-type array of a sweep references the SAME span; this
fused array-bytes codec strips the framing, walks the Sigmet 16-bit-word
RLE stream, and returns its configured interleave ordinal. The walk decodes
every data type at once and is cached per span
(:func:`~xradar.io.virtual.iris.format.walk_span`), so the sweep's other
moments do not walk the same bytes again::

    "codecs": [
        {"name": "xradar-iris-sweep",
         "configuration": {"moment_index": 2, "ndatatypes": 12,
                           "sort_rays": false}},
    ]

There is no separate bytes-bytes stage — IRIS RAW has no container
compression; the RLE is undone here. Decode-only (virtual sources are
read-only). The configuration deliberately holds only the interleave
ordinal, the stride, and the row-layout flags so arrays from different
volumes of the same task remain concatenation-compatible; gate count and
ray-slot count come from the chunk shape, the word size from the dtype.

Registered under the ``zarr.codecs`` entry-point group in
``pyproject.toml`` (and via :func:`zarr.registry.register_codec` on
import), so any zarr reader in an environment with xradar installed
resolves it by name with no explicit import. Only the ``xradar-`` prefixed
name is registered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from xradar.io.virtual._checks import check_output
from xradar.io.virtual._codec import (
    DecodeOnlyCodec,
    check_chunk_shape,
    normalize_fields,
    parse_config,
    register_codec_name,
)
from xradar.io.virtual.iris.format import decode_sweep_moment

if TYPE_CHECKING:
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer, NDBuffer

CODEC_NAME = "xradar-iris-sweep"

__all__ = ["IrisSweepCodec", "CODEC_NAME"]

#: Frozen configuration contract (key -> type); order is the JSON order.
_FIELDS = {
    "moment_index": int,
    "ndatatypes": int,
    "sort_rays": bool,
    "pad_missing_rays": bool,
}


@dataclass(frozen=True)
class IrisSweepCodec(DecodeOnlyCodec):
    """Decode one Sigmet data type from a whole-sweep RAW byte span."""

    moment_index: int = 0
    ndatatypes: int = 1
    sort_rays: bool = False
    pad_missing_rays: bool = False

    def __post_init__(self) -> None:
        normalize_fields(self, CODEC_NAME, _FIELDS)
        if not 0 <= self.moment_index < self.ndatatypes:
            raise ValueError(
                f"moment_index {self.moment_index} out of range for "
                f"ndatatypes={self.ndatatypes}"
            )

    @classmethod
    def from_dict(cls, data: dict) -> IrisSweepCodec:
        config = parse_config(
            data,
            CODEC_NAME,
            _FIELDS,
            required=("moment_index", "ndatatypes"),
        )
        return cls(**config)

    def to_dict(self) -> dict:
        return {
            "name": CODEC_NAME,
            "configuration": {key: getattr(self, key) for key in _FIELDS},
        }

    def _decode_sync(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        span = chunk_bytes.as_numpy_array()  # zero-copy view of the raw span
        shape = tuple(chunk_spec.shape)
        check_chunk_shape(CODEC_NAME, shape)
        dtype, fill_value = check_output(
            chunk_spec.dtype.to_native_dtype(), chunk_spec.fill_value, shape[-2:]
        )
        arr = decode_sweep_moment(
            span,
            self.moment_index,
            self.ndatatypes,
            (shape[-2], shape[-1]),
            dtype,
            fill_value,
            self.sort_rays,
            self.pad_missing_rays,
        ).reshape(shape)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(arr)


register_codec_name(IrisSweepCodec, CODEC_NAME)
