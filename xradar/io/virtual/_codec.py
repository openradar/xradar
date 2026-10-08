#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Shared plumbing for xradar's decode-only virtual-store codecs.

Read side only: imports zarr and numpy, never ``virtualizarr``. Every codec
configuration is a frozen public contract (published stores carry it
verbatim), so all codecs validate it the same way: exact name, no unknown
keys, required keys present, and strict value types. A ``bool`` never passes
as an ``int``, and nothing but a ``bool`` passes as a flag (``bool("false")``
is ``True``).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import zarr

from xradar.io.virtual._checks import MIN_ZARR, plain, zarr_supported

_zarr_version = getattr(zarr, "__version__", "unknown")
if not zarr_supported(_zarr_version):
    raise ImportError(
        f"xradar's virtual codecs need zarr>={MIN_ZARR} (zarr v3); "
        f"found zarr {_zarr_version}",
        name="zarr",
    )

from zarr.abc.codec import ArrayBytesCodec  # noqa: E402

if TYPE_CHECKING:
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer, NDBuffer

__all__ = [
    "DecodeOnlyCodec",
    "register_codec_name",
    "check_chunk_shape",
    "normalize_fields",
    "parse_config",
]


def _check_field(codec_name: str, key: str, value, kind: type) -> None:
    if kind is bool:
        ok = isinstance(value, bool)
    else:
        ok = isinstance(value, kind) and not isinstance(value, bool)
    if not ok:
        raise ValueError(
            f"{codec_name} codec configuration {key!r} must be "
            f"{kind.__name__}, got {value!r}"
        )


def parse_config(
    data,
    codec_name: str,
    fields: dict[str, type],
    required: tuple[str, ...],
) -> dict:
    """Validate a serialized codec configuration against its contract."""
    name = data.get("name") if isinstance(data, dict) else None
    if name != codec_name:
        raise ValueError(f"expected codec name {codec_name!r}, got {name!r}")
    config = data.get("configuration") or {}
    if not isinstance(config, dict):
        raise ValueError(f"{codec_name} codec configuration must be a mapping")
    unknown = set(config) - set(fields)
    if unknown:
        # a layout-affecting key from a newer writer must fail loudly, not be
        # silently dropped into a wrong-row-order decode
        raise ValueError(
            f"{codec_name} codec got unknown configuration keys "
            f"{sorted(unknown)}: reading this store may require a newer xradar"
        )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(
            f"{codec_name} codec configuration requires {missing}; "
            f"got {sorted(config)}"
        )
    for key, value in config.items():
        _check_field(codec_name, key, value, fields[key])
    return dict(config)


def normalize_fields(codec, codec_name: str, fields: dict[str, type]) -> None:
    """Coerce numpy scalars on a frozen codec dataclass, then type-check, so
    ``to_dict`` can only ever write configs that ``from_dict`` accepts."""
    for key, kind in fields.items():
        value = plain(getattr(codec, key))
        _check_field(codec_name, key, value, kind)
        object.__setattr__(codec, key, value)


def check_chunk_shape(codec_name: str, shape: tuple[int, ...]) -> None:
    """A chunk decodes to (rays, gates). Concatenating volumes along new
    leading dims (e.g. a volume-time axis) yields shapes like
    (1, rays, gates); all leading dims must be singletons."""
    if len(shape) < 2 or any(dim != 1 for dim in shape[:-2]):
        raise ValueError(
            f"{codec_name} decodes (rays, gates) chunks; got chunk shape "
            f"{shape} whose leading dimensions are not all 1"
        )


class DecodeOnlyCodec(ArrayBytesCodec):
    """zarr plumbing shared by the virtual codecs: the decode runs in a
    worker thread (it is CPU-bound), and there is no encode path."""

    is_fixed_size = False

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(self._decode_sync, chunk_bytes, chunk_spec)

    def _encode_sync(self, chunk_array: NDBuffer, chunk_spec: ArraySpec):
        raise NotImplementedError(f"{type(self).__name__} is decode-only")

    async def _encode_single(self, chunk_array: NDBuffer, chunk_spec: ArraySpec):
        raise NotImplementedError(f"{type(self).__name__} is decode-only")

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        return chunk_spec


def register_codec_name(codec_cls, codec_name: str) -> None:
    """Register a codec class under its name (the entry point does the same
    for readers that never import xradar)."""
    from zarr.registry import register_codec

    register_codec(codec_name, codec_cls)
