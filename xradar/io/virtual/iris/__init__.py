#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Virtual IRIS/Sigmet RAW
^^^^^^^^^^^^^^^^^^^^^^^

Byte-range virtualization of Vaisala Sigmet/IRIS RAW volumes: ``format``
walks the on-disk framing (pure numpy), ``codec`` provides the
``xradar-iris-sweep`` zarr v3 codec that decodes one moment out of a
whole-sweep byte span at read time.

Kept lazy so that resolving the ``xradar-iris-sweep`` ``zarr.codecs`` entry
point never imports ``virtualizarr``; it does import xradar's backends, whose
struct tables the format walker derives its layouts from.
"""

from xradar.io.virtual import __all__ as _available

# like the parent package: parsers are listed only where their extra is
__all__ = [name for name in ("IrisParser", "IrisSweepCodec") if name in _available]


def __getattr__(name):
    from xradar.io.virtual import _LAZY, lazy_attribute

    return lazy_attribute(
        __name__, name, {k: _LAZY[k] for k in ("IrisSweepCodec", "IrisParser")}
    )


def __dir__():
    return list(__all__)
