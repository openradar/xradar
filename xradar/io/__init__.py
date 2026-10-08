#!/usr/bin/env python
# Copyright (c) 2022, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radar Data IO
=============

.. toctree::
    :maxdepth: 4

.. automodule:: xradar.io.backends
.. automodule:: xradar.io.export
.. automodule:: xradar.io.virtual

"""

from .backends import *  # noqa
from .export import *  # noqa

__all__ = [s for s in dir() if not s.startswith("_")]


def __getattr__(name):
    # `virtual` stays lazy: importing it must not pull zarr/virtualizarr
    # into ordinary eager-reader sessions.
    if name == "virtual":
        import importlib

        return importlib.import_module("xradar.io.virtual")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
