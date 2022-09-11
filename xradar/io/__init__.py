#!/usr/bin/env python
# Copyright (c) 2022, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Radar Data IO

.. toctree::
    :maxdepth: 2

.. automodule:: xradar.io.backends

"""
from . import backends  # noqa

__all__ = [s for s in dir() if not s.startswith("_")]
