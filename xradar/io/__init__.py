#!/usr/bin/env python
# Copyright (c) 2022, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radar Data IO
=============

.. toctree::
    :maxdepth: 4

.. automodule:: xradar.io.backends

"""
from .backends import *  # noqa

__all__ = [s for s in dir() if not s.startswith("_")]
