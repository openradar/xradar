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

"""
from .backends import (
    CfRadial1BackendEntrypoint,  # noqa
    FurunoBackendEntrypoint,  # noqa
    GamicBackendEntrypoint,  # noqa
    IrisBackendEntrypoint,  # noqa
    NexradLevel2BackendEntrypoint,  # noqa
    OdimBackendEntrypoint,  # noqa
    RainbowBackendEntrypoint,  # noqa
    open_cfradial1_datatree,  # noqa
    open_furuno_datatree,  # noqa
    open_gamic_datatree,  # noqa
    open_iris_datatree,  # noqa
    open_nexradlevel2_datatree,  # noqa
    open_odim_datatree,  # noqa
    open_rainbow_datatree,  # noqa
)

# noqa
from .export import *  # noqa

__all__ = [s for s in dir() if not s.startswith("_")]
