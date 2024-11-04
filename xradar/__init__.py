#!/usr/bin/env python
# Copyright (c) 2022-2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
xradar
======

Top-level package for xradar.

"""

__author__ = """Open Radar Developers"""
__email__ = "mgroverwx@gmail.com"

# versioning
try:
    from .version import version as __version__
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"

# import subpackages
from . import accessors  # noqa
from . import georeference  # noqa
from . import io  # noqa
from . import model  # noqa
from . import util  # noqa
from .util import map_over_sweeps  # noqa
from . import transform  # noqa

__all__ = [s for s in dir() if not s.startswith("_")]
