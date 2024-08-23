#!/usr/bin/env python
# Copyright (c) 2022, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Xarray Backends
===============

.. toctree::
    :maxdepth: 4

.. automodule:: xradar.io.backends.cfradial1
.. automodule:: xradar.io.backends.gamic
.. automodule:: xradar.io.backends.odim
.. automodule:: xradar.io.backends.furuno
.. automodule:: xradar.io.backends.rainbow
.. automodule:: xradar.io.backends.hpl
.. automodule:: xradar.io.backends.iris
.. automodule:: xradar.io.backends.metek
.. automodule:: xradar.io.backends.hpl
.. automodule:: xradar.io.backends.nexrad_level2
.. automodule:: xradar.io.backends.datamet

"""

from .cfradial1 import *  # noqa
from .datamet import *  # noqa
from .furuno import *  # noqa
from .gamic import *  # noqa
from .hpl import *  # noqa
from .iris import *  # noqa
from .metek import *  # noqa
from .nexrad_level2 import *  # noqa
from .odim import *  # noqa
from .rainbow import *  # noqa
from .datamet import *  # noqa

__all__ = [s for s in dir() if not s.startswith("_")]
