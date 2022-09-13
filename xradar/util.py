#!/usr/bin/env python
# Copyright (c) 2022, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
XRadar Util
===========

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = [
    "has_import",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import importlib.util


def has_import(pkg_name):
    return importlib.util.find_spec(pkg_name)
