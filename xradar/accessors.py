#!/usr/bin/env python
# Copyright (c) 2022, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
XRadar Accessors
================

To extend :py:class:`xarray:xarray.DataArray` and  :py:class:`xarray:xarray.Dataset`
xradar provides accessors which downstream libraries can hook into.

This module contains the functionality to create those accessors.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = ["create_xradar_dataarray_accessor"]

__doc__ = __doc__.format("\n   ".join(__all__))

import sys

import xarray as xr


def accessor_constructor(self, xarray_obj):
    self._obj = xarray_obj


def create_function(func):
    def function(self):
        return func(self._obj)

    return function


def create_methods(funcs):
    methods = {}
    for name, func in funcs.items():
        methods[name] = create_function(func)
    return methods


def create_xradar_dataarray_accessor(name, funcs):
    if sys.version_info < (3, 9):
        methods = {"__init__": accessor_constructor, **create_methods(funcs)}
    else:
        methods = {"__init__": accessor_constructor} | create_methods(funcs)
    cls_name = "".join([name.capitalize(), "Accessor"])
    accessor = type(cls_name, (object,), methods)
    return xr.register_dataarray_accessor(name)(accessor)
