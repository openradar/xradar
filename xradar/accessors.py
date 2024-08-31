#!/usr/bin/env python
# Copyright (c) 2022-2023, openradar developers.
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

from __future__ import annotations  # noqa: F401

__all__ = ["create_xradar_dataarray_accessor"]

__doc__ = __doc__.format("\n   ".join(__all__))

import datatree as dt
import xarray as xr

from .georeference import add_crs, add_crs_tree, get_crs, get_x_y_z, get_x_y_z_tree


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
    methods = {"__init__": accessor_constructor} | create_methods(funcs)
    cls_name = "".join([name.capitalize(), "Accessor"])
    accessor = type(cls_name, (object,), methods)
    return xr.register_dataarray_accessor(name)(accessor)


class XradarAccessor:
    """
    Common Datatree, Dataset, DataArray accessor functionality.
    """

    def __init__(
        self, xarray_obj: xr.Dataset | xr.DataArray | dt.DataTree
    ) -> XradarAccessor:
        self.xarray_obj = xarray_obj


@xr.register_dataarray_accessor("xradar")
class XradarDataArrayAccessor(XradarAccessor):
    """Adds a number of xradar specific methods to xarray.DataArray objects."""

    def georeference(
        self, earth_radius=None, effective_radius_fraction=None
    ) -> xr.DataArray:
        """
        Parameters
        ----------
        earth_radius: float
            Radius of the earth. Defaults to a latitude-dependent radius derived from
            WGS84 ellipsoid.
        effective_radius_fraction: float
            Fraction of earth to use for the effective radius (default is 4/3).
        Returns
        -------
        da = xr.DataArray
            Dataset including x, y, and z as coordinates.
        """
        radar = self.xarray_obj

        return radar.pipe(
            get_x_y_z,
            earth_radius=earth_radius,
            effective_radius_fraction=effective_radius_fraction,
        )

    def add_crs(self) -> xr.DataArray:
        """Add 'spatial_ref' coordinate derived from pyproj.CRS

        Returns
        -------
        da : xr.DataArray
            DataArray including spatial_ref coordinate.
        """
        ds = self.xarray_obj
        return ds.pipe(add_crs)

    def get_crs(self):
        """Retrieve pyproj.CRS from 'spatial_ref' coordinate

        Returns
        -------
        proj_crs : :py:class:`~pyproj.crs.CoordinateSystem`
        """
        radar = self.xarray_obj
        return radar.pipe(get_crs)

    def apply(self, func, *args, **kwargs):
        """
        Apply a function to the DataArray object.
        """
        da = self.xarray_obj
        return da.pipe(func, *args, **kwargs)


@xr.register_dataset_accessor("xradar")
class XradarDataSetAccessor(XradarAccessor):
    """Adds a number of xradar specific methods to xarray.DataArray objects."""

    def georeference(
        self, earth_radius=None, effective_radius_fraction=None
    ) -> xr.Dataset:
        """
        Add georeference information to an xarray dataset
        Parameters
        ----------
        earth_radius: float
            Radius of the earth. Defaults to a latitude-dependent radius derived from
            WGS84 ellipsoid.
        effective_radius_fraction: float
            Fraction of earth to use for the effective radius (default is 4/3).
        Returns
        -------
        da = xarray.Dataset
            Dataset including x, y, and z as coordinates.
        """
        radar = self.xarray_obj
        return radar.pipe(
            get_x_y_z,
            earth_radius=earth_radius,
            effective_radius_fraction=effective_radius_fraction,
        )

    def add_crs(self) -> xr.DataSet:
        """Add 'spatial_ref' coordinate derived from pyproj.CRS

        Returns
        -------
        ds : xarray.Dataset
            Dataset including spatial_ref coordinate.
        """
        radar = self.xarray_obj
        return radar.pipe(add_crs)

    def get_crs(self):
        """Retrieve pyproj.CRS from 'spatial_ref' coordinate

        Returns
        -------
        proj_crs : :py:class:`~pyproj.crs.CoordinateSystem`
        """
        radar = self.xarray_obj
        return radar.pipe(get_crs)

    def apply(self, func, *args, **kwargs):
        """
        Apply a function to the Dataset object.
        """
        ds = self.xarray_obj
        return ds.pipe(func, *args, **kwargs)


@dt.register_datatree_accessor("xradar")
class XradarDataTreeAccessor(XradarAccessor):
    """Adds a number of xradar specific methods to datatree.DataTree objects."""

    def georeference(
        self, earth_radius=None, effective_radius_fraction=None
    ) -> dt.DataTree:
        """
        Add georeference information to an xradar datatree object
        Parameters
        ----------
        earth_radius: float
            Radius of the earth. Defaults to a latitude-dependent radius derived from
            WGS84 ellipsoid.
        effective_radius_fraction: float
            Fraction of earth to use for the effective radius (default is 4/3).
        Returns
        -------
        da = datatree.Datatree
            Datatree including x, y, and z as coordinates.
        """
        radar = self.xarray_obj
        return radar.pipe(
            get_x_y_z_tree,
            earth_radius=earth_radius,
            effective_radius_fraction=effective_radius_fraction,
        )

    def add_crs(self) -> dt.DataTree:
        """Add 'spatial_ref' coordinate derived from pyproj.CRS

        Returns
        -------
        da : datatree.DataTree
            Datatree including spatial_ref coordinate.
        """
        ds = self.xarray_obj
        return ds.pipe(add_crs_tree)

    def apply(self, func, *args, **kwargs):
        """
        Applies a given function to all sweeps in the radar volume.

        Parameters
        ----------
        func : function
            The function to apply to each sweep.
        *args : tuple
            Additional positional arguments to pass to the function.
        **kwargs : dict
            Additional keyword arguments to pass to the function.

        Returns
        -------
        DataTree
            The DataTree object after applying the function to all sweeps.
        """
        radar = self.xarray_obj
        for key in list(radar.children):
            if "sweep" in key:
                radar[key].ds = func(radar[key].to_dataset(), *args, **kwargs)
        return radar
        # return apply_to_sweeps(self.xarray_obj, func, *args, **kwargs)
