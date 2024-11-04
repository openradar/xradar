#!/usr/bin/env python
# Copyright (c) 2022-2024, openradar developers.
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


import xarray as xr

from .georeference import add_crs, add_crs_tree, get_crs, get_x_y_z, get_x_y_z_tree
from .transform import to_cfradial1, to_cfradial2
from .util import map_over_sweeps


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
        self, xarray_obj: xr.Dataset | xr.DataArray | xr.DataTree
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

    def to_cfradial2_datatree(self):
        """Convert a CfRadial1 Dataset to CfRadial2 DataTree."""
        return to_cfradial2(self.xarray_obj)

    def to_cfradial2(self):
        """Convert a CfRadial1 Dataset to CfRadial2 DataTree."""
        return self.to_cfradial2_datatree()

    def to_cf2(self):
        """Alias for CfRadial1 to CfRadial2."""
        return self.to_cfradial2_datatree()


@xr.register_datatree_accessor("xradar")
class XradarDataTreeAccessor(XradarAccessor):
    """Adds a number of xradar specific methods to xarray.DataTree objects."""

    def georeference(
        self, earth_radius=None, effective_radius_fraction=None
    ) -> xr.DataTree:
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
        da = xarray.DataTree
            Datatree including x, y, and z as coordinates.
        """
        radar = self.xarray_obj
        return radar.pipe(
            get_x_y_z_tree,
            earth_radius=earth_radius,
            effective_radius_fraction=effective_radius_fraction,
        )

    def add_crs(self) -> xr.DataTree:
        """Add 'spatial_ref' coordinate derived from pyproj.CRS

        Returns
        -------
        da : xarray.DataTree
            Datatree including spatial_ref coordinate.
        """
        ds = self.xarray_obj
        return ds.pipe(add_crs_tree)

    def map_over_sweeps(self, func, *args, **kwargs):
        """
        Apply a function across all sweep nodes in the DataTree using the xradar accessor.

        This method wraps a given function with the `map_over_sweeps` decorator and applies it
        to all sweep nodes using xarray's pipe mechanism.

        Parameters
        ----------
        func : callable
            A function that operates on an xarray Dataset. This function will be applied to each
            sweep node in the DataTree.
        *args : tuple
            Additional positional arguments passed to the function.
        **kwargs : dict
            Additional keyword arguments passed to the function.

        Returns
        -------
        DataTree
            The modified DataTree with the function applied to sweep nodes.

        Examples
        --------
        >>> dtree2 = dtree.xradar.map_over_sweeps(calculate_rain_rate, ref_field='DBZH')
        >>> display(dtree2["sweep_0"])
        """

        @map_over_sweeps
        def _func(*args, **kwargs):
            return func(*args, **kwargs)

        return self.xarray_obj.pipe(_func, *args, **kwargs)

    def to_cfradial1_dataset(self, calibs=True):
        """Convert a CfRadial2 DataTree to CfRadial1 dataset."""
        return to_cfradial1(self.xarray_obj, calibs=calibs)

    def to_cfradial1(self, calibs=True):
        """Convert a CfRadial2 DataTree to CfRadial1 dataset."""
        return to_cfradial1(self.xarray_obj, calibs=calibs)

    def to_cf1(self):
        """Alias for converting to CfRadial1 dataset."""
        return self.to_cfradial1()
