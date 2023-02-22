#!/usr/bin/env python
# Copyright (c) 2023, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Projection
^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = ["get_earth_radius", "add_crs", "add_crs_tree", "get_crs"]

__doc__ = __doc__.format("\n   ".join(__all__))

import numpy as np
import pyproj
from xarray import Variable


def get_earth_radius(crs, latitude):
    """Return earth radius (in m) for a given Spheroid model (crs) at a given latitude.

    .. math::

        R^2 = \\frac{a^4 \\cos(f)^2 + b^4 \\sin(f)^2}
        {a^2 \\cos(f)^2 + b^2 \\sin(f)^2}

    Parameters
    ----------
    crs : :py:class:`~pyproj.crs.CoordinateSystem`
        spatial reference object
    latitude : float
        geodetic latitude in degrees

    Returns
    -------
    radius : float
        earth radius in meter
    """
    geod = crs.get_geod()
    latitude = np.radians(latitude)
    radius = np.sqrt(
        (
            np.power(geod.a, 4) * np.power(np.cos(latitude), 2)
            + np.power(geod.b, 4) * np.power(np.sin(latitude), 2)
        )
        / (
            np.power(geod.a, 2) * np.power(np.cos(latitude), 2)
            + np.power(geod.b, 2) * np.power(np.sin(latitude), 2)
        )
    )
    return radius


def get_crs(ds, datum="WGS84"):
    """Return :py:class:`pyproj.crs.CoordinateSystem` from ``spatial_ref`` coordinate.

    Parameters
    ----------
    ds : xarray.Dataset
    datum : str
        datum string, defaults to 'WGS84'

    Returns
    -------
    proj_crs : :py:class:`~pyproj.crs.CoordinateSystem`
    """
    if "spatial_ref" in ds:
        proj_crs = pyproj.CRS.from_cf(ds["spatial_ref"].attrs)
    else:
        proj_crs = pyproj.CRS(
            proj="aeqd",
            datum=datum,
            lon_0=ds.longitude.values,
            lat_0=ds.latitude.values,
        )
    return proj_crs


def add_crs(ds, crs=None, datum="WGS84"):
    """Add ``spatial_ref`` coordinate derived from :py:class:`pyproj.crs.CoordinateSystem`.

    Parameters
    ----------
    ds : xarray.Dataset
    crs : :py:class:`~pyproj.crs.CoordinateSystem`
        ``spatial_ref`` to be added, defaults to ``AEQD`` (with given datum)
    datum : str
        datum string, defaults to 'WGS84'

    Returns
    -------
    ds : xarray.Dataset
        Dataset including ``spatial_ref`` coordinate.
    """
    spatial_ref = Variable((), 0)
    if crs is None:
        proj_crs = get_crs(ds, datum=datum)
    else:
        proj_crs = crs
    spatial_ref.attrs.update(proj_crs.to_cf())
    ds = ds.assign_coords(spatial_ref=spatial_ref)
    return ds


def add_crs_tree(radar, datum="WGS84"):
    """Add ``spatial_ref`` coordinate derived from :py:class:`pyproj.crs.CoordinateSystem`.

    Parameters
    ----------
    radar : xarray.Dataset
    crs : :py:class:`~pyproj.crs.CoordinateSystem`
        ``spatial_ref`` to be added, defaults to ``AEQD`` (with given datum)
    datum : str
        datum string, defaults to 'WGS84'

    Returns
    -------
    radar : datatree.DataTree
        Datatree with sweep datasets including ``spatial_ref`` coordinate.
    """
    for key in list(radar.children):
        if "sweep" in key:
            radar[key].ds = add_crs(radar[key].to_dataset(), datum=datum)
    return radar
