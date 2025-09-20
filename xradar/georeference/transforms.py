#!/usr/bin/env python
# Copyright (c) 2022-2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Transformations
^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = ["antenna_to_cartesian", "get_x_y_z", "get_x_y_z_tree"]

__doc__ = __doc__.format("\n   ".join(__all__))


import numpy as np
import pyproj
import xarray as xr

from .projection import add_crs, get_earth_radius, radar_crs


def antenna_to_cartesian(
    ranges,
    azimuths,
    elevations,
    earth_radius=6371000,
    effective_radius_fraction=None,
    site_altitude=0,
):
    """Return Cartesian coordinates from antenna coordinates.

    Parameters
    ----------
    ranges : array-like
        Distances to the center of the radar gates (bins) in meters.
    azimuths : array-like
        Azimuth angle of the radar in degrees.
    elevations : array-like
        Elevation angle of the radar in degrees.
    earth_radius: float
        Radius of the earth (default is 6371000 m).
    effective_radius_fraction: float
        Fraction of earth to use for the effective radius (default is 4/3).
    site_altitude: float
        Altitude amsl of radar site

    Returns
    -------
    x, y, z : array
        Cartesian coordinates in meters from the radar.

    Notes
    -----
    The calculation for Cartesian coordinate is adapted from equations
    2.28(b) and 2.28(c) of Doviak and Zrnić [1]_ assuming a
    standard atmosphere (4/3 Earth's radius model).

    .. math::
        :nowrap:

        \\begin{gather*}
        z = \\sqrt{r^2+R^2+2*r*R*sin(\\theta_e)} - R
        \\\\
        s = R * arcsin(\\frac{r*cos(\\theta_e)}{R+z})
        \\\\
        x = s * sin(\\theta_a)
        \\\\
        y = s * cos(\\theta_a)
        \\end{gather*}

    Where r is the distance from the radar to the center of the gate,
    :math:`\\theta_a` is the azimuth angle, :math:`\\theta_e` is the
    elevation angle, s is the arc length, and R is the effective radius
    of the earth, taken to be 4/3 the mean radius of earth (6371 km).

    References
    ----------
    .. [1] Doviak and Zrnić, Doppler Radar and Weather Observations, Second
        Edition, 1993, p. 21.
    """
    if effective_radius_fraction is None:
        effective_radius_fraction = 4.0 / 3.0
    # Convert the elevation angle from degrees to radians
    theta_e = np.deg2rad(elevations)
    theta_a = np.deg2rad(azimuths)  # azimuth angle in radians.
    R = earth_radius * effective_radius_fraction  # effective radius of earth in meters.
    r = ranges  # distances to gates in meters.

    # take site altitude into account
    sr = R + site_altitude

    z = (2.0 * np.sin(theta_e) * r * sr + r**2 + sr**2) ** 0.5 - R
    s = R * np.arcsin(r * np.cos(theta_e) / (R + z))  # arc length in m.
    x = np.sin(theta_a) * s
    y = np.cos(theta_a) * s
    return x, y, z


def get_x_y_z(ds, earth_radius=None, effective_radius_fraction=None, target_crs=None):
    """
    Add Cartesian coordinates from antenna coordinates.

    Parameters
    ----------
    ds: xarray.Dataset
        Xarray dataset containing range, azimuth, and elevation
    earth_radius: float
        Radius of the earth. Defaults to a latitude-dependent radius derived from
        WGS84 ellipsoid.
    effective_radius_fraction: float
        Fraction of earth to use for the effective radius (default is 4/3).
    target_crs: pyproj.CRS
        Coordinate reference system. If not provided, radar_crs() is used.

    Returns
    -------
    ds : xarray.Dataset
        Dataset including x, y, and z as coordinates.

    Notes
    -----
    The calculation for Cartesian coordinate is adapted from equations
    2.28(b) and 2.28(c) of Doviak and Zrnić [1]_ assuming a
    standard atmosphere (4/3 Earth's radius model).

    .. math::
        :nowrap:

        \\begin{gather*}
        z = \\sqrt{r^2+R^2+2*r*R*sin(\\theta_e)} - R
        \\\\
        s = R * arcsin(\\frac{r*cos(\\theta_e)}{R+z})
        \\\\
        x = s * sin(\\theta_a)
        \\\\
        y = s * cos(\\theta_a)
        \\end{gather*}

    Where r is the distance from the radar to the center of the gate,
    :math:`\\theta_a` is the azimuth angle, :math:`\\theta_e` is the
    elevation angle, s is the arc length, and R is the effective radius
    of the earth, taken to be 4/3 the mean radius of earth (6371 km).

    References
    ----------
    .. [1] Doviak and Zrnić, Doppler Radar and Weather Observations, Second
        Edition, 1993, p. 21.
    """

    ds = ds.copy()

    crs = radar_crs(ds)

    if earth_radius is None:
        earth_radius = get_earth_radius(crs, ds.latitude.values)

    # Calculate x, y, and z from the dataset
    x, y, z = antenna_to_cartesian(
        ds.range,
        ds.azimuth,
        ds.elevation,
        earth_radius=earth_radius,
        effective_radius_fraction=effective_radius_fraction,
        site_altitude=ds.altitude.values,
    )
    ds["x"] = x
    ds["y"] = y
    ds["z"] = z

    # Transform coordinates to target_crs if provided
    if target_crs is not None:

        target_crs = pyproj.CRS(target_crs)

        transformer = pyproj.Transformer.from_crs(crs, target_crs, always_xy=True)
        x_new, y_new = transformer.transform(ds.x.values, ds.y.values)

        ds["x"] = (("azimuth", "range"), x_new.reshape(ds.x.shape))
        ds["y"] = (("azimuth", "range"), y_new.reshape(ds.y.shape))

        crs = target_crs

    # Set the attributes for the dataset
    # todo: possible utilize crs.cs_to_cf() for x/y

    if crs.is_geographic:
        ds.x.attrs = {
            "standard_name": "longitude",
            "long_name": "Geographic Longitude",
            "units": "degrees_east",
            "axis": "X",
        }
        ds.y.attrs = {
            "standard_name": "latitude",
            "long_name": "Geographic Latitude",
            "units": "degrees_north",
            "axis": "Y",
        }
    else:
        ds.x.attrs = {
            "standard_name": "easting",
            "long_name": "Easting",
            "units": "meters",
            "axis": "X",
        }
        ds.y.attrs = {
            "standard_name": "northing",
            "long_name": "Northing",
            "units": "meters",
            "axis": "Y",
        }

    # Add the crs information
    ds = ds.pipe(add_crs, crs)

    # Make sure the coordinates are set properly if it is a dataset
    if isinstance(ds, xr.Dataset):
        ds = ds.set_coords(["x", "y", "z"])

    return ds


def get_x_y_z_tree(
    radar, earth_radius=None, effective_radius_fraction=None, target_crs=None
):
    """
    Applies the georeferencing to a xradar datatree

    Parameters
    ----------
    radar: xarray.DataTree
        Xradar datatree object with radar information.
    earth_radius: float
        Radius of the earth. Defaults to a latitude-dependent radius derived from
        WGS84 ellipsoid.
    effective_radius_fraction: float
        Fraction of earth to use for the effective radius (default is 4/3).
    target_crs: pyproj.CRS
        Coordinate reference system. If not provided, radar_crs() is used.

    Returns
    -------
    radar: xarray.DataTree
        Datatree with sweep datasets including georeferenced coordinates
    """
    for key in list(radar.children):
        if "sweep" in key:
            radar[key].ds = get_x_y_z(
                radar[key].to_dataset(),
                earth_radius=earth_radius,
                effective_radius_fraction=effective_radius_fraction,
                target_crs=target_crs,
            )
    return radar
