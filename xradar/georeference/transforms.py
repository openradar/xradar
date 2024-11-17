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

__all__ = [
    "antenna_to_cartesian",
    "cartesian_to_geographic_aeqd",
    "get_x_y_z",
    "get_lat_lon_alt",
    "get_x_y_z_tree",
]

__doc__ = __doc__.format("\n   ".join(__all__))


import warnings

import numpy as np
import xarray as xr

from .projection import add_crs, get_crs, get_earth_radius


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


def cartesian_to_geographic_aeqd(x, y, lon_0, lat_0, earth_radius):
    """
    Transform Cartesian coordinates (x, y) to geographic coordinates (latitude, longitude)
    using the Azimuthal Equidistant (AEQD) map projection.

    Parameters
    ----------
    x, y : array-like
        Cartesian coordinates in the same units as the Earth's radius, typically meters.
    lon_0, lat_0 : float
        Longitude and latitude, in degrees, of the center of the projection.
    earth_radius : float
        Radius of the Earth in meters.

    Returns
    -------
    lon, lat : array
        Longitude and latitude of Cartesian coordinates in degrees.

    Notes
    -----
    The calculations follow the AEQD projection equations, where the Earth's radius is used
    to define the distance metric.
    """
    # Ensure x and y are at least 1D arrays
    x = np.atleast_1d(np.asarray(x))
    y = np.atleast_1d(np.asarray(y))

    # Convert reference latitude and longitude to radians
    lat_0_rad = np.deg2rad(lat_0)
    lon_0_rad = np.deg2rad(lon_0)

    # Calculate distance (rho) and angular distance (c)
    rho = np.sqrt(x**2 + y**2)
    c = rho / earth_radius

    # Suppress warnings for potential division by zero
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lat_rad = np.arcsin(
            np.cos(c) * np.sin(lat_0_rad) + (y * np.sin(c) * np.cos(lat_0_rad) / rho)
        )

    # Convert latitude to degrees and handle edge cases where rho == 0
    lat_deg = np.rad2deg(lat_rad)
    lat_deg[rho == 0] = lat_0

    # Calculate longitude in radians
    x1 = x * np.sin(c)
    x2 = rho * np.cos(lat_0_rad) * np.cos(c) - y * np.sin(lat_0_rad) * np.sin(c)
    lon_rad = lon_0_rad + np.arctan2(x1, x2)

    # Convert longitude to degrees and normalize to [-180, 180]
    lon_deg = np.rad2deg(lon_rad)
    lon_deg = (lon_deg + 180) % 360 - 180  # Normalize longitude

    return lon_deg, lat_deg


def get_x_y_z(ds, earth_radius=None, effective_radius_fraction=None):
    """
    Return Cartesian coordinates from antenna coordinates.

    Parameters
    ----------
    ds: xarray.Dataset
        Xarray dataset containing range, azimuth, and elevation
    earth_radius: float
        Radius of the earth. Defaults to a latitude-dependent radius derived from
        WGS84 ellipsoid.
    effective_radius_fraction: float
        Fraction of earth to use for the effective radius (default is 4/3).

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
    if earth_radius is None:
        crs = get_crs(ds)
        earth_radius = get_earth_radius(crs, ds.latitude.values)
        ds = ds.pipe(add_crs, crs)

    # Calculate x, y, and z from the dataset
    ds["x"], ds["y"], ds["z"] = antenna_to_cartesian(
        ds.range,
        ds.azimuth,
        ds.elevation,
        earth_radius=earth_radius,
        effective_radius_fraction=effective_radius_fraction,
        site_altitude=ds.altitude.values,
    )

    # Set the attributes for the dataset
    # todo: possible utilize crs.cs_to_cf() for x/y
    ds.x.attrs = {"standard_name": "east_west_distance_from_radar", "units": "meters"}

    ds.y.attrs = {"standard_name": "north_south_distance_from_radar", "units": "meters"}

    ds.z.attrs = {"standard_name": "height_above_ground", "units": "meters"}

    # Make sure the coordinates are set properly if it is a dataset
    if isinstance(ds, xr.Dataset):
        ds = ds.set_coords(["x", "y", "z"])

    return ds


# def get_x_y_z_tree(radar, earth_radius=None, effective_radius_fraction=None):
#     """
#     Applies the georeferencing to a xradar datatree

#     Parameters
#     ----------
#     radar: xarray.DataTree
#         Xradar datatree object with radar information.
#     earth_radius: float
#         Radius of the earth. Defaults to a latitude-dependent radius derived from
#         WGS84 ellipsoid.
#     effective_radius_fraction: float
#         Fraction of earth to use for the effective radius (default is 4/3).

#     Returns
#     -------
#     radar: xarray.DataTree
#         Datatree with sweep datasets including georeferenced coordinates
#     """
#     for key in list(radar.children):
#         if "sweep" in key:
#             radar[key].ds = get_x_y_z(
#                 radar[key].to_dataset(),
#                 earth_radius=earth_radius,
#                 effective_radius_fraction=effective_radius_fraction,
#             )
#     return radar


def get_lat_lon_alt(ds, earth_radius=None, effective_radius_fraction=None):
    """
    Add geographic coordinates (lat, lon, alt) to the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Xarray dataset containing range, azimuth, and elevation.
    earth_radius : float
        Radius of the earth. Defaults to a latitude-dependent radius derived from
        WGS84 ellipsoid.
    effective_radius_fraction : float
        Fraction of earth to use for the effective radius (default is 4/3).

    Returns
    -------
    ds : xarray.Dataset
        Dataset including lat, lon, and alt as coordinates.
    """
    if earth_radius is None:
        crs = get_crs(ds)
        earth_radius = get_earth_radius(crs, ds.latitude.values)
        ds = ds.pipe(add_crs, crs)

    # Calculate Cartesian coordinates first
    x, y, z = antenna_to_cartesian(
        ds.range,
        ds.azimuth,
        ds.elevation,
        earth_radius=earth_radius,
        effective_radius_fraction=effective_radius_fraction,
        site_altitude=ds.altitude.values,
    )

    # Convert Cartesian coordinates to geographic coordinates
    lon, lat = cartesian_to_geographic_aeqd(
        x, y, ds.longitude.values, ds.latitude.values, earth_radius=earth_radius
    )
    # alt = z - ds.altitude.values
    alt = z

    # Add coordinates to the dataset
    ds["lat"] = xr.DataArray(lat, dims=["azimuth", "range"])
    ds["lat"].attrs = {"standard_name": "latitude", "units": "degrees_north"}

    ds["lon"] = xr.DataArray(lon, dims=["azimuth", "range"])
    ds["lon"].attrs = {"standard_name": "longitude", "units": "degrees_east"}

    ds["alt"] = xr.DataArray(alt, dims=["azimuth", "range"])
    ds["alt"].attrs = {"standard_name": "altitude", "units": "meters"}

    # Make sure the coordinates are set properly
    if isinstance(ds, xr.Dataset):
        ds = ds.set_coords(["lat", "lon", "alt"])

    return ds


def get_x_y_z_tree(radar, earth_radius=None, effective_radius_fraction=None, geo=False):
    """
    Applies the georeferencing to a xradar datatree.

    Parameters
    ----------
    radar : xarray.DataTree
        Xradar datatree object with radar information.
    earth_radius : float
        Radius of the earth. Defaults to a latitude-dependent radius derived from
        WGS84 ellipsoid.
    effective_radius_fraction : float
        Fraction of earth to use for the effective radius (default is 4/3).
    geo : bool, optional
        If True, adds geographic coordinates (lat, lon, alt) instead of Cartesian
        coordinates (x, y, z). Default is False.

    Returns
    -------
    radar : xarray.DataTree
        Datatree with sweep datasets including georeferenced coordinates.
    """
    for key in list(radar.children):
        if "sweep" in key:
            if geo:
                radar[key].ds = get_lat_lon_alt(
                    radar[key].to_dataset(),
                    earth_radius=earth_radius,
                    effective_radius_fraction=effective_radius_fraction,
                )
            else:
                radar[key].ds = get_x_y_z(
                    radar[key].to_dataset(),
                    earth_radius=earth_radius,
                    effective_radius_fraction=effective_radius_fraction,
                )
    return radar
