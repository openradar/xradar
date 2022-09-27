#!/usr/bin/env python
# Copyright (c) 2022, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

import numpy as np


def antenna_to_cartesian(
    ranges, azimuths, elevations, earth_radius=6371000, effective_radius_fraction=None
):
    """
    Return Cartesian coordinates from antenna coordinates.
    Parameters
    ----------
    ranges : array
        Distances to the center of the radar gates (bins) in kilometers.
    azimuths : array
        Azimuth angle of the radar in degrees.
    elevations : array
        Elevation angle of the radar in degrees.
    earth_radius: float
        Radius of the earth (default is 6371000 m).
    effective_radius_fraction: float
        Fraction of earth to use for the effective radius (default is 4/3).
    Returns
    -------
    x, y, z : array
        Cartesian coordinates in meters from the radar.
    Notes
    -----
    The calculation for Cartesian coordinate is adapted from equations
    2.28(b) and 2.28(c) of Doviak and Zrnic [1]_ assuming a
    standard atmosphere (4/3 Earth's radius model).
    .. math::
        z = \\sqrt{r^2+R^2+2*r*R*sin(\\theta_e)} - R
        s = R * arcsin(\\frac{r*cos(\\theta_e)}{R+z})
        x = s * sin(\\theta_a)
        y = s * cos(\\theta_a)
    Where r is the distance from the radar to the center of the gate,
    :math:`\\theta_a` is the azimuth angle, :math:`\\theta_e` is the
    elevation angle, s is the arc length, and R is the effective radius
    of the earth, taken to be 4/3 the mean radius of earth (6371 km).
    References
    ----------
    .. [1] Doviak and Zrnic, Doppler Radar and Weather Observations, Second
        Edition, 1993, p. 21.
    """
    if effective_radius_fraction is None:
        effective_radius_fraction = 4.0 / 3.0
    theta_e = np.deg2rad(elevations)  # elevation angle in radians.
    theta_a = np.deg2rad(azimuths)  # azimuth angle in radians.
    R = earth_radius * effective_radius_fraction  # effective radius of earth in meters.
    r = ranges  # distances to gates in meters.

    z = (r**2 + R**2 + 2.0 * r * R * np.sin(theta_e)) ** 0.5 - R
    s = R * np.arcsin(r * np.cos(theta_e) / (R + z))  # arc length in m.
    x = s * np.sin(theta_a)
    y = s * np.cos(theta_a)
    return x, y, z


def get_x_y_z(ds, earth_radius=6371000, effective_radius_fraction=None):
    """
    Return Cartesian coordinates from antenna coordinates.
    Parameters
    ----------
    ds: xarray.Dataset
        Xarray dataset containing range, azimuth, and elevation
    earth_radius: float
        Radius of the earth (default is 6371000 m).
    effective_radius_fraction: float
        Fraction of earth to use for the effective radius (default is 4/3).
    Returns
    -------
    ds : xarray.Dataset
        Dataset including x, y, and z as coordinates.
    Notes
    -----
    The calculation for Cartesian coordinate is adapted from equations
    2.28(b) and 2.28(c) of Doviak and Zrnic [1]_ assuming a
    standard atmosphere (4/3 Earth's radius model).
    .. math::
        z = \\sqrt{r^2+R^2+2*r*R*sin(\\theta_e)} - R
        s = R * arcsin(\\frac{r*cos(\\theta_e)}{R+z})
        x = s * sin(\\theta_a)
        y = s * cos(\\theta_a)
    Where r is the distance from the radar to the center of the gate,
    :math:`\\theta_a` is the azimuth angle, :math:`\\theta_e` is the
    elevation angle, s is the arc length, and R is the effective radius
    of the earth, taken to be 4/3 the mean radius of earth (6371 km).
    References
    ----------
    .. [1] Doviak and Zrnic, Doppler Radar and Weather Observations, Second
        Edition, 1993, p. 21.
    """

    # Calculate x, y, and z from the dataset
    ds['x'], ds['y'], ds['z'] = antenna_to_cartesian(
        ds.range,
        ds.azimuth,
        ds.elevation,
        earth_radius=earth_radius,
        effective_radius_fraction=None,
    )

    # Set the attributes for the dataset
    ds.x.attrs = {'standard_name': 'east_west_distance_from_radar', 'units': 'meters'}

    ds.y.attrs = {'standard_name': 'north_south_distance_from_radar', 'units': 'meters'}

    ds.z.attrs = {'standard_name': 'height_above_ground', 'units': 'meters'}

    # Make sure these
    ds.set_coords(['x', 'y', 'z'])

    return ds
