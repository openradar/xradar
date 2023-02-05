#!/usr/bin/env python
# Copyright (c) 2022-2023, openradar developers.
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
    "get_first_angle",
    "get_second_angle",
    "remove_duplicate_rays",
    "reindex_angle",
    "extract_angle_parameters",
    "ipol_time",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import contextlib
import gzip
import importlib.util
import io

import numpy as np
from scipy import interpolate


def has_import(pkg_name):
    return importlib.util.find_spec(pkg_name)


def get_first_angle(ds):
    """Return name of first angle dimension from given dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dateset to get first angle name from.

    Returns
    -------
    first_angle : str
        Name of first angle dimension.
    """
    first_angle = list(set(ds.dims) & set(ds.coords) ^ {"range"})[0]
    if first_angle == "time":
        raise ValueError(
            "first dimension is ``time``, but needed ``azimuth`` or ``elevation``"
        )
    return first_angle


def get_second_angle(ds):
    """Return name of second angle coordinate from given dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dateset to get second angle name from.

    Returns
    -------
    out : str
        Name of second angle coordinate.
    """
    return list(
        ((set(ds.coords) | set(ds.variables)) ^ set(ds.dims)) & {"azimuth", "elevation"}
    )[0]


def remove_duplicate_rays(ds):
    """Remove duplicate rays.

    Parameters
    ----------
    ds : xarray.Dataset
        Dateset to remove duplicate rays.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with duplicate rays removed
    """
    first_angle = get_first_angle(ds)

    _, idx = np.unique(ds[first_angle], return_index=True)
    if len(idx) < len(ds[first_angle]):
        # todo:
        # if times have been calculated with wrong number of rays
        # (ODIM_H5, we would need to recalculate the times
        # how should we find out?
        ds = ds.isel({first_angle: idx})

    return ds


def _reindex_angle(ds, array, tolerance, method="nearest"):
    """Reindex first angle.

    Missing values will be filled by variable's _FillValue.

    Parameters
    ----------
    ds : xarray.Dataset
        Dateset to reindex first angle.
    array : array
        Array with angle values which the Dataset should reindex to.
    tolerance : float
        Angle tolerance up to which angles should be considered for used method.

    Keyword Arguments
    -----------------
    method : str
        Reindexing method, defaults to "nearest". See xarray.Dataset.reindex().

    Returns
    -------
    ds : xarray.Dataset
        Reindexed dataset
    """
    # handle fill value
    fill_value = {
        k: np.asarray(v._FillValue).astype(v.dtype)
        for k, v in ds.items()
        if hasattr(v, "_FillValue")
    }

    angle = get_first_angle(ds)

    # reindex
    ds = ds.reindex(
        {angle: array},
        method=method,
        tolerance=tolerance,
        fill_value=fill_value,
    )
    return ds


def reindex_angle(
    ds,
    start_angle=None,
    stop_angle=None,
    angle_res=None,
    direction=None,
    method="nearest",
    tolerance=None,
):
    """Reindex along first angle.

    Missing values will be filled by variable's _FillValue.

    Parameters
    ----------
    ds : xarray.Dataset
        Dateset to reindex first angle.

    Keyword Arguments
    -----------------
    start_angle : float
        Start angle of dataset.
    stop_angle : float
        Stop angle of dataset.
    angle_res : float
        Angle resolution of the dataset.
    direction : int
        Sweep direction, -1 -> CCW, 1 -> CW.
    method : str
        Reindexing method, defaults to "nearest". See xarray.Dataset.reindex().
    tolerance : float
        Angle tolerance up to which angles should be considered for used method.
        Defaults to angle_res / 2.

    Returns
    -------
    ds : xarray.Dataset
        Reindexed dataset
    """
    if tolerance is None:
        tolerance = angle_res / 2.0

    # handle angle order, angle dimension
    first_angle = get_first_angle(ds)
    second_angle = get_second_angle(ds)

    expected_angle_span = abs(stop_angle - start_angle)
    expected_number_rays = int(np.round(expected_angle_span / angle_res, decimals=0))

    # create reindexing angle
    ang = start_angle + direction * np.arange(
        angle_res / 2.0,
        expected_number_rays * angle_res,
        angle_res,
        dtype=ds[first_angle].dtype,
    )

    ds = ds.sortby(first_angle)
    ds = _reindex_angle(ds, ang, tolerance, method=method)

    # check secondary angle coordinate (no nan)
    # set nan values to reasonable median
    sang = ds[second_angle]
    if np.count_nonzero(np.isnan(sang)):
        ds[second_angle] = sang.fillna(sang.median(skipna=True))

    return ds


def extract_angle_parameters(ds):
    """Extract sweep angle parameters from dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dateset to reindex first angle.

    Returns
    -------
    angle_dict : dict
        Dictionary with parameters.
    """
    angle_dict = {}

    # 1. get first and second angle (az/el)
    first_angle = get_first_angle(ds)
    second_angle = get_second_angle(ds)
    angle_dict.update(first_angle=first_angle, second_angle=second_angle)

    # 2. keep min/max angle and time
    min_angle = ds[first_angle].min(skipna=True)
    max_angle = ds[first_angle].max(skipna=True)
    min_time = ds.time.min(skipna=True)
    max_time = ds.time.max(skipna=True)
    angle_dict.update(
        min_angle=min_angle.values,
        max_angle=max_angle.values,
        min_time=min_time.values,
        max_time=max_time.values,
    )

    # 3. get unique angles
    unique_angles = np.unique(ds[first_angle])

    # check if azimuth can be sorted
    # check if time can be sorted
    angles_are_unique = len(unique_angles) == ds[first_angle].size
    times_are_unique = len(np.unique(ds.time)) == ds.time.size
    angle_dict.update(
        angles_are_unique=angles_are_unique, times_are_unique=times_are_unique
    )

    # 4. get index and value of first measured angle, aka a1gate
    a1gate_idx = ds.time.argmin(first_angle).values
    a1gate_val = ds.time.idxmin(first_angle).values
    angle_dict.update(a1gate_idx=a1gate_idx, a1gate_val=a1gate_val)

    # 5. angle differences
    fdim = ds[first_angle]
    diff = fdim.diff(first_angle)

    # this captures different angle spacing
    # catches also missing rays and double rays
    # and other erroneous ray alignments which result in different first_angle values

    # 6. this finds the median of those diff-values which are over some quantile
    # infact this removes angle differences which are too small
    median_diff = diff.where(diff >= diff.quantile(0.30)).median(skipna=True)
    # unique differences
    diffset = set(diff.values)
    # if there are more than 1 unique angle differences that means
    # non uniform angle spacing
    uniform_angle_spacing = len(diffset) == 1
    angle_dict.update(uniform_angle_spacing=uniform_angle_spacing)

    # 7. ascending/descending/direction
    # PPI ascending CW, descending CCW
    ascending = median_diff > 0
    direction = 1 if ascending else -1
    angle_dict.update(ascending=ascending.values, direction=direction)

    # 8. calculate angle resolution from median_diff and rounding
    angle_res = np.round(median_diff, decimals=2)
    angle_dict.update(angle_res=angle_res.values)

    # 9. start/stop angle
    # so for PPI we assume 0 to 360, for RHI range we try to guess it from the values
    # unfortunately this doesn't work if rays are missing at start or end of the sweep
    # so we need to have the possibility to override
    # also accounts for direction of sweep (CW/CCW, up/down)
    mode = ds.sweep_mode
    if mode == "azimuth_surveillance":
        start_ang = 0
        stop_ang = 360
    elif mode == "rhi":
        # start/stop angle, round to nearest multiple of angle_res
        start_ang = int(np.round(min_angle / angle_res) * angle_res)
        stop_ang = int(np.round(max_angle / angle_res) * angle_res)

    start_ang, stop_ang = (start_ang, stop_ang)[::direction]
    angle_dict.update(start_angle=start_ang, stop_angle=stop_ang)

    # 10. expected angle span
    expected_angle_span = abs(stop_ang - start_ang)
    angle_dict.update(expected_angle_span=expected_angle_span)

    # 11. incomplete sweep
    missing_rays = (angle_res * fdim.size) < expected_angle_span
    excess_rays = (angle_res * fdim.size) > expected_angle_span
    angle_dict.update(missing_rays=missing_rays.values, excess_rays=excess_rays.values)

    # 12. expected number of rays
    expected_number_rays = int(np.round(expected_angle_span / angle_res, decimals=0))
    angle_dict.update(expected_number_rays=expected_number_rays)

    return angle_dict


def _ipol_time(da, start=0, stop=-1):
    """Interpolate/extrapolate missing time steps (NaT).

    Parameters
    ----------
    da : xarray.DataArray
        DataArray to interpolate/extrapolate missing timesteps.

    Returns
    -------
    da : xarray.DataArray
        DataArray with interpolated/extrapolated timesteps.
    """
    dtype = da.dtype
    da_sel = da[start:stop].dropna("azimuth")
    ipol = interpolate.interp1d(
        da_sel.azimuth,
        da_sel.astype(int),
        fill_value="extrapolate",
    )
    da.data[start:stop] = ipol(da.azimuth[start:stop]).astype(dtype)
    return da


def ipol_time(ds):
    """Interpolate/extrapolate missing time steps (NaT).

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to interpolate/extrapolate missing timesteps.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with interpolated/extrapolated timesteps.
    """
    time = ds.time.astype(int)
    amin = time.where(time > 0).argmin("azimuth", skipna=True).values
    amax = time.where(time > 0).argmax("azimuth", skipna=True).values
    time = ds.time
    if amin > amax:
        time = time.pipe(_ipol_time, 0, amin)
        time = time.pipe(_ipol_time, amin, -1)
    else:
        time = time.pipe(_ipol_time)
    ds["time"] = time
    return ds


@contextlib.contextmanager
def _get_data_file(file, file_or_filelike):
    if file_or_filelike == "filelike":
        _open = open
        if file[-3:] == ".gz":
            _open = gzip.open
        with _open(file, mode="r+b") as f:
            yield io.BytesIO(f.read())
    else:
        yield file
