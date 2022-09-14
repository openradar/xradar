#!/usr/bin/env python
# Copyright (c) 2022, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
XRadar Data Model
=================

The data model for the different xradar DataArrays, Datasets and DataTrees is based on NetCDF4/CfRadial2.1.
It will be aligned to the upcoming WMO standard FM301.

This module contains several helper functions to create minimal DataArrays and Datasets, as well as Datatrees.

Must read:

* `CfRadial`_
* `WMO CF extensions`_
* `OPERA/ODIM_H5`_

Code ported from wradlib.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = [
    "create_sweep_dataset",
    "get_azimuth_attrs",
    "get_latitude_attrs",
    "get_longitude_attrs",
    "get_altitude_attrs",
    "get_elevation_attrs",
    "get_moment_attrs",
    "get_range_attrs",
    "get_time_attrs",
    "moment_attrs",
    "sweep_vars_mapping",
    "get_azimuth_dataarray",
    "get_elevation_dataarray",
    "get_range_dataarray",
    "get_sweep_dataarray",
    "get_time_dataarray",
    "required_global_attrs",
    "optional_root_attrs",
    "required_root_vars",
    "optional_root_vars",
    "sweep_coordinate_vars",
    "required_sweep_metadata_vars",
    "optional_sweep_metadata_vars",
    "sweep_dataset_vars",
    "non_standard_sweep_dataset_vars",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import numpy as np
from xarray import DataArray, Dataset, decode_cf

# This is a temporary setup, since CfRadial2.1 and FM301 are not yet finalized.
# Todo: adhere to standards when they are published

#: required global attributes (root-group)
required_global_attrs = dict(
    [
        ("Conventions", "Cf/Radial"),
        ("version", "Cf/Radial version number"),
        ("title", "short description of file contents"),
        ("instrument_name", "nameThe  of radar or lidar"),
        ("institution", "where the original data were produced"),
        (
            "references",
            ("references that describe the data or the methods used to produce it"),
        ),
        ("source", "method of production of the original data"),
        ("history", "list of modifications to the original data"),
        ("comment", "miscellaneous information"),
        ("platform_is_mobile", "'true' or 'false', assumed 'false' if missing"),
    ]
)

# optional global attributes (root-group)
optional_root_attrs = [
    ("site_name", "name of site where data were gathered"),
    ("scan_name", "name of scan strategy used, if applicable"),
    ("scan_id", "scan strategy id, if applicable. assumed 0 if missing"),
    (
        "ray_times_increase",
        (
            "'true' or 'false', assumed 'true' if missing. "
            "This is set to true if ray times increase monotonically "
            "throughout all of the sweeps in the volume."
        ),
    ),
    (
        "simulated",
        (
            "'true' or 'false', assumed 'false' if missing. "
            "data in this file are simulated"
        ),
    ),
]

# required global variables (root-group)
required_root_vars = {
    "volume_number",
    "time_coverage_start",
    "time_coverage_end",
    "latitude",
    "longitude",
    "altitude",
    "platform_type",
    "instrument_type",
}

# optional global attributes (root-group)
optional_root_vars = {
    "altitude_agl",
    "primary_axis",
    "status_str",
}

# sweep-group coordinate variables
sweep_coordinate_vars = {
    "time",
    "range",
    "frequency",
}

# required sweep-group metadata variables
required_sweep_metadata_vars = {
    "sweep_number",
    "sweep_mode",
    "follow_mode",
    "prt_mode",
    "fixed_angle",
    "azimuth",
    "elevation",
}

# optional sweep-group metadata variables
optional_sweep_metadata_vars = {
    "polarization_mode",
    "polarization_sequence",
    "rays_are_indexed",
    "rays_angle_resolution",
    "qc_procedures",
    "target_scan_rate",
    "scan_rate",
    "antenna_transition",
    "pulse_width",
    "calib_index",
    "rx_range_resolution",
    "prt",
    "prt_ratio",
    "prt_sequence",
    "nyquist_velocity",
    "unambiguous_range",
    "n_samples",
}

# sweep dataset variable names
sweep_dataset_vars = {
    "DBZH",
    "DBZV",
    "ZH",
    "ZV",
    "DBTH",
    "DBTV",
    "TH",
    "TV",
    "VRADH",
    "VRADV",
    "WRADH",
    "WRADV",
    "ZDR",
    "LDR",
    "LDRH",
    "LDRV",
    "PHIDP",
    "KDP",
    "PHIHX",
    "RHOHV",
    "RHOHX",
    "RHOVX",
    "DBM",
    "DBMHC",
    "DBMHX",
    "DBMVC",
    "DBMVX",
    "SNR",
    "SNRHC",
    "SNRHX",
    "SNRVC",
    "SNRVX",
    "NCP",
    "NCPH",
    "NCPV",
    "RR",
    "REC",
}

# non-standard sweep dataset variable names
non_standard_sweep_dataset_vars = {
    "DBZ",
    "VEL",
    "VR",
}

# required range attributes
range_attrs = {
    "units": "meters",
    "standard_name": "projection_range_coordinate",
    "long_name": "range_to_measurement_volume",
    "axis": "radial_range_coordinate",
    "spacing_is_constant": "true",
    "meters_to_center_of_first_gate": "{}",
    "meters_between_gates": "{}",
}

# required time attributes
time_attrs = {
    "standard_name": "time",
    "units": "seconds since {}",
}

# required frequency attributes
frequency_attrs = {
    "standard_name": "",
    "units": "s-1",
}

lon_attrs = {
    "long_name": "longitude",
    "units": "degrees_east",
    "standard_name": "longitude",
}
lat_attrs = {
    "long_name": "latitude",
    "units": "degrees_north",
    "positive": "up",
    "standard_name": "latitude",
}
alt_attrs = {
    "long_name": "altitude",
    "units": "meters",
    "standard_name": "altitude",
}

moment_attrs = {"standard_name", "long_name", "units"}

# todo: align this with sweep_dataset_vars
# CfRadial 2.1 / FM301 / ODIM_H5 mapping
sweep_vars_mapping = {
    "DBZH": {
        "standard_name": "radar_equivalent_reflectivity_factor_h",
        "long_name": "Equivalent reflectivity factor H",
        "short_name": "DBZH",
        "units": "dBZ",
    },
    "DBZH_CLEAN": {
        "standard_name": "radar_equivalent_reflectivity_factor_h",
        "long_name": "Equivalent reflectivity factor H",
        "short_name": "DBZH_CLEAN",
        "units": "dBZ",
    },
    "DBZV": {
        "standard_name": "radar_equivalent_reflectivity_factor_v",
        "long_name": "Equivalent reflectivity factor V",
        "short_name": "DBZV",
        "units": "dBZ",
    },
    "ZH": {
        "standard_name": "radar_linear_equivalent_reflectivity_factor_h",
        "long_name": "Linear equivalent reflectivity factor H",
        "short_name": "ZH",
        "units": "unitless",
    },
    "ZV": {
        "standard_name": "radar_equivalent_reflectivity_factor_v",
        "long_name": "Linear equivalent reflectivity factor V",
        "short_name": "ZV",
        "units": "unitless",
    },
    "DBZ": {
        "standard_name": "radar_equivalent_reflectivity_factor",
        "long_name": "Equivalent reflectivity factor",
        "short_name": "DBZ",
        "units": "dBZ",
    },
    "DBTH": {
        "standard_name": "radar_equivalent_reflectivity_factor_h",
        "long_name": "Total power H (uncorrected reflectivity)",
        "short_name": "DBTH",
        "units": "dBZ",
    },
    "DBTV": {
        "standard_name": "radar_equivalent_reflectivity_factor_v",
        "long_name": "Total power V (uncorrected reflectivity)",
        "short_name": "DBTV",
        "units": "dBZ",
    },
    "TH": {
        "standard_name": "radar_linear_equivalent_reflectivity_factor_h",
        "long_name": "Linear total power H (uncorrected reflectivity)",
        "short_name": "TH",
        "units": "unitless",
    },
    "TV": {
        "standard_name": "radar_linear_equivalent_reflectivity_factor_v",
        "long_name": "Linear total power V (uncorrected reflectivity)",
        "short_name": "TV",
        "units": "unitless",
    },
    "VRADH": {
        "standard_name": "radial_velocity_of_scatterers_away_" "from_instrument_h",
        "long_name": "Radial velocity of scatterers away from instrument H",
        "short_name": "VRADH",
        "units": "meters per seconds",
    },
    "VRADV": {
        "standard_name": "radial_velocity_of_scatterers_" "away_from_instrument_v",
        "long_name": "Radial velocity of scatterers away from instrument V",
        "short_name": "VRADV",
        "units": "meters per second",
    },
    "VR": {
        "standard_name": "radial_velocity_of_scatterers_away_" "from_instrument",
        "long_name": "Radial velocity of scatterers away from instrument",
        "short_name": "VR",
        "units": "meters per seconds",
    },
    "VRAD": {
        "standard_name": "radial_velocity_of_scatterers_away_" "from_instrument",
        "long_name": "Radial velocity of scatterers away from instrument",
        "short_name": "VRAD",
        "units": "meters per seconds",
    },
    "VRADDH": {
        "standard_name": "radial_velocity_of_scatterers_away_" "from_instrument_h",
        "long_name": "Radial velocity of scatterers away from instrument H",
        "short_name": "VRADDH",
        "units": "meters per seconds",
    },
    "WRADH": {
        "standard_name": "radar_doppler_spectrum_width_h",
        "long_name": "Doppler spectrum width H",
        "short_name": "WRADH",
        "units": "meters per seconds",
    },
    "UWRADH": {
        "standard_name": "radar_doppler_spectrum_width_h",
        "long_name": "Doppler spectrum width H",
        "short_name": "UWRADH",
        "units": "meters per seconds",
    },
    "WRADV": {
        "standard_name": "radar_doppler_spectrum_width_v",
        "long_name": "Doppler spectrum width V",
        "short_name": "WRADV",
        "units": "meters per second",
    },
    "WRAD": {
        "standard_name": "radar_doppler_spectrum_width",
        "long_name": "Doppler spectrum width",
        "short_name": "WRAD",
        "units": "meters per second",
    },
    "ZDR": {
        "standard_name": "radar_differential_reflectivity_hv",
        "long_name": "Log differential reflectivity H/V",
        "short_name": "ZDR",
        "units": "dB",
    },
    "UZDR": {
        "standard_name": "radar_differential_reflectivity_hv",
        "long_name": "Log differential reflectivity H/V",
        "short_name": "UZDR",
        "units": "dB",
    },
    "LDR": {
        "standard_name": "radar_linear_depolarization_ratio",
        "long_name": "Log-linear depolarization ratio HV",
        "short_name": "LDR",
        "units": "dB",
    },
    "PHIDP": {
        "standard_name": "radar_differential_phase_hv",
        "long_name": "Differential phase HV",
        "short_name": "PHIDP",
        "units": "degrees",
    },
    "UPHIDP": {
        "standard_name": "radar_differential_phase_hv",
        "long_name": "Differential phase HV",
        "short_name": "UPHIDP",
        "units": "degrees",
    },
    "KDP": {
        "standard_name": "radar_specific_differential_phase_hv",
        "long_name": "Specific differential phase HV",
        "short_name": "KDP",
        "units": "degrees per kilometer",
    },
    "RHOHV": {
        "standard_name": "radar_correlation_coefficient_hv",
        "long_name": "Correlation coefficient HV",
        "short_name": "RHOHV",
        "units": "unitless",
    },
    "URHOHV": {
        "standard_name": "radar_correlation_coefficient_hv",
        "long_name": "Correlation coefficient HV",
        "short_name": "URHOHV",
        "units": "unitless",
    },
    "SNRH": {
        "standard_name": "signal_noise_ratio_h",
        "long_name": "Signal Noise Ratio H",
        "short_name": "SNRH",
        "units": "unitless",
    },
    "SNRV": {
        "standard_name": "signal_noise_ratio_v",
        "long_name": "Signal Noise Ratio V",
        "short_name": "SNRV",
        "units": "unitless",
    },
    "SQIH": {
        "standard_name": "signal_quality_index_h",
        "long_name": "Signal Quality H",
        "short_name": "SQIH",
        "units": "unitless",
    },
    "SQIV": {
        "standard_name": "signal_quality_index_v",
        "long_name": "Signal Quality V",
        "short_name": "SQIV",
        "units": "unitless",
    },
    "CCORH": {
        "standard_name": "clutter_correction_h",
        "long_name": "Clutter Correction H",
        "short_name": "CCORH",
        "units": "unitless",
    },
    "CCORV": {
        "standard_name": "clutter_correction_v",
        "long_name": "Clutter Correction V",
        "short_name": "CCORV",
        "units": "unitless",
    },
    "CMAP": {
        "standard_name": "clutter_map",
        "long_name": "Clutter Map",
        "short_name": "CMAP",
        "units": "unitless",
    },
    "RATE": {
        "standard_name": "rainfall_rate",
        "long_name": "rainfall_rate",
        "short_name": "RATE",
        "units": "mm h-1",
    },
}


def get_longitude_attrs():
    lon_attrs = {
        "long_name": "longitude",
        "units": "degrees_east",
        "standard_name": "longitude",
    }
    return lon_attrs


def get_latitude_attrs():
    lat_attrs = {
        "long_name": "latitude",
        "units": "degrees_north",
        "positive": "up",
        "standard_name": "latitude",
    }
    return lat_attrs


def get_altitude_attrs():
    alt_attrs = {
        "long_name": "altitude",
        "units": "meters",
        "standard_name": "altitude",
    }
    return alt_attrs


def get_range_attrs(rng):
    """Get Range CF attributes.

    Parameters
    ----------
    rng : :class:`numpy:numpy.ndarray`
        Array with range values.

    Returns
    -------
    range_attrs : dict
        Dictionary with Range CF attributes.
    """
    diff = np.diff(rng)
    unique = np.unique(diff)
    range_attrs = {
        "units": "meters",
        "standard_name": "projection_range_coordinate",
        "long_name": "range_to_measurement_volume",
        "axis": "radial_range_coordinate",
    }
    if unique:
        spacing = "true"
        range_attrs["meters_between_gates"] = diff[0]
    else:
        spacing = "false"
    range_attrs["spacing_is_constant"] = spacing
    range_attrs["meters_to_center_of_first_gate"] = rng[0]

    return range_attrs


def get_azimuth_attrs(azi=None):
    """Get Azimuth CF attributes.

    Parameters
    ----------
    azi : :class:`numpy:numpy.ndarray`
        Array with azimuth values

    Returns
    -------
    az_attrs : dict
        Dictionary with Azimuth CF attributes.
    """
    az_attrs = {
        "standard_name": "ray_azimuth_angle",
        "long_name": "azimuth_angle_from_true_north",
        "units": "degrees",
        "axis": "radial_azimuth_coordinate",
    }
    if azi is not None:
        unique = len(np.unique(azi)) == 1
        if not unique:
            a1gate = np.argsort(np.argsort(azi))[0]
            az_attrs["a1gate"] = a1gate
    return az_attrs


def get_elevation_attrs(ele=None):
    """Get Elevation CF attributes.

    Parameters
    ----------
    ele : :class:`numpy:numpy.ndarray`
        Array with elevation values

    Returns
    -------
    el_attrs : dict
        Dictionary with Elevation CF attributes.
    """
    el_attrs = {
        "standard_name": "ray_elevation_angle",
        "long_name": "elevation_angle_from_horizontal_plane",
        "units": "degrees",
        "axis": "radial_elevation_coordinate",
    }
    return el_attrs


def get_time_attrs(date_str):
    time_attrs = {
        "standard_name": "time",
        "units": f"seconds since {date_str}",
    }
    return time_attrs


def get_moment_attrs(moment):
    """Get Radar Moment CF attributes.

    Parameters
    ----------
    moment : str
        String describing radar moment.

    Returns
    -------
    moment_attrs : dict
        Dictionary with Radar Moment CF attributes.
    """
    return sweep_vars_mapping[moment].copy()


def get_range_dataarray(rng, nbins=None):
    """Create Range DataArray.

    Parameters
    ----------
    rng : float | :class:`numpy:numpy.ndarray`
        range resolution or array with range values.
    nbins: int
        number of bins, must be passed if rng is provided as resolution

    Returns
    -------
    rng_da : xr.DataArray
        Range DataArray.
    """
    if np.isscalar(rng):
        res2 = rng / 2.0
        rng = np.arange(res2, rng * nbins, rng)
    rng_da = DataArray(data=rng, dims=["range"])
    rng_da.attrs.update(get_range_attrs(rng))
    rng_da.name = "range"
    return rng_da


def get_azimuth_dataarray(azimuth, nrays=None, a1gate=0):
    """Create Azimuth DataArray.

    Parameters
    ----------
    azimuth : float | :class:`numpy:numpy.ndarray`
        Azimuth resolution, fixed azimuth value or array with azimuth values.

    Keyword Arguments
    -----------------
    nrays : int
        number of rays
        If int -> set all rays to same azimuth value.
        If None, given azimuth value is used as resolution.
    a1gate : int
        First measured ray. Defaults to 0.

    Returns
    -------
    azi_da : xr.DataArray
        Azimuth DataArray.
    """
    if np.isscalar(azimuth):
        if nrays is None:
            res2 = azimuth / 2.0
            nrays = int(360 / azimuth)
            azimuth = np.linspace(res2, 360 - res2, nrays)
        else:
            azimuth = np.full(nrays, azimuth)

    azimuth = np.roll(azimuth, -a1gate)
    azi_da = DataArray(data=azimuth, dims=["time"])
    azi_da.attrs.update(get_azimuth_attrs(azimuth))
    azi_da.name = "azimuth"
    return azi_da


def get_elevation_dataarray(elevation, nrays=None):
    """Create Elevation DataArray.

    Parameters
    ----------
    elevation : float | :class:`numpy:numpy.ndarray`
        elevation resolution or array with elevation values.

    Keyword Arguments
    -----------------
    nrays : int
        If int -> set all rays to same elevation value.
        If None, given elevation value is used as resolution.

    Returns
    -------
    ele_da : xr.DataArray
        Elevation DataArray.
    """
    if np.isscalar(elevation):
        if nrays is None:
            res2 = elevation / 2.0
            nrays = int(90 / elevation)
            elevation = np.linspace(res2, 90 - res2, nrays)
        else:
            elevation = np.full(nrays, elevation)
    ele_da = DataArray(data=elevation, dims=["time"])
    ele_da.attrs.update(get_elevation_attrs(elevation))
    ele_da.name = "elevation"
    return ele_da


def get_time_dataarray(time, nrays, date_str):
    """Create Time DataArray.

    Parameters
    ----------
    time : float | :class:`numpy:numpy.ndarray`
        time resolution or array with time values ('seconds since'-notation).
    nrays : int
        number of rays
    date_str : str
        String with the form 'YYYY-mm-ddTHH:MM:SS' depicting the reference time.

    Returns
    -------
    time_da : xr.DataArray
        Time DataArray.
    """
    if np.isscalar(time):
        time = np.arange(0, nrays * time - time / 2, time)
    time_attrs = {
        "standard_name": "time",
        "units": f"seconds since {date_str}",
    }
    time_da = DataArray(data=time, dims=["time"])
    time_da.attrs.update(time_attrs)
    time_da.name = "time"
    return time_da


def get_sweep_dataarray(data, moment, fill=None):
    """Create Sweep Moment DataArray.

    Parameters
    ----------
    data : tuple | :class:`numpy:numpy.ndarray`
        tuple with shape of DataArray or 2d-array.
    moment : str
        String describing radar moment.

    Keyword Arguments
    -----------------
    fill : float
        Fill Value for new created array. Only used if data is tuple.

    Returns
    -------
    swp_da : xr.DataArray
        Sweep Moment DataArray.
    """
    if isinstance(data, tuple):
        if fill is None:
            data = np.tile(np.arange(data[1]), (data[0], 1))
        else:
            data = np.full(data, fill)
    swp_da = DataArray(data=data, dims=["time", "range"])
    swp_da.attrs.update(get_moment_attrs(moment))
    swp_da.name = moment
    return swp_da


def create_sweep_dataset(**kwargs):
    """Create Sweep Dataset with Coordinates.

    This function creates a simple Dataset with all needed coordinates.

    Radar moment variables can be assigned to this Dataset.

    Keyword Arguments
    -----------------
    shape : tuple
        tuple with shape of DataArray
    time : float | :class:`numpy:numpy.ndarray`
        time resolution or array with time values ('seconds since'-notation).
    date_str : str
        String with the form 'YYYY-mm-ddTHH:MM:SS' depicting the reference time.
    rng : float | :class:`numpy:numpy.ndarray`
        range resolution or array with range values.
    azimuth : float | :class:`numpy:numpy.ndarray`
        azimuth resolution or array with azimuth values.
    a1gate : int
        First measured ray. Defaults to 0. Only used for PPI.
    elevation : float | :class:`numpy:numpy.ndarray`
        elevation resolution or array with elevation values.
    sweep : str
        "PPI" or "RHI".

    Returns
    -------
    swp_ds : xr.Dataset
        Sweep Dataset.
    """
    shape = kwargs.pop("shape", None)
    sweep = kwargs.pop("sweep", "PPI")

    # set default shape for nrays, nbins
    if shape is None:
        nbins = 1000
        # default resolution/values (1deg azimuth, 1deg elevation)
        if sweep == "PPI":
            azimuth = kwargs.pop("azimuth", 1.0)
            elevation = kwargs.pop("elevation", 1.0)
        else:
            azimuth = kwargs.pop("azimuth", 1.0)
            elevation = kwargs.pop("elevation", 1.0)
    else:
        nbins = shape[1]
        azimuth = kwargs.pop("azimuth", None)
        elevation = kwargs.pop("elevation", None)
        if azimuth is not None and elevation is not None:
            raise ValueError(
                "Either `shape` or `azimuth` and `elevation` can be specified."
            )
        if sweep == "PPI":
            if elevation is None:
                raise ValueError("elevation need to be specified for PPI sweep")
            if azimuth is not None:
                raise ValueError("given azimuth value will be ignored for PPI sweep")
            azimuth = 360 / shape[0]
        else:
            if azimuth is None:
                raise ValueError("azimuth need to be specified for RHI sweep")
            if elevation is not None:
                raise ValueError("given elevation value will be ignored for RHI sweep")
            elevation = 90 / shape[0]

    a1gate = kwargs.pop("a1gate", 0)
    time = kwargs.pop("time", 0.25)
    date_str = kwargs.pop("date_str", "2022-08-27T10:00:00")
    rng = kwargs.pop("rng", 100)

    # calculate according to PPI/RHI
    if sweep == "PPI":
        azimuth = get_azimuth_dataarray(azimuth, nrays=None, a1gate=a1gate)
        nrays = azimuth.shape[0]
        elevation = get_elevation_dataarray(elevation, nrays=nrays)
    else:
        elevation = get_elevation_dataarray(elevation, nrays=None)
        nrays = elevation.shape[0]
        azimuth = get_azimuth_dataarray(azimuth, nrays=nrays, a1gate=a1gate)

    # get time array using retrieved number of rays
    time = get_time_dataarray(time, nrays, date_str)
    rng = get_range_dataarray(rng, nbins)

    ds = Dataset(
        coords=dict(
            time=time,
            range=rng,
            azimuth=azimuth,
            elevation=elevation,
        )
    )

    return decode_cf(ds)
