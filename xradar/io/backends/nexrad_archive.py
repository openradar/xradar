import warnings

import numpy as np

from ...model import (
    get_altitude_attrs,
    get_azimuth_attrs,
    get_elevation_attrs,
    get_latitude_attrs,
    get_longitude_attrs,
    get_moment_attrs,
    get_range_attrs,
    get_time_attrs,
)
from .common import LazyLoadDict, prepare_for_read
from .nexrad_common import get_nexrad_location
from .nexrad_interpolate import _fast_interpolate_scan_2, _fast_interpolate_scan_4
from .nexrad_level2 import NEXRADLevel2File

nexrad_mapping = {
    "REF": "DBZH",
    "VEL": "VRADH",
    "SW": "WRADH",
    "ZDR": "ZDR",
    "PHI": "PHIDP",
    "RHO": "RHOHV",
    "CFP": "CCORH",
}


class _NEXRADLevel2StagedField:
    """
    A class to facilitate on demand loading of field data from a Level 2 file.
    """

    def __init__(self, nfile, moment, max_ngates, scans):
        """initialize."""
        self.nfile = nfile
        self.moment = moment
        self.max_ngates = max_ngates
        self.scans = scans

    def __call__(self):
        """Return the array containing the field data."""
        return self.nfile.get_data(self.moment, self.max_ngates, scans=self.scans)


def _find_range_params(scan_info):
    """Return range parameters, first_gate, gate_spacing, last_gate."""
    min_first_gate = 999999
    min_gate_spacing = 999999
    max_last_gate = 0
    for scan_params in scan_info:
        ngates = scan_params["ngates"][0]
        for i, moment in enumerate(scan_params["moments"]):
            first_gate = scan_params["first_gate"][i]
            gate_spacing = scan_params["gate_spacing"][i]
            last_gate = first_gate + gate_spacing * (ngates - 0.5)

            min_first_gate = min(min_first_gate, first_gate)
            min_gate_spacing = min(min_gate_spacing, gate_spacing)
            max_last_gate = max(max_last_gate, last_gate)
    return min_first_gate, min_gate_spacing, max_last_gate


def _find_scans_to_interp(scan_info, first_gate, gate_spacing):
    """Return a dict indicating what moments/scans need interpolation."""
    moments = {m for scan in scan_info for m in scan["moments"]}
    interpolate = {moment: [] for moment in moments}
    for scan_num, scan in enumerate(scan_info):
        for moment in moments:
            if moment not in scan["moments"]:
                continue
            index = scan["moments"].index(moment)
            first = scan["first_gate"][index]
            spacing = scan["gate_spacing"][index]
            if first != first_gate or spacing != gate_spacing:
                interpolate[moment].append(scan_num)
                # for proper interpolation the gate spacing of the scan to be
                # interpolated should be 1/4th the spacing of the radar
                if spacing == gate_spacing * 4:
                    interpolate["multiplier"] = "4"
                elif spacing == gate_spacing * 2:
                    interpolate["multiplier"] = "2"
                else:
                    raise ValueError("Gate spacing is neither 1/4 or 1/2")
                # assert first_gate + 1.5 * gate_spacing == first
    # remove moments with no scans needing interpolation
    interpolate = {k: v for k, v in interpolate.items() if len(v) != 0}
    return interpolate


def _interpolate_scan(mdata, start, end, moment_ngates, multiplier, linear_interp=True):
    """Interpolate a single NEXRAD moment scan from 1000 m to 250 m."""
    fill_value = -9999
    data = mdata.filled(fill_value)
    scratch_ray = np.empty((data.shape[1],), dtype=data.dtype)
    if multiplier == "4":
        _fast_interpolate_scan_4(
            data, scratch_ray, fill_value, start, end, moment_ngates, linear_interp
        )
    else:
        _fast_interpolate_scan_2(
            data, scratch_ray, fill_value, start, end, moment_ngates, linear_interp
        )
    mdata[:] = np.ma.array(data, mask=(data == fill_value))


def open_dataset(
    filename,
    field_names=None,
    additional_metadata=None,
    file_field_names=False,
    exclude_fields=None,
    include_fields=None,
    delay_field_loading=False,
    station=None,
    scans=None,
    linear_interp=True,
    storage_options={"anon": True},
    **kwargs,
):
    # Load the data file in using NEXRADLevel2File Class
    nfile = NEXRADLevel2File(prepare_for_read(filename))

    # Access the scan info and load in the available moments
    scan_info = nfile.scan_info(scans)
    available_moments = {m for scan in scan_info for m in scan["moments"]}
    first_gate, gate_spacing, last_gate = _find_range_params(scan_info)

    # Interpolate to 360 degrees where neccessary
    interpolate = _find_scans_to_interp(scan_info, first_gate, gate_spacing)

    # Deal with time
    time_start, _time = nfile.get_times(scans)
    time = get_time_attrs(time_start)
    time["data"] = _time

    # range
    _range = get_range_attrs()
    first_gate, gate_spacing, last_gate = _find_range_params(scan_info)
    _range["data"] = np.arange(first_gate, last_gate, gate_spacing, "float32")
    _range["meters_to_center_of_first_gate"] = float(first_gate)
    _range["meters_between_gates"] = float(gate_spacing)

    # metadata
    metadata = {
        "Conventions": "CF/Radial instrument_parameters",
        "version": "1.3",
        "title": "",
        "institution": "",
        "references": "",
        "source": "",
        "history": "",
        "comment": "",
        "intrument_name": "",
        "original_container": "NEXRAD Level II",
    }

    vcp_pattern = nfile.get_vcp_pattern()
    if vcp_pattern is not None:
        metadata["vcp_pattern"] = vcp_pattern
    if "icao" in nfile.volume_header.keys():
        metadata["instrument_name"] = nfile.volume_header["icao"].decode()

    # scan_type

    # latitude, longitude, altitude
    latitude = get_latitude_attrs()
    longitude = get_longitude_attrs()
    altitude = get_altitude_attrs()

    if nfile._msg_type == "1" and station is not None:
        lat, lon, alt = get_nexrad_location(station)
    elif (
        "icao" in nfile.volume_header.keys()
        and nfile.volume_header["icao"].decode()[0] == "T"
    ):
        lat, lon, alt = get_nexrad_location(nfile.volume_header["icao"].decode())
    else:
        lat, lon, alt = nfile.location()
    latitude["data"] = np.array([lat], dtype="float64")
    longitude["data"] = np.array([lon], dtype="float64")
    altitude["data"] = np.array([alt], dtype="float64")

    # Sweep information
    sweep_number = {
        "units": "count",
        "standard_name": "sweep_number",
        "long_name": "Sweep number",
    }

    sweep_mode = {
        "units": "unitless",
        "standard_name": "sweep_mode",
        "long_name": "Sweep mode",
        "comment": 'Options are: "sector", "coplane", "rhi", "vertical_pointing", "idle", "azimuth_surveillance", "elevation_surveillance", "sunscan", "pointing", "manual_ppi", "manual_rhi"',
    }
    sweep_start_ray_index = {
        "long_name": "Index of first ray in sweep, 0-based",
        "units": "count",
    }
    sweep_end_ray_index = {
        "long_name": "Index of last ray in sweep, 0-based",
        "units": "count",
    }

    if scans is None:
        nsweeps = int(nfile.nscans)
    else:
        nsweeps = len(scans)
    sweep_number["data"] = np.arange(nsweeps, dtype="int32")
    sweep_mode["data"] = np.array(nsweeps * ["azimuth_surveillance"], dtype="S")

    rays_per_scan = [s["nrays"] for s in scan_info]
    sweep_end_ray_index["data"] = np.cumsum(rays_per_scan, dtype="int32") - 1

    rays_per_scan.insert(0, 0)
    sweep_start_ray_index["data"] = np.cumsum(rays_per_scan[:-1], dtype="int32")

    # azimuth, elevation, fixed_angle
    azimuth = get_azimuth_attrs()
    elevation = get_elevation_attrs()
    fixed_angle = {
        "long_name": "Target angle for sweep",
        "units": "degrees",
        "standard_name": "target_fixed_angle",
    }

    azimuth["data"] = nfile.get_azimuth_angles(scans)
    elevation["data"] = nfile.get_elevation_angles(scans).astype("float32")
    fixed_agl = []
    for i in nfile.get_target_angles(scans):
        if i > 180:
            i = i - 360.0
            warnings.warn(
                "Fixed_angle(s) greater than 180 degrees present."
                " Assuming angle to be negative so subtrating 360",
                UserWarning,
            )
        else:
            i = i
        fixed_agl.append(i)
    fixed_angles = np.array(fixed_agl, dtype="float32")
    fixed_angle["data"] = fixed_angles

    # fields
    max_ngates = len(_range["data"])
    available_moments = {m for scan in scan_info for m in scan["moments"]}
    interpolate = _find_scans_to_interp(
        scan_info,
        first_gate,
        gate_spacing,
    )

    fields = {}
    for moment in available_moments:
        dic = get_moment_attrs(nexrad_mapping[moment])
        dic["_FillValue"] = -9999
        if delay_field_loading and moment not in interpolate:
            dic = LazyLoadDict(dic)
            data_call = _NEXRADLevel2StagedField(nfile, moment, max_ngates, scans)
            dic.set_lazy("data", data_call)
        else:
            mdata = nfile.get_data(moment, max_ngates, scans=scans)
            if moment in interpolate:
                interp_scans = interpolate[moment]
                warnings.warn(
                    "Gate spacing is not constant, interpolating data in "
                    + f"scans {interp_scans} for moment {moment}.",
                    UserWarning,
                )
                for scan in interp_scans:
                    idx = scan_info[scan]["moments"].index(moment)
                    moment_ngates = scan_info[scan]["ngates"][idx]
                    start = sweep_start_ray_index["data"][scan]
                    end = sweep_end_ray_index["data"][scan]
                    if interpolate["multiplier"] == "4":
                        multiplier = "4"
                    else:
                        multiplier = "2"
                    _interpolate_scan(
                        mdata, start, end, moment_ngates, multiplier, linear_interp
                    )
            dic["data"] = mdata
        fields[nexrad_mapping[moment]] = dic
    return fields
