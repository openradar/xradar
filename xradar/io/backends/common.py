#!/usr/bin/env python
# Copyright (c) 2022-2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Common Backend Functions
========================

This submodule contains helper functions for data and metadata alignment.

Currently, all private and not part of the public API.

"""

import io
import struct
import textwrap
import warnings
from collections import OrderedDict

import h5netcdf
import numpy as np
import xarray as xr

from ...model import (
    georeferencing_correction_subgroup,
    optional_root_attrs,
    optional_root_vars,
    radar_calibration_subgroup,
    radar_parameters_subgroup,
    required_global_attrs,
    required_root_vars,
)


def _maybe_decode(attr):
    try:
        # Decode the xr.DataArray differently than a byte string
        if type(attr) is xr.core.dataarray.DataArray:
            decoded_attr = attr.astype(str).str.rstrip()
        else:
            decoded_attr = attr.decode()
        return decoded_attr
    except AttributeError:
        return attr


def _maybe_recover_surrogate(txt):
    """Recover text that was decoded with surrogate escapes."""
    if not isinstance(txt, str):
        return txt

    if any("\udc80" <= ch <= "\udcff" for ch in txt):
        try:
            return txt.encode("utf-8", "surrogateescape").decode("utf-8")
        except UnicodeError:
            pass

    return txt


def _calculate_angle_res(dim):
    # need to sort dim first
    angle_diff = np.diff(sorted(dim))
    angle_diff2 = np.abs(np.diff(angle_diff))

    # only select angle_diff, where angle_diff2 is less than 0.1 deg
    # Todo: currently 0.05 is working in most cases
    #  make this robust or parameterisable
    angle_diff_wanted = angle_diff[:-1][angle_diff2 < 0.05]
    return np.round(np.nanmean(angle_diff_wanted), decimals=2)


def _fix_angle(da):
    # fix elevation outliers
    if len(set(da.values)) > 1:
        med = da.median(skipna=True)
        da = da.where(da == med).fillna(med)
    return da


_STATION_VARS = {"latitude", "longitude", "altitude"}


def _apply_site_as_coords(ds, site_as_coords):
    """Promote or demote station coordinates on a sweep Dataset.

    When *site_as_coords* is true the latitude / longitude / altitude
    variables are promoted to coordinates.  When false they are demoted
    back to data variables so the root node owns the single authoritative
    copy in a DataTree context.

    Parameters
    ----------
    ds : xr.Dataset
        Sweep dataset to modify.
    site_as_coords : bool
        If True, promote station vars to coordinates.
        If False, demote them to data variables.

    Returns
    -------
    xr.Dataset
    """
    if site_as_coords:
        present = _STATION_VARS & (set(ds.data_vars) | set(ds.coords))
        if present:
            return ds.assign_coords({v: ds[v] for v in present})
        return ds
    to_demote = _STATION_VARS & set(ds.coords)
    if to_demote:
        return ds.reset_coords(list(to_demote))
    return ds


def _attach_sweep_groups(dtree, sweeps):
    """Attach sweep groups to DataTree."""
    for i, sw in enumerate(sweeps):
        sw = sw.drop_vars(_STATION_VARS, errors="ignore").drop_attrs(deep=False)
        dtree[f"sweep_{i}"] = xr.DataTree(sw)
    return dtree


def _get_h5group_names(filename, engine):
    if engine == "odim":
        groupname = "dataset"
        off = 1
    elif engine == "gamic":
        groupname = "scan"
        off = 0
    else:
        raise ValueError(f"xradar: unknown engine `{engine}`.")
    with h5netcdf.File(filename, "r", decode_vlen_strings=True) as fh:
        groups = ["/".join(["", grp]) for grp in fh.groups if groupname in grp.lower()]
        # h5py/h5netcdf might return groups with alphanumeric sorting
        # just sort in any case
        groups = sorted(groups, key=lambda x: int(x[len(groupname) + 1 :]))
        groups = [f"sweep_{int(sw[len(groupname) + 1 :]) - off}" for sw in groups]
    if isinstance(filename, io.BytesIO):
        filename.seek(0)
    return groups


def _assign_root(sweeps):
    """(Re-)Create root object according CfRadial2 standard.

    Returns
    -------
    root : xr.Dataset
        Root dataset with station vars promoted to coordinates.
    sweeps : list[xr.Dataset]
        Input list with station vars dropped from sweep datasets (index 1+).
    """
    # extract time coverage
    times = np.array(
        [[ts.time.values.min(), ts.time.values.max()] for ts in sweeps[1:]]
    ).flatten()
    time_coverage_start = min(times)
    time_coverage_end = max(times)

    time_coverage_start_str = str(time_coverage_start)[:19] + "Z"
    time_coverage_end_str = str(time_coverage_end)[:19] + "Z"

    # create root group from scratch
    root = xr.Dataset()  # data_vars=wrl.io.xarray.global_variables,
    # attrs=wrl.io.xarray.global_attrs)

    # take first dataset/file for retrieval of location
    # site = self.site

    # assign root variables
    # reset_coords as root doesn't have coordinates
    root = root.assign(
        {
            "volume_number": 0,
            "platform_type": "fixed",
            "instrument_type": "radar",
            "time_coverage_start": time_coverage_start_str,
            "time_coverage_end": time_coverage_end_str,
            "latitude": sweeps[1]["latitude"],
            "longitude": sweeps[1]["longitude"],
            "altitude": sweeps[1]["altitude"],
        }
    ).reset_coords()

    # Promote station location to coordinates on the root node.
    # Sweep children inherit these via DataTree coordinate inheritance.
    promote = _STATION_VARS & set(root.data_vars)
    if promote:
        root = root.set_coords(list(promote))

    # assign root attributes
    attrs = {}
    attrs["Conventions"] = sweeps[0].attrs.get("Conventions", "None")
    attrs["instrument_name"] = sweeps[0].attrs.get("instrument_name", "None")
    comment = sweeps[0].attrs.get("comment", None)
    attrs.update(
        {
            "version": "None",
            "title": "None",
            "institution": "None",
            "references": "None",
            "source": "None",
            "history": "None",
            "comment": "im/exported using xradar",
        }
    )
    if comment is not None:
        attrs["comment"] = attrs["comment"] + ",\n" + comment
    root = root.assign_attrs(attrs)
    # todo: pull in only CF attributes
    root = root.assign_attrs(sweeps[1].attrs)

    # Drop station vars from sweeps so the root owns the single copy.
    cleaned = [sweeps[0]] + [
        ds.drop_vars(_STATION_VARS, errors="ignore") for ds in sweeps[1:]
    ]
    return root, cleaned


def _get_fmt_string(dictionary, retsub=False, byte_order="<"):
    """Get Format String from given dictionary.

    Parameters
    ----------
    dictionary : dict
        Dictionary containing data structure with fmt-strings.
    retsub : bool
        If True, return sub structures.

    Returns
    -------
    fmt : str
        struct format string
    sub : dict
        Dictionary containing substructure
    """
    fmt = f"{byte_order}"
    if retsub:
        sub = OrderedDict()
    for k, v in dictionary.items():
        try:
            fmt += v["fmt"]
        except KeyError:
            # remember sub-structures
            if retsub:
                sub[k] = v
            if "size" in v:
                fmt += v["size"]
            else:
                fmt += f"{struct.calcsize(_get_fmt_string(v))}s"
    if retsub:
        return fmt, sub
    else:
        return fmt


def _unpack_dictionary(buffer, dictionary, rawdata=False):
    """Unpacks binary data using the given dictionary structure.

    Parameters
    ----------
    buffer : array-like
    dictionary : dict
        data structure in dictionary, keys are names and values are structure formats

    Returns
    -------
    data : dict
        Ordered Dictionary with unpacked data
    """
    # get format and substructures of dictionary
    fmt, sub = _get_fmt_string(dictionary, retsub=True)

    # unpack into OrderedDict
    data = OrderedDict(zip(dictionary, struct.unpack(fmt, buffer)))

    # remove spares
    if not rawdata:
        keys_to_remove = [k for k in data.keys() if k.startswith("spare")]
        keys_to_remove.extend([k for k in data.keys() if k.startswith("reserved")])
        for k in keys_to_remove:
            data.pop(k, None)

    # iterate over sub dictionary and unpack/read/decode
    for k, v in sub.items():
        if not rawdata:
            # read/decode data
            for k1 in ["read", "func"]:
                try:
                    data[k] = v[k1](data[k], **v[k1[0] + "kw"])
                except KeyError:
                    pass
                except UnicodeDecodeError:
                    pass
        # unpack sub dictionary
        try:
            data[k] = _unpack_dictionary(data[k], v, rawdata=rawdata)
        except TypeError:
            pass

    return data


def _get_required_root_dataset(ls_ds, optional=True):
    """Extract Root Dataset."""
    # keep only defined mandatory and defined optional variables per default
    # by checking in all nodes
    data_var = {x for xs in [sweep.variables.keys() for sweep in ls_ds] for x in xs}
    remove_root = set(data_var) ^ set(required_root_vars)
    if optional:
        remove_root ^= set(optional_root_vars)
    remove_root ^= {"sweep_number", "fixed_angle"}
    remove_root &= data_var
    # ignore errors for variables which exist in one sweep but not the other
    root = [sweep.drop_vars(remove_root, errors="ignore") for sweep in ls_ds]
    root_vars = {x for xs in [sweep.variables.keys() for sweep in root] for x in xs}
    # rename variables
    # todo: find a more easy method not iterating over all variables
    for k in root_vars:
        rename = optional_root_vars.get(k, None)
        if rename:
            root = [sweep.rename_vars({k: rename}) for sweep in root]

    ds_vars = [sweep[root_vars] for sweep in ls_ds]
    _vars = xr.concat(ds_vars, dim="sweep").reset_coords()

    # Creating the root group using _assign_root function
    ls = ls_ds.copy()
    ls.insert(0, xr.Dataset())
    root, ls = _assign_root(ls)

    # Drop station coords from _vars to avoid merge conflict
    # (they are already placed as coordinates on root by _assign_root)
    to_drop = _STATION_VARS & set(_vars.data_vars)
    if to_drop:
        _vars = _vars.drop_vars(to_drop)

    # merging both the created and the variables within each dataset
    root = xr.merge([root, _vars], compat="override")
    attrs = root.attrs.keys()
    remove_attrs = set(attrs) ^ set(required_global_attrs)
    if optional:
        remove_attrs ^= set(optional_root_attrs)
    for k in remove_attrs:
        root.attrs.pop(k, None)
    # Renaming variable
    if "sweep_number" in data_var and "sweep_group_name" not in data_var:
        root = root.rename_vars({"sweep_number": "sweep_group_name"})
    elif "sweep_group_name" in data_var:
        root["sweep_group_name"].values = np.array(
            [f"sweep_{i}" for i in range(len(root["sweep_group_name"].values))]
        )
    return root


def _get_subgroup(ls_ds: list[xr.Dataset], subdict):
    """Get iris-sigmet root metadata group.
    Variables are fetched from the provided Dataset according to the subdict dictionary.
    """
    meta_vars = subdict
    data_vars = {x for xs in [ds.variables.keys() for ds in ls_ds] for x in xs}
    extract_vars = set(data_vars) & set(meta_vars)
    subgroup = xr.merge([ds[extract_vars] for ds in ls_ds], compat="no_conflicts")
    for k in subgroup.data_vars:
        rename = meta_vars[k]
        if rename:
            subgroup = subgroup.rename_vars({k: rename})
    subgroup.attrs = {}
    return subgroup


def _get_radar_calibration(ls_ds: list[xr.Dataset], subdict: dict) -> xr.Dataset:
    """Get radar calibration root metadata group."""
    meta_vars = subdict
    data_vars = {x for xs in [ds.attrs for ds in ls_ds] for x in xs}
    extract_vars = set(data_vars) & set(meta_vars)
    if extract_vars:
        var_dict = {var: ls_ds[0].attrs[var] for var in extract_vars}
        return xr.Dataset({key: xr.DataArray(value) for key, value in var_dict.items()})
    else:
        return xr.Dataset()


def _prepare_backend_ds(ds):
    """wrap variables in CopyOnWriteArray and create indexes

    Needed for hdf5-based `odim` and `gamic` backends to work with
    file-like objects (see https://github.com/openradar/xradar/issues/189),
    as the wrapping in standard xarray pipeline happens after returning the
    dataset.
    """
    for name, variable in ds.variables.items():
        if name not in ds._indexes:
            data = xr.core.indexing.CopyOnWriteArray(variable._data)
            variable.data = data
    # create indexes
    ds = ds.set_index({dim: dim for dim in ds.dims})
    return ds


def _build_groups_dict(ls_ds, optional=True, optional_groups=False):
    """Build CfRadial2 groups dict from a list of sweep Datasets.

    Parameters
    ----------
    ls_ds : list of xr.Dataset
        List of sweep Datasets.
    optional : bool
        Import optional metadata, defaults to True.
    optional_groups : bool
        If True, includes ``/radar_parameters``, ``/georeferencing_correction``
        and ``/radar_calibration`` metadata subgroups. Default is False.

    Returns
    -------
    groups_dict : dict[str, xr.Dataset]
        Dictionary with CfRadial2 group structure.
    """
    groups_dict = {
        "/": _get_required_root_dataset(ls_ds, optional=optional),
    }
    if optional_groups:
        groups_dict["/radar_parameters"] = _get_subgroup(
            ls_ds, radar_parameters_subgroup
        )
        groups_dict["/georeferencing_correction"] = _get_subgroup(
            ls_ds, georeferencing_correction_subgroup
        )
        groups_dict["/radar_calibration"] = _get_radar_calibration(
            ls_ds, radar_calibration_subgroup
        )
    for i, ds in enumerate(ls_ds):
        sw = ds.drop_vars(_STATION_VARS, errors="ignore").drop_attrs(deep=False)
        groups_dict[f"/sweep_{i}"] = sw
    return groups_dict


def _deprecation_warning(old_name, engine):
    """Emit FutureWarning for deprecated standalone open_*_datatree functions."""
    warnings.warn(
        f"`{old_name}` is deprecated. Use "
        f'`xd.open_datatree(file, engine="{engine}")` or '
        f'`xr.open_datatree(file, engine="{engine}")` instead.',
        FutureWarning,
        stacklevel=4,
    )


#: NumPy-style Parameters block shared across all `open_groups_as_dict`
#: methods. Backend-specific blocks are appended via :func:`_compose_docstring`.
#: The CF decoder kwargs (`mask_and_scale`, `decode_times`, ...) thread
#: through to :py:func:`xarray.open_dataset`; see xarray's documentation for
#: full semantics.
COMMON_BACKEND_PARAMS_DOC = """
Parameters
----------
filename_or_obj : str, Path, or file-like
    Path or file-like object understood by the underlying reader.
mask_and_scale : bool or dict-like, optional
    Replace fill values with NA and apply ``scale_factor``/``add_offset``
    decoding. See :py:func:`xarray.open_dataset`. Defaults to ``True``.
decode_times : bool or dict-like, optional
    Decode CF time variables (calendar, units) into ``np.datetime64``.
    Defaults to ``True``.
concat_characters : bool or dict-like, optional
    Concatenate character arrays into strings along their trailing
    dimension. Defaults to ``True``.
decode_coords : bool or {"coordinates", "all"}, optional
    Decode the CF ``coordinates`` attribute. Defaults to ``True``
    (equivalent to ``"coordinates"``).
drop_variables : str or iterable of str, optional
    Names of variables to drop before processing.
use_cftime : bool, optional
    Force ``cftime`` decoding for time variables (instead of
    ``np.datetime64``). Defaults to ``None`` (auto).
decode_timedelta : bool, optional
    Decode CF timedelta variables. Default mirrors ``decode_times``
    unless the backend overrides it (cfradial1, cfradial2, and imd
    default to ``False``).
sweep : int, str, or list of int/str, optional
    Sweep selection. ``None`` (default) returns all sweeps. An ``int``
    or ``"sweep_N"`` string returns one sweep; a list returns the
    named subset.
first_dim : {"auto", "time"}, optional
    Leading dimension of each sweep dataset. ``"auto"`` picks
    ``azimuth`` (PPI) or ``elevation`` (RHI); ``"time"`` keeps the
    raw time axis. Default ``"auto"`` (``"time"`` for cfradial2).
optional : bool, optional
    Include optional root variables when available. Defaults to ``True``.
optional_groups : bool, optional
    Include the ``/radar_parameters``, ``/georeferencing_correction``,
    and ``/radar_calibration`` metadata subgroups under the root.
    Defaults to ``False``.
"""


#: Reindex/angle parameter block — shared by backends that resample
#: rays onto a regular angular grid (odim, gamic, nexrad, cfradial1,
#: iris, furuno, uf).
REINDEX_PARAMS_DOC = """
reindex_angle : bool or dict, optional
    Resample rays onto a regular angular grid when truthy. A dict is
    passed as kwargs to :func:`xradar.util.reindex_angle` (e.g.
    ``{"start_angle": 0.0, "stop_angle": 360.0, "angle_res": 1.0}``).
    Only invoked when ``decode_coords=True``. Defaults to ``False``.
fix_second_angle : bool, optional
    Correct erroneous secondary-angle values (azimuth on RHI,
    elevation on PPI). Only effective with ``first_dim="auto"``.
    Defaults to ``False``.
"""

#: Site-coordinate parameter block. Most multi-sweep backends spell this
#: `site_coords`; IMD uses the legacy `site_as_coords`.
SITE_COORDS_PARAM_DOC = """
site_coords : bool, optional
    Attach ``latitude``/``longitude``/``altitude`` as coordinates on
    the root dataset (and on per-sweep datasets where the backend
    supports it). Defaults to ``True``.
"""

#: HDF5/h5netcdf options shared by ODIM, GAMIC, HPL, Metek.
HDF5_PARAMS_DOC = """
format : str, optional
    h5netcdf format string. Defaults to ``None``.
invalid_netcdf : bool, optional
    Accept HDF5 files that are not strictly NetCDF-conformant.
phony_dims : {"access", "sort", None}, optional
    How h5netcdf labels unnamed dimensions. Defaults to ``"access"``.
decode_vlen_strings : bool, optional
    Decode variable-length strings stored in HDF5. Defaults to ``True``.
"""

#: Reader-lock parameter shared by NEXRAD, IRIS, UF.
LOCK_PARAM_DOC = """
lock : threading.Lock or None, optional
    Reader lock for thread-safe access. Defaults to ``None``.
"""


def _compose_docstring(summary, *extra_blocks):
    """Compose a NumPy-style docstring from a summary plus parameter blocks.

    The composed result always opens with the shared
    :data:`COMMON_BACKEND_PARAMS_DOC` Parameters block and closes with a
    fixed Returns section. Per-backend blocks (e.g. :data:`HDF5_PARAMS_DOC`,
    :data:`REINDEX_PARAMS_DOC`) are inserted between the common block and
    the Returns section in the order given.

    Each block is independently de-indented and re-indented with four
    spaces, so block authors do not need to keep the indentation in sync
    by hand — write a block at any indent level and this helper
    normalises it.

    Parameters
    ----------
    summary : str
        One-paragraph summary that opens the docstring.
    *extra_blocks : str
        Optional backend-specific parameter blocks. Each may use any
        indentation; the helper normalises them to four-space indent.

    Returns
    -------
    str
        Complete docstring suitable for ``method.__doc__ = ...``.
    """

    def _block(text):
        return textwrap.indent(textwrap.dedent(text).strip("\n"), "    ")

    parts = [summary.strip("\n"), "", _block(COMMON_BACKEND_PARAMS_DOC)]
    for block in extra_blocks:
        if block:
            parts.append(_block(block))
    returns_body = (
        "dict[str, xarray.Dataset]\n"
        "    CfRadial2 group paths (``/``, ``/sweep_N``, optional\n"
        "    ``/radar_parameters`` etc.) mapped to their datasets,\n"
        "    ready for :py:meth:`xarray.DataTree.from_dict`."
    )
    parts += ["", "    Returns", "    -------", _block(returns_body)]
    return "\n".join(parts) + "\n"


def _resolve_sweeps(sweep, discover_fn):
    """Normalise the sweep parameter into a list of sweep group names.

    Parameters
    ----------
    sweep : int, str, list, or None
        User-supplied sweep selection.
    discover_fn : callable
        Zero-arg function returning all sweep group names for the file.

    Returns
    -------
    list[str]
        List of sweep group name strings.
    """
    if isinstance(sweep, str):
        return [sweep]
    if isinstance(sweep, int):
        return [f"sweep_{sweep}"]
    if isinstance(sweep, list):
        if not sweep:
            raise ValueError("sweep list is empty.")
        if isinstance(sweep[0], int):
            return [f"sweep_{i}" for i in sweep]
        return list(sweep)
    if sweep is None:
        return discover_fn()
    raise TypeError(f"Unsupported sweep type: {type(sweep)}")


# IRIS Data Types and corresponding python struct format characters
# 4.2 Scalar Definitions, Page 23
# https://docs.python.org/3/library/struct.html#format-characters
# also used for Furuno data types
SINT2 = {"fmt": "h", "dtype": "int16"}
SINT4 = {"fmt": "i", "dtype": "int32"}
UINT1 = {"fmt": "B", "dtype": "unit8"}
UINT2 = {"fmt": "H", "dtype": "uint16"}
UINT4 = {"fmt": "I", "dtype": "unint32"}
