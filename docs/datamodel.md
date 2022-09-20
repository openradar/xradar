# Data Model

With the forthcoming standard [FM301], which is a subset of [CfRadial2.0], as
a basis for the xradar data model we can take full advantage of [Xarray] and the whole software stack.

We facilitate {py:class}`datatree:datatree.DataTree` to bundle the different sweeps of a radar volume into one structure. These sweep datasets are essentially {py:class}`xarray:xarray.Dataset` which contain metadata attributes and variables ({py:class}`xarray:xarray.DataArray`).

## DataTree

The DataTree consists of one global root group and several sweep groups. Optionally, groups containing parameter and calibration can be part of the datatree.

Internal representation: {py:class}`datatree:datatree.DataTree`

## Global Scope / Root Group

This group holds data and metadata relevant to the entire volume. These are:

- attributes
- ancillary variables

Internal representation: {py:class}`xarray:xarray.Dataset`

## Sweep Groups

Each of the sweep groups contains data and metadata belonging to a specific sweep. These are:

- dimensions
- coordinates
- ancillary variables
- dataset variables

Internal representation: {py:class}`xarray:xarray.Dataset`

## Dimensions

- time
- range
- frequency
- prt, optional

## Coordinates

- time
- range
- frequency

Internal Representation: {py:class}`xarray:xarray.DataArray`

## Ancillary Variables

- sweep_number
- sweep_mode
- follow_mode
- prt_mode
- fixed_angle
- azimuth
- elevation

Internal Representation: {py:class}`xarray:xarray.DataArray`

## Dataset Variables

- DBZH, radar_equivalent_reflectivity_factor_h
- DBZV, radar_equivalent_reflectivity_factor_v
- and many more, see {class}`xradar.model.sweep_dataset_vars`

Internal Representation: {py:class}`xarray:xarray.DataArray`

[CfRadial2.0]: https://dx.doi.org/10.5065/fy2k-x587
[FM301]: https://wmoomm.sharepoint.com/:b:/s/wmocpdb/EVcEDwHwf6FJtB6anuyBH3QBgA5bE_Uz9jI4FaSkAyowSg?e=jcazi4
[Xarray]: https://docs.xarray.dev
[xarray-datatree]: https://xarray-datatree.readthedocs.io

