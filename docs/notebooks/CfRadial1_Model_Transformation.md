---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
  main_language: python
kernelspec:
  display_name: Python 3
  name: python3
---

# CfRadial1 to CfRadial2
## A data model transformation

In this notebook we show how to transform the CfRadial1 Data model to a CfRadial2 representation.

We use some internal functions to show how xradar is working inside.

Within this notebook we reference to the [CfRadial2.1 draft](https://github.com/NCAR/CfRadial/tree/master/docs). As long as the FM301 WMO standard is not finalized we will rely on the drafts presented.

```{code-cell}
import os

import xarray as xr
from open_radar_data import DATASETS

import xradar as xd
```

## Download

Fetching CfRadial1 radar data file from [open-radar-data](https://github.com/openradar/open-radar-data) repository.

```{code-cell}
filename = DATASETS.fetch("cfrad.20080604_002217_000_SPOL_v36_SUR.nc")
```

## Open CfRadial1 file using xr.open_dataset

Making use of the xarray `netcdf4` backend. We get back all data and metadata in one single CfRadial1 Dataset. Since xarray 2025.04.0 we have to use `decode_timedalte=False` to prevent erroneous decoding of timedelta values for eg. pulse widths.

```{code-cell}
ds = xr.open_dataset(filename, engine="netcdf4", decode_timedelta=False)
with xr.set_options(
    display_expand_data_vars=True, display_expand_attrs=True, display_max_rows=1000
):
    display(ds.load())
```

## Extract CfRadial2 Groups and Subgroups

Now as we have the CfRadial1 Dataset we can work towards extracting the CfRadial2 groups and subgroups.

+++

### Extract CfRadial2 Root-Group

The following sections present the details of the information in the top-level (root) group of the
data set.

We use a convenience function to extract the CfRadial2 root group from the CfRadial1 Dataset. We can call this function with one kwarg:

- `optional=False` - only mandatory data and metadata is imported, defaults to True

+++

#### optional=True

```{code-cell}
root = xd.io.backends.cfradial1._get_required_root_dataset(ds)
with xr.set_options(
    display_expand_data_vars=True, display_expand_attrs=True, display_max_rows=1000
):
    display(root.load())
```

#### optional=False

```{code-cell}
root = xd.io.backends.cfradial1._get_required_root_dataset(ds, optional=False)
with xr.set_options(
    display_expand_data_vars=True, display_expand_attrs=True, display_max_rows=1000
):
    display(root)
```

### Extract Root-Group metadata groups

The Cfradial2 Data Model has a notion of root group metadata groups. Those groups provide additional metadata covering other aspects of the radar system.

#### The radar_parameters sub-group

This group holds radar parameters specific to a radar instrument. It's implemented as dictionary where the value can be used to override the name.

```{code-cell}
display(xd.model.radar_parameters_subgroup)
```

Again we use a convenience function to extract the group.

```{code-cell}
radar_parameters = xd.io.backends.cfradial1._get_subgroup(
    ds, xd.model.radar_parameters_subgroup
)
display(radar_parameters.load())
```

#### The radar_calibration sub-group

For a radar, a different calibration is required for each pulse width. Therefore the calibration
variables are arrays. If only one calibration is available it is squeezed by the reader.

```{code-cell}
display(xd.model.radar_calibration_subgroup)
```

Again we use a convenience function to extract the group.

```{code-cell}
radar_calibration = xd.io.backends.cfradial1._get_radar_calibration(ds)
with xr.set_options(display_expand_data_vars=True):
    if radar_calibration:
        display(radar_calibration.load())
```

#### The georeference_correction sub-group

The following additional variables are used to quantify errors in the georeference data for moving
platforms. These are constant for a volume.

```{code-cell}
display(xd.model.georeferencing_correction_subgroup)
```

Again we use a convenience function to extract the group.

```{code-cell}
georeference_correction = xd.io.backends.cfradial1._get_subgroup(
    ds, xd.model.georeferencing_correction_subgroup
)
with xr.set_options(display_expand_data_vars=True):
    display(georeference_correction.load())
```

### Sweep groups

This section provides details of the information in each sweep group. The name of the sweep groups is found in the sweep_group_name array variable in the root group.

```{code-cell}
root.sweep_group_name
```

Again we use a convenience function to extract the different sweep groups.  We can call this function with kwargs:

- `optional=False` - only mandatory data and metadata is imported, defaults to `True`
- `first_dim="time` - return first dimension as `time`, defaults to`auto` (return either as `azimuth` (PPI) or `elevation` (RHI)to `time`
- `site_as_coords=False` - do not add radar site coordinates to the Sweep-Dataset, defaults to `True`

#### Examining first sweep with default kwargs.

```{code-cell}
sweeps = xd.io.backends.cfradial1._get_sweep_groups(ds)
with xr.set_options(display_expand_data_vars=True):
    display(sweeps["sweep_0"])
```

#### Examining first sweep with `optional=False`

```{code-cell}
sweeps = xd.io.backends.cfradial1._get_sweep_groups(ds, optional=False)
with xr.set_options(display_expand_data_vars=True):
    display(sweeps["sweep_0"])
```

#### `optional=False` and `site_as_coords=False`

```{code-cell}
sweeps = xd.io.backends.cfradial1._get_sweep_groups(
    ds, optional=False, site_as_coords=False
)
with xr.set_options(display_expand_data_vars=True):
    display(sweeps["sweep_0"])
```

#### `optional=False`, `site_as_coords=True` and `first_dim="auto"`

```{code-cell}
sweeps = xd.io.backends.cfradial1._get_sweep_groups(
    ds, optional=False, site_as_coords=False, first_dim="time"
)
with xr.set_options(display_expand_data_vars=True):
    display(sweeps["sweep_0"])
```

## Read as CfRadial2 data representation

xradar provides two easy ways to retrieve the CfRadial1 data as CfRadial2 groups.

### DataTree

This is the most complete representation as a DataTree. All groups and subgroups are represented in a tree-like structure. Can be parameterized using kwargs. Easy write to netCDF4.

```{code-cell}
dtree = xd.io.open_cfradial1_datatree(filename, optional_groups=True)
with xr.set_options(display_expand_data_vars=True, display_expand_attrs=True):
    display(dtree)
```

Each DataTree-node itself represents another DataTree.

```{code-cell}
display(dtree["radar_parameters"].load())
```

```{code-cell}
with xr.set_options(display_expand_data_vars=True):
    display(dtree["sweep_7"].load())
```

#### Roundtrip with `to_netcdf`

+++

Write DataTree to netCDF4 file, reopen and compare with source. This just tets if roundtripping the DataTree works.

```{code-cell}
outfile = "test_dtree.nc"
if os.path.exists(outfile):
    os.unlink(outfile)
dtree.to_netcdf(outfile)
```

```{code-cell}
dtree2 = xr.open_datatree(outfile, decode_timedelta=False)
with xr.set_options(display_expand_data_vars=True, display_expand_attrs=True):
    display(dtree2)
```

```{code-cell}
for grp in dtree.groups:
    print(grp)
    xr.testing.assert_equal(dtree[grp].ds, dtree2[grp].ds)
```

#### Roundtrip with `xradar.io.to_cfradial2`

```{code-cell}
dtree3 = xd.io.open_cfradial1_datatree(filename, optional_groups=True)
```

```{code-cell}
display(dtree3)
```

```{code-cell}
outfile = "test_cfradial2.nc"
if os.path.exists(outfile):
    os.unlink(outfile)
xd.io.to_cfradial2(dtree3, outfile)
```

```{code-cell}
dtree4 = xr.open_datatree("test_cfradial2.nc", decode_timedelta=False)
with xr.set_options(display_expand_data_vars=True, display_expand_attrs=True):
    display(dtree4)
```

```{code-cell}
for grp in dtree3.groups:
    print(grp)
    xr.testing.assert_equal(dtree3[grp].ds, dtree4[grp].ds)
```

### Datasets

Using xarray.open_dataset and the cfradial1-backend we can easily load specific groups side-stepping the DataTree.  Can be parameterized using kwargs.

```{code-cell}
ds = xr.open_dataset(filename, group="sweep_1", engine="cfradial1", first_dim="time")
with xr.set_options(display_expand_data_vars=True):
    display(ds.load())
```

```{code-cell}
ds = xr.open_dataset(filename, group="radar_parameters", engine="cfradial1")
display(ds.load())
```

## Conclusion

CfRadial1 and CfRadial2 are based on the same principles with slightly different data representation. Nevertheless the conversion is relatively straighforward as has been shown here.

As the implementation with the cfradial1 xarray backend on one hand and the DataTree on the other hand is very versatile users can pick the most usable approach for their workflows.
