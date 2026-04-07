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

# Angle Reindexing

As a legacy from wradlib we have complex code for angle reindexing in xradar's codebase.

## High precision angle coordinates

As radar angle coordinates (`azimuth` or `elevation`) are measured constantly by different techniques of detection of the antenna pointing direction the values in the data files are mostly floating point numbers. In many cases these floating point numbers are not rounded to a certain decimal but keep the full possible range of the used dtype.

Problems of that:

- 1D angle coordinate arrays yield no equidistant vector.
- 1D angle coordinate arrays are not equivalent for different timesteps but same scan setup

## Missing rays, duplicate or additional rays

Sometimes rays (even sectors) are missing from the dataset, sometimes there are duplicate rays. Another problem with radar data are additional rays, which I call "antenna hickup" (two rays measured with within one resolution interval).

## What is angle reindexing?

Angle reindexing takes care of these problems by trying to determine the wanted layout from the radar metadata and the angle coordinates. With that newly created angle coordinate xarray machinery is used to reindex the radar moment data to that by nearest neighbor lookup (up to a tolerance). Missing rays will be filled with NaN.

## Why should it be used?

For most operations this is not a real problem. It will turn into a problem, if you want to stack your xarray.Dataset radar data on a third dimension (eg. `time`, by using `open_mfdataset`). Then all coordinates need alignment to keep things simple and manageable (eg. `azimuth=[0.5, 1.5, 2.5,..., 359.5]`)

## How should we treat it?

Currently the reindexing code relies on some internals which make things a bit hard to maintain. My suggestion would be to disentangle the reindexing code from the internals but feed the needed values as parameters. Then every reader can call this per activated `reindex_angle` kwarg.

+++

## Angle Reindexing Example

```{code-cell}
import matplotlib.pyplot as plt
import xarray as xr
from open_radar_data import DATASETS

import xradar as xd
```

```{code-cell}
filename = DATASETS.fetch("DWD-Vol-2_99999_20180601054047_00.h5")
```

```{code-cell}
def fix_angle(ds):
    angle_dict = xd.util.extract_angle_parameters(ds)
    display(angle_dict)
    start_ang = angle_dict["start_angle"]
    stop_ang = angle_dict["stop_angle"]
    angle_res = angle_dict["angle_res"]
    direction = angle_dict["direction"]

    # first find exact duplicates and remove
    ds = xd.util.remove_duplicate_rays(ds)

    # second reindex according to retrieved parameters
    ds = xd.util.reindex_angle(
        ds, start_ang, stop_ang, angle_res, direction, method="nearest"
    )

    return ds
```

### Read example data with one additional ray

```{code-cell}
ds0 = xr.open_dataset(filename, group="sweep_7", engine="gamic", first_dim="auto")
display(ds0.load())
```

```{code-cell}
ds0.DBTH.plot()
```

### Prepare sweep with several sections removed

```{code-cell}
ds_in = xr.concat(
    [
        ds0.isel(azimuth=slice(0, 100)),
        ds0.isel(azimuth=slice(150, 200)),
        ds0.isel(azimuth=slice(243, 300)),
        ds0.isel(azimuth=slice(330, 361)),
    ],
    "azimuth",
    data_vars="minimal",
)
display(ds_in)
```

```{code-cell}
ds_in.DBTH.plot()
```

### Reindex angle

First output is the extracted angle/time dictionary.

```{code-cell}
ds_out = fix_angle(ds_in)
display(ds_out)
```

```{code-cell}
ds_out.time.plot(marker=".")
plt.gca().grid()
```

We can observe that the dataset is aligned to it's expected number of rays.

```{code-cell}
ds_out.DBTH.plot()
```

### Fix timestamps

As reindexing instantiates the variables/coordinates added rays with `NaN`/`NaT` we need to take care of the coordinates.
The second angle (`elevation` in this case is already treated while reindexing by inserting it's median value, the time coordinate needs special handling.

```{code-cell}
ds_out2 = ds_out.copy(deep=True)
ds_out2 = ds_out2.pipe(xd.util.ipol_time)
```

```{code-cell}
ds_out2.time.plot(marker=".")
plt.gca().grid()
```

```{code-cell}
ds_out2.DBTH.sortby("time").plot(y="time")
```
