---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
---

# Iris/Sigmet - Reader

```{code-cell}
import xarray as xr
from open_radar_data import DATASETS

import xradar as xd
```

## Download

Fetching Iris radar data file from [open-radar-data](https://github.com/openradar/open-radar-data) repository.

```{code-cell}
filename_single = DATASETS.fetch("SUR210819000227.RAWKPJV")
filename_volume = DATASETS.fetch("cor-main131125105503.RAW2049")
```

## xr.open_dataset

Making use of the xarray `iris` backend. We also need to provide the group. We use the group notation from `CfRadial2`.

```{code-cell}
ds = xr.open_dataset(filename_single, group="sweep_0", engine="iris")
display(ds)
```

### Plot Time vs. Azimuth

```{code-cell}
ds.azimuth.plot(y="time")
```

### Plot Range vs. Time

We need to sort by time and specify the y-coordinate.

```{code-cell}
ds.DBZH.sortby("time").plot(y="time")
```

### Plot Range vs. Azimuth

```{code-cell}
ds.DBZH.plot(y="azimuth")
```

## backend_kwargs

Beside `first_dim` there are several additional backend_kwargs for the iris backend, which handle different aspects of angle alignment. This comes into play, when azimuth and/or elevation arrays are not evenly spacend and other issues.

```{code-cell}
help(xd.io.IrisBackendEntrypoint)
```

```{code-cell}
ds = xr.open_dataset(filename_single, group="sweep_0", engine="iris", first_dim="time")
display(ds)
```

## open_iris_datatree

The same works analoguous with the datatree loader. But additionally we can provide a sweep string, number or list. The underlying xarray.Dataset can be accessed with property `.ds`.

```{code-cell}
help(xd.io.open_iris_datatree)
```

```{code-cell}
dtree = xd.io.open_iris_datatree(filename_volume)
display(dtree)
```

```{code-cell}
dtree = xd.io.open_iris_datatree(filename_volume, sweep="sweep_8")
display(dtree)
```

```{code-cell}
dtree = xd.io.open_iris_datatree(filename_volume, sweep=[1, 2, 8])
display(dtree)
```

```{code-cell}
dtree = xd.io.open_iris_datatree(
    filename_volume,
    sweep=["sweep_0", "sweep_1", "sweep_8"],
)
display(dtree)
```

### Plot Time vs. Azimuth

```{code-cell}
dtree["sweep_0"].ds.azimuth.plot(y="time")
```

### Plot Sweep Range vs. Time

We need to sort by time and specify the y-coordinate. Please also observe the different resolutions of this plot, compared to the `Azimuth vs. Range` plot below. This is due to second-resolution of the time coordinate.

```{code-cell}
dtree["sweep_0"].ds.DBZH.sortby("time").plot(y="time")
```

### Plot Sweep Range vs. Azimuth

```{code-cell}
dtree["sweep_0"].ds.DBZH.plot()
```

```{code-cell}
import matplotlib.pyplot as plt

sweep = dtree["sweep_0"].to_dataset(inherit="all_coords")
sweep = sweep.sel(range=slice(0, 60000))
sweep = xd.georeference.get_x_y_z(sweep)
for var in xd.util.get_sweep_dataset_vars(sweep):
    plt.figure()
    sweep[var].plot(x="x", y="y")
    plt.title(var)
```

```{code-cell}

```
