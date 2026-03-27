---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
---

# Rainbow

```{code-cell}
import xarray as xr
from open_radar_data import DATASETS

import xradar as xd
```

## Download

Fetching Rainbow radar data file from [open-radar-data](https://github.com/openradar/open-radar-data) repository.

```{code-cell}
filename = DATASETS.fetch("2013051000000600dBZ.vol")
```

## xr.open_dataset

Making use of the xarray `rainbow` backend. We also need to provide the group. Note, that we are using CfRadial2 group access pattern.

```{code-cell}
ds = xr.open_dataset(filename, group="sweep_0", engine="rainbow")
display(ds)
```

### Plot Time vs. Azimuth

We need to sort by time and specify the coordinate.

```{code-cell}
ds.azimuth.sortby("time").plot(x="time")
```

### Plot Range vs. Time

We need to sort by time and specify the coordinate.

```{code-cell}
ds.DBZH.sortby("time").plot(y="time")
```

### Plot Range vs. Azimuth

```{code-cell}
ds.DBZH.plot()
```

## backend_kwargs

Beside `first_dim` there are several additional backend_kwargs for the rainbow backend, which handle different aspects of angle alignment. This comes into play, when azimuth and/or elevation arrays are not evenly spacend and other issues.

```{code-cell}
help(xd.io.RainbowBackendEntrypoint)
```

```{code-cell}
ds = xr.open_dataset(filename, group="sweep_1", engine="rainbow", first_dim="time")
display(ds)
```

## open_odim_datatree

The same works analoguous with the datatree loader. But additionally we can provide a sweep string, number or list.

```{code-cell}
help(xd.io.open_rainbow_datatree)
```

```{code-cell}
dtree = xd.io.open_rainbow_datatree(filename, sweep="sweep_8")
display(dtree)
```

### Plot Sweep Range vs. Time

```{code-cell}
dtree["sweep_0"].ds.DBZH.sortby("time").plot(y="time")
```

### Plot Sweep Range vs. Azimuth

```{code-cell}
dtree["sweep_0"].ds.DBZH.plot()
```

```{code-cell}
dtree = xd.io.open_rainbow_datatree(filename, sweep="sweep_8")
display(dtree)
```

```{code-cell}
dtree = xd.io.open_rainbow_datatree(filename, sweep=[0, 1, 8])
display(dtree)
```

```{code-cell}
dtree = xd.io.open_rainbow_datatree(filename, sweep=["sweep_1", "sweep_2", "sweep_8"])
display(dtree)
```
