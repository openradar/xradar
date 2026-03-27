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

# GAMIC

```{code-cell}
import xarray as xr
from open_radar_data import DATASETS

import xradar as xd
```

## Download

Fetching GAMIC radar data file from [open-radar-data](https://github.com/openradar/open-radar-data) repository.

```{code-cell}
filename = DATASETS.fetch("DWD-Vol-2_99999_20180601054047_00.h5")
```

## xr.open_dataset

Making use of the xarray `gamic` backend. We also need to provide the group. Note, that we are using CfRadial2 group access pattern.

```{code-cell}
ds = xr.open_dataset(filename, group="sweep_9", engine="gamic")
display(ds)
```

### Plot Time vs. Azimuth

```{code-cell}
ds.azimuth.plot()
```

### Plot Range vs. Time

```{code-cell}
ds.DBZH.plot()
```

### Plot Range vs. Azimuth

We need to sort by azimuth and specify the y-coordinate.

```{code-cell}
ds.DBZH.sortby("azimuth").plot(y="azimuth")
```

## backend_kwargs

Beside `first_dim` there are several additional backend_kwargs for the odim backend, which handle different aspects of angle alignment. This comes into play, when azimuth and/or elevation arrays are not evenly spacend and other issues.

```{code-cell}
help(xd.io.GamicBackendEntrypoint)
```

```{code-cell}
ds = xr.open_dataset(filename, group="sweep_9", engine="gamic", first_dim="time")
display(ds)
```

## open_odim_datatree

The same works analoguous with the datatree loader. But additionally we can provide a sweep string, number or list.

```{code-cell}
help(xd.io.open_gamic_datatree)
```

```{code-cell}
dtree = xd.io.open_gamic_datatree(filename, sweep=8)
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
dtree = xd.io.open_gamic_datatree(filename, sweep="sweep_8")
display(dtree)
```

```{code-cell}
dtree = xd.io.open_gamic_datatree(filename, sweep=[0, 1, 8])
display(dtree)
```

```{code-cell}
dtree = xd.io.open_gamic_datatree(filename, sweep=["sweep_1", "sweep_2", "sweep_8"])
display(dtree)
```
