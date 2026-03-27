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

# NEXRAD Level 2

```{code-cell}
import cmweather  # noqa
import xarray as xr
from open_radar_data import DATASETS

import xradar as xd
```

## Download

Fetching NEXRAD Level2 radar data file from [open-radar-data](https://github.com/openradar/open-radar-data) repository.

```{code-cell}
filename = DATASETS.fetch("KATX20130717_195021_V06")
```

## xr.open_dataset

Making use of the xarray `nexradlevel2` backend. We also need to provide the group. Note, that we are using CfRadial2 group access pattern.

```{code-cell}
ds = xr.open_dataset(filename, group="sweep_0", engine="nexradlevel2")
display(ds)
```

```{code-cell}
ds
```

```{code-cell}
import numpy as np

np.testing.assert_almost_equal(ds.sweep_fixed_angle.values, 0.4833984)
```

### Plot Time vs. Azimuth

```{code-cell}
ds.azimuth.plot()
```

### Plot Range vs. Time

We need to sort by time and specify the y-coordinate.

```{code-cell}
ds.DBZH.sortby("time").plot(y="time", cmap="HomeyerRainbow")
```

### Plot Range vs. Azimuth

```{code-cell}
ds.DBZH.plot(cmap="HomeyerRainbow")
```

## backend_kwargs

Beside `first_dim` there are several additional backend_kwargs for the nexradlevel2 backend, which handle different aspects of angle alignment. This comes into play, when azimuth and/or elevation arrays are not evenly spacend and other issues.

```{code-cell}
help(xd.io.NexradLevel2BackendEntrypoint)
```

```{code-cell}
ds = xr.open_dataset(filename, group="sweep_0", engine="nexradlevel2", first_dim="time")
display(ds)
```

## open_nexradlevel2_datatree

The same works analoguous with the datatree loader. But additionally we can provide a sweep string, number or list.

```{code-cell}
help(xd.io.open_nexradlevel2_datatree)
```

```{code-cell}
dtree = xd.io.open_nexradlevel2_datatree(filename, sweep=4)
display(dtree)
```

### Plot Sweep Range vs. Time

```{code-cell}
dtree["sweep_4"].ds.DBZH.sortby("time").plot(y="time", cmap="HomeyerRainbow")
```

### Plot Sweep Range vs. Azimuth

```{code-cell}
dtree["sweep_4"].ds.DBZH.plot(cmap="HomeyerRainbow")
```

```{code-cell}
dtree = xd.io.open_nexradlevel2_datatree(filename, sweep="sweep_8")
display(dtree)
```

```{code-cell}
dtree = xd.io.open_nexradlevel2_datatree(filename, sweep=[0, 1, 8])
display(dtree)
```

```{code-cell}
dtree["sweep_0"]["sweep_fixed_angle"].values
```

```{code-cell}
dtree["sweep_8"]["sweep_fixed_angle"].values
```

```{code-cell}
dtree = xd.io.open_nexradlevel2_datatree(
    filename,
)
display(dtree)
```

```{code-cell}
dtree["sweep_1"]
```

```{code-cell}

```
