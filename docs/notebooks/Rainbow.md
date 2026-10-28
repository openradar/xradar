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
help(xd.open_datatree)
```

```{code-cell}
dtree = xd.open_datatree(filename, engine="rainbow", sweep="sweep_8")
display(dtree)
```

### Plot Sweep Range vs. Time

```{code-cell}
dtree["sweep_0"].dataset.DBZH.sortby("time").plot(y="time")
```

### Plot Sweep Range vs. Azimuth

```{code-cell}
dtree["sweep_0"].dataset.DBZH.plot()
```

```{code-cell}
dtree = xd.open_datatree(filename, engine="rainbow", sweep="sweep_8")
display(dtree)
```

```{code-cell}
dtree = xd.open_datatree(filename, engine="rainbow", sweep=[0, 1, 8])
display(dtree)
```

```{code-cell}
dtree = xd.open_datatree(filename, engine="rainbow", sweep=["sweep_1", "sweep_2", "sweep_8"])
display(dtree)
```

## Optional groups: radar_parameters and radar_calibration

With `optional_groups=True` the datatree also has the metadata groups. The Rainbow reader fills them from the slice metadata of the file (beam widths, the radar constants).

```{code-cell}
dtree = xd.open_datatree(filename, engine="rainbow", sweep=0, optional_groups=True)
display(dtree["radar_parameters"].dataset)
display(dtree["radar_calibration"].dataset)
```

```{code-cell}
import numpy as np

np.testing.assert_almost_equal(dtree["radar_parameters"].radar_beam_width_h, 1.326, 3)
np.testing.assert_almost_equal(dtree["radar_calibration"].radar_constant_h, 74.09, 2)
```

The groups hold one set of values: if the slices of a file disagree about a value, that value is left out of the group and a warning lists it. Select the sweep whose values you need with `sweep=`.
