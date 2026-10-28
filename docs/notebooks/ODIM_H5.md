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

# ODIM_H5

```{code-cell}
import xarray as xr
from open_radar_data import DATASETS

import xradar as xd
```

## Download

Fetching ODIM_H5 radar data file from [open-radar-data](https://github.com/openradar/open-radar-data) repository.

```{code-cell}
filename = DATASETS.fetch("71_20181220_060628.pvol.h5")
```

## xr.open_dataset

Making use of the xarray `odim` backend. We also need to provide the group. We use CfRadial2 group access pattern.

```{code-cell}
ds = xr.open_dataset(filename, group="sweep_0", engine="odim")
display(ds)
```

```{code-cell}
ds.sweep_fixed_angle.values
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
?xd.io.OdimBackendEntrypoint
```

```{code-cell}
ds = xr.open_dataset(filename, group="sweep_0", engine="odim", first_dim="time")
display(ds)
```

## open_odim_datatree

The same works analoguous with the datatree loader. But additionally we can provide a sweep string, number or list.

```{code-cell}
?xd.open_datatree
```

```{code-cell}
dtree = xd.open_datatree(filename, engine="odim", sweep=8)
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
dtree = xd.open_datatree(filename, engine="odim", sweep="sweep_8")
display(dtree)
```

```{code-cell}
dtree = xd.open_datatree(filename, engine="odim", sweep=[0, 1, 8])
display(dtree)
```

```{code-cell}
dtree = xd.open_datatree(filename, engine="odim", sweep=["sweep_0", "sweep_1", "sweep_8"])
display(dtree)
```

## Optional groups: radar_parameters and radar_calibration

With `optional_groups=True` the datatree also has the metadata groups. The ODIM_H5 reader fills them from the `how` attributes of the file (`beamwH/V`, `antgainH/V`, `radconstH/V`, `NEZH/V` and `radomelossH/V`), with the names of the data model.

```{code-cell}
dtree = xd.open_datatree(filename, engine="odim", sweep=0, optional_groups=True)
display(dtree["radar_parameters"].dataset)
```

```{code-cell}
import numpy as np

np.testing.assert_almost_equal(dtree["radar_parameters"].radar_beam_width_h, 0.896)
np.testing.assert_almost_equal(dtree["radar_parameters"].radar_beam_width_v, 0.896)
```

This file has no calibration attributes, so `radar_calibration` is empty. The groups hold one set of values: if the datasets of a file disagree about a value, that value is left out of the group and a warning lists it. Select the sweep whose values you need with `sweep=`.

```{code-cell}
display(dtree["radar_calibration"].dataset)
```
