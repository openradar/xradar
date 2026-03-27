---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
---

# CfRadial1

```{code-cell}
import xarray as xr
from open_radar_data import DATASETS

import xradar as xd
```

## Download

Fetching CfRadial1 radar data file from [open-radar-data](https://github.com/openradar/open-radar-data) repository.

```{code-cell}
filename = DATASETS.fetch("cfrad.20080604_002217_000_SPOL_v36_SUR.nc")
```

## xr.open_dataset

Making use of the xarray `cfradial1` backend. We also need to provide the group.

```{code-cell}
ds = xr.open_dataset(filename, group="sweep_0", engine="cfradial1")
display(ds)
```

### Plot Time vs. Azimuth

We need to sort by time and specify the coordinate.

```{code-cell}
ds.azimuth.plot(y="time")
```

### Plot Range vs. Time

We need to sort by time and specify the coordinate.

```{code-cell}
ds.DBZ.sortby("time").plot(y="time")
```

### Plot Range vs. Azimuth

```{code-cell}
ds.DBZ.plot()
```

## backend_kwargs

The cfradial1 backend can be parameterized via kwargs. Please observe the possibilities below.

```{code-cell}
?xd.io.CfRadial1BackendEntrypoint
```

```{code-cell}
ds = xr.open_dataset(filename, group="sweep_0", engine="cfradial1", first_dim="time")
display(ds)
```

```{code-cell}
ds = xr.open_dataset(
    filename, group="sweep_1", engine="cfradial1", backend_kwargs=dict(first_dim="time")
)
display(ds)
```

## open_cfradial1_datatree

The same works analoguous with the datatree loader. But additionally we can provide a sweep number or list.

```{code-cell}
?xd.io.open_cfradial1_datatree
```

```{code-cell}
dtree = xd.io.open_cfradial1_datatree(
    filename,
    first_dim="time",
    optional=False,
)
display(dtree)
```

### Plot Sweep Range vs. Time

```{code-cell}
dtree["sweep_0"].ds.DBZ.plot()
```

### Plot Sweep Range vs. Azimuth

```{code-cell}
dtree["sweep_0"].ds.DBZ.sortby("azimuth").plot(y="azimuth")
```

```{code-cell}
dtree = xd.io.open_cfradial1_datatree(filename, sweep=[0, 1, 8])
display(dtree)
```

```{code-cell}
dtree = xd.io.open_cfradial1_datatree(filename, sweep=["sweep_0", "sweep_4", "sweep_8"])
display(dtree)
```
