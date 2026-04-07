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

# Furuno

```{code-cell}
import xarray as xr
from open_radar_data import DATASETS

import xradar as xd
```

## Download

Fetching Furuno radar data file from [open-radar-data](https://github.com/openradar/open-radar-data) repository.

```{code-cell}
filename_scnx = DATASETS.fetch("2006_20220324_000000_000.scnx.gz")
filename_scn = DATASETS.fetch("0080_20210730_160000_01_02.scn.gz")
```

## xr.open_dataset

Making use of the xarray `furuno` backend.

+++

### scn format

```{code-cell}
ds = xr.open_dataset(filename_scn, engine="furuno")
display(ds)
```

#### Plot Time vs. Azimuth

```{code-cell}
ds.azimuth.plot(y="time")
```

#### Plot Range vs. Time

We need to sort by `time` and specify the y-coordinate.

```{code-cell}
ds.DBZH.sortby("time").plot(y="time")
```

#### Plot Range vs. Azimuth

```{code-cell}
ds.DBZH.sortby("azimuth").plot(y="azimuth")
```

### scnx format

```{code-cell}
ds = xr.open_dataset(filename_scnx, engine="furuno")
display(ds)
```

#### Plot Time vs. Azimuth

```{code-cell}
ds.azimuth.plot(y="time")
```

#### Plot Range vs. Time

We need to sort by `time` and specify the y-coordinate.

```{code-cell}
ds.DBZH.sortby("time").plot(y="time")
```

#### Plot Range vs. Azimuth

```{code-cell}
ds.DBZH.sortby("azimuth").plot(y="azimuth")
```

## open_furuno_datatree

Furuno scn/scnx files consist only of one sweep. But we might load and combine several sweeps into one DataTree.

```{code-cell}
dtree = xd.io.open_furuno_datatree(filename_scn)
display(dtree)
```

### Plot Sweep Range vs. Time

```{code-cell}
dtree["sweep_0"].ds.DBZH.plot()
```

### Plot Sweep Range vs. Azimuth

```{code-cell}
dtree["sweep_0"].ds.DBZH.sortby("azimuth").plot(y="azimuth")
```
