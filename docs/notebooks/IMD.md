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

# IMD - Reader

The India Meteorological Department (IMD) publishes radar data as NetCDF4 files
with an IRIS-inspired variable layout. Each file holds **one sweep**; a
complete volume is assembled from multiple files: typically 2-3 files for
long-range PPI and 9-10 files for short-range, high-resolution PPI.

```{code-cell}
import xarray as xr
from open_radar_data import DATASETS

import xradar as xd
```

## Download

Fetching IMD radar data files from [open-radar-data](https://github.com/openradar/open-radar-data) repository.

```{code-cell}
filename_sweep_0 = DATASETS.fetch("IMD/JPR220822135253-IMD-B.nc")
filename_sweep_1 = DATASETS.fetch("IMD/JPR220822135253-IMD-B.nc.1")
volume_files = [
    DATASETS.fetch(f"IMD/JPR220822135253-IMD-B.nc{s}")
    for s in ["", ".1", ".2", ".3", ".4", ".5", ".6", ".7", ".8", ".9"]
]
```

## `xr.open_dataset` — a single sweep

The xarray `imd` backend reads a single IMD NetCDF file and returns a
CfRadial2-compatible sweep Dataset. Moments are renamed (`T`→`DBTH`,
`Z`→`DBZH`, `V`→`VRADH`, `W`→`WRADH`; dual-pol sites additionally carry
`ZDR` and `HCLASS`) and canonical CfRadial2 moment attributes
(`standard_name`, `long_name`, `units`) are applied.

```{code-cell}
ds = xr.open_dataset(filename_sweep_0, engine="imd")
display(ds)
```

### Plot Range vs. Azimuth

```{code-cell}
ds.DBZH.plot(y="azimuth")
```

## backend_kwargs

```{code-cell}
help(xd.io.IMDBackendEntrypoint)
```

```{code-cell}
ds = xr.open_dataset(filename_sweep_0, engine="imd", first_dim="time")
display(ds)
```

## `open_imd_datatree` — one volume

`open_imd_datatree` accepts **either** a single file path (→ single-sweep
DataTree) **or** a list of file paths (→ one volume). Multi-file stacking
is delegated to `xradar.util.create_volume`, which sorts sweeps by time
and supports `min_angle` / `max_angle` / `time_coverage_start` /
`time_coverage_end` / `volume_number` filtering.

```{code-cell}
help(xd.io.open_imd_datatree)
```

### Single sweep

```{code-cell}
dtree = xd.io.open_imd_datatree(filename_sweep_0)
display(dtree)
```

### Volume from multiple files

```{code-cell}
dtree = xd.io.open_imd_datatree(volume_files)
display(dtree)
```

### Plot a georeferenced PPI

```{code-cell}
import matplotlib.pyplot as plt

sweep = dtree["sweep_0"].to_dataset(inherit="all_coords")
sweep = sweep.sel(range=slice(0, 150000))
sweep = xd.georeference.get_x_y_z(sweep)
sweep.DBZH.plot(x="x", y="y")
plt.title("IMD Jaipur S-band DBZH")
plt.gca().set_aspect("equal")
```

### Filter by elevation angle

```{code-cell}
dtree = xd.io.open_imd_datatree(volume_files, max_angle=5.0)
display(dtree)
```

## `group_imd_files` — split a mixed directory

A single directory usually holds many volumes back-to-back. Use
`group_imd_files` to split a directory (or glob, or list) into per-volume
file lists by filename stem:

```{code-cell}
help(xd.io.group_imd_files)
```

```{code-cell}
xd.io.group_imd_files(volume_files)
```

Typical loop pattern over a directory of many volumes:

```python
for files in xd.io.group_imd_files("/data/imd"):
    dtree = xd.io.open_imd_datatree(files)
    # ... process this volume ...
```

## `open_imd_volumes` — all volumes in one DataTree

`open_imd_volumes` opens every volume in a directory at once and nests
them under zero-padded `vcp_NN` child nodes (VCP = *volume coverage
pattern*). The padding width is chosen so the child names sort lexically:

```
/
├── vcp_00/
│   ├── (root: sweep_group_name, sweep_fixed_angle, ...)
│   ├── sweep_0
│   ├── sweep_1
│   └── ...
├── vcp_01/
└── ...
```

All `open_imd_datatree` kwargs (`first_dim`, `reindex_angle`,
`optional_groups`, `min_angle`/`max_angle`, ...) are forwarded and applied
per volume.

```{code-cell}
help(xd.io.open_imd_volumes)
```

```{code-cell}
volume_files_2 = [
    DATASETS.fetch(f"IMD/JPR220822140253-IMD-B.nc{s}")
    for s in ["", ".1", ".2", ".3", ".4", ".5", ".6", ".7", ".8", ".9"]
]
tree = xd.io.open_imd_volumes(volume_files + volume_files_2)
display(tree)
```

```{code-cell}
tree["vcp_00/sweep_0"].ds["DBZH"]
```
