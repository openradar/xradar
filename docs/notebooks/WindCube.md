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

# Vaisala WindCube Doppler Lidar

```{code-cell}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from open_radar_data import DATASETS

import xradar as xd
```

Vaisala (formerly Leosphere) WindCube scanning lidars write NetCDF-4 files with CfRadial 2 style sweep groups. The files are netCDF, so xarray's own netCDF backend would claim them first. `xd.io.discover_engine` runs the `guess_can_open` checks of the xradar backends and returns the matching engine (`"windcube"`), which we pass to `xr.open_datatree`.

## Doppler beam swinging (DBS)

A DBS scan measures four inclined beams and a vertical beam. The beams have different gate geometries (the range along the inclined beams is longer for the same height), so the reader returns one sweep per geometry.

```{code-cell}
dbs_file = DATASETS.fetch("WLS200s-218_2022-10-07_00-51-38_dbs_1823_75m.nc")
engine = xd.io.discover_engine(dbs_file)
print(engine)
dtree = xr.open_datatree(dbs_file, engine=engine)
display(dtree)
```

```{code-cell}
assert engine == "windcube"
assert dtree.ds.sweep_fixed_angle.values.tolist() == [75.0, 90.0]
assert dtree["sweep_0"].sweep_mode.item() == "doppler_beam_swinging"
```

The vendor sweep mode is kept in the `windcube_sweep_mode` attribute, the `range` of the sweeps differ:

```{code-cell}
for name in ("sweep_0", "sweep_1"):
    ds = dtree[name].ds
    print(
        name,
        ds.sweep_mode.attrs["windcube_sweep_mode"],
        f"elevation {ds.sweep_fixed_angle.item()}",
        f"range {ds.range.values[0]:.0f} - {ds.range.values[-1]:.0f} m",
    )
```

### Radial wind speed of the inclined beams

```{code-cell}
dtree["sweep_0"]["radial_wind_speed"].plot(y="time", cmap="Spectral", vmin=-20, vmax=20)
```

## PPI sector scan

A PPI which does not cover the full circle is a `sector`.

```{code-cell}
ppi_file = DATASETS.fetch("WLS400s-118_2025-02-06_22-26-47_ppi_50_150m.nc")
dtree = xr.open_datatree(ppi_file, engine=xd.io.discover_engine(ppi_file))
display(dtree)
```

```{code-cell}
assert dtree["sweep_0"].sweep_mode.item() == "sector"
assert dtree["sweep_0"].sweep_fixed_angle.item() == 2.5
```

```{code-cell}
dtree["sweep_0"]["radial_wind_speed"].plot(cmap="Spectral", vmin=-20, vmax=20)
```

## RHI

```{code-cell}
rhi_file = DATASETS.fetch("WLS400s-118_2025-02-20_02-46-06_rhi_45_150m.nc")
dtree = xr.open_datatree(rhi_file, engine=xd.io.discover_engine(rhi_file))
display(dtree)
```

```{code-cell}
assert dtree["sweep_0"].sweep_mode.item() == "rhi"
assert dtree["sweep_0"].sweep_fixed_angle.item() == 320.0
```

The measurements are read lazily. The file does not hold the position of the instrument, so we set a site to georeference the sweep (only the shape of the cross section matters here).

```{code-cell}
ds = dtree["sweep_0"].to_dataset(inherit="all_coords")
ds = ds.assign_coords(latitude=0.0, longitude=0.0, altitude=0.0)
ds = ds.xradar.georeference()
fig, ax = plt.subplots()
ds["radial_wind_speed"].plot(
    x="x", y="z", cmap="Spectral", vmin=-20, vmax=20, ax=ax
)
ax.set_aspect("equal")
```
