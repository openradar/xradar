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

# Resample sweep coordinates

`xradar.util.resample_coords` puts a reflectivity sweep onto a coarser azimuth/elevation and range grid, which reduces the data size for faster processing and smaller storage. The dB values are averaged in linear space.

+++

## Imports

```{code-cell}
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from open_radar_data import DATASETS

import xradar as xd
```

## Read a dual-polarization sweep

```{code-cell}
filename = DATASETS.fetch("71_20181220_060628.pvol.h5")
tree = xd.open_datatree(filename, engine="odim", first_dim="auto")
sweep = tree["sweep_0"].to_dataset()

print(xd.util.get_sweep_dataset_vars(sweep, non_standard=True))
print(
    f"{sweep.sizes['azimuth']} rays x {sweep.sizes['range']} gates, "
    f"{sweep.azimuth.values[1] - sweep.azimuth.values[0]:.1f} deg x "
    f"{sweep.range.values[1] - sweep.range.values[0]:.0f} m"
)
```

## Select the moment

The selection keeps the metadata variables:

```{code-cell}
dbzh = xd.util.select_sweep_dataset_vars(sweep, ["DBZH"])
list(dbzh.data_vars)
```

The whole sweep is refused:

```{code-cell}
try:
    xd.util.resample_coords(sweep, 2.0, 1000.0)
except ValueError as err:
    print(err)
```

## Resample onto a 2 degree / 1 km grid

```{code-cell}
resampled = xd.util.resample_coords(dbzh, 2.0, 1000.0)
print(dict(dbzh.sizes), "->", dict(resampled.sizes))
```

The packing of the input is kept:

```{code-cell}
def size(dataset):
    buffer = BytesIO()
    dataset.to_netcdf(buffer)
    return len(buffer.getvalue()) / 1e6


print(f"{size(dbzh):.2f} MB -> {size(resampled):.2f} MB")
```

## Comparison up to 60 km

```{code-cell}
original = dbzh.sel(range=dbzh.range <= 60_000.0)
new = resampled.sel(range=resampled.range <= 60_000.0)
near = original.interp(range=new.range, azimuth=new.azimuth, method="nearest")
diff = new.DBZH - near.DBZH

pd.DataFrame(
    [
        {
            "original mean": float(original.DBZH.mean()),
            "original std": float(original.DBZH.std()),
            "resampled mean": float(new.DBZH.mean()),
            "resampled std": float(new.DBZH.std()),
            "bias": float(diff.mean()),
            "MAE": float(abs(diff).mean()),
            "RMSE": float(np.sqrt((diff**2).mean())),
            "max |diff|": float(abs(diff).max()),
        }
    ],
    index=["DBZH"],
).round(2)
```

```{code-cell}
fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 6.5),
    constrained_layout=True,
    subplot_kw={"projection": "polar"},
)
for ax, (label, ds) in zip(axes, [("original", original), ("resampled", new)]):
    mesh = ax.pcolormesh(
        np.deg2rad(ds.azimuth.values),
        ds.range.values / 1000.0,
        ds.DBZH.values.T,
        shading="auto",
        cmap="turbo",
        vmin=0,
        vmax=60,
    )
    fig.colorbar(mesh, ax=ax, shrink=0.8)
    ax.set_title(label)
    ax.set_title("DBZH", loc="left")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 60)
```
