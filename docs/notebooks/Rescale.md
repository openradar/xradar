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

# Rescale a sweep

`xradar.util.rescale` resamples a sweep onto a coarser azimuth/elevation and range grid. dB-valued moments are averaged in linear space, cyclic moments (`PHIDP`, `VRADH`) on the circle, the other known moments in their own units and all remaining moments nearest-neighbour.

+++

## Imports

```{code-cell}
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

moments = ["DBZH", "VRADH", "ZDR", "RHOHV", "PHIDP"]
print(
    f"{sweep.sizes['azimuth']} rays x {sweep.sizes['range']} gates, "
    f"{sweep.azimuth.values[1] - sweep.azimuth.values[0]:.1f} deg x "
    f"{sweep.range.values[1] - sweep.range.values[0]:.0f} m, up to "
    f"{sweep.range.values[-1] / 1000:.0f} km"
)
print({m: sweep[m].attrs.get("units") for m in moments})
```

## Rescale onto a 2 degree / 1 km grid

```{code-cell}
rescaled = xd.util.rescale(sweep, 2.0, 1000.0)
print(dict(sweep.sizes), "->", dict(rescaled.sizes))
```

## Comparison up to 60 km

```{code-cell}
nyq = float(sweep.nyquist_velocity)
original = sweep.sel(range=sweep.range <= 60_000.0)
new = rescaled.sel(range=rescaled.range <= 60_000.0)
near = original.interp(range=new.range, azimuth=new.azimuth, method="nearest")


def diff(res, ref, moment):
    d = res - ref
    if moment == "PHIDP":  # wrapped at 360 degrees
        return (d + 180.0) % 360.0 - 180.0
    if moment == "VRADH":  # aliased at +/- nyquist
        return (d + nyq) % (2 * nyq) - nyq
    return d


rows = {}
for m in moments:
    a, b = original[m], new[m]
    d = diff(b, near[m], m)
    rows[m] = {
        "original mean": float(a.mean()),
        "original std": float(a.std()),
        "rescaled mean": float(b.mean()),
        "rescaled std": float(b.std()),
        "bias": float(d.mean()),
        "MAE": float(abs(d).mean()),
        "RMSE": float(np.sqrt((d**2).mean())),
        "max |diff|": float(abs(d).max()),
    }
pd.DataFrame(rows).T.round(2)
```

```{code-cell}
cmaps = {
    "DBZH": "turbo",
    "VRADH": "RdBu_r",
    "ZDR": "RdBu_r",
    "RHOHV": "viridis",
    "PHIDP": "twilight",
}
limits = {
    "DBZH": (0, 60),
    "VRADH": (-nyq, nyq),
    "ZDR": (-2, 4),
    "RHOHV": (0, 1),
    "PHIDP": (0, 360),
}

fig, axes = plt.subplots(
    len(moments), 2, figsize=(12, 24), constrained_layout=True,
    subplot_kw={"projection": "polar"},
)
for row, moment in enumerate(moments):
    for col, (label, ds) in enumerate(
        [("original", original), ("rescaled", new)]
    ):
        ax = axes[row, col]
        mesh = ax.pcolormesh(
            np.deg2rad(ds.azimuth.values),
            ds.range.values / 1000.0,
            ds[moment].values.T,
            shading="auto",
            cmap=cmaps[moment],
            vmin=limits[moment][0],
            vmax=limits[moment][1],
        )
        fig.colorbar(mesh, ax=ax, shrink=0.8)
        if row == 0:
            ax.set_title(label)
        if col == 0:
            ax.set_title(moment, loc="left")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlim(0, 60)
```
