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

# NEXRAD Level 3

NEXRAD Level 3 (NIDS) products are small, derived, single-product files —
super-resolution reflectivity and velocity, dual-pol moments, hydrometeor
classification and precipitation products — distributed in real time and
archived back to ~2020 on AWS at `s3://unidata-nexrad-level3/`. This notebook
demonstrates:

1. Opening a Level 3 product straight from S3
2. Plotting a georeferenced PPI
3. Assembling a multi-tilt volume from same-product files

```{code-cell}
import cmweather  # noqa: F401 -- registers colormaps
import fsspec
import matplotlib.pyplot as plt
import xarray as xr

import xradar as xd
```

## Open a product from S3

Level 3 keys are flat: `SSS_PPP_YYYY_MM_DD_HH_MM_SS` (site without the
leading K, product code, timestamp). `N0B` is super-resolution base
reflectivity on the lowest cut:

```{code-cell}
fs = fsspec.filesystem("s3", anon=True)

f = fs.open("unidata-nexrad-level3/LOT_N0B_2026_07_17_19_30_15")
ds = xr.open_dataset(f, engine="nexradlevel3")
ds
```

The raw data levels are decoded into physical values, and the sweep follows
the same CfRadial2/FM301 layout as every other xradar backend — so
georeferencing and plotting work unchanged:

```{code-cell}
ds = ds.xradar.georeference()

fig, ax = plt.subplots(figsize=(6, 5))
ds["DBZH"].plot(x="x", y="y", cmap="ChaseSpectral", vmin=-10, vmax=70, ax=ax)
ax.set_title(f"{ds.attrs['instrument_name']} N0B "
             f"{str(ds['time'].values[0])[:19]}")
ax.set_aspect("equal")
```

## Dual-pol products

Each product lives in its own file. Correlation coefficient (`N0C`) for the
same volume scan:

```{code-cell}
f = fs.open("unidata-nexrad-level3/LOT_N0C_2026_07_17_19_30_15")
cc = xr.open_dataset(f, engine="nexradlevel3").xradar.georeference()

fig, ax = plt.subplots(figsize=(6, 5))
cc["RHOHV"].plot(x="x", y="y", cmap="LangRainbow12", vmin=0.7, vmax=1.05, ax=ax)
ax.set_aspect("equal")
```

## Multi-tilt volume

Files of the same product at different cuts (N0B/N1B/N2B/N3B) assemble into
a volume DataTree ordered by fixed angle:

```{code-cell}
keys = [
    "LOT_N0B_2026_07_17_19_30_15",
    "LOT_N1B_2026_07_17_19_30_15",
    "LOT_N2B_2026_07_17_19_30_15",
    "LOT_N3B_2026_07_17_19_30_15",
]
files = [fs.open(f"unidata-nexrad-level3/{key}") for key in keys]

dtree = xd.io.open_nexradlevel3_datatree(files)
dtree.root["sweep_fixed_angle"].values
```

```{code-cell}
dtree = dtree.xradar.georeference()

fig, axs = plt.subplots(2, 2, figsize=(11, 9))
for ax, (name, sweep) in zip(axs.flat, dtree.children.items()):
    sw = sweep.to_dataset()
    sw["DBZH"].plot(x="x", y="y", cmap="ChaseSpectral", vmin=-10, vmax=70, ax=ax)
    ax.set_title(f"{name}: {sw['sweep_fixed_angle'].values.item():.1f} deg")
    ax.set_aspect("equal")
fig.tight_layout()
```
