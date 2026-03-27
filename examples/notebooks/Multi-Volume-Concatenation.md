---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
---

# Timeseries of Sweeps

+++

## Imports

```{code-cell}
import warnings

import cartopy.crs as ccrs
import cmweather  # noqa
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from open_radar_data import DATASETS

import xradar as xd

warnings.filterwarnings("ignore")
```

## Access Radar Data from the Open Radar Data Package

```{code-cell}
radar_files = [
    "gucxprecipradarcmacppiS2.c1.20220314.021559.nc",
    "gucxprecipradarcmacppiS2.c1.20220314.024239.nc",
    "gucxprecipradarcmacppiS2.c1.20220314.025840.nc",
]
files = [DATASETS.fetch(file) for file in radar_files]
```

## Read the Data using Xradar
We can read the data into xradar by using the `xr.open_mfdataset` function, but first, we need to align the angles of the different radar volumes.

```{code-cell}
def fix_angle(ds):
    """
    Aligns the radar volumes
    """
    ds["time"] = ds.time.load()  # Convert time from dask to numpy

    start_ang = 0  # Set consistent start/end values
    stop_ang = 360

    # Find the median angle resolution
    angle_res = ds.azimuth.diff("azimuth").median()

    # Determine whether the radar is spinning clockwise or counterclockwise
    median_diff = ds.azimuth.diff("time").median()
    ascending = median_diff > 0
    direction = 1 if ascending else -1

    # first find exact duplicates and remove
    ds = xd.util.remove_duplicate_rays(ds)

    # second reindex according to retrieved parameters
    ds = xd.util.reindex_angle(
        ds, start_ang, stop_ang, angle_res, direction, method="nearest"
    )

    ds = ds.expand_dims("volume_time")  # Expand for volumes for concatenation

    ds["volume_time"] = [np.nanmin(ds.time.values)]

    return ds
```

```{code-cell}
# Concatenate in xarray ds
ds = xr.open_mfdataset(
    files,
    preprocess=fix_angle,
    engine="cfradial1",
    group="sweep_0",
    concat_dim="volume_time",
    combine="nested",
)
ds
```

## Visualize the Dataset
Now that we have our dataset, we can visualize it.

We need to georeference first, then plot it!

+++

### Georeference the Dataset

```{code-cell}
ds = ds.xradar.georeference()
ds
```

### Extract the geoaxis information

```{code-cell}
proj_crs = xd.georeference.get_crs(ds)
cart_crs = ccrs.Projection(proj_crs)
```

### Use the FacetGrid to visualize

```{code-cell}
# Create our facets grid
fg = ds.DBZ.plot.pcolormesh(
    x="x",
    y="y",
    vmin=-20,
    vmax=40,
    cmap="ChaseSpectral",
    col="volume_time",
    edgecolors="face",
    figsize=(14, 4),
    transform=cart_crs,
    subplot_kws={"projection": ccrs.PlateCarree()},
)
# Set the title
fg.set_titles("{value}")

# Fix the geo-axes labels
first_axis = True
for ax in fg.axes.flat:
    ax.coastlines()
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=1,
        color="gray",
        alpha=0.3,
        linestyle="--",
    )
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    gl.top_labels = False
    gl.right_labels = False
    if first_axis:
        gl.left_labels = True
    else:
        gl.left_labels = False
    first_axis = False
```

```{code-cell}

```
