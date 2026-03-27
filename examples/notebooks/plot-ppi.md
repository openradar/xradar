---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
---

# Plan Position Indicator
A Plan Position Indicator (PPI) plot is a common plot requested by radar scientists. Let's show how to create this plot using `xradar`

+++

## Imports

```{code-cell}
import cmweather  # noqa
from open_radar_data import DATASETS

import xradar as xd
```

```{code-cell}
import cartopy
import matplotlib.pyplot as plt
```

## Read in some data

Fetching CfRadial1 radar data file from [open-radar-data](https://github.com/openradar/open-radar-data) repository.

```{code-cell}
filename = DATASETS.fetch("cfrad.20080604_002217_000_SPOL_v36_SUR.nc")
```

Read the data using the `cfradial1` engine

```{code-cell}
radar = xd.io.open_cfradial1_datatree(filename, first_dim="auto")
display(radar)
```

## Add georeferencing
We can use the georeference function, or the accessor to add our georeference information!

+++

### Georeference Accessor
If you prefer the accessor (`.xradar.georefence()`), this is how you would add georeference information to your radar object.

```{code-cell}
radar = radar.xradar.georeference()
display(radar)
```

Please observe, that the additional coordinates `x`, `y`, `z` have been added to the dataset. This will also add `spatial_ref` CRS information on the used Azimuthal Equidistant Projection.

```{code-cell}
radar["sweep_0"]
```

### Use the Function
We can also use the function `xd.geoference.get_x_y_z_tree` function if you prefer that method.

```{code-cell}
radar = xd.georeference.get_x_y_z_tree(radar)
display(radar["sweep_0"])
```

## Plot our Data

### Plot simple PPI

Now, let's create our PPI plot! We just use the newly created 2D-coordinates `x` and `y` to create a meshplot.

```{code-cell}
radar["sweep_0"]["DBZ"].plot(x="x", y="y", cmap="ChaseSpectral")
```

### Plot PPI with geographic coordinates

```{code-cell}
fig, ax = plt.subplots(figsize=(10, 8))
radar = xd.georeference.get_x_y_z_tree(radar, target_crs=4326)
sweep = radar["sweep_0"]
sweep["DBZ"].plot(x="x", y="y", cmap="ChaseSpectral")
ax.grid(True, which="both", linestyle="--", color="gray", alpha=0.5)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Radar Reflectivity Sweep")
```

### Plot PPI on map with cartopy

If you have `cartopy` installed, you can easily plot on maps. We first have to extract the CRS from the dataset and to wrap it in a `cartopy.crs.Projection`.

```{code-cell}
proj_crs = xd.georeference.get_crs(radar["sweep_0"].to_dataset(inherit="all_coords"))
cart_crs = cartopy.crs.Projection(proj_crs)
```

Second, we create a matplotlib GeoAxes and a nice map.

```{code-cell}
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection=cartopy.crs.PlateCarree())
radar["sweep_0"]["DBZ"].plot(
    x="x",
    y="y",
    cmap="ChaseSpectral",
    transform=cart_crs,
    cbar_kwargs=dict(pad=0.075, shrink=0.75),
)
ax.coastlines()
ax.gridlines(draw_labels=True)
```

```{code-cell}

```
