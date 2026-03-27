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

# Assign GeoCoords

+++

This notebook demonstrates how to work with geocoordinates (longitude and latitude) instead of radar-centric x and y coordinates.

```{code-cell}
import cmweather  # noqa: F401
import matplotlib.pyplot as plt
import xarray as xr
from open_radar_data import DATASETS

import xradar as xd
```

## Define Functions

```{code-cell}
def get_geocoords(ds):
    """
    Converts Cartesian coordinates (x, y, z) in a radar dataset to geographic
    coordinates (longitude, latitude, altitude) using CRS transformation.

    Parameters
    ----------
    ds : xarray.Dataset
        Radar dataset with Cartesian coordinates.

    Returns
    -------
    xarray.Dataset
        Dataset with added 'lon', 'lat', and 'alt' coordinates and their attributes.
    """
    from pyproj import CRS, Transformer

    # Convert the dataset to georeferenced coordinates
    ds = ds.xradar.georeference()
    # Define source and target coordinate reference systems (CRS)
    src_crs = ds.xradar.get_crs()
    trg_crs = CRS.from_user_input(4326)  # EPSG:4326 (WGS 84)
    # Create a transformer for coordinate conversion
    transformer = Transformer.from_crs(src_crs, trg_crs)
    # Transform x, y, z coordinates to latitude, longitude, and altitude
    trg_y, trg_x, trg_z = transformer.transform(ds.x, ds.y, ds.z)
    # Assign new coordinates with appropriate attributes
    ds = ds.assign_coords(
        {
            "lon": (ds.x.dims, trg_x, xd.model.get_longitude_attrs()),
            "lat": (ds.y.dims, trg_y, xd.model.get_latitude_attrs()),
            "alt": (ds.z.dims, trg_z, xd.model.get_altitude_attrs()),
        }
    )
    return ds


def fix_sitecoords(ds):
    coords = ["longitude", "latitude", "altitude", "altitude_agl"]
    for coord in coords:
        if coord not in ds:
            continue
        # Skip coords that are already scalar
        if ds[coord].ndim == 0:
            continue
        # Compute median excluding NaN
        data = ds[coord].median(skipna=True).item()
        attrs = ds[coord].attrs
        ds = ds.assign_coords({coord: xr.DataArray(data=data, attrs=attrs)})
    return ds
```

## Load Data

```{code-cell}
file1 = DATASETS.fetch("cfrad.20080604_002217_000_SPOL_v36_SUR.nc")
file2 = DATASETS.fetch("cfrad.20211011_201557.188_to_20211011_201617.720_DOW8_PPI.nc")
```

## Example #1

+++

**Note:** Station coordinates (`latitude`, `longitude`, `altitude`) are stored on the root node of the DataTree. When accessing a sweep dataset directly, use `.to_dataset(inherit="all_coords")` to inherit these coordinates from the root. The `.xradar.georeference()` accessor handles this automatically.

```{code-cell}
dtree1 = xd.io.open_cfradial1_datatree(file1)
```

```{code-cell}
dtree1 = dtree1.xradar.georeference()
```

```{code-cell}
display(dtree1["sweep_0"].ds)
```

## Assign lat, lon, and alt

```{code-cell}
dtree1 = dtree1.xradar.map_over_sweeps(get_geocoords)
```

```{code-cell}
display(dtree1["sweep_0"].ds)
```

```{code-cell}
ds = dtree1["sweep_0"].to_dataset()
```

```{code-cell}
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ds["DBZ"].plot(x="x", y="y", vmin=-10, vmax=75, cmap="HomeyerRainbow", ax=ax[0])

ds["DBZ"].plot(x="lon", y="lat", vmin=-10, vmax=75, cmap="HomeyerRainbow", ax=ax[1])
plt.show()
```

## Example #2

```{code-cell}
dtree2 = xd.io.open_cfradial1_datatree(file2)
```

```{code-cell}
try:
    dtree2 = dtree2.xradar.georeference()
except Exception:
    print("Georeferencing failed!")
```

```{code-cell}
ds2 = dtree2["sweep_0"].to_dataset(inherit="all_coords")
print("Longitude", ds2["longitude"].shape)
print("Latitude", ds2["latitude"].shape)
```

<div class="alert alert-info">
    <p style="font-weight:bold; margin:0;">Important Note:</p>
    <p>
        This radar data is from a mobile research radar called <b>Doppler on Wheels (DOW)</b>. In previous versions of xradar,
        site coordinates (latitude, longitude) were stored per-ray on each sweep, resulting in shape <code>(731,)</code>.
        Now, station coordinates are stored as scalar values on the <b>root node</b> of the DataTree and inherited by sweeps
        via <code>to_dataset(inherit="all_coords")</code>. The <code>fix_sitecoords</code> helper gracefully handles both
        cases — it skips coordinates that are already scalar.
    </p>
</div>

+++

## Fix Coords

```{code-cell}
# Fix per-ray station coords on the root node by taking the median.
# Mobile radars like DOW have per-ray lat/lon which must be collapsed
# to scalar values before georeferencing.
root_ds = dtree2.ds
for coord in ["longitude", "latitude", "altitude"]:
    if coord in root_ds.coords and root_ds[coord].ndim > 0:
        median_val = root_ds[coord].median(skipna=True).item()
        root_ds = root_ds.assign_coords({coord: median_val})
dtree2.ds = root_ds
```

```{code-cell}
dtree2 = dtree2.xradar.map_over_sweeps(get_geocoords)
```

```{code-cell}
display(dtree2["sweep_0"].ds)
```

```{code-cell}
ds = dtree2["sweep_0"].to_dataset()
ref = ds.where(ds.DBZHC >= 5)["DBZHC"]

fig = plt.figure(figsize=(6, 5))
ax = plt.axes()
pl = ref.plot(
    x="lon",
    y="lat",
    vmin=-10,
    vmax=70,
    cmap="HomeyerRainbow",
    ax=ax,
)

ax.minorticks_on()
ax.grid(ls="--", lw=0.5)
ax.set_aspect("auto")
title = (
    dtree2.attrs["instrument_name"]
    + " "
    + str(ds.time.mean().dt.strftime("%Y-%m-%d %H:%M:%S").values)
)
ax.set_title(title)
plt.show()
```

```{code-cell}

```
