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

# Reproject Radar Coordinates

This notebook demonstrates the ``target_crs`` option now exposed on
``.xradar.georeference()`` for ``xarray.Dataset`` and ``xarray.DataTree``.

It lets you reproject radar ``x, y`` coordinates from the radar-native
Azimuthal Equidistant (AEQD) projection into any ``pyproj``-compatible CRS
in a single accessor call:

```python
radar = radar.xradar.georeference(target_crs=4326)
```

+++

## Imports

```{code-cell}
import cartopy.crs as ccrs
import cmweather  # noqa
import matplotlib.pyplot as plt
import pyproj
from open_radar_data import DATASETS

import xradar as xd
```

## Read a sample radar file

```{code-cell}
filename = DATASETS.fetch("cfrad.20080604_002217_000_SPOL_v36_SUR.nc")
radar = xd.io.open_cfradial1_datatree(filename, first_dim="auto")
radar
```

+++

## 1. Default georeferencing (AEQD, meters)

Without ``target_crs``, the accessor adds ``x``, ``y``, ``z`` in the
radar-native AEQD projection (distances in meters from the radar site).

```{code-cell}
radar_aeqd = radar.xradar.georeference()
sweep = radar_aeqd["sweep_0"].to_dataset(inherit="all_coords")
crs = sweep.xradar.get_crs()
print("projection:", crs.coordinate_operation.method_name if crs.coordinate_operation else crs.name)
print("is_projected / is_geographic:", crs.is_projected, "/", crs.is_geographic)
print("x units:", sweep.x.attrs.get("units"))
print("x range:", float(sweep.x.min()), float(sweep.x.max()))
```

```{code-cell}
fig, ax = plt.subplots(figsize=(7, 6))
radar_aeqd["sweep_0"]["DBZ"].plot(x="x", y="y", cmap="ChaseSpectral", ax=ax)
ax.set_title("AEQD (default) — x, y in meters from radar")
ax.set_aspect("equal")
```

+++

## 2. Reproject to geographic lon/lat (EPSG:4326)

Pass ``target_crs=4326`` to get ``x`` as longitude and ``y`` as latitude.
Attributes are updated automatically (``standard_name``, ``units``).

```{code-cell}
radar_geo = radar.xradar.georeference(target_crs=4326)
sweep = radar_geo["sweep_0"].to_dataset(inherit="all_coords")
crs = sweep.xradar.get_crs()
print("CRS name:", crs.name, "| EPSG:", crs.to_epsg())
print("x attrs:", dict(sweep.x.attrs))
print("y attrs:", dict(sweep.y.attrs))
```

```{code-cell}
fig, ax = plt.subplots(figsize=(8, 6))
radar_geo["sweep_0"]["DBZ"].plot(x="x", y="y", cmap="ChaseSpectral", ax=ax)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("EPSG:4326 — x=lon, y=lat")
```

+++

## 3. Reproject to a projected CRS (UTM, Web Mercator)

Any ``pyproj``-compatible CRS input works: EPSG code, WKT string, or a
``pyproj.CRS`` instance.

```{code-cell}
# Web Mercator (meters)
radar_wm = radar.xradar.georeference(target_crs="EPSG:3857")
sweep_wm = radar_wm["sweep_0"].to_dataset(inherit="all_coords")
crs = sweep_wm.xradar.get_crs()
print("CRS:", crs.name, "| EPSG:", crs.to_epsg())
```

```{code-cell}
utm_crs = pyproj.CRS("EPSG:32633")  # UTM zone 33N (tweak for your radar site)
radar_utm = radar.xradar.georeference(target_crs=utm_crs)
sweep_utm = radar_utm["sweep_0"].to_dataset(inherit="all_coords")
crs = sweep_utm.xradar.get_crs()
print("CRS:", crs.name, "| EPSG:", crs.to_epsg())
```

+++

## 4. Plot on a cartopy map

Once data is in a known CRS, hand it off to cartopy for map plotting.

```{code-cell}
radar_geo = radar.xradar.georeference(target_crs=4326)

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
radar_geo["sweep_0"]["DBZ"].plot(
    x="x",
    y="y",
    cmap="ChaseSpectral",
    transform=ccrs.PlateCarree(),
    ax=ax,
    cbar_kwargs=dict(pad=0.05, shrink=0.7),
)
ax.coastlines()
ax.gridlines(draw_labels=True)
ax.set_title("Georeferenced PPI on PlateCarree map")
```

+++

## 5. Side-by-side comparison of projections

Compare the same sweep in AEQD (meters), geographic (lon/lat), and
Lambert Conformal Conic — a projection commonly used for mid-latitude
weather maps.

```{code-cell}
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# AEQD (default)
radar_aeqd = radar.xradar.georeference()
radar_aeqd["sweep_0"]["DBZ"].plot(
    x="x", y="y", cmap="ChaseSpectral", ax=axes[0], add_colorbar=False,
)
axes[0].set_title("AEQD (meters)")
axes[0].set_aspect("equal")

# EPSG:4326 (lon/lat)
radar_geo = radar.xradar.georeference(target_crs=4326)
radar_geo["sweep_0"]["DBZ"].plot(
    x="x", y="y", cmap="ChaseSpectral", ax=axes[1], add_colorbar=False,
)
axes[1].set_title("EPSG:4326 (lon/lat)")
axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Latitude")

# Lambert Conformal Conic
lcc_crs = pyproj.CRS.from_proj4(
    "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=39.5 +lon_0=-105 +datum=WGS84"
)
radar_lcc = radar.xradar.georeference(target_crs=lcc_crs)
radar_lcc["sweep_0"]["DBZ"].plot(
    x="x", y="y", cmap="ChaseSpectral", ax=axes[2], add_colorbar=False,
)
axes[2].set_title("Lambert Conformal Conic")
axes[2].set_aspect("equal")

fig.tight_layout()
```

+++

## 6. Radar data on different map projections

The same georeferenced radar data rendered on four different cartopy map
projections — coastlines and gridlines make the projection distortion visible.

```{code-cell}
import cartopy.feature as cfeature

radar_geo = radar.xradar.georeference(target_crs=4326)
lon0 = float(radar_geo.ds.coords["longitude"].values)
lat0 = float(radar_geo.ds.coords["latitude"].values)

projections = [
    ("PlateCarree", ccrs.PlateCarree()),
    ("Lambert Conformal", ccrs.LambertConformal(
        central_longitude=lon0, central_latitude=lat0,
    )),
    ("Orthographic (globe)", ccrs.Orthographic(
        central_longitude=lon0, central_latitude=lat0,
    )),
    ("Mercator", ccrs.Mercator(central_longitude=lon0)),
]

fig = plt.figure(figsize=(14, 12))

for i, (name, proj) in enumerate(projections, 1):
    ax = fig.add_subplot(2, 2, i, projection=proj)
    radar_geo["sweep_0"]["DBZ"].plot(
        x="x", y="y", cmap="ChaseSpectral",
        transform=ccrs.PlateCarree(), ax=ax,
        add_colorbar=False,
    )
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.gridlines(draw_labels=isinstance(proj, (ccrs.PlateCarree, ccrs.Mercator)))
    ax.set_title(name)

fig.tight_layout()
```

+++

## 7. Works on a single Dataset too

The accessor is available on ``Dataset`` and ``DataArray`` as well, not just
``DataTree``.

```{code-cell}
ds = radar["sweep_0"].to_dataset(inherit="all_coords")
ds_geo = ds.xradar.georeference(target_crs=4326)
ds_geo[["DBZ"]]
```

+++

## Summary

* ``target_crs`` accepts anything ``pyproj.CRS(...)`` accepts: int
  [EPSG codes](https://epsg.io/), ``"EPSG:xxxx"`` strings, WKT, or
  ``pyproj.CRS`` instances.
* The ``spatial_ref`` / ``crs_wkt`` coordinate is updated to reflect the
  target CRS — ``sweep.xradar.get_crs()`` will return it.
* Currently only ``x`` and ``y`` are reprojected; ``z`` stays as AEQD
  altitude above the radar site.
* If you want **separate** ``lon`` / ``lat`` / ``alt`` coordinates (without
  overwriting ``x, y``), see the [Assign_GeoCoords](Assign_GeoCoords.md) notebook.
