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

# Gridding with Py-ART and wradlib

xradar reads and writes radar data in its native polar geometry
(sweeps with ``azimuth``/``elevation`` and ``range``). It does not
interpolate onto Cartesian grids itself. For that, the xradar
{py:class}`xarray:xarray.DataTree` and its sweep datasets can be handed
directly to downstream packages:

- [Py-ART](https://arm-doe.github.io/pyart/) grids a whole volume into a
  3D ``(z, y, x)`` grid with [``pyart.map.grid_from_radars``](https://arm-doe.github.io/pyart/API/generated/pyart.map.grid_from_radars.html).
- [wradlib](https://docs.wradlib.org/) interpolates sweeps between polar and
  Cartesian coordinates with its xarray interpolation API
  (``.wrl.ipol.interpolate``).

This notebook shows both, starting from the same xradar volume.

+++

## Imports

```{code-cell}
import cmweather  # noqa
import matplotlib.pyplot as plt
import numpy as np
import pyart
import wradlib as wrl  # noqa, registers the .wrl accessor
import xarray as xr
from open_radar_data import DATASETS

import xradar as xd
```

## Read a radar volume

```{code-cell}
filename = DATASETS.fetch("71_20181220_060628.pvol.h5")
radar = xd.io.open_odim_datatree(filename)
radar
```

+++

## 3D grid with Py-ART

[``pyart.xradar.Xradar``](https://arm-doe.github.io/pyart/API/generated/pyart.xradar.Xradar.html) wraps the xradar ``DataTree`` as a Py-ART
radar object, without copying the data into Py-ART's own structure.
[``pyart.map.grid_from_radars``](https://arm-doe.github.io/pyart/API/generated/pyart.map.grid_from_radars.html) then maps all sweeps onto a regular
grid, here 0-10 km height and ±100 km around the radar with 1 km spacing.

```{code-cell}
pyart_radar = pyart.xradar.Xradar(radar)

grid = pyart.map.grid_from_radars(
    (pyart_radar,),
    grid_shape=(11, 201, 201),
    grid_limits=((0, 10_000), (-100_000, 100_000), (-100_000, 100_000)),
    fields=["DBZH"],
)
```

``Grid.to_xarray()`` returns an {py:class}`xarray:xarray.Dataset` with
``(time, z, y, x)`` dimensions and ``lat``/``lon`` coordinates.

```{code-cell}
grid_ds = grid.to_xarray()
grid_ds
```

```{code-cell}
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for ax, height in zip(axs, [2000, 5000]):
    grid_ds.DBZH.isel(time=0).sel(z=height).plot(
        ax=ax, cmap="HomeyerRainbow", vmin=-10, vmax=60
    )
    ax.set_title(f"Py-ART grid, z = {height / 1000:.0f} km")
    ax.set_aspect("equal")
plt.tight_layout()
```

To use the grid with Py-ART's grid-based functions (e.g. PyDDA), wrap the
dataset again with ``pyart.xradar.Xgrid``. Note that ``Xgrid`` expects
undecoded times, see its docstring.

+++

## Polar to Cartesian with wradlib

wradlib interpolates single sweeps. The sweep needs Cartesian ``x``/``y``
coordinates, which ``.xradar.georeference()`` adds in the radar-centred
azimuthal equidistant projection.

```{code-cell}
radar = radar.xradar.georeference()
swp = radar["sweep_0"].to_dataset(inherit="all_coords")
swp
```

The target is any dataset with ``x``/``y`` coordinates in the same
projection, here the same 1 km grid as above.

```{code-cell}
trg = xr.Dataset(
    coords={
        "x": np.arange(-100_000, 100_001, 1000.0),
        "y": np.arange(-100_000, 100_001, 1000.0),
    }
)
```

``method`` selects the interpolator, e.g. ``"nearest"`` or
``"inverse_distance"`` (both based on [``scipy.spatial.cKDTree``](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html)).
``distance_upper_bound`` keeps grid points far from any radar bin empty.

```{code-cell}
nearest = swp.wrl.ipol.interpolate(
    trg, method="nearest", distance_upper_bound=1500
)
idw = swp.wrl.ipol.interpolate(
    trg, method="inverse_distance", k=4, idw_p=2, distance_upper_bound=1500
)
nearest
```

```{code-cell}
fig, axs = plt.subplots(1, 3, figsize=(17, 5))
swp.DBZH.plot(
    x="x", y="y", ax=axs[0], cmap="HomeyerRainbow", vmin=-10, vmax=60
)
axs[0].set_title(f"Sweep {float(swp.sweep_fixed_angle):.1f}° (polar)")
nearest.DBZH.plot(ax=axs[1], cmap="HomeyerRainbow", vmin=-10, vmax=60)
axs[1].set_title("wradlib nearest")
idw.DBZH.plot(ax=axs[2], cmap="HomeyerRainbow", vmin=-10, vmax=60)
axs[2].set_title("wradlib inverse distance")
for ax in axs:
    ax.set_xlim(-100_000, 100_000)
    ax.set_ylim(-100_000, 100_000)
    ax.set_aspect("equal")
plt.tight_layout()
```

The same API works in the other direction (Cartesian to polar, e.g. a DEM
onto the radar bins), see the wradlib
[interpolation examples](https://docs.wradlib.org/en/latest/notebooks/interpolation/interpolation.html).

+++

## Which one to use?

- **Py-ART** grids the whole volume into a 3D grid in one call, with
  weighting functions designed for radar data (e.g. ``Barnes2``, ``Cressman``),
  and works with multiple radars.
- **wradlib** works on xarray objects throughout, keeps sweep and coordinate
  metadata, and offers several interpolators (nearest, inverse distance,
  kriging, ``griddata``, ``map_coordinates``) for single sweeps or other
  2D fields.
