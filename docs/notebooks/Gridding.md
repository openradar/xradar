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
- [wradlib](https://docs.wradlib.org/) grids a whole volume with
  ``wradlib.vpr.CAPPI`` / ``wradlib.vpr.PseudoCAPPI``, and interpolates single
  sweeps between polar and Cartesian coordinates with its xarray interpolation
  API (``.wrl.ipol.interpolate``).

This notebook grids the same xradar volume with both, onto the same grid:
0-10 km above the radar and ±100 km around it, with 1 km spacing.

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

```{code-cell}
z_levels = np.arange(0, 10_001, 1000.0)
xy = np.arange(-100_000, 100_001, 1000.0)
```

+++

## 3D grid with Py-ART

[``pyart.xradar.Xradar``](https://arm-doe.github.io/pyart/API/generated/pyart.xradar.Xradar.html)
wraps the xradar ``DataTree`` as a Py-ART radar object, without copying the
data into Py-ART's own structure.
[``pyart.map.grid_from_radars``](https://arm-doe.github.io/pyart/API/generated/pyart.map.grid_from_radars.html)
then maps all sweeps onto the grid.

```{code-cell}
pyart_radar = pyart.xradar.Xradar(radar)

grid = pyart.map.grid_from_radars(
    (pyart_radar,),
    grid_shape=(z_levels.size, xy.size, xy.size),
    grid_limits=(
        (z_levels[0], z_levels[-1]),
        (xy[0], xy[-1]),
        (xy[0], xy[-1]),
    ),
    fields=["DBZH"],
)
```

``Grid.to_xarray()`` returns an {py:class}`xarray:xarray.Dataset` with
``(time, z, y, x)`` dimensions and ``lat``/``lon`` coordinates.

```{code-cell}
pyart_grid = grid.to_xarray()
pyart_grid
```

To use the grid with Py-ART's grid-based functions (e.g. PyDDA), wrap the
dataset again with ``pyart.xradar.Xgrid``. Note that ``Xgrid`` expects
undecoded times, see its docstring.

+++

## 3D grid with wradlib

``.xradar.georeference()`` adds Cartesian ``x``, ``y``, ``z`` coordinates
to every radar bin, in the radar-centred azimuthal equidistant projection
(``z`` is the altitude above sea level).

```{code-cell}
radar = radar.xradar.georeference()
sweeps = [
    radar[name].to_dataset(inherit="all_coords")
    for name in radar.children
    if name.startswith("sweep_")
]
altitude = float(sweeps[0].altitude)
```

``wradlib.vpr.CAPPI`` takes the coordinates of all bins of the volume and of
all grid points as ``(n, 3)`` arrays. We stack the bins of all sweeps into one
point dimension.

```{code-cell}
bins = xr.concat(
    [
        swp.DBZH.where(swp.range <= 150_000, drop=True)
        .stack(bin=("azimuth", "range"))
        .reset_index("bin", drop=True)
        for swp in sweeps
    ],
    dim="bin",
)
bin_xyz = np.column_stack([bins.x.values, bins.y.values, bins.z.values])

# grid levels relative to the radar, like the Py-ART grid
grid_xyz = wrl.util.gridaspoints(z_levels + altitude, xy, xy)
```

``CAPPI`` masks grid points below the lowest and above the highest elevation
and beyond ``maxrange``. ``ipclass`` selects the interpolator from
``wradlib.ipol``, here nearest neighbour; ``maxdist`` leaves grid points
farther than 2 km from any bin empty.

```{code-cell}
elevations = [float(swp.sweep_fixed_angle) for swp in sweeps]
gridder = wrl.vpr.CAPPI(
    bin_xyz,
    grid_xyz,
    maxrange=150_000,
    minelev=min(elevations),
    maxelev=max(elevations),
    site=(0.0, 0.0, altitude),
    ipclass=wrl.ipol.Nearest,
)
wradlib_grid = xr.DataArray(
    gridder(bins.values, maxdist=2000).reshape(z_levels.size, xy.size, xy.size),
    dims=("z", "y", "x"),
    coords={"z": z_levels, "y": xy, "x": xy},
    name="DBZH",
    attrs=bins.attrs,
)
wradlib_grid
```

## Compare

```{code-cell}
fig, axs = plt.subplots(2, 3, figsize=(17, 10))
grids = {"Py-ART": pyart_grid.DBZH.isel(time=0), "wradlib CAPPI": wradlib_grid}
for row, (name, da) in zip(axs, grids.items()):
    for ax, height in zip(row[:2], [2000, 5000]):
        da.sel(z=height).plot(ax=ax, cmap="HomeyerRainbow", vmin=-10, vmax=60)
        ax.set_title(f"{name}, z = {height / 1000:.0f} km")
        ax.set_aspect("equal")
    da.sel(y=-50_000).plot(ax=row[2], cmap="HomeyerRainbow", vmin=-10, vmax=60)
    row[2].set_title(f"{name}, y = -50 km")
plt.tight_layout()
```

The differences mainly come from the methods: Py-ART weights all bins within
a radius of influence that grows with range (``roi_func``,
``weighting_function``), so it also fills the gaps between the beams and below
the lowest beam, while the ``CAPPI`` above uses the nearest bin and masks the
blind areas (use ``wradlib.vpr.PseudoCAPPI`` to fill them).

+++

## Single sweep to Cartesian with wradlib

For single sweeps or other 2D fields, wradlib's xarray interpolation API works
directly on the georeferenced sweep. The target is any dataset with ``x``/``y``
coordinates in the same projection.

```{code-cell}
swp = sweeps[0]
trg = xr.Dataset(coords={"x": xy, "y": xy})
```

``method`` selects the interpolator, e.g. ``"nearest"`` or
``"inverse_distance"`` (both based on
[``scipy.spatial.cKDTree``](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html)).
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
    ax.set_xlim(xy[0], xy[-1])
    ax.set_ylim(xy[0], xy[-1])
    ax.set_aspect("equal")
plt.tight_layout()
```

The same API works in the other direction (Cartesian to polar, e.g. a DEM
onto the radar bins), see the wradlib
[interpolation examples](https://docs.wradlib.org/en/latest/notebooks/interpolation/interpolation.html).

+++

## More examples

Py-ART:

- [Map a single radar to a Cartesian grid](https://arm-doe.github.io/pyart/examples/mapping/plot_map_one_radar_to_grid.html) (Py-ART gallery)
- [Py-ART Gridding](https://projectpythia.org/radar-cookbook/notebooks/foundations/pyart-gridding) and [Fast Barnes interpolation of RHI scans](https://projectpythia.org/radar-cookbook/notebooks/example-workflows/fastbarnes-interpolation-rhi) (Project Pythia Radar Cookbook)

wradlib:

- [Interpolation: polar to Cartesian and Cartesian to polar](https://docs.wradlib.org/en/latest/notebooks/interpolation/interpolation.html)
- [Recipe #2: 3D Cartesian grid from an ODIM_H5 polar volume](https://github.com/wradlib/wradlib-notebooks/blob/main/notebooks/workflow/recipe2.md) (``wradlib.vpr.CAPPI``)

Open radar short courses at ERAD:

- ERAD 2026: [Gridding Polar Data](https://openradarscience.org/erad2026/notebooks/workflow/gridding-data), [Composite To Grid](https://openradarscience.org/erad2026/notebooks/workflow/composite-togrid), [Py-ART Gridding](https://openradarscience.org/erad2026/notebooks/pyart/pyart-gridding)
- ERAD 2022: [Py-ART Gridding](https://github.com/openradar/erad2022/blob/main/notebooks/pyart/pyart-gridding.ipynb)
