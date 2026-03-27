---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
---

# NDPointIndex

This uses one of the new indexes xarray is providing. See [xarray-indexes](https://xarray-indexes.readthedocs.io/) for more in-depth details on that.

Needs latest xarray '2025.7.1'

```{code-cell}
import cmweather  # noqa
import numpy as np
import xarray as xr
from open_radar_data import DATASETS
```

```{code-cell}
xr.__version__
```

# Step-by-step guide

+++

## Load radar data

```{code-cell}
fname = DATASETS.fetch("DWD-Vol-2_99999_20180601054047_00.h5")
ds = xr.open_dataset(fname, engine="gamic", group="sweep_9")
display(ds)
```

## Georeference

Add x,y,z - 2D-coordinates.

```{code-cell}
ds = ds.xradar.georeference()
display(ds)
```

## Add NDPointIndex

This uses scipy.KDTree under the hood. See also Indexes-Section in the html-repr below.

```{code-cell}
ds = ds.set_xindex(("x", "y"), xr.indexes.NDPointIndex)
display(ds)
```

## Plot

This works as usual with the new NDPointIndex.

```{code-cell}
ds.DBZH.plot(
    x="x", y="y", xlim=(-10e3, 10e3), ylim=(-10e3, 10e3), cmap="HomeyerRainbow", vmin=0
)
```

## Nearest neighbour interpolation with NDPointIndex

+++

### Create 1D DataArrays for x and y selection

```{code-cell}
y = xr.DataArray(np.arange(-100e3, 100e3, 500), dims="y", name="y", attrs=ds.y.attrs)
x = xr.DataArray(np.arange(-100e3, 100e3, 500), dims="x", name="x", attrs=ds.x.attrs)
```

### Select with above 1D DataArrays

```{code-cell}
actual = ds.sel(y=y, x=x, method="nearest")
```

### Assign the 1D DataArrays

```{code-cell}
actual = actual.assign(x=x, y=y)
```

```{code-cell}
display(actual)
```

## Plot cartesian representation

```{code-cell}
actual.DBZH.plot(xlim=(-10e3, 10e3), ylim=(-10e3, 10e3), cmap="HomeyerRainbow", vmin=0)
```
