---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
---

# CfRadial1 - Export

+++

### Imports

```{code-cell}
import cmweather  # noqa
import xarray as xr
from open_radar_data import DATASETS

import xradar as xd
```

### Download

Fetching CfRadial1 radar data file from [open-radar-data](https://github.com/openradar/open-radar-data) repository.

```{code-cell}
filename = DATASETS.fetch("cfrad.20080604_002217_000_SPOL_v36_SUR.nc")
```

```{code-cell}
radar = xd.io.open_cfradial1_datatree(filename, first_dim="auto")
display(radar)
```

### Plot Azimuth vs. Range

```{code-cell}
radar.sweep_0.DBZ.plot(cmap="ChaseSpectral", vmin=-10, vmax=70)
```

### Plot Time vs. Range

```{code-cell}
radar.sweep_0.DBZ.swap_dims({"azimuth": "time"}).sortby("time").plot(
    cmap="ChaseSpectral", vmin=-10, vmax=70
)
```

### Georeference

```{code-cell}
radar = radar.xradar.georeference()
display(radar)
```

### Plot PPI

```{code-cell}
radar["sweep_0"]["DBZ"].plot(x="x", y="y", cmap="ChaseSpectral", vmin=-10, vmax=70)
```

### Filter

Apply basic reflectivity filter. This is just a demonstration.

```{code-cell}
def ref_filter(dtree, sweep="sweep_0", field="DBZ"):
    ds = dtree[sweep].ds
    ds = ds.where((ds[field] >= -10) & (ds[field] <= 70))
    red_patch = ds.where(
        (
            (ds[field] >= ds[field].max().values - 0.5)
            & (ds[field] <= ds[field].max().values + 0.5)
        ),
        drop=True,
    )
    rmin, rmax = int(red_patch.range.min().values - 150), int(
        red_patch.range.max().values + 150
    )
    out_of_range_mask = (ds.range < rmin) | (ds.range > rmax)
    ds[field] = ds[field].where(out_of_range_mask)
    # Interpolate missing values using the slinear method along the 'range' dimension
    ds[field] = ds[field].interpolate_na(dim="range", method="slinear")
    dtree[sweep][f"corr_{field}"] = ds[field].copy()
    return dtree[sweep]
```

```{code-cell}
swp0 = ref_filter(radar, sweep="sweep_0", field="DBZ")
```

```{code-cell}
swp0.corr_DBZ.plot(x="x", y="y", cmap="ChaseSpectral", vmin=-10, vmax=70)
```

### Filter full volume

```{code-cell}
# Initialize an empty DataTree
result_tree = xr.DataTree()

for sweep in radar.sweep_group_name.values:
    corrected_data = ref_filter(radar, sweep, field="DBZ")

    # Convert the xarray Dataset to a DataTree and add it to the result_tree
    data_tree = xr.DataTree.from_dict(corrected_data.to_dict())

    # Copy the contents of data_tree into result_tree
    for key, value in data_tree.items():
        result_tree[key] = value
```

```{code-cell}
radar.sweep_6.corr_DBZ.plot(x="x", y="y", cmap="ChaseSpectral", vmin=-10, vmax=70)
```

### Export

Export to CfRadial1

```{code-cell}
xd.io.to_cfradial1(dtree=radar, filename="cfradial1_qced.nc", calibs=True)
```

```{code-cell}
?xd.io.to_cfradial1
```

### Note

If `filename` is `None` in the `xd.io.to_cfradial1` function, it will automatically generate a<br>
filename using the instrument name and the first available timestamp from the data.
