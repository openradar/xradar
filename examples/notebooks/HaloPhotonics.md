---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
---

## Halo Photonics Doppler Lidar

```{code-cell}
import matplotlib.pyplot as plt
from open_radar_data import DATASETS

import xradar as xd
```

Opening a Halo Photonics Doppler lidar .hpl file.

We use the `xd.io.open_hpl_datatree` in order to load the Halo Photonics Doppler lidar data. After that we will need to enter in the latitude and longitude in order to properly georeference the data. The .hpl file does not contain the latitude, longitude, or altitude of the lidar, so these need to be entered in as keywords as a part of the  `backend_kwargs` argument to `xd.io.open_hpl_datatree`.

In this example, we are using the coordinates of the Doppler lidar at the Nantucket Wastewater Management Facility, deployed as as part of the DOE Energy Effciency and Renewable Energy Office's [3rd Wind Forecast Improvement Project](https://www2.whoi.edu/site/wfip3/).

```{code-cell}
ds = xd.io.open_hpl_datatree(
    DATASETS.fetch("User1_184_20240601_013257.hpl"),
    sweep=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    backend_kwargs=dict(latitude=41.24276244459537, longitude=-70.1070364814594),
)
```

```{code-cell}
ds["sweep_2"]["mean_doppler_velocity"].plot(vmin=-20, vmax=0, cmap="Spectral")
```

In order to plot each sweep, we need to georeference the underlying sweeps.

```{code-cell}
fig, ax = plt.subplots(3, 3, figsize=(12, 10))
for sweep in range(9):
    sweep_ds = xd.georeference.get_x_y_z(
        ds[f"sweep_{sweep}"].to_dataset(inherit="all_coords")
    )
    sweep_ds = sweep_ds.set_coords(["x", "y", "z", "time", "range"])
    sweep_ds["mean_doppler_velocity"].plot(
        x="x", y="y", ax=ax[int(sweep / 3), sweep % 3]
    )
    ax[int(sweep / 3), sweep % 3].set_title(
        "{angle:2.1f} degree scan".format(angle=sweep_ds["sweep_fixed_angle"].values)
    )
    ax[int(sweep / 3), sweep % 3].set_ylim([-4000, 0])
    ax[int(sweep / 3), sweep % 3].set_xlim([-4000, 1000])
fig.tight_layout()
```
