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

# Halo Photonics Doppler Lidar

```{code-cell}
import matplotlib.pyplot as plt
from open_radar_data import DATASETS

import xradar as xd
```

Opening a Halo Photonics Doppler lidar .hpl file.

We use `xd.open_datatree(file, engine="hpl")` to load the Halo Photonics Doppler lidar data. The .hpl file does not contain the latitude, longitude, or altitude of the lidar, so those need to be passed as `latitude=`, `longitude=`, and `altitude=` keyword arguments.

In this example, we are using the coordinates of the Doppler lidar at the Nantucket Wastewater Management Facility, deployed as as part of the DOE Energy Efficiency and Renewable Energy Office's [3rd Wind Forecast Improvement Project](https://www2.whoi.edu/site/wfip3/).

```{code-cell}
ds = xd.open_datatree(
    DATASETS.fetch("User1_184_20240601_013257.hpl"),
    engine="hpl",
    sweep=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    latitude=41.24276244459537,
    longitude=-70.1070364814594,
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
