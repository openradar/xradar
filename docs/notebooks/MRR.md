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

# Metek MRR2

```{code-cell}
import cmweather  # noqa
import matplotlib.pyplot as plt
from open_radar_data import DATASETS

import xradar as xd
```

`xd.io.open_metek_datatree` supports the Metek MRR2 processed (.pro, .ave) and raw (.raw) files. The initialized datatree will contain all vertically pointing radar data in one sweep.

In this example, we are loading the 60 s average files from the MRR2 sampling a rain event over the Argonne Testbed for Multiscale Observational Science at Argonne National Laboratory in the Chicago suburbs.

```{code-cell}
mrr_test_file = DATASETS.fetch("0308.pro.gz")
import gzip
import shutil

decompressed_file = mrr_test_file[:-3]
with gzip.open(mrr_test_file, "rb") as f_in:
    with open(decompressed_file, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
with xd.io.open_metek_datatree(decompressed_file) as ds:
    display(ds)
```

View the structure of the loaded datatree.

```{code-cell}
ds["sweep_0"]
```

## Plot MRR timeseries

One can use the typical xarray plotting functions for plotting the velocity or other MRR2 variables.

```{code-cell}
plt.figure(figsize=(10, 3))
ds["sweep_0"]["velocity"].T.plot(cmap="balance", vmin=0, vmax=12)
```

## Plot MRR spectra

In order to plot the spectra, you first need to locate the index that corresponds to the given time period. This is done using xarray .sel() functionality to get the indicies.

```{code-cell}
indicies = ds["sweep_0"]["spectrum_index"].sel(
    time="2024-03-08T23:01:00", method="nearest"
)
indicies
ds["sweep_0"]["spectral_reflectivity"].isel(index=indicies).T.plot(
    cmap="ChaseSpectral", x="velocity_bins"
)
```

## Calculate rainfall accumulation estimated from Doppler velocity spectra

```{code-cell}
rainfall = ds["sweep_0"]["rainfall_rate"].isel(range=0).cumsum() / 60.0
rainfall.plot()
plt.ylabel("Cumulative rainfall [mm]")
```
