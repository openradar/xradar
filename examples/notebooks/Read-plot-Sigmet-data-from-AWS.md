---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
---

# Work with AWS
This example shows how to access radar data from the Colombian national radar network public on Amazon Web Services. We will look at the bucket structure and plot a PPI using the Xradar library. Radar reflectivity is filtered using some polarimetric values and xarray functionality.

+++

## Imports

```{code-cell}
from datetime import datetime

import boto3
import botocore
import cmweather  # noqa
import fsspec
import matplotlib.pyplot as plt
import xarray as xr
from botocore.client import Config
from pandas import to_datetime

import xradar as xd
```

## IDEAM AWS Bucket
Instituto de Hidrología, Meteorología y Estudios Ambientales - IDEAM (Colombian National Weather Service) has made public the weather radar data. Data can be found [here](https://registry.opendata.aws/ideam-radares/), and documentation [here](http://www.pronosticosyalertas.gov.co/archivos-radar#:~:text=RED%20DE%20RADARES%20DE%20IDEAM%20EN%20AWS).

The bucket structure is s3://s3-radaresideam/l2_data/YYYY/MM/DD/Radar_name/RRRAAMMDDTTTTTT.RAWXXXX where:
* YYYY is the 4-digit year
* MM is the 2-digit month
* DD is the 2-digit day
* Radar_name radar name. Options are Guaviare, Munchique, Barrancabermja, and Carimagua
* RRRAAMMDDTTTTTT.RAWXXXX is the radar filename with the following:
    - RRR three first letters of the radar name (e.g., GUA for Guaviare radar)
    - YY is the 2-digit year
    - MM is the 2-digit month
    - DD is the 2-digit day
    - TTTTTT is the time at which the scan was made (GTM)
    - RAWXXXX Sigmet file format and unique code provided by IRIS software

This is too complicated! No worries. We created a function to help you list files within the bucket.

```{code-cell}
def create_query(date, radar_site):
    """
    Creates a string for quering the IDEAM radar files stored in AWS bucket
    :param date: date to be queried. e.g datetime(2021, 10, 3, 12). Datetime python object
    :param radar_site: radar site e.g. Guaviare
    :return: string with a IDEAM radar bucket format
    """
    if (date.hour != 0) and (date.minute != 0):
        return f"l2_data/{date:%Y}/{date:%m}/{date:%d}/{radar_site}/{radar_site[:3].upper()}{date:%y%m%d%H%M}"
    elif (date.hour != 0) and (date.minute == 0):
        return f"l2_data/{date:%Y}/{date:%m}/{date:%d}/{radar_site}/{radar_site[:3].upper()}{date:%y%m%d%H}"
    else:
        return f"l2_data/{date:%Y}/{date:%m}/{date:%d}/{radar_site}/{radar_site[:3].upper()}{date:%y%m%d}"
```

Let's suppose we want to check the radar files on **2022-10-6** from the Guaviare radar

```{code-cell}
date_query = datetime(2022, 10, 6)
radar_name = "Guaviare"
query = create_query(date=date_query, radar_site=radar_name)
query
```

### Connecting to the AWS bucket
Once the query is defined, we can procced to list all the available files in the bucket using **boto3** and **botocore** libraries

```{code-cell}
str_bucket = "s3://s3-radaresideam/"
s3 = boto3.resource(
    "s3",
    config=Config(signature_version=botocore.UNSIGNED, user_agent_extra="Resource"),
)

bucket = s3.Bucket("s3-radaresideam")

radar_files = [f"{str_bucket}{i.key}" for i in bucket.objects.filter(Prefix=f"{query}")]
radar_files[:5]
```

We can use the Filesystem interfaces for Python [fsspec](https://filesystem-spec.readthedocs.io/en/latest/) to access the data from the s3 bucket

```{code-cell}
file = fsspec.open_local(
    f"simplecache::{radar_files[0]}",
    s3={"anon": True},
    filecache={"cache_storage": "."},
)
```

```{code-cell}
ds = xr.open_dataset(file, engine="iris", group="sweep_0")
```

```{code-cell}
display(ds)
```

## Reflectivity and Correlation coefficient plot

```{code-cell}
fig, (ax, ax1) = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
ds.DBZH.plot(cmap="ChaseSpectral", vmin=-10, vmax=50, ax=ax)
ds.RHOHV.plot(cmap="ChaseSpectral", vmin=0, vmax=1, ax=ax1)
fig.tight_layout()
```

The dataset object has range and azimuth as coordinates. To create a polar plot, we need to add the georeference information using  `xd.georeference.get_x_y_z()` module from [xradar](https://docs.openradarscience.org/projects/xradar/en/stable/index.html)

```{code-cell}
ds = xd.georeference.get_x_y_z(ds)
display(ds)
```

Now x, y, and z have been added to the dataset coordinates. Let's create the new plot using the georeference information.

```{code-cell}
fig, (ax, ax1) = plt.subplots(1, 2, figsize=(10, 4))
ds.DBZH.plot(x="x", y="y", cmap="ChaseSpectral", vmin=-10, vmax=50, ax=ax)
ds.RHOHV.plot(x="x", y="y", cmap="ChaseSpectral", vmin=0, vmax=1, ax=ax1)
ax.set_title("")
ax1.set_title("")
fig.tight_layout()
```

## Filtering data

The blue background color indicates that the radar reflectivity is less than -10 dBZ. we can filter radar data using [xarray.where](https://docs.xarray.dev/en/stable/generated/xarray.where.html) module

```{code-cell}
fig, (ax, ax1) = plt.subplots(1, 2, figsize=(10, 4))
ds.DBZH.where(ds.DBZH >= -10).plot(
    x="x", y="y", cmap="ChaseSpectral", vmin=-10, vmax=50, ax=ax
)
ds.RHOHV.where(ds.DBZH >= -10).plot(
    x="x", y="y", cmap="ChaseSpectral", vmin=0, vmax=1, ax=ax1
)
ax.set_title("")
ax1.set_title("")
fig.tight_layout()
```

Polarimetric variables can also be used as indicators to remove different noises from different sources. For example, the $\rho_{HV}$ measures the consistency of the shapes and sizes of targets within the radar beam. Thus, the greater the $\rho_{HV}$, the more consistent the measurement. For this example we can use $\rho_{HV} > 0.80$ as an acceptable threshold

```{code-cell}
fig, (ax, ax1) = plt.subplots(1, 2, figsize=(10, 4))
ds.DBZH.where(ds.DBZH >= -10).where(ds.RHOHV >= 0.80).plot(
    x="x", y="y", cmap="ChaseSpectral", vmin=-10, vmax=50, ax=ax
)

ds.DBZH.where(ds.DBZH >= -10).where(ds.RHOHV >= 0.85).plot(
    x="x", y="y", cmap="ChaseSpectral", vmin=0, vmax=1, ax=ax1
)
ax.set_title("")
ax1.set_title("")
fig.tight_layout()
```

##  Axis labels and titles
We can change some axis labels as well as the colorbar label

```{code-cell}
fig, (ax, ax1) = plt.subplots(1, 2, figsize=(10, 4))
ds.DBZH.where(ds.DBZH >= -10).where(ds.RHOHV >= 0.85).plot(
    x="x",
    y="y",
    cmap="ChaseSpectral",
    vmin=-10,
    vmax=50,
    ax=ax,
    cbar_kwargs={"label": r"$Reflectivity \ [dBZ]$"},
)

ds.RHOHV.where(ds.DBZH >= -10).where(ds.RHOHV >= 0.85).plot(
    x="x",
    y="y",
    cmap="ChaseSpectral",
    vmin=0,
    vmax=1,
    ax=ax1,
    cbar_kwargs={"label": r"$Corr. \ Coef. \  [unitless]$"},
)

# lambda fucntion for unit trasformation m->km
m2km = lambda x, _: f"{x/1000:g}"
# set new ticks
ax.xaxis.set_major_formatter(m2km)
ax.yaxis.set_major_formatter(m2km)
ax1.xaxis.set_major_formatter(m2km)
ax1.yaxis.set_major_formatter(m2km)
# removing the title in both plots
ax.set_title("")
ax1.set_title("")

# renaming the axis
ax.set_ylabel(r"$North - South \ distance \ [km]$")
ax.set_xlabel(r"$East - West \ distance \ [km]$")
ax1.set_ylabel(r"$North - South \ distance \ [km]$")
ax1.set_xlabel(r"$East - West \ distance \ [km]$")

# setting up the title
ax.set_title(
    r"$Guaviare \ radar$"
    + "\n"
    + f"${to_datetime(ds.time.values[0]): %Y-%m-%d - %X}$"
    + "$ UTC$"
)
ax1.set_title(
    r"$Guaviare \ radar$"
    + "\n"
    + f"${to_datetime(ds.time.values[0]): %Y-%m-%d - %X}$"
    + "$ UTC$"
)
fig.tight_layout()
```
