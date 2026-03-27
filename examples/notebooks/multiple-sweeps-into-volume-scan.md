---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
---

# AWS Volumes to ARCO
This example shows how to create a volume scan from multiple sweep files stored on AWS. The volume scan structure is based on [tree-like](https://xarray-datatree.readthedocs.io/en/latest/generated/datatree.DataTree.html) hierarchical collection of xarray objects

+++

## Imports

```{code-cell}
import warnings
from datetime import datetime

import cartopy.crs as ccrs
import cmweather  # noqa
import fsspec
import matplotlib.pyplot as plt
import xarray as xr
from IPython.display import HTML
from matplotlib.animation import FuncAnimation

import xradar as xd

warnings.simplefilter("ignore")
```

## Access radar data from the Colombian radar network on AWS
Access data from the IDEAM bucket on AWS. Detailed information can be found [here](https://openradar-docs--102.org.readthedocs.build/projects/xradar/en/102/notebooks/Read-plot-Sigmet-data-from-AWS.html)

```{code-cell}
def create_query(date, radar_site):
    """
    Creates a string for quering the IDEAM radar files stored in AWS bucket
    :param date: date to be queried. e.g datetime(2021, 10, 3, 12). Datetime python object
    :param radar_site: radar site e.g. Guaviare
    :return: string with a IDEAM radar bucket format
    """
    if (date.hour != 0) and (date.hour != 0):
        return f"l2_data/{date:%Y}/{date:%m}/{date:%d}/{radar_site}/{radar_site[:3].upper()}{date:%y%m%d%H}"
    elif (date.hour != 0) and (date.hour == 0):
        return f"l2_data/{date:%Y}/{date:%m}/{date:%d}/{radar_site}/{radar_site[:3].upper()}{date:%y%m%d}"
    else:
        return f"l2_data/{date:%Y}/{date:%m}/{date:%d}/{radar_site}/{radar_site[:3].upper()}{date:%y%m%d}"
```

```{code-cell}
date_query = datetime(2023, 4, 7, 3)
radar_name = "Barrancabermeja"
query = create_query(date=date_query, radar_site=radar_name)
str_bucket = "s3://s3-radaresideam/"
fs = fsspec.filesystem("s3", anon=True)
```

```{code-cell}
query
```

```{code-cell}
radar_files = sorted(fs.glob(f"{str_bucket}{query}*"))
radar_files[:4]
```

## Let's check the elevation at each file using `xradar.datatree` module

IDEAM radar network operates with a volume scan every five minutes. Each volume scan has four different tasks
* *SURVP* "super-resolution" sweep at the lowest elevation angle, usually 0.5 deg,  with 720 degrees in azimuth (every 0.5 deg)
* *PRECA* task with 1.5, 2.4, 3.0, and 5.0 elevation angles and shorter range than *SURVP*
* *PRECB* task with 6.4 and 8.0 elevation angles and a shorter range than the previous task
* *PRECC* task with 10.0, 12.5, and 15.0 with a shorter range than all the previous tasks.

```{code-cell}
# List of first four task files
task_files = [
    fsspec.open_local(
        f"simplecache::s3://{i}", s3={"anon": True}, filecache={"cache_storage": "."}
    )
    for i in radar_files[:4]
]
# list of xradar datatrees
ls_dt = [xd.io.open_iris_datatree(i).xradar.georeference() for i in task_files]

# sweeps and elevations within each task
for i in ls_dt:
    sweeps = list(i.children.keys())
    print(f"task sweeps: {sweeps}")
    for j in sweeps:
        if j.startswith("sweep"):
            print(
                f"{j}: {i[j].sweep_fixed_angle.values: .1f} [deg], {i[j].range.values[-1] / 1e3:.1f} [km]"
            )
    print("----------------------------------------------------------------")
```

## Create a single-volume scan
Let's use the first four files, tasks *SURVP*, *PRECA*, *PRECB*, *PRECC*, to create a single volume scan using each task as a datatree. The new volume scan is a tree-like hierarchical object with all four tasks as children.

```{code-cell}
vcp_dt = xr.DataTree(
    name="root",
    children=dict(SURVP=ls_dt[0], PRECA=ls_dt[1], PRECB=ls_dt[2], PRECC=ls_dt[3]),
)
```

```{code-cell}
vcp_dt.groups
```

```{code-cell}
print(f"Size of data in tree = {vcp_dt.nbytes / 1e6 :.2f} MB")
```

## PPI plot from the Datatree object

Now that we have a tree-like hierarchical volume scan object. We can access data at each scan/sweep using dot method `vcp_dt.SURVP` or dictionary-key method `vcp_dt['PRECB']`

```{code-cell}
fig, (ax, ax1) = plt.subplots(1, 2, figsize=(10, 4))
# dot method
vcp_dt.SURVP.sweep_0.DBZH.plot(
    x="x", y="y", cmap="ChaseSpectral", vmin=-10, vmax=50, ax=ax
)

ax.set_title(
    f"SURPV sweep_0 ({vcp_dt.SURVP.sweep_0.sweep_fixed_angle.values: .1f} [deg])"
)
m2km = lambda x, _: f"{x/1000:g}"
ax.xaxis.set_major_formatter(m2km)
ax.yaxis.set_major_formatter(m2km)
ax.set_ylabel(r"$North - South \ distance \ [km]$")
ax.set_xlabel(r"$East - West \ distance \ [km]$")

# Dictionary-key method
vcp_dt["PRECB"]["sweep_0"].DBZH.plot(
    x="x", y="y", cmap="ChaseSpectral", vmin=-10, vmax=50, ax=ax1
)

ax1.set_title(
    f"PRECB sweep_0 ({vcp_dt.PRECB.sweep_0.sweep_fixed_angle.values: .1f} [deg])"
)
m2km = lambda x, _: f"{x/1000:g}"
ax1.xaxis.set_major_formatter(m2km)
ax1.yaxis.set_major_formatter(m2km)
ax1.set_xlim(ax.get_xlim())
ax1.set_ylim(ax.get_ylim())
ax1.set_ylabel(r"$North - South \ distance \ [km]$")
ax1.set_xlabel(r"$East - West \ distance \ [km]$")
fig.tight_layout()
```

## Multiple volumes scan into one `datatree` object

Similarly, we can create a tree-like hierarchical object with multiple volume scans.

```{code-cell}
def data_accessor(file):
    """
    Open AWS S3 file(s), which can be resolved locally by file caching
    """
    return fsspec.open_local(
        f"simplecache::s3://{file}",
        s3={"anon": True},
        filecache={"cache_storage": "./tmp/"},
    )


def create_vcp(ls_dt):
    """
    Creates a tree-like object for each volume scan
    """
    return xr.DataTree(
        name="root",
        children=dict(SURVP=ls_dt[0], PRECA=ls_dt[1], PRECB=ls_dt[2], PRECC=ls_dt[3]),
    )


def mult_vcp(radar_files):
    """
    Creates a tree-like object for multiple volumes scan every 4th file in the bucket
    """
    ls_files = [radar_files[i : i + 4] for i in range(len(radar_files)) if i % 4 == 0]
    ls_sigmet = [
        [xd.io.open_iris_datatree(data_accessor(i)).xradar.georeference() for i in j]
        for j in ls_files
    ]
    ls_dt = [create_vcp(i) for i in ls_sigmet]
    return xr.DataTree.from_dict({f"vcp_{idx}": i for idx, i in enumerate(ls_dt)})
```

```{code-cell}
# let's test it using the first 24 files in the bucket. We can include more files for visualization. e.g. radar_files[:96]
vcps_dt = mult_vcp(radar_files[:24])
```

Now we have 6 vcps in one tree-like hierarchical object.

```{code-cell}
list(vcps_dt.keys())
```

```{code-cell}
print(f"Size of data in tree = {vcps_dt.nbytes / 1e9 :.2f} GB")
```

### PPI animation using the lowest elevation angle

We can create an animation using the `FuncAnimation` module from `matplotlib` package

```{code-cell}
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
proj_crs = xd.georeference.get_crs(vcps_dt.vcp_1.SURVP)
cart_crs = ccrs.Projection(proj_crs)
sc = vcps_dt.vcp_1.SURVP.sweep_0.DBZH.plot.pcolormesh(
    x="x",
    y="y",
    vmin=-10,
    vmax=50,
    cmap="ChaseSpectral",
    edgecolors="face",
    transform=cart_crs,
    ax=ax,
)

title = f"SURVP - {vcps_dt.vcp_1.SURVP.sweep_0.sweep_fixed_angle.values: .1f} [deg]"
ax.set_title(title)
gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=1,
    color="gray",
    alpha=0.3,
    linestyle="--",
)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
gl.top_labels = False
gl.right_labels = False
ax.coastlines()


def update_plot(vcp):
    sc.set_array(vcps_dt[vcp].SURVP.sweep_0.DBZH.values.ravel())


ani = FuncAnimation(fig, update_plot, frames=list(vcps_dt.keys()), interval=150)
plt.close()
HTML(ani.to_html5_video())
```

## Bonus!!
### Analysis-ready data, cloud-optimized (ARCO) format

Tree-like hierarchical data can be stored using ARCO format.

```{code-cell}
zarr_store = "./multiple_vcp_test.zarr"
_ = vcps_dt.to_zarr(zarr_store)
```

ARCO format can be read by using `open_datatree` module

```{code-cell}
vcps_back = xr.open_datatree(zarr_store, engine="zarr")
```

```{code-cell}
display(vcps_back)
```

```{code-cell}

```
