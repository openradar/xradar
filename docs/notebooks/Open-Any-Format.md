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

# Open any format without `engine`

`xd.open_datatree(file)` finds the reader itself: every xradar backend checks the file with its `guess_can_open` (format signature, input type) and {func}`xradar.io.discover_engine` returns the engine which can read it. This notebook opens the radar and lidar files of [open-radar-data](https://github.com/openradar/open-radar-data) of all supported formats with the same call.

```{code-cell}
import atexit
import gzip
import shutil
import tempfile
from pathlib import Path

import cmweather  # noqa
import matplotlib.pyplot as plt
import pandas as pd
from open_radar_data import DATASETS

import xradar as xd
```

## Download

Some files are gzipped, and the backends read them uncompressed.

```{code-cell}
_tmpdir_obj = tempfile.TemporaryDirectory()
atexit.register(_tmpdir_obj.cleanup)
tmpdir = Path(_tmpdir_obj.name)


def fetch_ungzipped(name):
    src = Path(DATASETS.fetch(name))
    dst = tmpdir / src.stem
    with gzip.open(src) as fin, open(dst, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    return dst


files = {
    "ODIM_H5": DATASETS.fetch("71_20181220_060628.pvol.h5"),
    "GAMIC": DATASETS.fetch("DWD-Vol-2_99999_20180601054047_00.h5"),
    "CfRadial1": DATASETS.fetch("cfrad.20080604_002217_000_SPOL_v36_SUR.nc"),
    "NEXRAD Level II": DATASETS.fetch("KATX20130717_195021_V06"),
    "IRIS/Sigmet": DATASETS.fetch("cor-main131125105503.RAW2049"),
    "Furuno": DATASETS.fetch("0080_20210730_160000_01_02.scn.gz"),
    "Rainbow": DATASETS.fetch("2013051000000600dBZ.vol"),
    "DataMet": DATASETS.fetch("H-000-VOL-ILMONTE-201907100700.tar.gz"),
    "UF": fetch_ungzipped("20110427_164233_rvp8-rel_v001_SUR.uf.gz"),
    "IMD": DATASETS.fetch("IMD/JPR220822135253-IMD-B.nc"),
    "Halo Photonics lidar": DATASETS.fetch("User1_100_20240714_122137.hpl"),
    "Metek MRR2": fetch_ungzipped("0308.ave.gz"),
}
```

## Open all files with the same call

```{code-cell}
trees = {}
rows = []
for name, path in files.items():
    trees[name] = dtree = xd.open_datatree(path)
    sweeps = [child for child in dtree.children if child.startswith("sweep_")]
    rows.append(
        {
            "format": name,
            "file": Path(path).name,
            "engine": xd.io.discover_engine(path),
            "sweeps": len(sweeps),
            "variables of sweep_0": ", ".join(dtree["sweep_0"].ds.data_vars),
        }
    )
pd.DataFrame(rows).set_index("format")
```

Every result has the same structure: a root with the station and the sweep list, and one `sweep_N` child per sweep.

```{code-cell}
for name, dtree in trees.items():
    assert any(child.startswith("sweep_") for child in dtree.children), name
```

```{code-cell}
trees["IRIS/Sigmet"]
```

## The lowest sweep of each radar volume

Because of the common structure the same code works for every format. The sweeps are georeferenced with the `xradar` accessor of the tree.

```{code-cell}
radars = [
    "ODIM_H5",
    "GAMIC",
    "CfRadial1",
    "NEXRAD Level II",
    "IRIS/Sigmet",
    "Furuno",
    "Rainbow",
    "DataMet",
    "UF",
    "IMD",
]
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for ax, name in zip(axes.flat, radars, strict=True):
    sweep = trees[name].xradar.georeference()["sweep_0"].ds
    moment = next(m for m in ("DBZH", "DBZ", "DBTH") if m in sweep)
    # mask the no-echo values (below 0 dBZ) for the display
    sweep[moment].where(sweep[moment] >= 0).plot(
        x="x",
        y="y",
        ax=ax,
        cmap="HomeyerRainbow",
        vmin=0,
        vmax=60,
        add_colorbar=False,
    )
    ax.set_title(f"{name}\n{moment}")
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
fig.tight_layout()
```

## Lidar and profiler

The Halo Photonics lidar and the Metek MRR2 are opened the same way.

```{code-cell}
trees["Halo Photonics lidar"]["sweep_0"]
```

```{code-cell}
trees["Metek MRR2"]["sweep_0"]
```

## Files with one sweep each

IMD writes one sweep per file. A list of files is stacked into one volume (sorted by time, see {func}`xradar.util.create_volume`), again without `engine`:

```{code-cell}
imd_sweep_files = [
    DATASETS.fetch(f"IMD/JPR220822135253-IMD-B.nc{s}")
    for s in ["", ".1", ".2", ".3", ".4", ".5", ".6", ".7", ".8", ".9"]
]
volume = xd.open_datatree(imd_sweep_files)
print(volume.ds.sweep_fixed_angle.values)
assert len([c for c in volume.children if c.startswith("sweep_")]) == 10
```

## What if no engine matches?

```{code-cell}
unknown = tmpdir / "notes.txt"
unknown.write_text("not a radar file\n")
try:
    xd.open_datatree(unknown)
except ValueError as err:
    print(err)
```

## `xr.open_datatree` without `engine`

xarray's own `xr.open_datatree(file)` chooses the xradar engine for formats which no xarray backend reads (NEXRAD Level II, IRIS, Rainbow, UF, ...). The netCDF and HDF5 based formats (ODIM_H5, GAMIC, CfRadial1, IMD) are claimed by xarray's netCDF/HDF5 backends first. Use `xd.open_datatree` for all formats, see [Open-Datatree-Engine](Open-Datatree-Engine.md).

## clean up

```{code-cell}
for dtree in [*trees.values(), volume]:
    dtree.close()
```
