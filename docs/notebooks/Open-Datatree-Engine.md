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

# `open_datatree` with `engine=`

xradar registers each of its readers as an `xarray.backends.BackendEntrypoint`,
so you can load any supported radar format into an `xarray.DataTree` directly
through the xarray-native API:

```python
import xarray as xr

dtree = xr.open_datatree(file, engine="<format>")
```

The same call is also exposed under `xradar` for convenience:

```python
import xradar as xd

dtree = xd.open_datatree(file, engine="<format>")
```

Both paths return a CfRadial2-shaped `xarray.DataTree` with a root dataset
and one `sweep_N` child per sweep. The xarray-native form is preferred in
most cases; the xradar-prefixed form is a thin shim that resolves the engine
through xradar's registry.

```{code-cell}
import atexit
import gzip
import shutil
import tempfile
import warnings
from pathlib import Path

import xarray as xr
from open_radar_data import DATASETS

import xradar as xd

# Some sample files in the open-radar-data repository ship gzipped but the
# corresponding backends expect a raw binary stream. Helper to decompress on
# demand into a tmpdir cleaned up at interpreter exit.
_tmpdir_obj = tempfile.TemporaryDirectory()
atexit.register(_tmpdir_obj.cleanup)
_tmpdir = Path(_tmpdir_obj.name)


def fetch_ungzipped(name):
    src = Path(DATASETS.fetch(name))
    dst = _tmpdir / src.stem
    with gzip.open(src) as fin, open(dst, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    return str(dst)
```

## Supported engines

The current registry covers thirteen radar formats:

```{code-cell}
xd.io.list_engines()
```

## ODIM_H5

```{code-cell}
odim_file = DATASETS.fetch("71_20181220_060628.pvol.h5")
dtree = xr.open_datatree(odim_file, engine="odim")
display(dtree)
```

## CfRadial1

```{code-cell}
cfradial1_file = DATASETS.fetch("cfrad.20080604_002217_000_SPOL_v36_SUR.nc")
dtree = xr.open_datatree(cfradial1_file, engine="cfradial1")
display(dtree)
```

## CfRadial2

CfRadial2 files are already group-native; the backend normalizes common
institutional variations onto the FM301 layout.

```{code-cell}
# Round-trip a CfRadial1 file to CfRadial2 so we have a demo input.
tmp_cfradial2 = _tmpdir / "demo_cfradial2.nc"
xd.io.to_cfradial2(
    xr.open_datatree(cfradial1_file, engine="cfradial1", first_dim="time").copy(),
    tmp_cfradial2,
    engine="netcdf4",
)

dtree = xr.open_datatree(str(tmp_cfradial2), engine="cfradial2")
display(dtree)
```

## NEXRAD Level II

```{code-cell}
nexrad_file = DATASETS.fetch("KATX20130717_195021_V06")
dtree = xr.open_datatree(nexrad_file, engine="nexradlevel2")
display(dtree)
```

## GAMIC

```{code-cell}
gamic_file = DATASETS.fetch("DWD-Vol-2_99999_20180601054047_00.h5")
dtree = xr.open_datatree(gamic_file, engine="gamic")
display(dtree)
```

## IRIS

```{code-cell}
iris_file = DATASETS.fetch("cor-main131125105503.RAW2049")
dtree = xr.open_datatree(iris_file, engine="iris")
display(dtree)
```

## Furuno

```{code-cell}
furuno_file = DATASETS.fetch("0080_20210730_160000_01_02.scn.gz")
dtree = xr.open_datatree(furuno_file, engine="furuno")
display(dtree)
```

## Rainbow

```{code-cell}
rainbow_file = DATASETS.fetch("2013051000000600dBZ.vol")
dtree = xr.open_datatree(rainbow_file, engine="rainbow")
display(dtree)
```

## DataMet

```{code-cell}
datamet_file = DATASETS.fetch("H-000-VOL-ILMONTE-201907100700.tar.gz")
dtree = xr.open_datatree(datamet_file, engine="datamet")
display(dtree)
```

## HPL (Halo Photonics)

```{code-cell}
hpl_file = DATASETS.fetch("User1_100_20240714_122137.hpl")
dtree = xr.open_datatree(hpl_file, engine="hpl")
display(dtree)
```

## Metek MRR

```{code-cell}
metek_file = fetch_ungzipped("0308.ave.gz")
dtree = xr.open_datatree(metek_file, engine="metek")
display(dtree)
```

## Universal Format (UF)

```{code-cell}
uf_file = fetch_ungzipped("20110427_164233_rvp8-rel_v001_SUR.uf.gz")
dtree = xr.open_datatree(uf_file, engine="uf")
display(dtree)
```

## IMD - single file via `engine="imd"`

IMD distributes one sweep per NetCDF file. The `engine="imd"` entry serves
the **single-file** case:

```{code-cell}
imd_file = DATASETS.fetch("IMD/JPR220822135253-IMD-B.nc")
dtree = xr.open_datatree(imd_file, engine="imd")
display(dtree)
```

## IMD - multi-file volume

To assemble a full IMD volume you supply a list of sweep files, see
[Files with one sweep each](#files-with-one-sweep-each). `xd.open_datatree` calls
`xd.io.open_imd_datatree` for the IMD engine, which delegates to
`xradar.util.create_volume`, and you can call it directly as well:

```{code-cell}
imd_volume = [
    DATASETS.fetch(f"IMD/JPR220822135253-IMD-B.nc{s}")
    for s in ["", ".1", ".2", ".3", ".4", ".5", ".6", ".7", ".8", ".9"]
]
dtree = xd.io.open_imd_datatree(imd_volume)
display(dtree)
```

## Common parameters

Every backend accepts a `sweep` selector (int, str, or list), `first_dim`
(`"auto"` or `"time"`), `optional`, and `optional_groups`. They behave
uniformly across all engines:

```{code-cell}
# Single sweep by index
dtree = xr.open_datatree(odim_file, engine="odim", sweep=0)
list(dtree.children)
```

```{code-cell}
# Multiple sweeps by index
dtree = xr.open_datatree(odim_file, engine="odim", sweep=[0, 2, 4])
list(dtree.children)
```

```{code-cell}
# Sweeps by name
dtree = xr.open_datatree(
    cfradial1_file, engine="cfradial1", sweep=["sweep_0", "sweep_3"]
)
list(dtree.children)
```

## `open_groups_as_dict` — work with the raw dict

If you want the pre-`DataTree` dict directly (useful for inspection or
custom assembly), instantiate the backend entrypoint and call
`open_groups_as_dict`:

```{code-cell}
groups = xd.io.OdimBackendEntrypoint().open_groups_as_dict(odim_file, sweep=[0, 1])
list(groups)
```

## Deprecated `open_*_datatree` shims

Most legacy `xd.io.open_<format>_datatree(...)` functions still work but
emit a `FutureWarning` directing users to the engine API. The one documented
exception is `xd.io.open_imd_datatree`, which remains the supported API for
multi-file IMD volumes (lists of per-sweep paths) and does **not** emit a
deprecation warning.

```{code-cell}
with warnings.catch_warnings(record=True) as captured:
    warnings.simplefilter("always")
    xd.io.open_odim_datatree(odim_file, sweep=[0])
[w.message for w in captured if issubclass(w.category, FutureWarning)]
```

## Without `engine=`

Every backend implements `guess_can_open` (format signature and the input types
it reads). `xd.open_datatree` uses these checks, via
{func}`xradar.io.discover_engine`, when `engine` is left out, so one call opens
every supported format:

```{code-cell}
for name, path in {
    "ODIM_H5": odim_file,
    "GAMIC": gamic_file,
    "CfRadial1": cfradial1_file,
    "NEXRAD Level II": nexrad_file,
    "IRIS/Sigmet": iris_file,
    "Rainbow": rainbow_file,
    "UF": uf_file,
    "IMD": imd_file,
}.items():
    dtree = xd.open_datatree(path)
    sweeps = [child for child in dtree.children if child.startswith("sweep_")]
    print(f"{name}: engine {xd.io.discover_engine(path)}, {len(sweeps)} sweeps")
    assert sweeps
```

xarray's own `xr.open_datatree(path)` without `engine` does the same only for
formats which xarray's backends do not read (NEXRAD Level II, IRIS, Rainbow,
UF, ...). netCDF and HDF5 based formats (ODIM_H5, GAMIC, CfRadial1, IMD) are
claimed by xarray's netCDF/HDF5 backends first, and opened as plain
netCDF/HDF5 groups, without the `sweep_N` layout and without the xradar
processing:

```{code-cell}
plain = xr.open_datatree(odim_file)
print("xr.open_datatree:", [child for child in plain.children][:4], "...")
assert not any(child.startswith("sweep_") for child in plain.children)
```

For these files use `xd.open_datatree(path)`, or pass the discovered engine to
xarray:

```{code-cell}
dtree = xr.open_datatree(odim_file, engine=xd.io.discover_engine(odim_file))
assert "sweep_0" in dtree.children
```

`discover_engine` raises a `ValueError` which lists the engines if none of them
reads the file.

## Files with one sweep each

Some systems write one sweep per file (IMD, IRIS/Sigmet files of single sweeps).
A list of files is stacked into one volume: the engine is discovered (all files
need the same one), the sweeps of all files are sorted by time and numbered in
this order. The same code works for every format:

```{code-cell}
imd_sweep_files = [
    DATASETS.fetch(f"IMD/JPR220822135253-IMD-B.nc{s}")
    for s in ["", ".1", ".2", ".3", ".4", ".5", ".6", ".7", ".8", ".9"]
]
volume = xd.open_datatree(imd_sweep_files)
display(volume)
assert len([child for child in volume.children if child.startswith("sweep_")]) == 10
```

The keyword arguments `time_coverage_start`, `time_coverage_end`, `min_angle`,
`max_angle` and `volume_number` of {func}`xradar.util.create_volume` select the
sweeps of the volume, e.g. `xd.open_datatree(files, max_angle=2.0)`. All other
arguments (`sweep`, `first_dim`, ...) are used for every file.

## Unknown engine

`xd.open_datatree` looks the engine up in xradar's registry and raises a
clear `ValueError` listing every supported name. (The xarray-native
`xr.open_datatree` uses xarray's own plugin discovery and raises a
different error from there.)

```{code-cell}
try:
    xd.open_datatree(odim_file, engine="nonexistent")
except ValueError as exc:
    print(exc)
```
