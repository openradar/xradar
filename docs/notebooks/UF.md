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

# Universal Format (UF)

```{code-cell}
import cmweather  # noqa
import xarray as xr
from open_radar_data import DATASETS

import xradar as xd
```

## Download

Fetching Universal Format radar data file from [open-radar-data](https://github.com/openradar/open-radar-data) repository.

```{code-cell}
import atexit
from pathlib import Path
from tempfile import TemporaryDirectory

tmpdir_obj = TemporaryDirectory()
atexit.register(tmpdir_obj.cleanup)  # remove even if you forget
tmpdir = Path(tmpdir_obj.name)


def get_temp_file(fname):
    import gzip
    import shutil

    fnamei = Path(DATASETS.fetch(fname))
    fnameo = tmpdir / fnamei.stem
    with gzip.open(fnamei) as fin:
        with open(fnameo, "wb") as fout:
            shutil.copyfileobj(fin, fout)
            fout.flush()
            fout.close()
    return fnameo


fname = get_temp_file("20110427_164233_rvp8-rel_v001_SUR.uf.gz")
```

## xr.open_dataset

Making use of the xarray `uf` backend. We also need to provide the group. Note, that we are using CfRadial2 group access pattern.

```{code-cell}
ds = xr.open_dataset(fname, group="sweep_0", engine="uf")
display(ds)
```

```{code-cell}
import numpy as np

np.testing.assert_almost_equal(ds.sweep_fixed_angle.values, 0.703125)
```

### Plot Time vs. Azimuth

```{code-cell}
ds.azimuth.plot()
```

### Plot Range vs. Time

We need to sort by time and specify the y-coordinate.

```{code-cell}
ds.DBZH.sortby("time").plot(y="time", cmap="HomeyerRainbow")
```

### Plot Range vs. Azimuth

```{code-cell}
ds.DBZH.plot(cmap="HomeyerRainbow")
```

## backend_kwargs

Beside `first_dim` there are several additional backend_kwargs for the `uf` backend, which handle different aspects of angle alignment. This comes into play, when azimuth and/or elevation arrays are not evenly spacend and other issues.

```{code-cell}
help(xd.io.UFBackendEntrypoint)
```

```{code-cell}
ds = xr.open_dataset(fname, group="sweep_0", engine="uf", first_dim="time")
display(ds)
```

## open_uf_datatree

The same works analoguous with the datatree loader. But additionally we can provide a sweep string, number or list.

```{code-cell}
help(xd.io.open_uf_datatree)
```

```{code-cell}
dtree = xd.io.open_uf_datatree(fname, sweep=4)
display(dtree)
```

### Plot Sweep Range vs. Time

```{code-cell}
dtree["sweep_4"].ds.DBZH.sortby("time").plot(y="time", cmap="HomeyerRainbow")
```

### Plot Sweep Range vs. Azimuth

```{code-cell}
dtree["sweep_4"].ds.DBZH.plot(cmap="HomeyerRainbow")
```

```{code-cell}
dtree = xd.io.open_uf_datatree(fname, sweep="sweep_8")
display(dtree)
```

```{code-cell}
dtree = xd.io.open_uf_datatree(fname, sweep=[0, 1, 8])
display(dtree)
```

```{code-cell}
dtree["sweep_0"]["sweep_fixed_angle"].values
```

```{code-cell}
dtree["sweep_8"]["sweep_fixed_angle"].values
```

```{code-cell}
dtree = xd.io.open_uf_datatree(fname)
display(dtree)
```

```{code-cell}
dtree["sweep_1"]
```

## clean up

```{code-cell}
import time

for node in dtree.values():
    if hasattr(node, "close"):
        node.close()
for file in tmpdir.iterdir():
    if file.is_file():
        for _ in range(5):
            try:
                file.unlink()
                break
            except PermissionError:
                time.sleep(0.5)
```
