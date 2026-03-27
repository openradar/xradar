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

# CfRadial1 <-> CfRadial2

+++

## Imports

```{code-cell}
import xarray as xr
from open_radar_data import DATASETS

import xradar as xd
```

## Load Data

```{code-cell}
file = DATASETS.fetch("cfrad.20080604_002217_000_SPOL_v36_SUR.nc")
dtree = xd.io.open_cfradial1_datatree(file)
display(dtree)
```

## Transform CF2 to CF1

```{code-cell}
ds = dtree.xradar.to_cfradial1_dataset()
display(ds)
```

## Transform CF1 to CF2

```{code-cell}
dtree = ds.xradar.to_cfradial2_datatree()
display(dtree)
```

```{code-cell}
del ds, dtree
```

## Alternate Method

+++

We can directly use xarray to read the data and then transform it to CF2 datatree.

```{code-cell}
ds = xr.open_dataset(file)
```

```{code-cell}
ds
```

```{code-cell}
radar = ds.xradar.to_cf2()
```

```{code-cell}
display(radar)
```

```{code-cell}

```
