---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
---

# Accessors

+++

To extend `xarray.DataArray` and  `xarray.Dataset`
xradar aims to provide accessors which downstream libraries can hook into.

Those accessors are yet to be defined. For starters we could implement purpose-based
accessors (like `.vis`, `.kdp` or `.trafo`) on `xarray.DataArray` level.

To not have to import downstream packages a similar approach to xarray.backends using
`importlib.metadata.entry_points` could be facilitated.

In this notebook the creation of such an accessor is showcased.

```{code-cell}
import numpy as np
import xarray as xr
from open_radar_data import DATASETS

import xradar as xd
```

## Import Data

Fetch data from [open-radar-data](https://github.com/openradar/open-radar-data) repository.

```{code-cell}
filename = DATASETS.fetch("71_20181220_060628.pvol.h5")
```

### Open data

```{code-cell}
ds = xr.open_dataset(filename, group="sweep_0", engine="odim")
display(ds.DBZH.values)
```

### Plot DBZH

```{code-cell}
ds.DBZH.plot()
```

## Define two example functions

Functions copied verbatim from wradlib.

```{code-cell}
def _decibel(x):
    """Calculates the decibel representation of the input values

    :math:`dBZ=10 \\cdot \\log_{10} z`

    Parameters
    ----------
    x : float or :class:`numpy:numpy.ndarray`
        (must not be <= 0.)

    Examples
    --------
    >>> from wradlib.trafo import decibel
    >>> print(decibel(100.))
    20.0
    """
    return 10.0 * np.log10(x)


def _idecibel(x):
    """Calculates the inverse of input decibel values

    :math:`z=10^{x \\over 10}`

    Parameters
    ----------
    x : float or :class:`numpy:numpy.ndarray`

    Examples
    --------
    >>> from wradlib.trafo import idecibel
    >>> print(idecibel(10.))
    10.0

    """
    return 10.0 ** (x / 10.0)
```

## Function dictionaries

To show the import of the functions, we put them in different dictionaries as we would get them via `entry_points`.

This is what the downstream libraries would have to provide.

```{code-cell}
package_1_func = {"trafo": {"decibel": _decibel}}
package_2_func = {"trafo": {"idecibel": _idecibel}}
```

## xradar internal functionality

This is how xradar would need to treat that input data.

```{code-cell}
downstream_functions = [package_1_func, package_2_func]
xradar_accessors = ["trafo"]
```

```{code-cell}
package_functions = {}
for accessor in xradar_accessors:
    package_functions[accessor] = {}
    for dfuncs in downstream_functions:
        package_functions[accessor].update(dfuncs[accessor])
print(package_functions)
```

## Create and register accessor

We bundle the different steps into one function, ``create_xradar_dataarray_accessor``.

```{code-cell}
for accessor in xradar_accessors:
    xd.accessors.create_xradar_dataarray_accessor(accessor, package_functions[accessor])
```

## Convert DBZH to linear and plot

```{code-cell}
z = ds.DBZH.trafo.idecibel()
z.plot()
```

## Convert z to decibel and plot()

```{code-cell}
dbz = z.trafo.decibel()
display(dbz)
```

```{code-cell}
dbz.plot()
```
