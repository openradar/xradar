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

# Iris/Sigmet - Virtual byte-range access

Instead of decoding a Sigmet/IRIS RAW volume into memory, {class}`~xradar.io.virtual.IrisParser`
(from the ``xradar[virtual]`` extra) indexes it into a VirtualiZarr
`ManifestStore` of **byte-range references**: one zarr chunk = one whole
sweep, every moment of a sweep pointing at the same byte span. The
decode-only `xradar-iris-sweep` zarr codec — shipped with xradar as a
`zarr.codecs` entry point — demultiplexes one moment out of the span when
you read. No data is copied; reading a persisted store needs only
`xradar` + `zarr`.

```{code-cell}
import cmweather  # noqa: F401  (registers the ChaseSpectral colormap)
import xarray as xr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from open_radar_data import DATASETS

from xradar.io.virtual import IrisParser
```

## Parse a RAW volume into virtual references

Fetching the same Corozal file used by the eager Iris notebook and parsing
it — the registry tells the parser (and later the codec) how to reach the
bytes; swap `LocalStore` for an S3 store to reference objects in a bucket
without downloading them.

```{code-cell}
filename = DATASETS.fetch("cor-main131125105503.RAW2049")
registry = ObjectStoreRegistry({"file://": LocalStore()})
store = IrisParser()(f"file://{filename}", registry)
```

The manifest behind each moment is just `{chunk_key: {path, offset,
length}}` — and every moment of a sweep shares the one span:

```{code-cell}
vtree = store.to_virtual_datatree()
dbzh = vtree["sweep_0"]["DBZH"].data
print(dbzh)
print(dbzh.manifest.dict())
print(vtree["sweep_0"]["VRADH"].data.manifest.dict() == dbzh.manifest.dict())
```

The codec configuration records the moment's slot in the ray-major
interleave:

```{code-cell}
dbzh.metadata.codecs
```

## Read it back

Opening the store decodes on demand through the codec — the result matches
{func}`~xradar.io.backends.iris.open_iris_datatree`:

```{code-cell}
dtree = xr.open_datatree(
    store, engine="zarr", consolidated=False, zarr_format=3
)
dtree["sweep_0"]
```

```{code-cell}
dtree["sweep_0"]["DBZH"].plot(x="azimuth", cmap="ChaseSpectral", vmin=-10, vmax=60)
```

## Notes

- The parser stores pure pointers: rays keep the file's order. A store
  builder that wants azimuth-sorted sweeps writes sorted coordinates and
  sets ``sort_rays: true`` in the codec configuration, which the codec
  honors at read time.
- ``pad_missing_rays=True`` keeps a slot for mechanically dropped rays so
  volumes of one task stay stackable on a fixed azimuth dimension.
- Persisting the references (e.g. with
  [icechunk](https://icechunk.io)) yields a store any xradar user can open —
  the codec resolves by name via the entry point, with no explicit import.
