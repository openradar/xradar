---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
---

# Streaming NEXRAD Level 2 Chunks from S3

xradar can now ingest a **list of NEXRAD Level 2 chunk byte objects** directly,
so you can stream real-time radar data from S3 without downloading full volume
files first. This notebook demonstrates:

1. Listing and downloading chunk files from the `unidata-nexrad-level2-chunks` bucket
2. Opening a full volume assembled from all chunks
3. Handling partial volumes with `incomplete_sweep="drop"` (default)
4. Handling partial volumes with `incomplete_sweep="pad"`
5. Early streaming with just a few chunks

```{code-cell}
import warnings

import cmweather  # noqa: F401 -- registers colormaps
import fsspec
import matplotlib.pyplot as plt
import numpy as np

import xradar as xd
```

## Background: NEXRAD chunk files on S3

NOAA publishes NEXRAD Level 2 data to two public S3 buckets:

| Bucket | Content | Latency |
|---|---|---|
| `noaa-nexrad-level2` | Complete volume files | Minutes after scan |
| `unidata-nexrad-level2-chunks` | Real-time chunk files | Seconds after scan |

Each radar volume is split into many small **chunk files** that arrive as the
radar scans. A volume directory typically contains:

- One **S** (start) chunk that includes the volume header
- Many **I** (intermediate) chunks with sweep data
- One **E** (end) chunk marking the volume boundary

For example:
```
KABR/903/KABR20250717_120038_V06_S  (start)
KABR/903/KABR20250717_120038_V06_I02  (intermediate)
KABR/903/KABR20250717_120038_V06_I03
...
KABR/903/KABR20250717_120038_V06_E   (end)
```

xradar accepts a **list** of chunk bytes (or file paths, or file-like objects)
directly via `open_nexradlevel2_datatree()`. The chunks are concatenated
internally, so you never need to assemble them manually.

+++

## Load chunk data

```{code-cell}
chunk_paths = []

try:
    fs = fsspec.filesystem("s3", anon=True)
    volumes = sorted(fs.ls("unidata-nexrad-level2-chunks/KABR/"))
    if volumes:
        chunk_paths = sorted(fs.ls(volumes[-1]))
except Exception:
    pass

if chunk_paths:
    print(f"Using live S3 chunks: {len(chunk_paths)} files")
    for p in chunk_paths[:3]:
        print(f"  {p.split('/')[-1]}")
    print(f"  ... {chunk_paths[-1].split('/')[-1]}")
else:
    print("S3 bucket empty or unreachable, using open-radar-data fixture")
```

## Download / load chunk bytes

```{code-cell}
if chunk_paths:
    all_bytes = [fs.open(p, "rb").read() for p in chunk_paths]
else:
    import tarfile
    from pathlib import Path

    from open_radar_data import DATASETS

    archive = DATASETS.fetch("nexrad_level2_chunks_KLOT.tar.gz")
    with tarfile.open(archive) as tar:
        tar.extractall("/tmp/nexrad_chunks", filter="data")
    chunk_files = sorted(Path("/tmp/nexrad_chunks/nexrad_chunks_KLOT").iterdir())
    all_bytes = [f.read_bytes() for f in chunk_files]

total_mb = sum(len(b) for b in all_bytes) / 1e6
print(f"Loaded {len(all_bytes)} chunks ({total_mb:.1f} MB total)")
```

## Full volume from all chunks

When all chunks (S through E) are available, passing the list to
`open_nexradlevel2_datatree` produces the same result as opening a
complete volume file.

```{code-cell}
dtree = xd.io.open_nexradlevel2_datatree(all_bytes)
display(dtree)
```

```{code-cell}
ds = xd.georeference.get_x_y_z(dtree["sweep_0"].to_dataset(inherit="all_coords"))

fig, ax = plt.subplots(figsize=(6, 5))
ds.DBZH.plot(x="x", y="y", cmap="ChaseSpectral", vmin=-10, vmax=70, ax=ax)
ax.set_title(f"Full volume - sweep_0 ({ds.sweep_fixed_angle.values:.1f} deg)")
ax.set_aspect("equal")
fig.tight_layout()
```

## Partial volume -- drop mode (default)

When only some chunks have arrived, the last sweep is usually **incomplete**
(fewer rays than a full 360-degree rotation). By default,
`incomplete_sweep="drop"` excludes these partial sweeps and emits a warning.

This is the safest option for downstream processing that expects complete
sweeps.

```{code-cell}
partial_chunks = all_bytes[:15]

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    dtree_drop = xd.io.open_nexradlevel2_datatree(
        partial_chunks, incomplete_sweep="drop"
    )

# Show warnings
for warning in w:
    print(f"WARNING: {warning.message}")

sweep_groups = list(dtree_drop.match("sweep_*").keys())
print(f"\nSweeps kept: {sweep_groups}")
```

```{code-cell}
if len(sweep_groups) >= 2:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, grp in zip(axes, sweep_groups[:2]):
        ds = xd.georeference.get_x_y_z(dtree_drop[grp].to_dataset(inherit="all_coords"))
        ds.DBZH.plot(x="x", y="y", cmap="ChaseSpectral", vmin=-10, vmax=70, ax=ax)
        ax.set_title(f"{grp} ({ds.sweep_fixed_angle.values:.1f} deg)")
        ax.set_aspect("equal")
    fig.suptitle("Drop mode: only complete sweeps", y=1.02, fontsize=13)
    fig.tight_layout()
elif len(sweep_groups) == 1:
    fig, ax = plt.subplots(figsize=(6, 5))
    ds = xd.georeference.get_x_y_z(
        dtree_drop[sweep_groups[0]].to_dataset(inherit="all_coords")
    )
    ds.DBZH.plot(x="x", y="y", cmap="ChaseSpectral", vmin=-10, vmax=70, ax=ax)
    ax.set_title(f"{sweep_groups[0]} ({ds.sweep_fixed_angle.values:.1f} deg)")
    ax.set_aspect("equal")
    fig.suptitle("Drop mode: only complete sweeps", y=1.02, fontsize=13)
    fig.tight_layout()
else:
    print("No complete sweeps in 15 chunks (all dropped).")
```

## Partial volume -- pad mode

With `incomplete_sweep="pad"`, incomplete sweeps are **kept** and reindexed
to a full azimuth grid. Missing rays are filled with `NaN`. The angular
resolution (0.5 or 1.0 degree) is auto-detected from the data.

This is useful for visualization and monitoring where you want to see all
available data as soon as it arrives.

```{code-cell}
dtree_pad = xd.io.open_nexradlevel2_datatree(partial_chunks, incomplete_sweep="pad")

sweep_groups_pad = list(dtree_pad.match("sweep_*").keys())
print(f"Sweeps available (pad mode): {sweep_groups_pad}")

# Show NaN percentage in each sweep
for grp in sweep_groups_pad:
    ds = dtree_pad[grp].to_dataset()
    if "DBZH" in ds:
        nan_pct = np.isnan(ds.DBZH.values).mean() * 100
        print(f"  {grp}: azimuth size={ds.sizes['azimuth']}, DBZH NaN={nan_pct:.1f}%")
```

```{code-cell}
n_sweeps = len(sweep_groups_pad)
fig, axes = plt.subplots(1, n_sweeps, figsize=(6 * n_sweeps, 5))
if n_sweeps == 1:
    axes = [axes]

for ax, grp in zip(axes, sweep_groups_pad):
    ds = xd.georeference.get_x_y_z(dtree_pad[grp].to_dataset(inherit="all_coords"))
    ds.DBZH.plot(x="x", y="y", cmap="ChaseSpectral", vmin=-10, vmax=70, ax=ax)
    ax.set_title(f"{grp} ({ds.sweep_fixed_angle.values:.1f} deg)")
    ax.set_aspect("equal")

fig.suptitle("Pad mode: incomplete sweeps filled with NaN", y=1.02, fontsize=13)
fig.tight_layout()
```

## Early streaming -- few chunks

Even with only 5 chunks (before the first sweep completes), pad mode shows
the partial data that has arrived. The NaN wedge makes it clear which azimuths
are still missing.

```{code-cell}
early_chunks = all_bytes[:5]

dtree_early = xd.io.open_nexradlevel2_datatree(early_chunks, incomplete_sweep="pad")

sweep_groups_early = list(dtree_early.match("sweep_*").keys())
print(f"Sweeps from 5 chunks: {sweep_groups_early}")

if sweep_groups_early:
    ds = xd.georeference.get_x_y_z(
        dtree_early[sweep_groups_early[0]].to_dataset(inherit="all_coords")
    )

    nan_pct = np.isnan(ds.DBZH.values).mean() * 100
    print(f"DBZH NaN percentage: {nan_pct:.1f}%")

    fig, ax = plt.subplots(figsize=(6, 5))
    ds.DBZH.plot(x="x", y="y", cmap="ChaseSpectral", vmin=-10, vmax=70, ax=ax)
    ax.set_title(
        f"Early stream: {sweep_groups_early[0]} "
        f"({ds.sweep_fixed_angle.values:.1f} deg) -- {nan_pct:.0f}% NaN"
    )
    ax.set_aspect("equal")
    fig.tight_layout()
else:
    print("No sweeps found in 5 chunks.")
```

## Summary

| Scenario | `incomplete_sweep` | Behavior |
|---|---|---|
| Full volume (all chunks) | `"drop"` or `"pad"` | All sweeps present, no difference |
| Partial volume | `"drop"` (default) | Incomplete sweeps excluded, warning emitted |
| Partial volume | `"pad"` | Incomplete sweeps kept, missing rays filled with NaN |
| Early stream (few chunks) | `"pad"` | Single partial sweep visible with NaN wedge |

**Note:** Single-file, bytes, and file-like inputs continue to work exactly as
before. The list input and `incomplete_sweep` parameter are additive features.
