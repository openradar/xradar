# Importers

The backends use different approaches to ingest the data.

## Common DataTree behavior

All ``open_*_datatree()`` functions share the following behavior:

### Station coordinates

Station location variables (``latitude``, ``longitude``, ``altitude``) are placed as
**coordinates** on the root node of the {py:class}`xarray:xarray.DataTree`, following
CfRadial 2.0 Section 4.4. Sweep child nodes also retain local copies of these variables
for compatibility with code that accesses them directly on sweep datasets (e.g.
georeferencing). Once xarray supports scalar coordinate inheritance
(`pydata/xarray#9077 <https://github.com/pydata/xarray/issues/9077>`_), the sweep-level
copies can be removed in a future release.

### Optional metadata subgroups

By default, the metadata subgroups ``/radar_parameters``, ``/georeferencing_correction``,
and ``/radar_calibration`` are **not** included in the DataTree. Pass
``optional_groups=True`` to include them:

```python
import xradar as xd

# Default: lean DataTree without metadata subgroups
dtree = xd.io.open_nexradlevel2_datatree(filename)

# Include optional metadata subgroups
dtree = xd.io.open_nexradlevel2_datatree(filename, optional_groups=True)
```

## CfRadial1

### CfRadial1BackendEntrypoint

The xarray backend {class}`~xradar.io.backends.cfradial1.CfRadial1BackendEntrypoint`
opens the file with {py:class}`xarray:xarray.backends.NetCDF4DataStore`. From the
xarray machinery a {py:class}`xarray:xarray.Dataset` with the complete file content is
returned. In a final step the wanted group (eg. ``sweep_0``) is extracted and returned.
Currently only mandatory data and metadata is provided. If needed the complete ``root``
group with all data and metadata can be returned.

### open_cfradial1_datatree

With {func}`~xradar.io.backends.cfradial1.open_cfradial1_datatree` all groups (eg.
``sweeps_0`` and ``root`` are extracted from the source file and added as ParentNodes
and ChildNodes to a {py:class}`xarray:xarray.DataTree`.

## ODIM_H5

### OdimBackendEntrypoint

The xarray backend {class}`~xradar.io.backends.odim.OdimBackendEntrypoint`
opens the file with {class}`~xradar.io.backends.odim.OdimStore`. For the ODIM_H5
subgroups ``dataN`` and ``qualityN`` a {class}`~xradar.io.backends.odim.OdimSubStore` is
implemented. Several private helper functions are used to conveniently access data and
metadata. Finally, the xarray machinery returns a {py:class}`xarray:xarray.Dataset`
with wanted group (eg. ``dataset1``). Depending on the used backend kwargs several
more functions are applied on that {py:class}`xarray:xarray.Dataset`.

### open_odim_datatree

With {func}`~xradar.io.backends.odim.open_odim_datatree` all groups (eg. ``datasetN``)
are extracted. From that the ``root`` group is processed. Everything is finally added as
ParentNodes and ChildNodes to a {py:class}`xarray:xarray.DataTree`.


## GAMIC HDF5

### GamicBackendEntrypoint

The xarray backend {class}`~xradar.io.backends.gamic.GamicBackendEntrypoint`
opens the file with {class}`~xradar.io.backends.gamic.GamicStore`. Several private helper functions are used to conveniently access data and
metadata. Finally, the xarray machinery returns a {py:class}`xarray:xarray.Dataset`
with wanted group (eg. ``scan0``). Depending on the used backend kwargs several
more functions are applied on that {py:class}`xarray:xarray.Dataset`.

### open_gamic_datatree

With {func}`~xradar.io.backends.gamic.open_gamic_datatree` all groups (eg. ``scanN``)
are extracted. From that the ``root`` group is processed. Everything is finally added as
ParentNodes and ChildNodes to a {py:class}`xarray:xarray.DataTree`.


## Furuno SCN and SCNX

### FurunoBackendEntrypoint

The xarray backend {class}`~xradar.io.backends.furuno.FurunoBackendEntrypoint`
opens the file with {class}`~xradar.io.backends.furuno.FurunoStore`.
Furuno SCN and SCNX data files contain only one sweep group, so the
group-keyword isn't used. Several private helper functions are used to
conveniently access data and metadata. Finally, the xarray machinery returns
a {py:class}`xarray:xarray.Dataset` with the sweep group.

### open_furuno_datatree

With {func}`~xradar.io.backends.furuno.open_furuno_datatree` the single group
is extracted. From that the ``root`` group is processed. Everything is finally
added as ParentNodes and ChildNodes to a {py:class}`xarray:xarray.DataTree`.

## Rainbow

### RainbowBackendEntrypoint

The xarray backend {class}`~xradar.io.backends.rainbow.RainbowBackendEntrypoint`
opens the file with {class}`~xradar.io.backends.rainbow.RainbowStore`. Several
private helper functions are used to conveniently access data and
metadata. Finally, the xarray machinery returns a {py:class}`xarray:xarray.Dataset`
with wanted group (eg. ``0``). Depending on the used backend kwargs several
more functions are applied on that {py:class}`xarray:xarray.Dataset`.

### open_rainbow_datatree

With {func}`~xradar.io.backends.rainbow.open_rainbow_datatree` all groups (eg. ``0``)
are extracted. From that the ``root`` group is processed. Everything is finally added as
ParentNodes and ChildNodes to a {py:class}`xarray:xarray.DataTree`.


## Iris/Sigmet

### IrisBackendEntrypoint

The xarray backend {class}`~xradar.io.backends.iris.IrisBackendEntrypoint`
opens the file with {class}`~xradar.io.backends.Iris.IrisStore`. Several
private helper functions are used to conveniently access data and
metadata. Finally, the xarray machinery returns a {py:class}`xarray:xarray.Dataset`
with wanted group (eg. ``0``). Depending on the used backend kwargs several
more functions are applied on that {py:class}`xarray:xarray.Dataset`.

### open_iris_datatree

With {func}`~xradar.io.backends.iris.open_iris_datatree` all groups (eg. ``1``)
are extracted. From that the ``root`` group is processed. Everything is finally added as
ParentNodes and ChildNodes to a {py:class}`xarray:xarray.DataTree`.


## NexradLevel2

### NexradLevel2BackendEntryPoint

The xarray backend {class}`~xradar.io.backends.nexrad_level2.NexradLevel2BackendEntrypoint`
opens the file with {class}`~xradar.io.backends.nexrad_level2.NexradLevel2Store`. Several
private helper functions are used to conveniently access data and
metadata. Finally, the xarray machinery returns a {py:class}`xarray:xarray.Dataset`
with wanted group (eg. ``0``). Depending on the used backend kwargs several
more functions are applied on that {py:class}`xarray:xarray.Dataset`.

### open_nexradlevel2_datatree

With {func}`~xradar.io.backends.nexrad_level2.open_nexradlevel2_datatree`
all groups (eg. ``1``) are extracted. From that the ``root`` group is processed.
Everything is finally added as ParentNodes and ChildNodes to a {py:class}`xarray:xarray.DataTree`.

#### Chunk file / list input

``open_nexradlevel2_datatree`` accepts a **list or tuple** of chunk sources
as the first argument. Each element can be ``bytes``, a file-like object with
a ``.read()`` method, or a ``str``/``os.PathLike`` path. The chunks are
concatenated internally before parsing.

This enables streaming NEXRAD Level 2 data directly from the
``unidata-nexrad-level2-chunks`` S3 bucket without downloading full volume
files:

```python
import fsspec
import xradar as xd

fs = fsspec.filesystem("s3", anon=True)
chunks = sorted(fs.ls("unidata-nexrad-level2-chunks/KABR/903"))
all_bytes = [fs.open(p, "rb").read() for p in chunks]

dtree = xd.io.open_nexradlevel2_datatree(all_bytes)
```

#### Handling incomplete sweeps

When working with partial volumes (not all chunks have arrived yet), the last
sweep is typically incomplete. The ``incomplete_sweep`` parameter controls how
these are handled:

- ``incomplete_sweep="drop"`` (default): Incomplete sweeps are excluded from
  the DataTree and a warning is emitted. This is the safest option for
  downstream processing that expects full 360-degree sweeps.

- ``incomplete_sweep="pad"``: Incomplete sweeps are kept and reindexed to a
  full azimuth grid (360 or 720 azimuths depending on the auto-detected
  angular resolution). Missing rays are filled with ``NaN``.

```python
# Drop mode (default) -- only complete sweeps
dtree = xd.io.open_nexradlevel2_datatree(
    partial_bytes, incomplete_sweep="drop"
)

# Pad mode -- all sweeps, missing rays filled with NaN
dtree = xd.io.open_nexradlevel2_datatree(
    partial_bytes, incomplete_sweep="pad"
)
```

See the {doc}`notebooks/nexrad_read_chunks` notebook for a full walkthrough.

## Datamet

### DataMetBackendEntrypoint

The xarray backend {class}`~xradar.io.backends.datamet.DataMetBackendEntrypoint`
opens the file with {class}`~xradar.io.backends.datamet.DataMetStore`. Several
private helper functions are used to conveniently access data and
metadata. Finally, the xarray machinery returns a {py:class}`xarray:xarray.Dataset`
with wanted group (eg. ``0``). Depending on the used backend kwargs several
more functions are applied on that {py:class}`xarray:xarray.Dataset`.

### open_datamet_datatree

With {func}`~xradar.io.backends.datamet.open_datamet_datatree`
all groups (eg. ``1``) are extracted. From that the ``root`` group is processed.
Everything is finally added as ParentNodes and ChildNodes to a {py:class}`xarray:xarray.DataTree`.

## Halo Photonics Lidar

### HPLBackendEntrypoint

The xarray backend {class}`~xradar.io.backends.hpl.HPLBackendEntrypoint`
opens the file with {class}`~xradar.io.backends.hpl.HplStore`. Several
private helper functions are used to conveniently access data and
metadata. Finally, the xarray machinery returns a {py:class}`xarray:xarray.Dataset`
with wanted group (eg. ``0``). Depending on the used backend kwargs several
more functions are applied on that {py:class}`xarray:xarray.Dataset`.

### open_hpl_datatree

With {func}`~xradar.io.backends.hpl.open_hpl_datatree`
all groups (eg. ``1``) are extracted. From that the ``root`` group is processed.
Everything is finally added as ParentNodes and ChildNodes to a {py:class}`xarray:xarray.DataTree`.

## Vaisala WindCube Lidar

Vaisala (formerly Leosphere) WindCube scanning Doppler lidars write NetCDF-4
files declaring ``Conventions = "CF/Radial 2.0 , CF-1.7"``. The sweep groups
(e.g. ``Sweep_152468-1``) are listed in the root ``sweep_group_name``, the fixed
angles in the root ``sweep_fixed_angle`` (azimuth for RHI, elevation otherwise).
Variables are documented in the files through ``long_name``, ``units`` and
``comments``; measurements include ``radial_wind_speed`` with confidence index
and status, ``cnr``, ``relative_beta`` and ``doppler_spectrum_width``.

The reader normalizes these files to the xradar data model:

- vendor sweep modes are mapped to CfRadial 2.1 sweep modes (``ppi``/``volume``/``vad``
  to ``azimuth_surveillance`` or ``sector``, ``fixed`` to ``vertical_pointing`` or
  ``pointing``, ``dbs`` to ``doppler_beam_swinging``, ``segment`` to
  ``complex_trajectory``); the original mode is kept in the
  ``windcube_sweep_mode`` attribute,
- DBS/VAD sweeps with ``range(time, gate_index)`` are split into one sweep per
  gate geometry (e.g. vertical and inclined beams),
- over-the-top RHIs are unfolded to elevations from 0 to 180 degrees,
- rays without valid angles are dropped, ``time`` is the end of each ray.

References: [CfRadial 2.1 format](https://github.com/NCAR/CfRadial/blob/master/docs/CfRadialDoc-v2.1-20190901.pdf),
[Vaisala WindCube Scan](https://vaisala.com/products/weather-environmental-sensors/windcube-scan-general-info), [WindCube Scan product spotlight](https://www.vaisala.com/sites/default/files/documents/WEA-MET-WindCube-Scan-Lidar-Product-Spotlight-B212058EN-A.pdf).

### WindCubeBackendEntrypoint

The xarray backend {class}`~xradar.io.backends.windcube.WindCubeBackendEntrypoint`
returns the wanted sweep (eg. ``sweep_0``) as {py:class}`xarray:xarray.Dataset`.

### open_windcube_datatree

With {func}`~xradar.io.backends.windcube.open_windcube_datatree`
all sweeps are extracted and the ``root`` group is processed.
Everything is finally added as ParentNodes and ChildNodes to a {py:class}`xarray:xarray.DataTree`.

## Metek MRR2

### MRRBackendEntrypoint

The xarray backend {class}`~xradar.io.backends.metek.MRRBackendEntrypoint`
opens the file with {class}`~xradar.io.backends.metek.MRR2DataStore`. Several
private helper functions are used to conveniently access data and
metadata. Finally, the xarray machinery returns a {py:class}`xarray:xarray.Dataset`
with wanted group (eg. ``0``). Depending on the used backend kwargs several
more functions are applied on that {py:class}`xarray:xarray.Dataset`.

### open_metek_datatree

With {func}`~xradar.io.backends.metek.open_metek_datatree`
all groups (eg. ``1``) are extracted. From that the ``root`` group is processed.
Everything is finally added as ParentNodes and ChildNodes to a {py:class}`xarray:xarray.DataTree`.

## India Meteorological Department (IMD)

### IMDBackendEntrypoint

The xarray backend {class}`~xradar.io.backends.imd.IMDBackendEntrypoint` reads a
single IMD NetCDF radar file and returns a CfRadial2-compatible
{py:class}`xarray:xarray.Dataset` containing one sweep. IMD files use a
NetCDF4 container with an IRIS-inspired variable layout (``radialAzim``,
``radialElev``, single-letter moment codes ``T``, ``Z``, ``V``, ``W``, etc.).
The backend renames dimensions/variables to the CfRadial2 convention and
maps moments to ``DBTH``, ``DBZH``, ``VRADH``, ``WRADH``.

### open_imd_datatree

IMD stores **one sweep per file**. A complete volume is assembled from
multiple files: typically 2-3 files for long-range PPI and 9-10 files for
short-range, high-resolution PPI. Pass a single file path to
{func}`~xradar.io.backends.imd.open_imd_datatree` to get a single-sweep
DataTree, or a list of file paths (one per sweep) to assemble a volume:

```python
import xradar as xd

# single sweep
dtree = xd.io.open_imd_datatree("GOA210515003646-IMD-C.nc")

# multi-sweep volume (stacked via xradar.util.create_volume)
files = sorted(glob.glob("GOA210515003646-IMD-C.nc*"))
dtree = xd.io.open_imd_datatree(files)
```

### group_imd_files and open_imd_volumes

A single directory usually holds many volumes back-to-back. Use
{func}`~xradar.io.backends.imd.group_imd_files` to split a directory
(or glob, or list) into per-volume file lists by filename stem, or
{func}`~xradar.io.backends.imd.open_imd_volumes` to load everything at
once into a nested DataTree with ``vcp_NN`` child nodes (VCP = *volume
coverage pattern*):

```python
import xradar as xd

# Iterate one volume at a time
for files in xd.io.group_imd_files("/data/imd"):
    dtree = xd.io.open_imd_datatree(files)

# Or load all volumes into a single tree
tree = xd.io.open_imd_volumes("/data/imd")
tree["vcp_00/sweep_0"].ds["DBZH"]
```

## Universal Format (UF))

### UFBackendEntryPoint

The xarray backend {class}`~xradar.io.backends.uf.UFBackendEntrypoint`
opens the file with {class}`~xradar.io.backends.uf.UFStore`. Several
private helper functions are used to conveniently access data and
metadata. Finally, the xarray machinery returns a {py:class}`xarray:xarray.Dataset`
with wanted group (eg. ``0``). Depending on the used backend kwargs several
more functions are applied on that {py:class}`xarray:xarray.Dataset`.

### open_uf_datatree

With {func}`~xradar.io.backends.uf.open_uf_datatree`
all groups (eg. ``1``) are extracted. From that the ``root`` group is processed.
Everything is finally added as ParentNodes and ChildNodes to a {py:class}`xarray:xarray.DataTree`.
