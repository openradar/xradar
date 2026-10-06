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

### Virtual byte-range access

{class}`~xradar.io.virtual.IrisParser` (requires the ``xradar[virtual]``
extra, Python 3.12+) indexes a RAW volume into a VirtualiZarr ``ManifestStore`` of
byte-range references instead of decoding it: one zarr chunk = one whole
sweep (rays are RLE-compressed with all data types interleaved ray-major, so
no smaller unit is both contiguous and independently decodable). Every
moment of a sweep references the same byte span; the decode-only zarr v3
codec ``xradar-iris-sweep`` — registered as a ``zarr.codecs`` entry point, so
**reading** a virtual store needs only ``xradar`` + ``zarr`` — demultiplexes
one data type out of the span at read time. Its configuration is:

| key                | meaning                                              |
|--------------------|------------------------------------------------------|
| `moment_index`     | this data type's slot in the ray-major interleave    |
| `ndatatypes`       | interleave stride (counts every type incl. `DB_XHDR`)|
| `sort_rays`        | reorder rows to azimuth order at decode (store-builder flag; the parser always writes `false`) |
| `pad_missing_rays` | keep canonical slots for dropped rays (fill rows)    |

The parser runs this backend's own code wherever the job is the same: the
headers come from `IrisRawFile(..., loaddata=False)`, and the moment names
(`iris_mapping`; a second type mapping to an already used name, e.g.
`DB_DBZ` + `DB_DBZ2`, keeps its Sigmet name), CF scaling and no-data fill
(`SIGMET_DATA_TYPES`), the FM301 `sweep_mode` (`sector` / `rhi` /
`azimuth_surveillance`), nyquist velocity, range gates, CF attributes
({mod}`xradar.model`) and the root group (source, stripped scan and site
names, task description) are the eager reader's. The parser emits pure
pointers: rays and per-ray coordinates keep the file's order (a sweep whose
first ray straddles north differs from the eager reader by a cyclic roll,
values identical). Ordering rays by azimuth is a store-builder decision: it
writes sorted coordinates with {func}`xradar.io.virtual.azimuth_sort_order`
and sets `sort_rays: true`, which the codec honors with the same
permutation.

Anything that is not a well-formed RAW file (partial records, wrong
structure identifiers, a truncated volume) raises `ValueError`. Documented
differences from `open_iris_datatree`:

- RHI tasks are refused (`NotImplementedError`): the eager reader lays them
  out as `(elevation, range)`, the parser builds `(azimuth, range)` sweeps.
- Ray times have 1-second resolution (`DB_XHDR` millisecond times are not
  read), so `time` and the coverage strings can differ by under a second
  on files with `DB_XHDR`.
- The root carries no `sweep_group_name` / `sweep_fixed_angle` (store
  builders assemble the volume-level index themselves).
- The `range` attributes `meters_between_gates` and
  `meters_to_center_of_first_gate` are derived from the range values in
  metres; `open_iris_datatree` still writes the raw header numbers there.

Stores written before the codec name gained its ``xradar-`` prefix declare
``sigmet-sweep``; xradar still reads them through that read-only alias.

See the {doc}`notebooks/IRIS_Virtual` notebook for a full walkthrough.


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
