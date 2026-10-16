# Exporters

Exporting is only supported for acknowledged standards.

Currently xradar can export:

## ODIM_H5

### to_odim

With {class}`~xradar.io.export.to_odim` an xradar {py:class}`xarray:xarray.DataTree`
can be saved to an ODIM_H5 file (v2.2 at the moment).

## CfRadial2

### to_cfradial2

With {class}`~xradar.io.export.to_cfradial2` an xradar {py:class}`xarray:xarray.DataTree`
can be saved to a CfRadial2-like file. The input DataTree is not modified. The export
works on a shallow copy by default; pass `deepcopy=True` to work on a deep copy instead.

## CfRadial1

### to_cfradial1

With {class}`~xradar.io.export.to_cfradial1` an xradar {py:class}`xarray:xarray.DataTree`
can be saved to a CfRadial1-like file.

CfRadial1 has one `range(range)` for all rays, so by default all sweeps need the
same first gate and gate spacing (the number of gates may differ). Volumes with
different gate geometries per sweep (e.g. 100 m and 150 m gates in one GAMIC
volume) raise an error. With `range_geometry="per_sweep"` they are written as they
are, with a 2-D `range(sweep, range)` and the attributes
`meters_to_center_of_first_gate` and `meters_between_gates` per sweep
(CfRadial 1.4 Section 2.5). xradar reads such files, not every other reader does
(see ARM-DOE/pyart#1900). If all sweeps share the geometry, the usual 1-D `range`
is written.

```python
xd.io.to_cfradial1(dtree, "volume.nc", range_geometry="per_sweep")
```
