# Importers

To import different radar data different approaches are used.

Currently xradar can import:

- CfRadial1
- ODIM_H5

## CfRadial1

### CfRadial1BackendEntrypoint

The xarray backend {class}`xradar.io.backends.cfradial1.CfRadial1BackendEntrypoint`
opens the file with {py:class}`xarray:xarray.backends.NetCDF4DataStore`. From the
xarray machinery a {py:class}`xarray:xarray.Dataset` with the complete file content is
returned. In a final step the wanted group (eg. ``sweep_0``) is extracted and returned.
Currently only mandatory data and metadata is provided. If needed the complete ``root``
group with all data and metadata can be returned.

### open_cfradial1_datatree

With {class}`xradar.io.backends.cfradial1.open_cfradial1_datatree` all groups (eg.
``sweeps_0`` and ``root`` are extracted from the source file and added as ParentNodes
and ChildNodes to a datatree.Datatree.

## ODIM_H5

### OdimBackendEntrypoint

The xarray backend {class}`xradar.io.backends.odim.OdimBackendEntrypoint`
opens the file with {class}`xradar.io.backends.odim.OdimStore`. For the ODIM_H5
subgroups ``dataN`` and ``qualityN`` a {class}`xradar.io.backends.odim.OdimSubStore` is
implemented. Several private helper functions are used to conveniently access data and
metadata. Finally, the xarray machinery returns a {py:class}`xarray:xarray.Dataset`
with wanted group (eg. ``dataset1``). Depending on the used backend kwargs several
more functions are applied on that {py:class}`xarray:xarray.Dataset`.

### open_odim_datatree

With {class}`xradar.io.backends.odim.open_odim_datatree` all groups (eg. ``datasetN``)
are extracted. From that the ``root`` group is processed. Everything is finally added as
ParentNodes and ChildNodes to a {py:class}`datatree:datatree.Datatree`.
