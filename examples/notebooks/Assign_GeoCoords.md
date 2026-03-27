{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "0",
   "metadata": {},
   "source": [
    "# Assign GeoCoords"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1",
   "metadata": {},
   "source": [
    "This notebook demonstrates how to work with geocoordinates (longitude and latitude) instead of radar-centric x and y coordinates."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2",
   "metadata": {},
   "outputs": [],
   "source": [
    "import cmweather  # noqa: F401\n",
    "import matplotlib.pyplot as plt\n",
    "import xarray as xr\n",
    "from open_radar_data import DATASETS\n",
    "\n",
    "import xradar as xd"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3",
   "metadata": {},
   "source": [
    "## Define Functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4",
   "metadata": {},
   "outputs": [],
   "source": "def get_geocoords(ds):\n    \"\"\"\n    Converts Cartesian coordinates (x, y, z) in a radar dataset to geographic\n    coordinates (longitude, latitude, altitude) using CRS transformation.\n\n    Parameters\n    ----------\n    ds : xarray.Dataset\n        Radar dataset with Cartesian coordinates.\n\n    Returns\n    -------\n    xarray.Dataset\n        Dataset with added 'lon', 'lat', and 'alt' coordinates and their attributes.\n    \"\"\"\n    from pyproj import CRS, Transformer\n\n    # Convert the dataset to georeferenced coordinates\n    ds = ds.xradar.georeference()\n    # Define source and target coordinate reference systems (CRS)\n    src_crs = ds.xradar.get_crs()\n    trg_crs = CRS.from_user_input(4326)  # EPSG:4326 (WGS 84)\n    # Create a transformer for coordinate conversion\n    transformer = Transformer.from_crs(src_crs, trg_crs)\n    # Transform x, y, z coordinates to latitude, longitude, and altitude\n    trg_y, trg_x, trg_z = transformer.transform(ds.x, ds.y, ds.z)\n    # Assign new coordinates with appropriate attributes\n    ds = ds.assign_coords(\n        {\n            \"lon\": (ds.x.dims, trg_x, xd.model.get_longitude_attrs()),\n            \"lat\": (ds.y.dims, trg_y, xd.model.get_latitude_attrs()),\n            \"alt\": (ds.z.dims, trg_z, xd.model.get_altitude_attrs()),\n        }\n    )\n    return ds\n\n\ndef fix_sitecoords(ds):\n    coords = [\"longitude\", \"latitude\", \"altitude\", \"altitude_agl\"]\n    for coord in coords:\n        if coord not in ds:\n            continue\n        # Skip coords that are already scalar\n        if ds[coord].ndim == 0:\n            continue\n        # Compute median excluding NaN\n        data = ds[coord].median(skipna=True).item()\n        attrs = ds[coord].attrs\n        ds = ds.assign_coords({coord: xr.DataArray(data=data, attrs=attrs)})\n    return ds"
  },
  {
   "cell_type": "markdown",
   "id": "5",
   "metadata": {},
   "source": [
    "## Load Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6",
   "metadata": {},
   "outputs": [],
   "source": [
    "file1 = DATASETS.fetch(\"cfrad.20080604_002217_000_SPOL_v36_SUR.nc\")\n",
    "file2 = DATASETS.fetch(\"cfrad.20211011_201557.188_to_20211011_201617.720_DOW8_PPI.nc\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7",
   "metadata": {},
   "source": [
    "## Example #1"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "42rlr4y97nb",
   "source": "**Note:** Station coordinates (`latitude`, `longitude`, `altitude`) are stored on the root node of the DataTree. When accessing a sweep dataset directly, use `.to_dataset(inherit=\"all_coords\")` to inherit these coordinates from the root. The `.xradar.georeference()` accessor handles this automatically.",
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8",
   "metadata": {},
   "outputs": [],
   "source": [
    "dtree1 = xd.io.open_cfradial1_datatree(file1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9",
   "metadata": {},
   "outputs": [],
   "source": [
    "dtree1 = dtree1.xradar.georeference()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "10",
   "metadata": {},
   "outputs": [],
   "source": [
    "display(dtree1[\"sweep_0\"].ds)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "11",
   "metadata": {},
   "source": [
    "## Assign lat, lon, and alt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "12",
   "metadata": {},
   "outputs": [],
   "source": [
    "dtree1 = dtree1.xradar.map_over_sweeps(get_geocoords)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "13",
   "metadata": {},
   "outputs": [],
   "source": [
    "display(dtree1[\"sweep_0\"].ds)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "14",
   "metadata": {},
   "outputs": [],
   "source": [
    "ds = dtree1[\"sweep_0\"].to_dataset()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "15",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, ax = plt.subplots(1, 2, figsize=(11, 4))\n",
    "ds[\"DBZ\"].plot(x=\"x\", y=\"y\", vmin=-10, vmax=75, cmap=\"HomeyerRainbow\", ax=ax[0])\n",
    "\n",
    "ds[\"DBZ\"].plot(x=\"lon\", y=\"lat\", vmin=-10, vmax=75, cmap=\"HomeyerRainbow\", ax=ax[1])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "16",
   "metadata": {},
   "source": [
    "## Example #2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "17",
   "metadata": {},
   "outputs": [],
   "source": [
    "dtree2 = xd.io.open_cfradial1_datatree(file2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "18",
   "metadata": {},
   "outputs": [],
   "source": [
    "try:\n",
    "    dtree2 = dtree2.xradar.georeference()\n",
    "except Exception:\n",
    "    print(\"Georeferencing failed!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "19",
   "metadata": {},
   "outputs": [],
   "source": "ds2 = dtree2[\"sweep_0\"].to_dataset(inherit=\"all_coords\")\nprint(\"Longitude\", ds2[\"longitude\"].shape)\nprint(\"Latitude\", ds2[\"latitude\"].shape)"
  },
  {
   "cell_type": "markdown",
   "id": "20",
   "metadata": {},
   "source": "<div class=\"alert alert-info\">\n    <p style=\"font-weight:bold; margin:0;\">Important Note:</p>\n    <p>\n        This radar data is from a mobile research radar called <b>Doppler on Wheels (DOW)</b>. In previous versions of xradar,\n        site coordinates (latitude, longitude) were stored per-ray on each sweep, resulting in shape <code>(731,)</code>.\n        Now, station coordinates are stored as scalar values on the <b>root node</b> of the DataTree and inherited by sweeps\n        via <code>to_dataset(inherit=\"all_coords\")</code>. The <code>fix_sitecoords</code> helper gracefully handles both\n        cases — it skips coordinates that are already scalar.\n    </p>\n</div>"
  },
  {
   "cell_type": "markdown",
   "id": "21",
   "metadata": {},
   "source": [
    "## Fix Coords"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "22",
   "metadata": {},
   "outputs": [],
   "source": "# Fix per-ray station coords on the root node by taking the median.\n# Mobile radars like DOW have per-ray lat/lon which must be collapsed\n# to scalar values before georeferencing.\nroot_ds = dtree2.ds\nfor coord in [\"longitude\", \"latitude\", \"altitude\"]:\n    if coord in root_ds.coords and root_ds[coord].ndim > 0:\n        median_val = root_ds[coord].median(skipna=True).item()\n        root_ds = root_ds.assign_coords({coord: median_val})\ndtree2.ds = root_ds"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "23",
   "metadata": {},
   "outputs": [],
   "source": "dtree2 = dtree2.xradar.map_over_sweeps(get_geocoords)"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "24",
   "metadata": {},
   "outputs": [],
   "source": [
    "display(dtree2[\"sweep_0\"].ds)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "25",
   "metadata": {},
   "outputs": [],
   "source": [
    "ds = dtree2[\"sweep_0\"].to_dataset()\n",
    "ref = ds.where(ds.DBZHC >= 5)[\"DBZHC\"]\n",
    "\n",
    "fig = plt.figure(figsize=(6, 5))\n",
    "ax = plt.axes()\n",
    "pl = ref.plot(\n",
    "    x=\"lon\",\n",
    "    y=\"lat\",\n",
    "    vmin=-10,\n",
    "    vmax=70,\n",
    "    cmap=\"HomeyerRainbow\",\n",
    "    ax=ax,\n",
    ")\n",
    "\n",
    "ax.minorticks_on()\n",
    "ax.grid(ls=\"--\", lw=0.5)\n",
    "ax.set_aspect(\"auto\")\n",
    "title = (\n",
    "    dtree2.attrs[\"instrument_name\"]\n",
    "    + \" \"\n",
    "    + str(ds.time.mean().dt.strftime(\"%Y-%m-%d %H:%M:%S\").values)\n",
    ")\n",
    "ax.set_title(title)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "26",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
