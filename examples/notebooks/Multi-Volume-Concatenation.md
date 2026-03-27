{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "0",
   "metadata": {},
   "source": [
    "# Timeseries of Sweeps"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1",
   "metadata": {},
   "source": [
    "## Imports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2",
   "metadata": {},
   "outputs": [],
   "source": [
    "import warnings\n",
    "\n",
    "import cartopy.crs as ccrs\n",
    "import cmweather  # noqa\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import xarray as xr\n",
    "from open_radar_data import DATASETS\n",
    "\n",
    "import xradar as xd\n",
    "\n",
    "warnings.filterwarnings(\"ignore\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3",
   "metadata": {},
   "source": [
    "## Access Radar Data from the Open Radar Data Package"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4",
   "metadata": {},
   "outputs": [],
   "source": [
    "radar_files = [\n",
    "    \"gucxprecipradarcmacppiS2.c1.20220314.021559.nc\",\n",
    "    \"gucxprecipradarcmacppiS2.c1.20220314.024239.nc\",\n",
    "    \"gucxprecipradarcmacppiS2.c1.20220314.025840.nc\",\n",
    "]\n",
    "files = [DATASETS.fetch(file) for file in radar_files]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5",
   "metadata": {},
   "source": [
    "## Read the Data using Xradar\n",
    "We can read the data into xradar by using the `xr.open_mfdataset` function, but first, we need to align the angles of the different radar volumes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6",
   "metadata": {},
   "outputs": [],
   "source": [
    "def fix_angle(ds):\n",
    "    \"\"\"\n",
    "    Aligns the radar volumes\n",
    "    \"\"\"\n",
    "    ds[\"time\"] = ds.time.load()  # Convert time from dask to numpy\n",
    "\n",
    "    start_ang = 0  # Set consistent start/end values\n",
    "    stop_ang = 360\n",
    "\n",
    "    # Find the median angle resolution\n",
    "    angle_res = ds.azimuth.diff(\"azimuth\").median()\n",
    "\n",
    "    # Determine whether the radar is spinning clockwise or counterclockwise\n",
    "    median_diff = ds.azimuth.diff(\"time\").median()\n",
    "    ascending = median_diff > 0\n",
    "    direction = 1 if ascending else -1\n",
    "\n",
    "    # first find exact duplicates and remove\n",
    "    ds = xd.util.remove_duplicate_rays(ds)\n",
    "\n",
    "    # second reindex according to retrieved parameters\n",
    "    ds = xd.util.reindex_angle(\n",
    "        ds, start_ang, stop_ang, angle_res, direction, method=\"nearest\"\n",
    "    )\n",
    "\n",
    "    ds = ds.expand_dims(\"volume_time\")  # Expand for volumes for concatenation\n",
    "\n",
    "    ds[\"volume_time\"] = [np.nanmin(ds.time.values)]\n",
    "\n",
    "    return ds"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Concatenate in xarray ds\n",
    "ds = xr.open_mfdataset(\n",
    "    files,\n",
    "    preprocess=fix_angle,\n",
    "    engine=\"cfradial1\",\n",
    "    group=\"sweep_0\",\n",
    "    concat_dim=\"volume_time\",\n",
    "    combine=\"nested\",\n",
    ")\n",
    "ds"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8",
   "metadata": {},
   "source": [
    "## Visualize the Dataset\n",
    "Now that we have our dataset, we can visualize it.\n",
    "\n",
    "We need to georeference first, then plot it!"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9",
   "metadata": {},
   "source": [
    "### Georeference the Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "10",
   "metadata": {},
   "outputs": [],
   "source": [
    "ds = ds.xradar.georeference()\n",
    "ds"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "11",
   "metadata": {},
   "source": [
    "### Extract the geoaxis information"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "12",
   "metadata": {},
   "outputs": [],
   "source": [
    "proj_crs = xd.georeference.get_crs(ds)\n",
    "cart_crs = ccrs.Projection(proj_crs)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "13",
   "metadata": {},
   "source": [
    "### Use the FacetGrid to visualize"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "14",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create our facets grid\n",
    "fg = ds.DBZ.plot.pcolormesh(\n",
    "    x=\"x\",\n",
    "    y=\"y\",\n",
    "    vmin=-20,\n",
    "    vmax=40,\n",
    "    cmap=\"ChaseSpectral\",\n",
    "    col=\"volume_time\",\n",
    "    edgecolors=\"face\",\n",
    "    figsize=(14, 4),\n",
    "    transform=cart_crs,\n",
    "    subplot_kws={\"projection\": ccrs.PlateCarree()},\n",
    ")\n",
    "# Set the title\n",
    "fg.set_titles(\"{value}\")\n",
    "\n",
    "# Fix the geo-axes labels\n",
    "first_axis = True\n",
    "for ax in fg.axes.flat:\n",
    "    ax.coastlines()\n",
    "    gl = ax.gridlines(\n",
    "        crs=ccrs.PlateCarree(),\n",
    "        draw_labels=True,\n",
    "        linewidth=1,\n",
    "        color=\"gray\",\n",
    "        alpha=0.3,\n",
    "        linestyle=\"--\",\n",
    "    )\n",
    "    plt.gca().xaxis.set_major_locator(plt.NullLocator())\n",
    "    gl.top_labels = False\n",
    "    gl.right_labels = False\n",
    "    if first_axis:\n",
    "        gl.left_labels = True\n",
    "    else:\n",
    "        gl.left_labels = False\n",
    "    first_axis = False"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "15",
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
