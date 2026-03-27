{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "0",
   "metadata": {},
   "source": [
    "# GAMIC"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1",
   "metadata": {},
   "outputs": [],
   "source": [
    "import xarray as xr\n",
    "from open_radar_data import DATASETS\n",
    "\n",
    "import xradar as xd"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2",
   "metadata": {},
   "source": [
    "## Download\n",
    "\n",
    "Fetching GAMIC radar data file from [open-radar-data](https://github.com/openradar/open-radar-data) repository."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3",
   "metadata": {},
   "outputs": [],
   "source": [
    "filename = DATASETS.fetch(\"DWD-Vol-2_99999_20180601054047_00.h5\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4",
   "metadata": {},
   "source": [
    "## xr.open_dataset\n",
    "\n",
    "Making use of the xarray `gamic` backend. We also need to provide the group. Note, that we are using CfRadial2 group access pattern."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5",
   "metadata": {},
   "outputs": [],
   "source": [
    "ds = xr.open_dataset(filename, group=\"sweep_9\", engine=\"gamic\")\n",
    "display(ds)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6",
   "metadata": {},
   "source": [
    "### Plot Time vs. Azimuth"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7",
   "metadata": {},
   "outputs": [],
   "source": [
    "ds.azimuth.plot()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8",
   "metadata": {},
   "source": [
    "### Plot Range vs. Time"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9",
   "metadata": {},
   "outputs": [],
   "source": [
    "ds.DBZH.plot()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "10",
   "metadata": {},
   "source": [
    "### Plot Range vs. Azimuth\n",
    "\n",
    "We need to sort by azimuth and specify the y-coordinate."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "11",
   "metadata": {},
   "outputs": [],
   "source": [
    "ds.DBZH.sortby(\"azimuth\").plot(y=\"azimuth\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "12",
   "metadata": {},
   "source": [
    "## backend_kwargs\n",
    "\n",
    "Beside `first_dim` there are several additional backend_kwargs for the odim backend, which handle different aspects of angle alignment. This comes into play, when azimuth and/or elevation arrays are not evenly spacend and other issues."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "13",
   "metadata": {},
   "outputs": [],
   "source": [
    "help(xd.io.GamicBackendEntrypoint)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "14",
   "metadata": {},
   "outputs": [],
   "source": [
    "ds = xr.open_dataset(filename, group=\"sweep_9\", engine=\"gamic\", first_dim=\"time\")\n",
    "display(ds)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "15",
   "metadata": {},
   "source": [
    "## open_odim_datatree\n",
    "\n",
    "The same works analoguous with the datatree loader. But additionally we can provide a sweep string, number or list."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "16",
   "metadata": {},
   "outputs": [],
   "source": [
    "help(xd.io.open_gamic_datatree)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "17",
   "metadata": {},
   "outputs": [],
   "source": [
    "dtree = xd.io.open_gamic_datatree(filename, sweep=8)\n",
    "display(dtree)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "18",
   "metadata": {},
   "source": [
    "### Plot Sweep Range vs. Time"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "19",
   "metadata": {},
   "outputs": [],
   "source": [
    "dtree[\"sweep_0\"].ds.DBZH.sortby(\"time\").plot(y=\"time\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "20",
   "metadata": {},
   "source": [
    "### Plot Sweep Range vs. Azimuth"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "21",
   "metadata": {},
   "outputs": [],
   "source": [
    "dtree[\"sweep_0\"].ds.DBZH.plot()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "22",
   "metadata": {},
   "outputs": [],
   "source": [
    "dtree = xd.io.open_gamic_datatree(filename, sweep=\"sweep_8\")\n",
    "display(dtree)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "23",
   "metadata": {},
   "outputs": [],
   "source": [
    "dtree = xd.io.open_gamic_datatree(filename, sweep=[0, 1, 8])\n",
    "display(dtree)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "24",
   "metadata": {},
   "outputs": [],
   "source": [
    "dtree = xd.io.open_gamic_datatree(filename, sweep=[\"sweep_1\", \"sweep_2\", \"sweep_8\"])\n",
    "display(dtree)"
   ]
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
