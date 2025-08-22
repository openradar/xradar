{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "0",
   "metadata": {},
   "source": [
    "# CfRadial1"
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
    "Fetching CfRadial1 radar data file from [open-radar-data](https://github.com/openradar/open-radar-data) repository."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3",
   "metadata": {},
   "outputs": [],
   "source": [
    "filename = DATASETS.fetch(\"cfrad.20080604_002217_000_SPOL_v36_SUR.nc\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4",
   "metadata": {},
   "source": [
    "## xr.open_dataset\n",
    "\n",
    "Making use of the xarray `cfradial1` backend. We also need to provide the group."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5",
   "metadata": {},
   "outputs": [],
   "source": [
    "ds = xr.open_dataset(filename, group=\"sweep_0\", engine=\"cfradial1\")\n",
    "display(ds)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6",
   "metadata": {},
   "source": [
    "### Plot Time vs. Azimuth\n",
    "\n",
    "We need to sort by time and specify the coordinate."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7",
   "metadata": {},
   "outputs": [],
   "source": [
    "ds.azimuth.plot(y=\"time\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8",
   "metadata": {},
   "source": [
    "### Plot Range vs. Time\n",
    "\n",
    "We need to sort by time and specify the coordinate."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9",
   "metadata": {},
   "outputs": [],
   "source": [
    "ds.DBZ.sortby(\"time\").plot(y=\"time\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "10",
   "metadata": {},
   "source": [
    "### Plot Range vs. Azimuth\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "11",
   "metadata": {},
   "outputs": [],
   "source": [
    "ds.DBZ.plot()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "12",
   "metadata": {},
   "source": [
    "## backend_kwargs\n",
    "\n",
    "The cfradial1 backend can be parameterized via kwargs. Please observe the possibilities below."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "13",
   "metadata": {},
   "outputs": [],
   "source": [
    "?xd.io.CfRadial1BackendEntrypoint"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "14",
   "metadata": {},
   "outputs": [],
   "source": [
    "ds = xr.open_dataset(filename, group=\"sweep_0\", engine=\"cfradial1\", first_dim=\"time\")\n",
    "display(ds)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "15",
   "metadata": {},
   "outputs": [],
   "source": [
    "ds = xr.open_dataset(\n",
    "    filename, group=\"sweep_1\", engine=\"cfradial1\", backend_kwargs=dict(first_dim=\"time\")\n",
    ")\n",
    "display(ds)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "16",
   "metadata": {},
   "source": [
    "## open_cfradial1_datatree\n",
    "\n",
    "The same works analoguous with the datatree loader. But additionally we can provide a sweep number or list."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "17",
   "metadata": {},
   "outputs": [],
   "source": [
    "?xd.io.open_cfradial1_datatree"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "18",
   "metadata": {},
   "outputs": [],
   "source": [
    "dtree = xd.io.open_cfradial1_datatree(\n",
    "    filename,\n",
    "    first_dim=\"time\",\n",
    "    optional=False,\n",
    ")\n",
    "display(dtree)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "19",
   "metadata": {},
   "source": [
    "### Plot Sweep Range vs. Time"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "20",
   "metadata": {},
   "outputs": [],
   "source": [
    "dtree[\"sweep_0\"].ds.DBZ.plot()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "21",
   "metadata": {},
   "source": [
    "### Plot Sweep Range vs. Azimuth"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "22",
   "metadata": {},
   "outputs": [],
   "source": [
    "dtree[\"sweep_0\"].ds.DBZ.sortby(\"azimuth\").plot(y=\"azimuth\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "23",
   "metadata": {},
   "outputs": [],
   "source": [
    "dtree = xd.io.open_cfradial1_datatree(filename, sweep=[0, 1, 8])\n",
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
    "dtree = xd.io.open_cfradial1_datatree(filename, sweep=[\"sweep_0\", \"sweep_4\", \"sweep_8\"])\n",
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
