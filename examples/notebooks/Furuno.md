{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "0",
   "metadata": {},
   "source": [
    "# Furuno"
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
    "Fetching Furuno radar data file from [open-radar-data](https://github.com/openradar/open-radar-data) repository."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3",
   "metadata": {},
   "outputs": [],
   "source": [
    "filename_scnx = DATASETS.fetch(\"2006_20220324_000000_000.scnx.gz\")\n",
    "filename_scn = DATASETS.fetch(\"0080_20210730_160000_01_02.scn.gz\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4",
   "metadata": {},
   "source": [
    "## xr.open_dataset\n",
    "\n",
    "Making use of the xarray `furuno` backend. "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5",
   "metadata": {},
   "source": [
    "### scn format"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6",
   "metadata": {},
   "outputs": [],
   "source": [
    "ds = xr.open_dataset(filename_scn, engine=\"furuno\")\n",
    "display(ds)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7",
   "metadata": {},
   "source": [
    "#### Plot Time vs. Azimuth"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8",
   "metadata": {},
   "outputs": [],
   "source": [
    "ds.azimuth.plot(y=\"time\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9",
   "metadata": {},
   "source": [
    "#### Plot Range vs. Time\n",
    "\n",
    "We need to sort by `time` and specify the y-coordinate."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "10",
   "metadata": {},
   "outputs": [],
   "source": [
    "ds.DBZH.sortby(\"time\").plot(y=\"time\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "11",
   "metadata": {},
   "source": [
    "#### Plot Range vs. Azimuth"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "12",
   "metadata": {},
   "outputs": [],
   "source": [
    "ds.DBZH.sortby(\"azimuth\").plot(y=\"azimuth\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "13",
   "metadata": {},
   "source": [
    "### scnx format"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "14",
   "metadata": {},
   "outputs": [],
   "source": [
    "ds = xr.open_dataset(filename_scnx, engine=\"furuno\")\n",
    "display(ds)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "15",
   "metadata": {},
   "source": [
    "#### Plot Time vs. Azimuth"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "16",
   "metadata": {},
   "outputs": [],
   "source": [
    "ds.azimuth.plot(y=\"time\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "17",
   "metadata": {},
   "source": [
    "#### Plot Range vs. Time\n",
    "\n",
    "We need to sort by `time` and specify the y-coordinate."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "18",
   "metadata": {},
   "outputs": [],
   "source": [
    "ds.DBZH.sortby(\"time\").plot(y=\"time\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "19",
   "metadata": {},
   "source": [
    "#### Plot Range vs. Azimuth"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "20",
   "metadata": {},
   "outputs": [],
   "source": [
    "ds.DBZH.sortby(\"azimuth\").plot(y=\"azimuth\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "21",
   "metadata": {},
   "source": [
    "## open_furuno_datatree\n",
    "\n",
    "Furuno scn/scnx files consist only of one sweep. But we might load and combine several sweeps into one DataTree."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "22",
   "metadata": {},
   "outputs": [],
   "source": [
    "dtree = xd.io.open_furuno_datatree(filename_scn)\n",
    "display(dtree)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "23",
   "metadata": {},
   "source": [
    "### Plot Sweep Range vs. Time"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "24",
   "metadata": {},
   "outputs": [],
   "source": [
    "dtree[\"sweep_0\"].ds.DBZH.plot()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "25",
   "metadata": {},
   "source": [
    "### Plot Sweep Range vs. Azimuth"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "26",
   "metadata": {},
   "outputs": [],
   "source": [
    "dtree[\"sweep_0\"].ds.DBZH.sortby(\"azimuth\").plot(y=\"azimuth\")"
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
