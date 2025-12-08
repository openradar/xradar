{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "0",
   "metadata": {},
   "source": [
    "# Accessors"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1",
   "metadata": {},
   "source": [
    "To extend `xarray.DataArray` and  `xarray.Dataset`\n",
    "xradar aims to provide accessors which downstream libraries can hook into.\n",
    "\n",
    "Those accessors are yet to be defined. For starters we could implement purpose-based\n",
    "accessors (like `.vis`, `.kdp` or `.trafo`) on `xarray.DataArray` level.\n",
    "\n",
    "To not have to import downstream packages a similar approach to xarray.backends using\n",
    "`importlib.metadata.entry_points` could be facilitated.\n",
    "\n",
    "In this notebook the creation of such an accessor is showcased."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
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
    "## Import Data\n",
    "\n",
    "Fetch data from [open-radar-data](https://github.com/openradar/open-radar-data) repository."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4",
   "metadata": {},
   "outputs": [],
   "source": [
    "filename = DATASETS.fetch(\"71_20181220_060628.pvol.h5\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5",
   "metadata": {},
   "source": [
    "### Open data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6",
   "metadata": {},
   "outputs": [],
   "source": [
    "ds = xr.open_dataset(filename, group=\"sweep_0\", engine=\"odim\")\n",
    "display(ds.DBZH.values)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7",
   "metadata": {},
   "source": [
    "### Plot DBZH"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8",
   "metadata": {},
   "outputs": [],
   "source": [
    "ds.DBZH.plot()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9",
   "metadata": {},
   "source": [
    "## Define two example functions\n",
    "\n",
    "Functions copied verbatim from wradlib."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "10",
   "metadata": {},
   "outputs": [],
   "source": [
    "def _decibel(x):\n",
    "    \"\"\"Calculates the decibel representation of the input values\n",
    "\n",
    "    :math:`dBZ=10 \\\\cdot \\\\log_{10} z`\n",
    "\n",
    "    Parameters\n",
    "    ----------\n",
    "    x : float or :class:`numpy:numpy.ndarray`\n",
    "        (must not be <= 0.)\n",
    "\n",
    "    Examples\n",
    "    --------\n",
    "    >>> from wradlib.trafo import decibel\n",
    "    >>> print(decibel(100.))\n",
    "    20.0\n",
    "    \"\"\"\n",
    "    return 10.0 * np.log10(x)\n",
    "\n",
    "\n",
    "def _idecibel(x):\n",
    "    \"\"\"Calculates the inverse of input decibel values\n",
    "\n",
    "    :math:`z=10^{x \\\\over 10}`\n",
    "\n",
    "    Parameters\n",
    "    ----------\n",
    "    x : float or :class:`numpy:numpy.ndarray`\n",
    "\n",
    "    Examples\n",
    "    --------\n",
    "    >>> from wradlib.trafo import idecibel\n",
    "    >>> print(idecibel(10.))\n",
    "    10.0\n",
    "\n",
    "    \"\"\"\n",
    "    return 10.0 ** (x / 10.0)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "11",
   "metadata": {},
   "source": [
    "## Function dictionaries\n",
    "\n",
    "To show the import of the functions, we put them in different dictionaries as we would get them via `entry_points`. \n",
    "\n",
    "This is what the downstream libraries would have to provide."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "12",
   "metadata": {},
   "outputs": [],
   "source": [
    "package_1_func = {\"trafo\": {\"decibel\": _decibel}}\n",
    "package_2_func = {\"trafo\": {\"idecibel\": _idecibel}}"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "13",
   "metadata": {},
   "source": [
    "## xradar internal functionality\n",
    "\n",
    "This is how xradar would need to treat that input data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "14",
   "metadata": {},
   "outputs": [],
   "source": [
    "downstream_functions = [package_1_func, package_2_func]\n",
    "xradar_accessors = [\"trafo\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "15",
   "metadata": {},
   "outputs": [],
   "source": [
    "package_functions = {}\n",
    "for accessor in xradar_accessors:\n",
    "    package_functions[accessor] = {}\n",
    "    for dfuncs in downstream_functions:\n",
    "        package_functions[accessor].update(dfuncs[accessor])\n",
    "print(package_functions)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "16",
   "metadata": {},
   "source": [
    "## Create and register accessor\n",
    "\n",
    "We bundle the different steps into one function, ``create_xradar_dataarray_accessor``."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "17",
   "metadata": {},
   "outputs": [],
   "source": [
    "for accessor in xradar_accessors:\n",
    "    xd.accessors.create_xradar_dataarray_accessor(accessor, package_functions[accessor])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "18",
   "metadata": {},
   "source": [
    "## Convert DBZH to linear and plot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "19",
   "metadata": {},
   "outputs": [],
   "source": [
    "z = ds.DBZH.trafo.idecibel()\n",
    "z.plot()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "20",
   "metadata": {},
   "source": [
    "## Convert z to decibel and plot()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "21",
   "metadata": {},
   "outputs": [],
   "source": [
    "dbz = z.trafo.decibel()\n",
    "display(dbz)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "22",
   "metadata": {},
   "outputs": [],
   "source": [
    "dbz.plot()"
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
