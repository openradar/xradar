# Installation

## Stable release

To install xradar, run this command in your terminal:

```bash
$ pip install xradar
```

This is the preferred method to install xradar, as it will always install the most recent stable release.

If you don't have [pip](https://pip.pypa.io) installed, this [Python installation guide](http://docs.python-guide.org/en/latest/starting/installation/) can guide
you through the process.

## From conda-forge

To install xradar into your conda environment, run this command:

```bash
(my-nifty-env) $ conda install xradar -c conda-forge
```

You might also use the `conda` drop-in `mamba`. Of course you can omit `-c conda-forge` if you are already on that channel.

## From sources

The sources for xradar can be downloaded from the [Github repo](https://github.com/openradar/xradar).

You can either clone the public repository (recommended):

```bash
$ git clone git://github.com/openradar/xradar
```

Or download the [tarball](https://github.com/openradar/xradar/tarball/master):

```bash
$ curl -OJL https://github.com/openradar/xradar/tarball/master
```

```{warning}
The github tarballs have no notion of the xradar version. xradar will thus be installed as version 999.
```

Once you have a copy of the source, you can install it with:

```bash
$ python -m pip install .
```

## From github commit, branch or tag

You might also install directly from a specific commit, branch or tag:

```bash
$ python -m pip install git+https://github.com/openradar/xradar.git@92e2e4
$ python -m pip install git+https://github.com/openradar/xradar.git@main
$ python -m pip install git+https://github.com/openradar/xradar.git@0.0.5
```
