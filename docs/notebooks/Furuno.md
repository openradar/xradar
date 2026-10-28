---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
  main_language: python
kernelspec:
  display_name: Python 3
  name: python3
---

# Furuno

```{code-cell}
import xarray as xr
from open_radar_data import DATASETS

import xradar as xd
```

## Download

Fetching Furuno radar data file from [open-radar-data](https://github.com/openradar/open-radar-data) repository.

```{code-cell}
filename_scnx = DATASETS.fetch("2006_20220324_000000_000.scnx.gz")
filename_scn = DATASETS.fetch("0080_20210730_160000_01_02.scn.gz")
```

## xr.open_dataset

Making use of the xarray `furuno` backend.

+++

### scn format

```{code-cell}
ds = xr.open_dataset(filename_scn, engine="furuno")
display(ds)
```

#### Plot Time vs. Azimuth

```{code-cell}
ds.azimuth.plot(y="time")
```

#### Plot Range vs. Time

We need to sort by `time` and specify the y-coordinate.

```{code-cell}
ds.DBZH.sortby("time").plot(y="time")
```

#### Plot Range vs. Azimuth

```{code-cell}
ds.DBZH.sortby("azimuth").plot(y="azimuth")
```

### scnx format

```{code-cell}
ds = xr.open_dataset(filename_scnx, engine="furuno")
display(ds)
```

#### Plot Time vs. Azimuth

```{code-cell}
ds.azimuth.plot(y="time")
```

#### Plot Range vs. Time

We need to sort by `time` and specify the y-coordinate.

```{code-cell}
ds.DBZH.sortby("time").plot(y="time")
```

#### Plot Range vs. Azimuth

```{code-cell}
ds.DBZH.sortby("azimuth").plot(y="azimuth")
```

## open_furuno_datatree

Furuno scn/scnx files consist only of one sweep. But we might load and combine several sweeps into one DataTree.

```{code-cell}
dtree = xd.open_datatree(filename_scn, engine="furuno")
display(dtree)
```

### Plot Sweep Range vs. Time

```{code-cell}
dtree["sweep_0"].dataset.DBZH.plot()
```

### Plot Sweep Range vs. Azimuth

```{code-cell}
dtree["sweep_0"].dataset.DBZH.sortby("azimuth").plot(y="azimuth")
```

## Optional groups: radar_parameters

With `optional_groups=True` the datatree also has the metadata groups. The Furuno reader fills `radar_parameters` with the half-power beam widths of the file header (the `scn` sample file has none).

```{code-cell}
dtree = xd.open_datatree(filename_scnx, engine="furuno", optional_groups=True)
display(dtree["radar_parameters"].dataset)
```

```{code-cell}
import numpy as np

np.testing.assert_almost_equal(dtree["radar_parameters"].radar_beam_width_h, 2.75)
np.testing.assert_almost_equal(dtree["radar_parameters"].radar_beam_width_v, 2.76)
```

## Calibration of scnx files

With `optional_groups=True` the datatree has the group `radar_calibration`. The scnx header gives the antenna gain, the transmit power, the radar constant and the noise power in tenths (Furuno WR2120 manual, data file type 3); xradar scales the values and sets `units`. The transmit power is given in W in the file, the group has it as `xmit_power_h`/`xmit_power_v` in dBm, the unit of CfRadial.

```{code-cell}
calib = dtree["radar_calibration"].dataset
display(calib)
```

```{code-cell}
for name, var in calib.data_vars.items():
    print(f"{name:28s} {float(var):9.2f} {var.attrs['units']}")

# inside the ranges of the manual: antenna gain 30 to 40 dBi, transmit power 50 to 150 W
assert 30 <= calib.antenna_gain_h <= 40
transmit_power_w = 10 ** (float(calib.xmit_power_h) / 10) / 1e3
print(f"transmit power {transmit_power_w:.1f} W")
assert 50 <= transmit_power_w <= 150
```

The values are written to CfRadial1 files with these names.

```{code-cell}
xd.io.to_cfradial1(dtree, "furuno_scnx_cfradial1.nc", calibs=True)
with xr.open_dataset("furuno_scnx_cfradial1.nc") as ds:
    print(ds.r_calib_xmit_power_h.values, ds.r_calib_xmit_power_h.attrs["units"])
```

The older scn format has no antenna gain or transmit power in its header and keeps the radar constant as it is in the file (mantissa and exponent), without `units`:

```{code-cell}
dtree_scn = xd.open_datatree(filename_scn, engine="furuno", optional_groups=True)
display(dtree_scn["radar_calibration"].dataset)
```
