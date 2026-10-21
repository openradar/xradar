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

# Name mapping between formats

xradar reads each format into the same CfRadial2 / FM301 shaped data model. This page shows how the names of moments and of the metadata are mapped. The tables are built from the mappings the readers and exporters use, so they always describe the installed version.

The standards and formats it refers to:

- WMO FM301, the CF extensions for weather radar data: [WMO CF extensions](https://community.wmo.int/activity-areas/wis/wmo-cf-extensions) and the [FM301 specification](https://wmoomm.sharepoint.com/:b:/s/wmocpdb/EVcEDwHwf6FJtB6anuyBH3QBgA5bE_Uz9jI4FaSkAyowSg?e=jcazi4)
- CfRadial 2.x (the data model of xradar): [CfRadial 2.0](https://dx.doi.org/10.5065/fy2k-x587) and the [CfRadial documents](https://github.com/NCAR/CfRadial/tree/master/docs) (2.1 draft, 1.x)
- CfRadial 1.x: [CfRadial documents](https://github.com/NCAR/CfRadial/tree/master/docs)
- ODIM_H5, the [EUMETNET OPERA data information model](https://www.eumetnet.eu/activities/observations-programme/current-activities/opera/)

The other formats are vendor or institute specific, see the [importers](../importers.md) for their descriptions.

```{code-cell}
import pandas as pd

from xradar import model
from xradar.io.backends import (
    datamet,
    gamic,
    imd,
    iris,
    nexrad_level2,
    rainbow,
    uf,
)
from xradar.io.export.odim import FM301_TO_ODIM_NAMES

pd.set_option("display.max_rows", None, "display.max_colwidth", 80)
```

## Moments

The names of the data model are the [FM301](https://community.wmo.int/activity-areas/wis/wmo-cf-extensions) / [ODIM_H5](https://www.eumetnet.eu/activities/observations-programme/current-activities/opera/) quantity names (`DBZH`, `VRADH`, ...), `xradar.model.sweep_vars_mapping` holds the `standard_name`, `long_name` and `units` of each one. Readers rename the names of their format to these names.

Reader tables (name in the file -> xradar name):

```{code-cell}
readers = {
    "GAMIC": gamic.gamic_mapping,
    "IRIS/Sigmet": iris.iris_mapping,
    "Rainbow": rainbow.rainbow_mapping,
    "DataMet": datamet.datamet_mapping,
    "UF": uf.uf_mapping,
    "NEXRAD Level II": nexrad_level2.nexrad_mapping,
    "IMD": imd.imd_mapping,
}


def names_of(mapping):
    """xradar name -> the names of the format which are mapped to it."""
    inverse = {}
    for source, name in mapping.items():
        inverse.setdefault(name, []).append(source.strip())
    return {name: ", ".join(sorted(sources)) for name, sources in inverse.items()}


table = pd.DataFrame(
    {
        "standard_name": {k: v["standard_name"] for k, v in model.sweep_vars_mapping.items()},
        "units": {k: v["units"] for k, v in model.sweep_vars_mapping.items()},
    }
)
for reader, mapping in readers.items():
    table[reader] = pd.Series(names_of(mapping))
table.fillna("")
```

Notes:

- ODIM_H5 (reader and exporter), CfRadial1 and CfRadial2 use the xradar names as they are, with one exception: ODIM_H5 calls the total power (uncorrected reflectivity) `TH` / `TV`, which are `DBTH` / `DBTV` in FM301 and xradar.
- Furuno files hold the data under the xradar names.
- Names without an entry are kept as they are, e.g. the IRIS type `DB_HCLASS`.

```{code-cell}
pd.Series(FM301_TO_ODIM_NAMES, name="ODIM_H5 quantity").rename_axis("xradar / FM301")
```

```{code-cell}
assert FM301_TO_ODIM_NAMES == {"DBTH": "TH", "DBTV": "TV"}
assert iris.iris_mapping["DB_DBZ"] == "DBZH"
assert gamic.gamic_mapping["zh"] == "DBZH"
```

## Metadata groups

The groups `/radar_parameters` and `/radar_calibration` (names of [CfRadial 2](https://dx.doi.org/10.5065/fy2k-x587) / FM301) are filled from the variables of the sweeps. Names of other formats and of other CfRadial versions are renamed to the CfRadial2 names; these are the names which are read (the others are used as they are):

```{code-cell}
def renames(subgroup):
    return pd.DataFrame(
        [(name, new) for name, new in subgroup.items() if new],
        columns=["name in the file", "xradar name"],
    )


renames(model.radar_parameters_subgroup)
```

```{code-cell}
renames(model.radar_calibration_subgroup)
```

All names which can be in the groups:

```{code-cell}
pd.DataFrame(
    {
        "radar_parameters": pd.Series(
            sorted({v or k for k, v in model.radar_parameters_subgroup.items()})
        ),
        "radar_calibration": pd.Series(
            sorted({v or k for k, v in model.radar_calibration_subgroup.items()})
        ),
    }
).fillna("")
```
