# xradar

[![PyPI Version](https://img.shields.io/pypi/v/xradar.svg)](https://pypi.python.org/pypi/xradar)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/xradar.svg)](https://anaconda.org/conda-forge/xradar)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7091737.svg)](https://doi.org/10.5281/zenodo.7091737)

[![CI](https://github.com/openradar/xradar/actions/workflows/ci.yml/badge.svg)](https://github.com/openradar/xradar/actions/workflows/ci.yml)
[![Build distribution](https://github.com/openradar/xradar/actions/workflows/upload_pypi.yml/badge.svg)](https://github.com/openradar/xradar/actions/workflows/upload_pypi.yml)
[![RTD Version](https://readthedocs.org/projects/xradar/badge/?version=latest)](https://xradar.readthedocs.io/en/latest/?version=latest)

Xradar includes all the tools to get your weather radar into the xarray data model.

* Free software: MIT license
* Documentation: [https://docs.openradarscience.org/projects/xradar](https://xradar.readthedocs.io)

## About

At a developer meeting held in the course of the ERAD2022 conference in Locarno, Switzerland, future plans and cross-package collaboration of the openradarscience community were intensively discussed.

The consensus was that a close collaboration that benefits the entire community can only be maximized through joint projects. So the idea of a common software project whose only task is to read and write radar data was born. The data import should include as many available data formats as possible, but the data export should be limited to the recognized standards, such as [ODIM_H5](https://www.eumetnet.eu/activities/observations-programme/current-activities/opera/) and [CfRadial](https://github.com/NCAR/CfRadial).

As memory representation an xarray based data model was chosen, which is internally adapted to the forthcoming standard CfRadial2.1/FM301. FM301 is enforced by the [Joint Expert Team on Operational Weather Radar (JET-OWR)](https://community.wmo.int/governance/commission-membership/commission-observation-infrastructure-and-information-systems-infcom/commission-infrastructure-officers/infcom-management-group/standing-committee-measurements-instrumentation-and-traceability-sc-mint/joint-expert-team). Information on FM301 is available at WMO as [WMO CF Extensions](https://community.wmo.int/activity-areas/wis/wmo-cf-extensions).

Any software package that uses xarray in any way will then be able to directly use the described data model and thus quickly and easily import and export radar data. Another advantage is the easy connection to already existing [open source radar processing software](https://openradarscience.org/pages/projects/#).

## Features

* Import CfRadial1 data
* Import/Export CfRadial2 data
* Import/Export ODIM_H5 data
* Import GAMIC HDF5
* Import Rainbow5
* Import Iris/Sigmet
* Import Furuno SCN/SCNX
* Georeferencing (AEQD)
* Angle Reindexing
