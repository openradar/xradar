# History

## 0.0.12 (2023-02-09)

* ENH: add IRIS ``DB_VELC`` decoding and tests ({issue}`78`), ({pull}`83`) by [@kmuehlbauer](https://github.com/kmuehlbauer)
* FIX: furuno backend inconsistencies ({issue}`77`), ({pull}`82`) by [@kmuehlbauer](https://github.com/kmuehlbauer)
* FIX: ODIM_H5 backend inconsistencies ({issue}`80`), ({pull}`81`) by [@kmuehlbauer](https://github.com/kmuehlbauer)

## 0.0.11 (2023-02-06)

* fix ``_Undetect``/``_FillValue`` in odim writer ({pull}`71`) by [@kmuehlbauer](https://github.com/kmuehlbauer)
* port more backend tests from wradlib ({pull}`73`) by [@kmuehlbauer](https://github.com/kmuehlbauer)

## 0.0.10 (2023-02-01)

* add WRN110 scn format to Furuno reader ({pull}`65`) by [@kmuehlbauer](https://github.com/kmuehlbauer)
* Adapt to new build process, pyproject.toml only, use `ruff` for linting ({pull}`67`) by [@kmuehlbauer](https://github.com/kmuehlbauer)

## 0.0.9 (2022-12-11)

* add ODIM_H5 exporter ({pull}`39`) by [@kmuehlbauer](https://github.com/kmuehlbauer)
* fetch radar data from open-radar-data ({pull}`44`) by [@kmuehlbauer](https://github.com/kmuehlbauer)
* align readers with CfRadial2, add CfRadial2 exporter ({pull}`45`), ({pull}`49`), ({pull}`53`), ({pull}`56`), ({pull}`57`) and ({pull}`58`) by [@kmuehlbauer](https://github.com/kmuehlbauer)
* add georeference accessor, update examples ({pull}`60`), ({pull}`61`) by [@mgrover1](https://github.com/mgrover1)
* refactored and partly reimplemented angle reindexing ({issue}`55`), ({pull}`62`) by [@kmuehlbauer](https://github.com/kmuehlbauer)

## 0.0.8 (2022-09-28)

* add GAMIC HDF5 importer ({pull}`29`) by [@kmuehlbauer](https://github.com/kmuehlbauer)
* add Furuno SCN/SCNX importer ({pull}`30`) by [@kmuehlbauer](https://github.com/kmuehlbauer)
* add Rainbow5 importer ({pull}`32`) by [@kmuehlbauer](https://github.com/kmuehlbauer)
* add Iris/Sigmet importer ({pull}`33`) by [@kmuehlbauer](https://github.com/kmuehlbauer)
* add georeferencing (AEQD) ({pull}`28`) by [@mgrover1](https://github.com/mgrover1)

## 0.0.7 (2022-09-21)

* Add zenodo badges to README.md ({pull}`22`) by [@mgrover1](https://github.com/mgrover1)
* Fix version on RTD ({pull}`23`) by [@kmuehlbauer](https://github.com/kmuehlbauer)
* Add minimal documentation for Datamodel ({pull}`24`) by [@kmuehlbauer](https://github.com/kmuehlbauer)

## 0.0.6 (2022-09-19)

* Improve installation and contribution guide, update README.md with more badges, add version and date of release to docs, update install process ({pull}`19`) by [@kmuehlbauer](https://github.com/kmuehlbauer)
* Add minimal documentation for CfRadial1 and ODIM_H5 importers ({pull}`20`) by [@kmuehlbauer](https://github.com/kmuehlbauer)
* Add accessors.py submodule, add accessors showcase  ({pull}`21`) by [@kmuehlbauer](https://github.com/kmuehlbauer)

## 0.0.5 (2022-09-14)

* Data Model, CfRadial1 Backend ({pull}`13`) by [@kmuehlbauer](https://github.com/kmuehlbauer)
* ODIM_H5 Backend ({pull}`14`) by [@kmuehlbauer](https://github.com/kmuehlbauer)

## 0.0.4 (2022-09-01)

Setting up CI workflow and build, [@mgrover1](https://github.com/mgrover1) and [@kmuehlbauer](https://github.com/kmuehlbauer)

## 0.0.1 (2022-09-01)

* First release on PyPI.
