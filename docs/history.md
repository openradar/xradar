# History

## 0.6.5 (2024-09-20)

FIX: Azimuth dimension now labelled correctly for Halo Photonics data ({pull}`206`) by [@rcjackson](https://github.com/rcjackson).
FIX: Do not apply scale/offset in datamet reader, leave it to xarray instead ({pull}`209`) by [@kmuehlbauer](https://github.com/kmuehlbauer).

## 0.6.4 (2024-08-30)

FIX: Notebooks are now conforming to ruff's style checks by [@rcjackson](https://github.com/rcjackson), ({pull}`199`) by [@rcjackson](https://github.com/rcjackson).
FIX: use dict.get() to retrieve attribute key and return "None" if not available, ({pull}`200`) by [@kmuehlbauer](https://github.com/kmuehlbauer)

## 0.6.3 (2024-08-13)

FIX: use rstart in meter for ODIM_H5/V2_4 ({issue}`196`) by [@kmuehlbauer](https://github.com/kmuehlbauer), ({pull}`197`) by [@kmuehlbauer](https://github.com/kmuehlbauer).

## 0.6.2 (2024-08-12)

* FIX: Passing 'engine' kwarg in "open_cfradial1_datatree" method to enable fsspec.open when using url ({issue}`194`) by [@aladinor](https://github.com/aladinor), ({pull}`195`) by [@aladinor](https://github.com/aladinor)

## 0.6.1 (2024-08-07)

* MNT: minimize CI ({pull}`192`) by [@kmuehlbauer](https://github.com/kmuehlbauer).
* FIX: properly read CfRadial1 n_points files ({issue}`188`) by [@aladinor](https://github.com/aladinor), ({pull}`190`) by [@kmuehlbauer](https://github.com/kmuehlbauer).

## 0.6.0 (2024-08-05)

* ADD: DataMet reader ({pull}`175`) by [@wolfidan](https://github.com/wolfidan).
* FIX: Nexrad level2 time offset of 1 day, skip reading missing elevations, introduce new radial_status of 5
 ({issue}`180`) by [@ghiggi](https://github.com/ghiggi), ({pull}`180`) by [@kmuehlbauer](https://github.com/kmuehlbauer).
* ADD: Reader for Halo Photonics Doppler lidar data by [@rcjackson](https://github.com/rcjackson)

## 0.5.1 (2024-07-05)

* ADD: Add Alfonso to citation doc ({pull}`169`) by [@mgrover1](https://github.com/mgrover1).
* ENH: Adding global variables and attributes to iris datatree ({pull}`166`) by [@aladinor](https://github.com/aladinor).
* FIX: Set fillvalue before applying scale/offset when exporting to odim ({issue}`122`) by [@pavlikp](https://github.com/pavlikp), ({pull}`173`) by [@kmuehlbauer](https://github.com/kmuehlbauer).
* FIX: Fix use of ruff, CI and numpy2 ({pull}`177`) by [@mgrover1](https://github.com/mgrover1) and [@kmuehlbauer](https://github.com/kmuehlbauer).

## 0.5.0 (2024-03-28)

* MNT: Update GitHub actions, address DeprecationWarnings ({pull}`153`) by [@kmuehlbauer](https://github.com/kmuehlbauer).
* MNT: restructure odim.py/gamic.py, add test_odim.py/test_gamic.py ({pull}`154`) by [@kmuehlbauer](https://github.com/kmuehlbauer).
* MNT: use CODECOV token ({pull}`155`) by [@kmuehlbauer](https://github.com/kmuehlbauer).
* MNT: fix path for notebook coverage ({pull}`157`) by [@kmuehlbauer](https://github.com/kmuehlbauer).
* ADD: NEXRAD Level2 structured reader ({pull}`158`) by [@kmuehlbauer](https://github.com/kmuehlbauer) and [@mgrover1](https://github.com/mgrover1).
* FIX: Add the proper elevation angle to fixed angle ({pull}`162`) by [@mgrover1](https://github.com/mgrover1).
* ENH: Add a utility for finding sweep number keys ({pull}`167`) by [@mgrover1](https://github.com/mgrover1).

## 0.4.3 (2024-02-24)

* MNT:  address black style changes, update pre-commit-config.yaml ({pull}`152`) by [@kmuehlbauer](https://github.com/kmuehlbauer).
* FIX: use len(unique) to estimate unique entry for odim range ({pull}`151`) by [@martinpaule](https://github.com/martinpaule).

## 0.4.2 (2023-11-02)

* FIX: Fix handling of sweep_mode attributes ({pull}`143`) by [@mgrover1](https://github.com/mgrover1)
* FIX: explicitely check for "False" in get_crs() {pull}`142`) by [@kmuehlbauer](https://github.com/kmuehlbauer).

## 0.4.1 (2023-10-26)

* FIX: Add history to cfradial1 output, and fix minor error in CfRadial1_Export.ipynb notebook({pull}`132`) by [@syedhamidali](https://github.com/syedhamidali)
* FIX: fix readthedocs build for python 3.12 ({pull}`140`) by [@kmuehlbauer](https://github.com/kmuehlbauer).
* FIX: align coordinates in backends, pin python >3.9,<=3.12 in environment.yml ({pull}`139`) by [@kmuehlbauer](https://github.com/kmuehlbauer)
* FIX: prevent integer overflow when calculating azimuth in FURUNO scn files ({issue}`137`) by [@giacant](https://github.com/giacant) , ({pull}`138`) by [@kmuehlbauer](https://github.com/kmuehlbauer)

## 0.4.0 (2023-09-27)

* ENH: Add cfradial1 exporter ({issue}`124`) by [@syedhamidali](https://github.com/syedhamidali), ({pull}`126`) by [@syedhamidali](https://github.com/syedhamidali)
* FIX: use datastore._group instead of variable["sweep_number"] ({issue}`121`) by [@aladinor](https://github.com/aladinor) , ({pull}`123`) by [@kmuehlbauer](https://github.com/kmuehlbauer)
* MIN: use "crs_wkt" instead of deprecated "spatial_ref" when adding CRS ({pull}`127`) by [@kmuehlbauer](https://github.com/kmuehlbauer)
* FIX: always read nodata and undetect attributes from ODIM file ({pull}`125`) by [@egouden](https://github.com/egouden)
* MIN: use `cmweather` colormaps in xradar ({pull}`128`) by [@kmuehlbauer](https://github.com/kmuehlbauer).

## 0.3.0 (2023-07-11)

* ENH: Add new examples using radar data on AWS s3 bucket ({pull}`102`) by [@aladinor](https://github.com/aladinor)
* FIX: Correct DB_DBTE8/DB_DBZE8 and DB_DBTE16/DB_DBZE16 decoding for iris-backend ({pull}`110`) by [@kmuehlbauer](https://github.com/kmuehlbauer)
* FIX: Cast boolean string to int in rainbow dictionary ({pull}`113`) by [@egouden](https://github.com/egouden)
* MNT: switch to mamba-org/setup-micromamba, split CI tests ({issue}`115`), ({pull}`116`) by [@kmuehlbauer](https://github.com/kmuehlbauer)
* FIX: time interpolation ({pull}`117`) by [@kmuehlbauer](https://github.com/kmuehlbauer)
* FIX: robust ``angle_res`` retrieval in ``extract_angle_parameters`` ({issue}`112`), ({pull}`118`) by [@kmuehlbauer](https://github.com/kmuehlbauer)
* FIX: robust radar identifier in ``to_odim()`` ({pull}`120`) by [@kmuehlbauer](https://github.com/kmuehlbauer)

## 0.2.0 (2023-03-24)

* ENH: switch to add optional how attributes in ODIM format writer ({pull}`97`) by [@egouden](https://github.com/egouden)
* FIX: add keyword argument for mandatory source attribute in ODIM format writer ({pull}`96`) by [@egouden](https://github.com/egouden)
* FIX: check for dim0 if not given, only swap_dims if needed ({issue}`92`), ({pull}`94`) by [@kmuehlbauer](https://github.com/kmuehlbauer)
* FIX+ENH: need array copy before overwriting and make compression available in to_odim ({pull}`95`) by [@kmuehlbauer](https://github.com/kmuehlbauer)

## 0.1.0 (2023-02-23)

* Add an example on reading multiple sweeps into a single object ({pull}`69`) by [@mgrover1](https://github.com/mgrover1)
* ENH: add spatial_ref with pyproj when georeferencing, add/adapt methods/tests ({issue}`38`), ({pull}`87`) by [@kmuehlbauer](https://github.com/kmuehlbauer)
* Docs/docstring updates, PULL_REQUEST_TEMPLATE.md ({pull}`89`) by [@kmuehlbauer](https://github.com/kmuehlbauer)
* Finalize release 0.1.0, add testpypi upload on push to main ({pull}`91`) by [@kmuehlbauer](https://github.com/kmuehlbauer)

## 0.0.13 (2023-02-09)

* FIX: only skip setting cf-attributes if both gain and offset are unused ({pull}`85`) by [@kmuehlbauer](https://github.com/kmuehlbauer)

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
