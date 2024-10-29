import xarray as xr
from open_radar_data import DATASETS

import xradar as xd


def test_open_datatree_hpl():
    dtree = xd.io.open_hpl_datatree(
        DATASETS.fetch("User1_184_20240601_013257.hpl"),
        sweep=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        backend_kwargs=dict(latitude=41.24276244459537, longitude=-70.1070364814594),
    )
    assert "/sweep_0" in list(dtree.groups)
    assert dtree["sweep_0"]["mean_doppler_velocity"].dims == ("azimuth", "range")
    assert dtree["sweep_0"]["mean_doppler_velocity"].max() == 19.5306


def test_open_dataset_hpl():
    with xr.open_dataset(
        DATASETS.fetch("User1_184_20240601_013257.hpl"),
        engine="hpl",
        backend_kwargs=dict(latitude=41.24276244459537, longitude=-70.1070364814594),
    ) as ds:

        assert ds["mean_doppler_velocity"].dims == ("azimuth", "range")
        assert ds["mean_doppler_velocity"].max() == 19.5306
