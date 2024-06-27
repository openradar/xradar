import tarfile
import numpy as np
import pytest 

from xradar.io.backends import datamet
from xradar.util import _get_data_file

@pytest.fixture
def data(datamet_file):
    with _get_data_file(datamet_file, 'file') as datametfile:
        data = datamet.DataMetFile(datametfile)
        assert data.filename == datametfile
    data.get_sweep(1)
    return data

def test_file_tyes(data):
    assert isinstance(data, datamet.DataMetFile)
    assert isinstance(data.tarfile, tarfile.TarFile)

def test_basic_content(data):
    assert data.moments == ['UZ', 'CZ', 'V', 'W', 'ZDR', 'PHIDP', 'RHOHV', 'KDP']
    assert len(data.data[1]) == len(data.moments)
    assert data.first_dimension == "azimuth"
    assert data.scan_metadata['origin'] == 'ILMONTE'
    assert data.scan_metadata['orig_lat'] == 41.9394
    assert data.scan_metadata['orig_lon'] == 14.6208
    assert data.scan_metadata['orig_alt'] == 710

def test_moment_metadata(data):
    mom_metadata = data.get_mom_metadata('UZ', 1)
    assert mom_metadata['Rangeoff'] == 0.0
    assert mom_metadata['Eloff'] == 16.05
    assert mom_metadata['nlines'] == 360
    assert mom_metadata['ncols'] == 493

@pytest.mark.parametrize(
    "moment, expected_value",
    [
        ('UZ', -3.5),
        ('CZ', -3.5),
        ('V', 2.3344999999999985),
        ('W', 16.0),
        ('ZDR', 0.6859999999999999),
        ('PHIDP', 94.06648),
        ('RHOHV', 1.9243000000000001),
        ('KDP', 0.5190000000000001)
    ]
)
def test_moment_data(data, moment, expected_value):
    assert data.data[1][moment][(4, 107)] == expected_value