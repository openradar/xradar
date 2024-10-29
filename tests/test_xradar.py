#!/usr/bin/env python
# Copyright (c) 2022, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for `xradar` package."""
import importlib
from unittest import mock

import pytest

import xradar


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_version_import_fallback():
    # Mock import of `version` to raise ImportError within xradar's __init__.py
    with mock.patch("importlib.import_module", side_effect=ImportError):
        importlib.reload(xradar)
        assert xradar.__version__ == "999"
