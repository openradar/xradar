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
    # Mock the version import to simulate an import error.
    with mock.patch("xradar.__init__.__version__", new="999"):
        importlib.reload(xradar)
        assert xradar.__version__ == "999"
