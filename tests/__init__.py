#!/usr/bin/env python
# Copyright (c) 2022-2025, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Unit test package for xradar."""

import importlib

import pytest


def skip_import(name):
    try:
        importlib.import_module(name)
        found = True
    except ImportError:
        found = False

    return pytest.mark.skipif(not found, reason=f"requires {name}")
