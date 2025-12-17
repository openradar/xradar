#!/usr/bin/env python
# Copyright (c) 2025, openradar developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import os
import sys

import nbformat
import pytest
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor


@pytest.mark.usefixtures("tmp_path")
def test_execution(notebook_path, tmp_path, request):
    # preserve environment
    old_cwd = os.getcwd()
    old_sys_path = sys.path.copy()

    # ensure project root is importable (coverage!)
    project_root = str(request.config.rootpath)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    os.chdir(tmp_path)

    try:
        # Windows asyncio workaround
        if sys.platform.startswith("win") and sys.version_info >= (3, 8):
            import asyncio

            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        with notebook_path.open(encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(
            kernel_name=f"python{sys.version_info[0]}",
            timeout=600,
        )

        ep.preprocess(
            nb,
            {"metadata": {"path": str(tmp_path)}},
        )

    except CellExecutionError as e:
        pytest.fail(str(e), pytrace=False)

    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_sys_path
