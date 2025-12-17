#!/usr/bin/env python
# Copyright (c) 2019-2022, wradlib developers.
# Copyright (c) 2022-2025, openradar developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

from pathlib import Path


def pytest_generate_tests(metafunc):
    if "notebook_path" not in metafunc.fixturenames:
        return

    notebooks = []

    for arg in metafunc.config.args:
        arg_path = Path(arg)
        if arg_path.is_dir():
            notebooks.extend(
                nb.resolve()
                for nb in arg_path.rglob("*.ipynb")
                if ".ipynb_checkpoints" not in nb.parts
            )
        elif arg_path.suffix == ".ipynb" and arg_path.exists():
            notebooks.append(arg_path.resolve())

    notebooks = sorted(set(notebooks))

    metafunc.parametrize(
        "notebook_path",
        notebooks,
        ids=[nb.stem for nb in notebooks],  # short, readable IDs
    )
