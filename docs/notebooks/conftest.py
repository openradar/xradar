#!/usr/bin/env python
# Copyright (c) 2019-2022, wradlib developers.
# Copyright (c) 2022-2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

import os
import sys
from pathlib import Path

import jupytext
import nbformat
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def pytest_configure(config):
    # prevent adding DataTree.ds by accident, turn warning into error
    config.addinivalue_line(
        "filterwarnings",
        "error:DataTree.ds is deprecated, use DataTree.dataset instead.:FutureWarning",
    )

def pytest_collect_file(parent, file_path):
    if file_path.suffix == ".md" and "-Copy" not in file_path.name:
        return NotebookFile.from_parent(parent=parent, path=file_path)


class NotebookFile(pytest.File):
    def collect(self):
        yield NotebookItem.from_parent(parent=self, name=self.path.name)


class NotebookItem(pytest.Item):
    def runtest(self):
        old_cwd = os.getcwd()
        old_sys_path = sys.path.copy()
        old_env = os.environ.copy()

        try:
            # ensure project root is importable
            project_root = Path(self.config.rootpath)
            # fix PYTHONPATH and COVERAGE_RCFILE  in environment
            existing = os.environ.get("PYTHONPATH")
            os.environ["PYTHONPATH"] = str(project_root) + (
                os.pathsep + existing if existing else ""
            )
            os.environ["COVERAGE_RCFILE"] = str(project_root / "pyproject.toml")

            # run in temp dir
            tmp_path = Path(self.config._tmp_path_factory.mktemp("notebook"))
            os.chdir(tmp_path)

            if self.parent.path.suffix == ".md":
                nb = jupytext.read(self.parent.path)

            if not isinstance(nb, nbformat.NotebookNode):
                nb = nbformat.from_dict(nb)

            # ensure kernel metadata
            nb.metadata.setdefault(
                "kernelspec",
                {
                    "name": "python3",
                    "display_name": "Python 3",
                    "language": "python",
                },
            )

            # clear outputs
            for cell in nb.cells:
                if cell.cell_type == "code":
                    cell.outputs = []
                    cell.execution_count = None

            client = NotebookClient(
                nb,
                kernel_name="python3",
                timeout=600,
                iopub_timeout=600,
                allow_errors=False,
                resources={
                    "metadata": {
                        "path": str(tmp_path),
                    }
                },
            )

            try:
                client.execute()
            except CellExecutionError as e:
                raise NotebookException(e) from e

            out_path = self.parent.path.with_suffix(".ipynb")

            out_path_parts = tuple(
                "render" if part == "notebooks" else part for part in out_path.parts
            )
            out_path = Path(*out_path_parts)

            out_path.parent.mkdir(parents=True, exist_ok=True)

            with open(out_path, "w", encoding="utf-8") as f:
                nbformat.write(nb, f)

        finally:
            # restore the original process state
            os.chdir(old_cwd)
            sys.path = old_sys_path
            os.environ.clear()
            os.environ.update(old_env)

    def repr_failure(self, excinfo):
        if isinstance(excinfo.value, NotebookException):
            return excinfo.exconly()
        return super().repr_failure(excinfo)

    def reportinfo(self):
        return self.parent.path, 0, f"Notebook: {self.name}"


class NotebookException(Exception):
    pass
