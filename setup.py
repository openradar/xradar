#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("docs/history.md") as history_file:
    history = history_file.read()

with open("requirements.txt", "r") as f:
    INSTALL_REQUIRES = [rq for rq in f.read().split("\n") if rq != ""]

with open("requirements_dev.txt", "r") as f:
    DEVEL_REQUIRES = [rq for rq in f.read().split("\n") if rq != ""]

test_requirements = [
    "pytest>=3",
]

setup(
    author=["Maxwell Grover, Kai Muehlbauer, Zachary Sherman"],
    author_email="mgrover@anl.gov",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    description="Xradar includes all the tools to get your weather radar into the xarray data model.",
    install_requires=INSTALL_REQUIRES,
    extras_require=dict(dev=DEVEL_REQUIRES),
    license="MIT license",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="xradar",
    name="xradar",
    packages=find_packages(include=["xradar", "xradar.*"]),
    entry_points={
        "xarray.backends": [
            "cfradial1 = xradar.io.backends:CfRadial1BackendEntrypoint",
            "gamic = xradar.io.backends:GamicBackendEntrypoint",
            "odim = xradar.io.backends:OdimBackendEntrypoint",
            "furuno = xradar.io.backends:FurunoBackendEntrypoint",
            "rainbow = xradar.io.backends:RainbowBackendEntrypoint",
            "iris = xradar.io.backends:IrisBackendEntrypoint",
        ]
    },
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/openradar/xradar",
    zip_safe=False,
)
