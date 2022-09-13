#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("docs/history.md") as history_file:
    history = history_file.read()

requirements = []

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
    ],
    description="Xradar includes all the tools to get your weather radar into the xarray data model.",
    install_requires=requirements,
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
            "odim = xradar.io.backends:OdimBackendEntrypoint",
        ]
    },
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/openradar/xradar",
    zip_safe=False,
)
