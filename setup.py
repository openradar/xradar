from Cython.Build import cythonize
from Cython.Compiler import Options
from setuptools import Extension, setup

# These are optional
Options.docstrings = True
Options.annotate = False

# Modules to be compiled and include_dirs when necessary
extensions = [
    Extension(
        "xradar.interpolate._nexrad_interpolate",
        sources=["xradar/interpolate/_nexrad_interpolate.pyx"],
    ),
]


# This is the function that is executed
setup(
    # external to be compiled
    ext_modules=cythonize(
        extensions, compiler_directives={"language_level": "3", "cpow": True}
    ),
)
