"""
xradar
======

Top-level package for xradar.

"""

__author__ = """Maxwell Grover"""
__email__ = "mgroverwx@gmail.com"

# versioning
try:
    from .version import version as __version__
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"

# import subpackages
from . import io  # noqa
from . import model  # noqa
from . import util  # noqa

__all__ = [s for s in dir() if not s.startswith("_")]
