#!/usr/bin/env python
# Copyright (c) 2022-2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
XRadar Auto Reader
==================

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = [
    "read",
]

__doc__ = __doc__.format("\n   ".join(__all__))

from .. import io  # noqa
import signal


# Custom exception for handling timeouts
class TimeoutException(Exception):
    pass


# Timeout handler
def timeout_handler(signum, frame):
    raise TimeoutException("Radar file reading timed out.")


# Decorator for handling timeouts
def with_timeout(timeout):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Set the signal handler and a timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            try:
                return func(*args, **kwargs)
            finally:
                # Disable the alarm
                signal.alarm(0)

        return wrapper

    return decorator


def read(file, georeference=True, verbose=False, timeout=None):
    """
    Attempt to read a radar file using all available formats in xradar.io.

    This function iterates over all the available file-opening functions in the
    xradar.io module and attempts to open the provided radar file. If successful,
    it can optionally georeference the data (adding x, y, z coordinates).

    Parameters
    ----------
    file : str or Path
        The path to the radar file to be read.
    georeference : bool, optional
        If True, georeference the radar data by adding x, y, z coordinates (default is True).
    verbose : bool, optional
        If True, prints out detailed processing information (default is False).
    timeout : int or None, optional
        Timeout in seconds for reading the radar file. If None, no timeout is applied (default is None).

    Returns
    -------
    dtree : DataTree
        A DataTree object containing the radar data.

    Raises
    ------
    ValueError
        If the file could not be opened by any supported format in xradar.io.
    TimeoutException
        If reading the file takes longer than the specified timeout.

    Examples
    --------
    >>> file = DATASETS.fetch('KATX20130717_195021_V06')
    >>> dtree = xd.io.read(file, verbose=True, timeout=10)
    Georeferencing radar data...
    File opened successfully using open_nexrad_archive.

    Notes
    -----
    This function relies on the `xradar` library to support various radar file formats.
    It tries to open the file using all available `open_` functions in the `xradar.io` module.
    """
    dtree = None

    # Wrap the read process with a timeout if specified
    if timeout:

        @with_timeout(timeout)
        def attempt_read(file):
            nonlocal dtree
            for key in io.__all__:
                if "open_" in key:
                    open_func = getattr(io, key)
                    try:
                        dtree = open_func(file)
                        if georeference:
                            if verbose:
                                print("Georeferencing radar data...")
                            dtree = dtree.xradar.georeference()
                        if verbose:
                            print(f"File opened successfully using {key}.")
                        break
                    except Exception as e:
                        if verbose:
                            print(f"Failed to open with {key}: {e}")
                        continue

            if dtree is None:
                raise ValueError(
                    "File could not be opened by any supported format in xradar.io."
                )
            return dtree

        return attempt_read(file)

    # Normal read process without timeout
    for key in io.__all__:
        if "open_" in key:
            open_func = getattr(io, key)
            try:
                dtree = open_func(file)
                if georeference:
                    if verbose:
                        print("Georeferencing radar data...")
                    dtree = dtree.xradar.georeference()
                if verbose:
                    print(f"File opened successfully using {key}.")
                break
            except Exception as e:
                if verbose:
                    print(f"Failed to open with {key}: {e}")
                continue

    if dtree is None:
        raise ValueError(
            "File could not be opened by any supported format in xradar.io."
        )

    return dtree
