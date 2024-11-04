#!/usr/bin/env python
# Copyright (c) 2022-2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
XRadar Auto Reader
==================

This module provides the ability to automatically read radar files using all
available formats in the `xradar.io` module. It supports handling various file
types, georeferencing, and logging, as well as a timeout mechanism for file reading.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = [
    "read",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import logging
import signal

from .. import io  # noqa

# Setup a logger for this module
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARNING
)  # Default log level to suppress debug logs unless verbose is set


class TimeoutException(Exception):
    """
    Custom exception to handle file read timeouts.

    This exception is raised when the radar file reading process exceeds the
    specified timeout duration.
    """

    pass


def timeout_handler(signum, frame):
    """
    Timeout handler to raise a TimeoutException.

    This function is triggered by the alarm signal when the radar file reading
    exceeds the allowed timeout duration.

    Parameters
    ----------
    signum : int
        The signal number (in this case, SIGALRM).
    frame : frame object
        The current stack frame at the point where the signal was received.

    Raises
    ------
    TimeoutException
        If the reading operation takes too long and exceeds the timeout.
    """
    raise TimeoutException("Radar file reading timed out.")


def with_timeout(timeout):
    """
    Decorator to enforce a timeout on the file reading process.

    This decorator wraps the file reading function to ensure that if it takes longer than the
    specified `timeout` duration, it raises a `TimeoutException`.

    Parameters
    ----------
    timeout : int
        The maximum number of seconds allowed for the file reading process.

    Returns
    -------
    function
        A wrapped function that raises `TimeoutException` if it exceeds the timeout.
    """

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
    `xradar.io` module and attempts to open the provided radar file. If successful,
    it can optionally georeference the data (adding x, y, z coordinates) and log
    detailed processing information.

    Parameters
    ----------
    file : str or Path
        The path to the radar file to be read.
    georeference : bool, optional
        If True, georeference the radar data by adding x, y, z coordinates (default is True).
    verbose : bool, optional
        If True, prints out detailed processing information (default is False). When set to True,
        debug-level logs are enabled. When False, only warnings and errors are logged.
    timeout : int or None, optional
        Timeout in seconds for reading the radar file. If None, no timeout is applied (default is None).

    Returns
    -------
    dtree : DataTree
        A `DataTree` object containing the radar data.

    Raises
    ------
    ValueError
        If the file could not be opened by any supported format in `xradar.io`.
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
    If a `comment` attribute exists in the radar file's metadata, this function appends the
    radar type to the comment. The default comment is set to "im/exported using xradar".
    """
    # Configure logger level based on 'verbose'
    if verbose:
        logger.setLevel(logging.DEBUG)  # Enable debug messages when verbose is True
        logger.debug("Verbose mode activated.")
    else:
        logger.setLevel(
            logging.WARNING
        )  # Suppress debug messages when verbose is False

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

                        # Ensure the 'comment' key exists; if not, create it
                        if "comment" not in dtree.attrs:
                            logger.debug("Creating new 'comment' key.")
                            dtree.attrs["comment"] = "im/exported using xradar"
                        else:
                            logger.debug(
                                f"Existing 'comment': {dtree.attrs['comment']}"
                            )

                        # Append the key information to the comment without quotes
                        dtree.attrs["comment"] += f",\n{key.split('_')[1]}"

                        # Log the updated comment
                        logger.debug(f"After update: {dtree.attrs['comment']}")

                        if georeference:
                            logger.debug("Georeferencing radar data...")
                            dtree = dtree.xradar.georeference()

                        logger.debug(f"File opened successfully using {key}.")
                        break
                    except Exception as e:
                        logger.debug(f"Failed to open with {key}: {e}")
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

                if "comment" not in dtree.attrs:
                    logger.debug("Creating new 'comment' key.")
                    dtree.attrs["comment"] = "im/exported using xradar"
                else:
                    logger.debug(
                        f"Existing 'comment' before update: {dtree.attrs['comment']}"
                    )

                dtree.attrs["comment"] += f",\n{key.split('_')[1]}"

                if georeference:
                    logger.debug("Georeferencing radar data...")
                    dtree = dtree.xradar.georeference()

                logger.debug(f"File opened successfully using {key}.")
                break
            except Exception as e:
                logger.debug(f"Failed to open with {key}: {e}")
                continue

    if dtree is None:
        raise ValueError(
            "File could not be opened by any supported format in xradar.io."
        )

    return dtree
