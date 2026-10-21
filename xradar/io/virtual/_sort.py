#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""The one ray-sort rule shared by every virtual codec and store builder.

Numpy only: imported by the format walkers (codec read path) and exported
from :mod:`xradar.io.virtual` for store builders, without pulling in zarr or
virtualizarr.
"""

from __future__ import annotations

import numpy as np

__all__ = ["azimuth_sort_order"]


def azimuth_sort_order(azimuths) -> np.ndarray:
    """Stable permutation sorting rays by azimuth (NaN keys sort last).

    A store builder that writes ``sort_rays: true`` into a sweep codec
    config must order its per-ray coordinate values with this permutation:
    the codecs apply the identical one to the decoded rows, so coordinates
    and data stay aligned. Ties keep acquisition order (stable sort).
    """
    return np.argsort(np.asarray(azimuths, dtype=np.float64), kind="stable")
