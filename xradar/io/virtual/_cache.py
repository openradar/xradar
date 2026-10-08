#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""A small thread-safe cache for decoded byte spans.

Every moment of a sweep references the same byte span, so the codecs walk
a span once and serve the sweep's other moments from here. zarr decodes
chunks on several threads at once, so concurrent callers of one key wait
for the first caller's result instead of walking the span again.

Numpy-free and zarr-free, like the format walkers that use it.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable
from concurrent.futures import Future
from typing import Generic, TypeVar

__all__ = ["SingleFlightCache"]

V = TypeVar("V")


class SingleFlightCache(Generic[V]):
    """LRU cache, bounded by the total size of its values, that computes
    each key once even under concurrent callers.

    Parameters
    ----------
    max_bytes : int
        Budget for the summed ``sizeof`` of the kept values. The least
        recently used values are evicted to stay within it; a value larger
        than the whole budget is returned but not kept.
    sizeof : callable
        Size in bytes of one value.

    Failures are not cached: every waiter of a failed key gets the
    exception, and the next call computes it again.
    """

    def __init__(self, max_bytes: int, sizeof: Callable[[V], int]) -> None:
        self.max_bytes = max_bytes
        self._sizeof = sizeof
        self._lock = threading.Lock()
        self._done: OrderedDict[Hashable, tuple[V, int]] = OrderedDict()
        self._pending: dict[Hashable, Future] = {}
        self._nbytes = 0

    def get(self, key: Hashable, compute: Callable[[], V]) -> V:
        """The cached value of ``key``, computed by ``compute`` on a miss."""
        with self._lock:
            if key in self._done:
                self._done.move_to_end(key)
                return self._done[key][0]
            future = self._pending.get(key)
            owner = future is None
            if owner:
                future = self._pending[key] = Future()
        if not owner:
            return future.result()

        try:
            value = compute()
        except BaseException as exc:
            with self._lock:
                del self._pending[key]
            future.set_exception(exc)
            raise
        size = self._sizeof(value)
        with self._lock:
            del self._pending[key]
            if size <= self.max_bytes:
                self._done[key] = (value, size)
                self._nbytes += size
                while self._nbytes > self.max_bytes:
                    _, (_, evicted) = self._done.popitem(last=False)
                    self._nbytes -= evicted
        future.set_result(value)
        return value

    def clear(self) -> None:
        """Drop every kept value (in-flight computations still finish)."""
        with self._lock:
            self._done.clear()
            self._nbytes = 0

    def __len__(self) -> int:
        return len(self._done)

    @property
    def nbytes(self) -> int:
        """Summed size of the kept values."""
        return self._nbytes
