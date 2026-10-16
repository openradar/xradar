#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for `io.virtual._cache` (the per-span walk cache)."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from xradar.io.virtual._cache import SingleFlightCache


def test_computes_once_and_serves_hits():
    calls = []
    cache = SingleFlightCache(max_bytes=100, sizeof=len)
    for _ in range(3):
        assert cache.get("a", lambda: calls.append(1) or "value") == "value"
    assert calls == [1]
    assert len(cache) == 1 and cache.nbytes == 5


def test_evicts_least_recently_used_to_stay_within_budget():
    cache = SingleFlightCache(max_bytes=10, sizeof=len)
    cache.get("a", lambda: "aaaa")
    cache.get("b", lambda: "bbbb")
    cache.get("a", lambda: "never")  # a is now the most recent
    cache.get("c", lambda: "cccc")  # 12 bytes: b, the oldest, goes
    assert cache.get("a", lambda: "never") == "aaaa"
    assert cache.get("c", lambda: "never") == "cccc"
    assert cache.get("b", lambda: "BBBB") == "BBBB"  # recomputed
    assert cache.nbytes <= 10


def test_value_larger_than_the_budget_is_returned_not_kept():
    cache = SingleFlightCache(max_bytes=3, sizeof=len)
    assert cache.get("a", lambda: "toolong") == "toolong"
    assert len(cache) == 0 and cache.nbytes == 0


def test_failures_are_not_cached():
    cache = SingleFlightCache(max_bytes=100, sizeof=len)

    def fail():
        raise ValueError("corrupt span")

    with pytest.raises(ValueError, match="corrupt span"):
        cache.get("a", fail)
    assert cache.get("a", lambda: "ok") == "ok"


def test_concurrent_callers_share_one_computation():
    """zarr decodes a sweep's moments on several threads at once: the
    first caller computes, the others wait for its result."""
    calls = []
    release = threading.Event()

    def slow():
        calls.append(1)
        release.wait(5)
        return "value"

    cache = SingleFlightCache(max_bytes=100, sizeof=len)
    with ThreadPoolExecutor(8) as pool:
        futures = [pool.submit(cache.get, "a", slow) for _ in range(8)]
        release.set()
        assert {f.result() for f in futures} == {"value"}
    assert calls == [1]


def test_every_caller_gets_the_failure():
    release = threading.Event()

    def fail():
        release.wait(5)
        raise ValueError("corrupt span")

    cache = SingleFlightCache(max_bytes=100, sizeof=len)
    with ThreadPoolExecutor(4) as pool:
        futures = [pool.submit(cache.get, "a", fail) for _ in range(4)]
        release.set()
        for future in futures:
            with pytest.raises(ValueError, match="corrupt span"):
                future.result()
    assert len(cache) == 0


def test_a_failing_sizeof_does_not_wedge_the_key():
    """If sizing the value fails, the key is released: the next call
    computes again instead of waiting forever."""
    sizes = iter([ValueError("boom"), 2])

    def sizeof(value):
        size = next(sizes)
        if isinstance(size, Exception):
            raise size
        return size

    cache = SingleFlightCache(max_bytes=100, sizeof=sizeof)
    with pytest.raises(ValueError, match="boom"):
        cache.get("a", lambda: "v1")
    assert cache.get("a", lambda: "v2") == "v2"
