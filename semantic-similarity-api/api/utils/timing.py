"""Timing helper for measuring inference latency."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, Iterator, List


@contextmanager
def measure_ms() -> Iterator[Callable[[], float]]:
    """Context manager yielding a callable that returns elapsed milliseconds.

    Usage::

        with measure_ms() as elapsed:
            do_work()
        latency = elapsed()
    """
    start = time.perf_counter()
    result: List[float] = []

    def elapsed() -> float:
        # After the block exits, ``result`` holds the frozen duration; while
        # inside the block it reflects the current elapsed time.
        if result:
            return result[0]
        return (time.perf_counter() - start) * 1000.0

    try:
        yield elapsed
    finally:
        result.append((time.perf_counter() - start) * 1000.0)
