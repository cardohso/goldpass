"""Run a function over items, optionally on several threads, returning results in input order.

With one worker it is a plain loop, so default behaviour is exactly sequential. Threads suit
goldpass: the work is waiting on HTTP APIs, and the shared RateLimiter keeps the request rate
within what each API allows however many workers there are.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def ordered_map(fn: Callable[[T], R], items: Sequence[T], workers: int = 1) -> list[R]:
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
        return list(pool.map(fn, items))
