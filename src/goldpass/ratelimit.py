"""One shared pace for every worker calling the same API.

`wait()` hands each caller its own start slot at least `min_interval_s` after the previous one,
so N threads together never exceed 1 / min_interval_s requests per second. `defer()` is for
`Retry-After`: when one worker is told to back off, all of them are, not just that one.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class RateLimiter:
    def __init__(
        self,
        min_interval_s: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.min_interval_s = min_interval_s
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next = float("-inf")  # earliest start time of the next call

    def wait(self) -> None:
        """Block until this caller's slot, and reserve the one after it."""
        with self._lock:
            now = self._clock()
            start = max(now, self._next)
            self._next = start + self.min_interval_s
        if start > now:
            self._sleep(start - now)

    def defer(self, seconds: float) -> None:
        """Nobody starts a new call for `seconds` (a longer existing deferral wins)."""
        with self._lock:
            self._next = max(self._next, self._clock() + seconds)
