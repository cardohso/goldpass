"""Retries with exponential backoff that honour `Retry-After`. Free tiers rate-limit hard, so a
429 is expected, not exceptional. No jitter: goldpass makes one call at a time."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 5  # total tries, including the first
    backoff_s: float = 1.0  # first wait; doubles after each failure
    max_wait_s: float = 120.0  # cap on any single wait, including a server's Retry-After


class RetryableError(Exception):
    def __init__(self, message: str, retry_after_s: float | None = None):
        super().__init__(message)
        self.retry_after_s = retry_after_s


class GaveUp(Exception):
    def __init__(self, attempts: int, last: RetryableError):
        super().__init__(f"gave up after {attempts} attempts: {last}")
        self.last = last


def parse_retry_after(value: str | None, now: datetime | None = None) -> float | None:
    """Seconds to wait from a Retry-After header: either delta-seconds or an HTTP date."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - (now or datetime.now(UTC))).total_seconds())


def call_with_retry(
    fn: Callable[[], T],
    policy: RetryPolicy,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[int, float, RetryableError], None] | None = None,
) -> T:
    """Call fn until it returns or raises something other than RetryableError."""
    for attempt in range(1, policy.attempts + 1):
        try:
            return fn()
        except RetryableError as e:
            if attempt == policy.attempts:
                raise GaveUp(attempt, e) from e
            wait = e.retry_after_s
            if wait is None:
                wait = policy.backoff_s * 2 ** (attempt - 1)
            wait = min(wait, policy.max_wait_s)
            if on_retry:
                on_retry(attempt, wait, e)
            sleep(wait)
    raise AssertionError("unreachable")  # attempts < 1
