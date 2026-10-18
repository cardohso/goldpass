from datetime import UTC, datetime

import pytest

from goldpass.retry import GaveUp, RetryableError, RetryPolicy, call_with_retry, parse_retry_after

NOW = datetime(2026, 10, 19, 12, 0, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("7", 7.0),
        (" 0 ", 0.0),
        ("Mon, 19 Oct 2026 12:00:30 GMT", 30.0),
        ("Mon, 19 Oct 2026 11:00:00 GMT", 0.0),  # in the past: retry now
        ("soon", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_retry_after(header: str | None, expected: float | None) -> None:
    assert parse_retry_after(header, now=NOW) == expected


class Flaky:
    def __init__(self, failures: list[RetryableError]):
        self.failures = failures
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return "ok"


def test_backoff_doubles_until_success() -> None:
    waits: list[float] = []
    fn = Flaky([RetryableError("429"), RetryableError("429"), RetryableError("503")])
    assert call_with_retry(fn, RetryPolicy(attempts=5, backoff_s=1.0), sleep=waits.append) == "ok"
    assert waits == [1.0, 2.0, 4.0]
    assert fn.calls == 4


def test_retry_after_wins_over_backoff_and_is_capped() -> None:
    waits: list[float] = []
    fn = Flaky([RetryableError("429", retry_after_s=9.0), RetryableError("429", 999.0)])
    call_with_retry(fn, RetryPolicy(max_wait_s=60.0), sleep=waits.append)
    assert waits == [9.0, 60.0]


def test_gives_up_after_attempts() -> None:
    waits: list[float] = []
    fn = Flaky([RetryableError(f"503 #{i}") for i in range(10)])
    with pytest.raises(GaveUp, match="gave up after 3 attempts: 503 #2"):
        call_with_retry(fn, RetryPolicy(attempts=3), sleep=waits.append)
    assert fn.calls == 3 and len(waits) == 2


def test_other_errors_are_not_retried() -> None:
    def boom() -> str:
        raise KeyError("x")

    with pytest.raises(KeyError):
        call_with_retry(boom, RetryPolicy(), sleep=lambda s: None)
