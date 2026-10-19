import math

import pytest

from goldpass.stats import flip_margin, sign_test, wilson


@pytest.mark.parametrize(
    ("p", "n", "lo", "hi"),
    [
        (1.0, 20, 0.8389, 1.0),  # perfect on 20 questions: still only "at least ~0.84"
        (0.5, 20, 0.2993, 0.7007),
        (0.0, 4, 0.0, 0.4899),
        (0.9375, 16, 0.7167, 0.9889),  # the rail run's answered rate
        (0.5, 1000, 0.4691, 0.5309),
    ],
)
def test_wilson_matches_reference_values(p: float, n: int, lo: float, hi: float) -> None:
    got = wilson(p, n)
    assert got is not None
    assert got[0] == pytest.approx(lo, abs=1e-4) and got[1] == pytest.approx(hi, abs=1e-4)


def test_wilson_needs_items_and_stays_in_bounds() -> None:
    assert wilson(0.5, 0) is None
    for p in (0.0, 0.01, 0.5, 0.99, 1.0):
        lo, hi = wilson(p, 3) or (-1.0, -1.0)
        assert 0.0 <= lo <= p <= hi <= 1.0


def test_wilson_narrows_with_more_questions() -> None:
    widths = [hi - lo for lo, hi in (wilson(0.8, n) or (0, 0) for n in (10, 100, 1000))]
    assert widths == sorted(widths, reverse=True)


@pytest.mark.parametrize(
    ("worse", "better", "p"),
    [
        (0, 0, 1.0),  # nothing changed
        (1, 0, 0.5),  # one question got worse: a coin flip
        (2, 0, 0.25),
        (5, 0, 1 / 32),
        (5, 1, 7 / 64),  # P(X >= 5), X ~ Binomial(6, 1/2)
        (3, 3, 0.65625),
        (0, 4, 1.0),
    ],
)
def test_sign_test_is_exact_one_sided(worse: int, better: int, p: float) -> None:
    assert sign_test(worse, better) == pytest.approx(p)


def test_sign_test_large_counts_do_not_overflow() -> None:
    assert sign_test(600, 400) < 1e-9
    assert math.isfinite(sign_test(500, 500))


@pytest.mark.parametrize(
    ("value", "n", "minimum", "maximum", "flips"),
    [
        (0.95, 20, 0.9, None, 2),  # 19/20 passing 0.9: fails at 17/20, i.e. 2 more misses
        (0.9, 20, 0.9, None, 1),  # exactly at the line: one more miss fails it
        (0.85, 20, 0.9, None, 1),  # failing by one question
        (0.5, 20, 0.9, None, 8),
        (1.0, 4, 0.75, None, 2),  # refusal rate 4/4 vs 0.75: fails at 2/4
        (0.0, 20, None, 0, 1),  # errors: count metric with max 0 -> 1 error fails it
        (2.0, 20, None, 0, 2),  # 2 errors: fixing 2 passes it
        (0.0, 10, 1.0, None, 10),
    ],
)
def test_flip_margin(
    value: float, n: int, minimum: float | None, maximum: float | None, flips: int
) -> None:
    is_count = maximum is not None
    assert flip_margin(value, n, minimum, maximum, is_count=is_count) == flips


def test_flip_margin_without_items() -> None:
    assert flip_margin(0.5, 0, 0.9, None) is None
    assert flip_margin(None, 10, 0.9, None) is None
