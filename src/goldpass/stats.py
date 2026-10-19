"""How much a golden set of 20 questions can and cannot tell you.

With n questions, one question moves a rate by 1/n: 0.05 at n = 20, 0.25 for 4 refusal
questions. These helpers put that next to every number the gate uses.

- `wilson`: 95% Wilson score interval. Exact (in the binomial sense) for yes/no metrics such as
  grounding or refusal rate. For metrics that are means of per-question scores in [0, 1]
  (recall, faithfulness, ...) it is conservative, because such a mean's variance is at most
  p(1 - p) / n, the binomial variance it assumes.
- `sign_test`: exact one-sided paired sign test for "did this run get worse than the baseline
  on the same questions?" (McNemar's exact test when the metric is yes/no).
- `flip_margin`: how few questions would have to change for a threshold decision to flip.

Standard library only; deterministic.
"""

from __future__ import annotations

import math
from math import comb

Z95 = 1.959963984540054  # two-sided 95% normal quantile


def wilson(p: float, n: int, z: float = Z95) -> tuple[float, float] | None:
    """Wilson score interval for a proportion p observed on n items; None without items."""
    if n <= 0:
        return None
    p = min(max(p, 0.0), 1.0)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # The interval always contains p; clamping keeps float noise from breaking that at 0 and 1.
    return max(0.0, min(centre - half, p)), min(1.0, max(centre + half, p))


def sign_test(worse: int, better: int) -> float:
    """P(at least `worse` of the changed questions got worse | no real change).

    Questions whose score did not change carry no information and are left out. Under the null
    hypothesis each changed question is equally likely to have moved either way.
    """
    m = worse + better
    if m == 0 or worse == 0:
        return 1.0
    tail: int = sum(comb(m, k) for k in range(worse, m + 1))
    return tail / (1 << m)  # 2**m, typed as int


def flip_margin(
    value: float | None,
    n: int,
    minimum: float | None,
    maximum: float | None,
    is_count: bool = False,
) -> int | None:
    """The fewest questions that would have to change for this threshold decision to flip.

    For a passing check: how many more questions must fail before it fails. For a failing one:
    how many must be fixed before it passes. A question moves a rate by at most 1/n, so for
    scores that are not yes/no this is a lower bound ("as few as k questions"). Count metrics
    (`errors`) move by 1 per question.
    """
    if value is None or n <= 0:
        return None
    scale = 1 if is_count else n
    eps = 5e-5 * scale + 1e-9  # values are rounded to 4 decimals

    def margin(gap: float) -> int:  # gap > 0: room before failing; gap < 0: shortfall
        x = gap * scale
        return math.floor(x + eps) + 1 if x >= -eps else math.ceil(-x - eps)

    gaps = []
    if minimum is not None:
        gaps.append(value - minimum)
    if maximum is not None:
        gaps.append(maximum - value)
    if not gaps:
        return None
    failing = [g for g in gaps if g * scale < -eps]
    if failing:
        return max(margin(g) for g in failing)
    return min(margin(g) for g in gaps)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pairs of a paired yes/no outcome
    (b: only the first judge missed, c: only the second missed)."""
    n = b + c
    if n == 0:
        return 1.0
    tail: int = sum(comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2 * tail / (1 << n))
