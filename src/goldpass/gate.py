"""Pass/fail: absolute thresholds, and regressions against a saved baseline run.

thresholds.yaml:

    metrics:
      recall_at_k:  {min: 0.70}
      grounding:    {min: 0.90, min_n: 10}   # and measured on at least 10 items
      errors:       {max: 0}
    tags:
      wrong-document:
        grounding:  {min: 0.90}
    baseline:
      max_drop: 0.05            # any metric may fall at most this much below the baseline
      metrics: {grounding: 0}   # per-metric override
      alpha: 0.05               # optional: a drop must also be significant to fail

Every check carries a 95% interval and a flip margin (how few questions would change the
decision); every baseline delta carries the questions that got worse and better and an exact
sign-test p-value (see goldpass.stats).

Gating works on results dicts (the JSON `goldpass run` writes), so `goldpass compare` can
re-gate saved runs without calling anything.
"""

from __future__ import annotations

import difflib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from goldpass.config import ConfigError
from goldpass.judging import JUDGE_METRICS
from goldpass.metrics import METRICS
from goldpass.stats import flip_margin, sign_test, wilson

WORST = 5  # questions listed per failing check

# names people bring from RAGAS or the rail repo, mapped to goldpass's
ALIASES = {
    "context_recall": "recall_at_k",
    "retrieval_recall_at_k": "retrieved_recall",
    "refusal": "refusal_rate",
    "answered": "answered_rate",
}


@dataclass(frozen=True)
class Rule:
    min: float | None = None
    max: float | None = None
    min_n: int | None = None

    def __str__(self) -> str:
        parts = []
        if self.min is not None:
            parts.append(f"≥ {self.min:g}")
        if self.max is not None:
            parts.append(f"≤ {self.max:g}")
        if self.min_n is not None:
            parts.append(f"n ≥ {self.min_n}")
        return ", ".join(parts)


@dataclass(frozen=True)
class Thresholds:
    metrics: Mapping[str, Rule] = field(default_factory=dict)
    tags: Mapping[str, Mapping[str, Rule]] = field(default_factory=dict)
    max_drop: float = 0.05
    max_drop_by_metric: Mapping[str, float] = field(default_factory=dict)
    alpha: float | None = None  # if set, a drop past max_drop fails only when p <= alpha

    def allowed_drop(self, metric: str) -> float:
        return self.max_drop_by_metric.get(metric, self.max_drop)


def _metric_name(name: Any, where: str) -> str:
    if name in METRICS:
        return str(name)
    close = (
        [ALIASES[name]]
        if name in ALIASES
        else difflib.get_close_matches(str(name), list(METRICS), n=1)
    )
    hint = f"; did you mean '{close[0]}'?" if close else f" (known: {', '.join(METRICS)})"
    raise ConfigError(f"{where}: unknown metric '{name}'{hint}")


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{where}: expected a number, got {type(value).__name__}")
    return float(value)


def _rule(data: Any, where: str) -> Rule:
    if not isinstance(data, dict) or not data or set(data) - {"min", "max", "min_n"}:
        raise ConfigError(f"{where}: expected a mapping with min, max and/or min_n")
    n = data.get("min_n")
    if n is not None and (isinstance(n, bool) or not isinstance(n, int) or n < 1):
        raise ConfigError(f"{where}.min_n: expected a positive integer")
    return Rule(
        min=_number(data["min"], f"{where}.min") if "min" in data else None,
        max=_number(data["max"], f"{where}.max") if "max" in data else None,
        min_n=n,
    )


def _rules(data: Any, where: str) -> dict[str, Rule]:
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping of metric -> rule")
    return {_metric_name(k, where): _rule(v, f"{where}.{k}") for k, v in data.items()}


def parse_thresholds(data: Any, where: str = "thresholds") -> Thresholds:
    if data is None:
        data = {}
    if not isinstance(data, dict) or set(data) - {"metrics", "tags", "baseline"}:
        raise ConfigError(f"{where}: expected keys metrics, tags and/or baseline")
    tags = data.get("tags") or {}
    if not isinstance(tags, dict):
        raise ConfigError(f"{where}.tags: expected a mapping of tag -> metrics")
    base = data.get("baseline") or {}
    if not isinstance(base, dict) or set(base) - {"max_drop", "metrics", "alpha"}:
        raise ConfigError(f"{where}.baseline: expected keys max_drop, metrics and/or alpha")
    alpha = None
    if base.get("alpha") is not None:
        alpha = _number(base["alpha"], f"{where}.baseline.alpha")
        if not 0 < alpha < 1:
            raise ConfigError(f"{where}.baseline.alpha: expected a number between 0 and 1")
    overrides = base.get("metrics") or {}
    if not isinstance(overrides, dict):
        raise ConfigError(f"{where}.baseline.metrics: expected a mapping of metric -> max drop")
    return Thresholds(
        metrics=_rules(data.get("metrics"), f"{where}.metrics"),
        tags={str(t): _rules(r, f"{where}.tags.{t}") for t, r in tags.items()},
        max_drop=_number(base.get("max_drop", 0.05), f"{where}.baseline.max_drop"),
        max_drop_by_metric={
            _metric_name(k, f"{where}.baseline.metrics"): _number(
                v, f"{where}.baseline.metrics.{k}"
            )
            for k, v in overrides.items()
        },
        alpha=alpha,
    )


def load_thresholds(path: Path) -> Thresholds:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except OSError as e:
        raise ConfigError(f"{path}: cannot read: {e}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: invalid YAML: {e}") from e
    return parse_thresholds(data, str(path))


# absolute thresholds


@dataclass(frozen=True)
class Check:
    metric: str
    scope: str  # "all" or "tag:<name>"
    value: float | None
    n: int
    rule: Rule
    ok: bool
    message: str
    worst: tuple[tuple[str, float], ...] = ()  # (item id, value), worst first
    ci: tuple[float, float] | None = None  # 95% Wilson interval for the value
    margin: int | None = None  # fewest questions whose change would flip `ok`


def _item_values(
    result: Mapping[str, Any], metric: str, tag: str | None
) -> list[tuple[str, float]]:
    out = []
    for it in result["items"]:
        if tag is not None and tag not in it.get("tags", ()):
            continue
        if metric == "errors":
            if it.get("error"):
                out.append((it["id"], 1.0))
            continue
        v = it["metrics"].get(metric)
        if v is not None:
            out.append((it["id"], float(v)))
    return out


def _worst(
    result: Mapping[str, Any], metric: str, tag: str | None
) -> tuple[tuple[str, float], ...]:
    values = _item_values(result, metric, tag)
    if METRICS[metric].higher_is_better:
        bad = sorted((x for x in values if x[1] < 1.0), key=lambda x: (x[1], x[0]))
    else:
        bad = sorted((x for x in values if x[1] > 0.0), key=lambda x: (-x[1], x[0]))
    return tuple(bad[:WORST])


def _check(result: Mapping[str, Any], agg: Mapping[str, Any], metric: str, rule: Rule,
           tag: str | None) -> Check:  # fmt: skip
    value, n = agg.get("value"), int(agg.get("n", 0))
    scope = "all" if tag is None else f"tag:{tag}"
    problems = []
    if value is None:
        problems.append("no value (nothing could be scored)")
    else:
        if rule.min is not None and value < rule.min:
            problems.append(f"{value:.3f} < {rule.min:g}")
        if rule.max is not None and value > rule.max:
            problems.append(f"{value:g} > {rule.max:g}")
    if rule.min_n is not None and n < rule.min_n:
        problems.append(f"measured on {n} items, needs {rule.min_n}")
    ok = not problems
    message = "ok" if ok else "; ".join(problems)
    worst = () if ok else _worst(result, metric, tag)
    is_count = metric == "errors"
    ci = None if is_count or value is None else wilson(value, n)
    scored = len(result["items"]) if is_count else n
    margin = (
        None
        if rule.min_n is not None and n < rule.min_n
        else (flip_margin(value, scored, rule.min, rule.max, is_count=is_count))
    )
    return Check(metric, scope, value, n, rule, ok, message, worst, ci, margin)


def check_thresholds(result: Mapping[str, Any], th: Thresholds) -> list[Check]:
    checks = [
        _check(result, result["metrics"].get(m, {}), m, rule, None)
        for m, rule in th.metrics.items()
    ]
    for tag, rules in th.tags.items():
        tagged = result.get("by_tag", {}).get(tag, {})
        checks += [_check(result, tagged.get(m, {}), m, rule, tag) for m, rule in rules.items()]
    return checks


# baseline comparison


@dataclass(frozen=True)
class Delta:
    metric: str
    baseline: float
    current: float
    allowed_drop: float
    ok: bool
    regressed: tuple[tuple[str, float, float], ...] = ()  # (item id, before, after)
    worse: int = 0  # questions present in both runs whose score got worse
    better: int = 0
    p_value: float = 1.0  # exact one-sided sign test: P(this many worse | no real change)
    within_noise: bool = False  # dropped past the limit, but not significantly (alpha)

    @property
    def change(self) -> float:
        return self.current - self.baseline

    @property
    def drop(self) -> float:
        """How much worse, in the metric's own direction (negative means it improved)."""
        return -self.change if METRICS[self.metric].higher_is_better else self.change


@dataclass(frozen=True)
class Comparison:
    deltas: tuple[Delta, ...]
    warnings: tuple[str, ...]
    alpha: float | None = None

    @property
    def ok(self) -> bool:
        return all(d.ok for d in self.deltas)


def _gating_judge(result: Mapping[str, Any]) -> str | None:
    judges = result.get("judges") or []
    return next((j["name"] for j in judges if j.get("gating")), None)


def _item_score(item: Mapping[str, Any], metric: str) -> float | None:
    """One question's score; errors count as 1 (errored) or 0."""
    if metric == "errors":
        return 1.0 if item.get("error") else 0.0
    raw = item.get("metrics", {}).get(metric)
    return None if raw is None else float(raw)


def compare_baseline(
    result: Mapping[str, Any], baseline: Mapping[str, Any], th: Thresholds | None = None
) -> Comparison:
    th = th or Thresholds()
    warnings = []
    ids_now = {it["id"] for it in result["items"]}
    ids_then = {it["id"] for it in baseline["items"]}
    if ids_now != ids_then:
        warnings.append(
            f"golden sets differ ({len(ids_now - ids_then)} items added, "
            f"{len(ids_then - ids_now)} removed): corpus metrics are not strictly comparable"
        )
    k_now, k_then = result["meta"].get("k"), baseline["meta"].get("k")
    if k_now != k_then:
        warnings.append(f"k changed from {k_then} to {k_now}")
    skip: set[str] = set()
    j_now, j_then = _gating_judge(result), _gating_judge(baseline)
    if j_now != j_then:
        skip = set(JUDGE_METRICS)
        if j_now and j_then:
            warnings.append(
                f"baseline was judged by {j_then}, this run by {j_now}: judge metrics not compared"
            )

    before = {it["id"]: it for it in baseline["items"]}
    deltas = []
    for metric, info in METRICS.items():
        if metric in skip:
            continue
        now = result["metrics"].get(metric, {}).get("value")
        then = baseline["metrics"].get(metric, {}).get("value")
        if now is None or then is None:
            continue
        pairs = []  # (item id, before, after) for questions scored in both runs
        for it in result["items"]:
            prev = before.get(it["id"])
            b = None if prev is None else _item_score(prev, metric)
            a = _item_score(it, metric)
            if a is not None and b is not None:
                pairs.append((it["id"], b, a))
        sign = 1 if info.higher_is_better else -1
        worse = [x for x in pairs if sign * (x[2] - x[1]) < -1e-9]
        better = sum(sign * (x[2] - x[1]) > 1e-9 for x in pairs)
        p_value = sign_test(len(worse), better)
        allowed = th.allowed_drop(metric)
        drop = (then - now) if info.higher_is_better else (now - then)
        past_limit = drop > allowed + 1e-9
        # errors are failures of the system, not sampling noise: alpha never excuses them
        alpha = None if metric == "errors" else th.alpha
        within_noise = past_limit and alpha is not None and p_value > alpha
        regressed = tuple(worse[:WORST]) if past_limit else ()
        deltas.append(
            Delta(
                metric,
                then,
                now,
                allowed,
                not past_limit or within_noise,
                regressed,
                len(worse),
                better,
                p_value,
                within_noise,
            )  # fmt: skip
        )
    return Comparison(tuple(deltas), tuple(warnings), th.alpha)


def is_fragile(c: Check) -> bool:
    """Passes, but one more failing question would fail it. Zero-tolerance rules (min 1, max 0)
    always are, by design, so they are not flagged."""
    zero_tolerance = c.rule.min == 1.0 or (c.rule.max == 0 and c.rule.min is None)
    return c.ok and c.margin == 1 and not zero_tolerance


def gate_section(checks: Sequence[Check], comparison: Comparison | None) -> dict[str, Any]:
    """The `gate` block of the results JSON."""
    failures = [f"{c.metric} ({c.scope}): {c.message}" for c in checks if not c.ok]
    if comparison:
        failures += [
            f"{d.metric}: dropped {d.drop:.3f} from the baseline (allowed {d.allowed_drop:g})"
            for d in comparison.deltas
            if not d.ok
        ]
    fragile = [f"{c.metric} ({c.scope})" for c in checks if is_fragile(c)]
    return {
        "passed": not failures,
        "failures": failures,
        "fragile": fragile,  # passing checks that one more failing question would fail
        "checks": [
            {
                "metric": c.metric,
                "scope": c.scope,
                "value": c.value,
                "n": c.n,
                "rule": str(c.rule),
                "ok": c.ok,
                "message": c.message,
                "worst": [list(w) for w in c.worst],
                "ci": None if c.ci is None else [round(c.ci[0], 4), round(c.ci[1], 4)],
                "margin": c.margin,
                "fragile": is_fragile(c),
            }
            for c in checks
        ],  # fmt: skip
        "baseline": None
        if comparison is None
        else {
            "alpha": comparison.alpha,
            "warnings": list(comparison.warnings),
            "deltas": [
                {
                    "metric": d.metric,
                    "baseline": d.baseline,
                    "current": d.current,
                    "change": round(d.change, 4),
                    "drop": round(d.drop, 4),
                    "allowed_drop": d.allowed_drop,
                    "ok": d.ok,
                    "regressed": [list(r) for r in d.regressed],
                    "worse": d.worse,
                    "better": d.better,
                    "p_value": round(d.p_value, 4),
                    "within_noise": d.within_noise,
                }
                for d in comparison.deltas
            ],  # fmt: skip
        },
    }
