"""Run the configured judges over recorded answers.

Judging is a second stage over saved Records: the same answers can be scored by several
judges, and re-judged later from a replay without calling the system again. Only the first
judge's scores feed the main metrics (and, later, the gate); the others are for comparison.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import httpx

from goldpass.cache import Cache, DiskCache, NoCache
from goldpass.config import ConfigError, Judging
from goldpass.golden import GoldenItem
from goldpass.judge import Judge, JudgeError, LLMJudge
from goldpass.parallel import ordered_map
from goldpass.record import Record

Log = Callable[[str], None]
JUDGE_METRICS = ("faithfulness", "answer_relevancy")

# (item id, scores, details, notes, judge failed)
Outcome = tuple[str, dict[str, float], dict[str, object], list[str], bool]

TRANSPORT: httpx.BaseTransport | None = None  # tests swap in a fake API here


@dataclass
class JudgeRun:
    name: str
    family: str | None
    scores: dict[str, dict[str, float]] = field(default_factory=dict)  # item id -> metric -> value
    details: dict[str, dict[str, object]] = field(default_factory=dict)
    notes: dict[str, list[str]] = field(default_factory=dict)
    attempted: int = 0
    failed: int = 0


def make_judges(
    judging: Judging, use_cache: bool = True, log: Log | None = None, offline: bool = False
) -> list[LLMJudge]:
    """Build every judge up front, so a missing API key fails before the system is called.
    Offline judges answer only from the cache and need no key."""
    if offline and (not use_cache or judging.cache_dir is None):
        raise ConfigError("--offline needs the judge cache (judging.cache_dir, no --no-cache)")
    cache: Cache = NoCache()
    if use_cache and judging.cache_dir is not None:
        cache = DiskCache(judging.cache_dir / "judge.sqlite")
    judges = []
    for spec in judging.judges:
        key = None
        if spec.api_key_env and not offline:
            key = os.environ.get(spec.api_key_env)
            if not key:
                raise ConfigError(
                    f"judge {spec.name}: environment variable {spec.api_key_env} is not set"
                )
        judges.append(
            LLMJudge(
                spec,
                key,
                cache=cache,
                retry=judging.retry,
                strip_pattern=judging.strip_pattern,
                transport=TRANSPORT,
                log=log,
                offline=offline,
            )
        )
    return judges


def judge_records(
    judge: Judge,
    items: Sequence[GoldenItem],
    records: Sequence[Record],
    log: Log | None = None,
    workers: int = 1,
) -> JudgeRun:
    """Judge 'answer' items the system answered. One call at a time by default: free-tier
    rate limits; `workers` > 1 overlaps items (the judge's rate limiter keeps the pace)."""
    run = JudgeRun(judge.name, judge.family)
    by_id = {r.id: r for r in records}
    todo = [
        (item, by_id[item.id])
        for item in items
        if item.expected == "answer"
        and item.id in by_id
        and not by_id[item.id].error
        and not by_id[item.id].refused
    ]

    def one(
        pair: tuple[GoldenItem, Record],
    ) -> tuple[str, dict[str, float], dict[str, object], list[str], bool]:
        item, r = pair
        notes: list[str] = []
        scores: dict[str, float] = {}
        details: dict[str, object] = {}
        failed = False
        try:
            texts = [p.text for p in r.contexts or ()]
            if not texts:
                notes.append("no contexts: faithfulness skipped")
            elif any(t is None for t in texts):
                notes.append("contexts have no text: faithfulness skipped")
            else:
                f = judge.faithfulness(item.question, r.answer, [t for t in texts if t])
                if f is None:
                    notes.append("answer makes no factual claims: faithfulness skipped")
                else:
                    scores["faithfulness"] = f.value
                    details["faithfulness"] = dict(f.detail)
            rel = judge.answer_relevancy(item.question, r.answer)
            scores["answer_relevancy"] = rel.value
            details["answer_relevancy"] = dict(rel.detail)
        except JudgeError as e:
            failed = True
            notes.append(f"judge {judge.name} failed: {e}")
        if log:
            shown = " ".join(f"{k}={v:.2f}" for k, v in scores.items()) or "; ".join(notes)
            log(f"  {judge.name} {item.id}: {shown}")
        return item.id, scores, details, notes, failed

    for item_id, scores, details, notes, failed in ordered_map(one, todo, workers):
        run.attempted += 1
        run.failed += failed
        run.scores[item_id], run.details[item_id], run.notes[item_id] = scores, details, notes
    return run
