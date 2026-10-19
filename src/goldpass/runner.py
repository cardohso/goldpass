"""Ask every golden question, score the answers, and write the results.

Answers from a live system are saved next to the results (`<out>.answers.json`, goldpass's
record format), so the same answers can be re-scored or re-judged later with a replay system,
without calling the system again.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from goldpass import __version__
from goldpass.adapters import Adapter
from goldpass.config import Generator
from goldpass.golden import GoldenItem
from goldpass.judging import JUDGE_METRICS, JudgeRun
from goldpass.metrics import ItemScore, aggregate, by_document, by_tag, mean_of, score_item
from goldpass.parallel import ordered_map
from goldpass.record import Record

Log = Callable[[str], None]


def collect(
    items: Sequence[GoldenItem], adapter: Adapter, log: Log | None = None, workers: int = 1
) -> list[Record]:
    """Ask every question. One at a time by default, because free-tier rate limits, not
    latency, are the limit; `workers` > 1 overlaps requests (the adapter keeps the pace)."""
    done = 0
    lock = threading.Lock()

    def ask(item: GoldenItem) -> Record:
        nonlocal done
        r = adapter.ask(item)
        if log:
            status = f"error: {r.error}" if r.error else ("refused" if r.refused else "answered")
            took = f" in {r.latency_s:.2f}s" if r.latency_s is not None else ""
            with lock:
                done += 1
                log(f"[{done:>{len(str(len(items)))}}/{len(items)}] {item.id} {status}{took}")
        return r

    return ordered_map(ask, items, workers)


def write_answers(path: Path, system: str, records: Sequence[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"system": system, "records": [asdict(r) for r in records]}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def commit() -> str | None:
    if sha := os.environ.get("GITHUB_SHA"):
        return sha[:7]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def skipped_summary(scores: Sequence[ItemScore]) -> dict[str, int]:
    """How many items each reason for skipping a metric applied to."""
    return dict(Counter(note for s in scores for note in s.notes).most_common())


def results(
    items: Sequence[GoldenItem],
    records: Sequence[Record],
    system: str,
    golden: Path,
    k: int | None,
    judge_runs: Sequence[JudgeRun] = (),
    generator: Generator | None = None,
    allow_same_family: bool = False,
) -> dict[str, Any]:
    by_id = {r.id: r for r in records}
    scores = [score_item(i, by_id.get(i.id), k) for i in items]
    if judge_runs:  # the first (gating) judge feeds the main metrics
        for s in scores:
            s.metrics.update(judge_runs[0].scores.get(s.id, {}))
            s.notes += judge_runs[0].notes.get(s.id, [])
    generator = generator or Generator()
    return {
        "goldpass_version": __version__,
        "meta": {
            "system": system,
            "golden": str(golden),
            "n_items": len(items),
            "k": k,
            "generator": {"model": generator.model, "family": generator.family},
            "judge_free": not judge_runs,
            "allow_same_family": allow_same_family,
            "commit": commit(),
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "metrics": {name: asdict(a) for name, a in aggregate(scores).items()},
        "by_tag": {
            tag: {name: asdict(a) for name, a in agg.items()} for tag, agg in by_tag(scores).items()
        },
        "by_document": [asdict(row) for row in by_document(scores)],
        "judges": [
            {
                "name": run.name,
                "family": run.family,
                "gating": n == 0,
                "metrics": {
                    m: asdict(mean_of(v.get(m) for v in run.scores.values())) for m in JUDGE_METRICS
                },
                "attempted": run.attempted,
                "failed": run.failed,
                "items": {
                    i: {"scores": run.scores[i], "details": run.details[i], "notes": run.notes[i]}
                    for i in run.scores
                },
            }
            for n, run in enumerate(judge_runs)
        ],
        "skipped": skipped_summary(scores),
        "items": [asdict(s) for s in scores],
    }
