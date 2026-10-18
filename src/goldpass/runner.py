"""Ask every golden question, score the answers, and write the results.

Answers from a live system are saved next to the results (`<out>.answers.json`, goldpass's
record format), so the same answers can be re-scored or re-judged later with a replay system,
without calling the system again.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from goldpass import __version__
from goldpass.adapters import Adapter
from goldpass.golden import GoldenItem
from goldpass.metrics import ItemScore, aggregate, by_document, by_tag, score_item
from goldpass.record import Record

Log = Callable[[str], None]


def collect(items: Sequence[GoldenItem], adapter: Adapter, log: Log | None = None) -> list[Record]:
    """One question at a time on purpose: free-tier rate limits, not latency, are the limit."""
    records = []
    for n, item in enumerate(items, start=1):
        r = adapter.ask(item)
        records.append(r)
        if log:
            status = f"error: {r.error}" if r.error else ("refused" if r.refused else "answered")
            took = f" in {r.latency_s:.2f}s" if r.latency_s is not None else ""
            log(f"[{n:>{len(str(len(items)))}}/{len(items)}] {item.id} {status}{took}")
    return records


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
) -> dict[str, Any]:
    by_id = {r.id: r for r in records}
    scores = [score_item(i, by_id.get(i.id), k) for i in items]
    return {
        "goldpass_version": __version__,
        "meta": {
            "system": system,
            "golden": str(golden),
            "n_items": len(items),
            "k": k,
            "commit": commit(),
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "metrics": {name: asdict(a) for name, a in aggregate(scores).items()},
        "by_tag": {
            tag: {name: asdict(a) for name, a in agg.items()} for tag, agg in by_tag(scores).items()
        },
        "by_document": [asdict(row) for row in by_document(scores)],
        "skipped": skipped_summary(scores),
        "items": [asdict(s) for s in scores],
    }
