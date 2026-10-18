"""goldpass must reproduce the judge-free metrics of a real rail-regulation-rag CI run."""

import json
from pathlib import Path
from typing import Any

import pytest

from goldpass.golden import load_golden
from goldpass.metrics import aggregate, by_document, by_tag, score_item
from goldpass.record import Record

FIXTURE = Path(__file__).parent / "fixtures" / "rail"

# rail metric name -> goldpass metric name
RENAMED = {
    "context_precision": "context_precision",
    "context_recall": "recall_at_k",
    "grounding": "grounding",
    "answered_rate": "answered_rate",
    "refusal_rate": "refusal_rate",
    "retrieval_recall_at_k": "retrieved_recall",
}


@pytest.fixture(scope="module")
def run() -> dict[str, Any]:
    items = load_golden(FIXTURE / "golden.jsonl")
    raw = json.loads((FIXTURE / "records.json").read_text(encoding="utf-8"))["records"]
    records = {
        r["id"]: Record(
            r["id"],
            refused=r["refused"],
            contexts=Record.passages(r["contexts"]),
            retrieved=Record.passages(r["retrieved"]),
            citations=Record.cites(r["citations"]),
        )
        for r in raw
    }
    scores = [score_item(i, records[i.id]) for i in items]
    expected = json.loads((FIXTURE / "expected.json").read_text(encoding="utf-8"))
    return {"scores": scores, "expected": expected}


@pytest.mark.parametrize(("rail", "ours"), RENAMED.items())
def test_corpus_metrics_match(run: dict[str, Any], rail: str, ours: str) -> None:
    value = aggregate(run["scores"])[ours].value
    assert value == pytest.approx(run["expected"]["metrics"][rail], abs=5e-5)


def test_wrong_document_slice_matches(run: dict[str, Any]) -> None:
    value = by_tag(run["scores"])["wrong-document"]["grounding"].value
    assert value == pytest.approx(run["expected"]["metrics"]["wrong_document_grounding"])


def test_per_document_matches(run: dict[str, Any]) -> None:
    ours = [
        {
            "doc_id": r.doc_id,
            "questions": r.questions,
            "answered": r.answered,
            "grounding": r.grounding.value,
            "cited_other_docs": list(r.cited_other_docs),
        }
        for r in by_document(run["scores"])
    ]
    assert ours == run["expected"]["per_document"]
