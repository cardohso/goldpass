import pytest

from goldpass.golden import GoldenItem
from goldpass.metrics import (
    Aggregate,
    aggregate,
    by_document,
    by_tag,
    context_precision,
    grounded,
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_item,
)
from goldpass.record import Citation, Passage, Record

GOLD = {"g1", "g2"}


# ranking functions


@pytest.mark.parametrize(
    ("ranked", "k", "expected"),
    [
        (["g1", "x", "g2", "y"], None, 0.5),
        (["g1", "x", "g2", "y"], 2, 0.5),
        (["g1", "g2", "x"], 2, 1.0),
        (["g1"], 4, 1.0),  # fewer results than k: divide by what was returned
        ([], 4, 0.0),
        (["x", "y"], None, 0.0),
    ],
)
def test_precision_at_k(ranked: list[str], k: int | None, expected: float) -> None:
    assert precision_at_k(ranked, GOLD, k) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("ranked", "k", "expected"),
    [
        (["g1", "x", "g2"], None, 1.0),
        (["g1", "x", "g2"], 2, 0.5),
        (["x"], None, 0.0),
        ([], None, 0.0),
        (["g1", "g1"], None, 0.5),  # a passage repeated in the ranking counts once
    ],
)
def test_recall_at_k(ranked: list[str], k: int | None, expected: float) -> None:
    assert recall_at_k(ranked, GOLD, k) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("ranked", "expected"),
    [
        (["g1", "x", "g2"], (1 / 1 + 2 / 3) / 2),
        (["x", "g1"], 1 / 2),
        (["g1", "g2"], 1.0),
        (["x", "y"], 0.0),
    ],
)
def test_context_precision(ranked: list[str], expected: float) -> None:
    assert context_precision(ranked, GOLD) == pytest.approx(expected)


def test_reciprocal_rank_and_hit() -> None:
    assert reciprocal_rank(["x", "y", "g2"], GOLD) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x", "y", "g2"], GOLD, k=2) == 0.0
    assert hit_at_k(["x", "g1"], GOLD, k=2) is True
    assert hit_at_k(["x", "g1"], GOLD, k=1) is False


def test_grounded() -> None:
    assert grounded(["a"], ["a", "b"]) is True
    assert grounded(["a", "c"], ["a", "b"]) is False
    assert grounded([], ["a"]) is False  # an answer with no citations is not grounded


# per item


def answer_item(**kw: object) -> GoldenItem:
    base: dict[str, object] = {
        "id": "q1",
        "question": "?",
        "expected": "answer",
        "gold_passage_ids": ("p1", "p2"),
        "source_doc_ids": ("docA",),
    }
    return GoldenItem(**(base | kw))  # type: ignore[arg-type]


def refuse_item(**kw: object) -> GoldenItem:
    return GoldenItem(**({"id": "r1", "question": "?", "expected": "refuse"} | kw))  # type: ignore[arg-type]


CTX = (Passage("p1", "docA"), Passage("x", "docA"), Passage("p2", "docA"))


def test_answered_item_gets_every_metric() -> None:
    rec = Record("q1", "answer", contexts=CTX, citations=(Citation("p1", "docA"),))
    s = score_item(answer_item(), rec)
    assert s.metrics == {
        "precision_at_k": pytest.approx(2 / 3),
        "recall_at_k": 1.0,
        "context_precision": pytest.approx((1 + 2 / 3) / 2),
        "mrr": 1.0,
        "hit_rate": True,
        "answered_rate": True,
        "grounding": True,
    }
    assert s.cited_docs == ("docA",)
    assert s.notes == []


def test_k_cuts_the_ranking() -> None:
    rec = Record("q1", contexts=CTX, citations=())
    s = score_item(answer_item(), rec, k=2)
    assert s.metrics["recall_at_k"] == 0.5
    assert s.metrics["precision_at_k"] == 0.5


def test_retrieved_list_gives_retrieved_recall() -> None:
    rec = Record("q1", contexts=CTX[:1], retrieved=CTX, citations=())
    assert score_item(answer_item(), rec).metrics["retrieved_recall"] == 1.0


def test_refused_answer_item_keeps_retrieval_metrics_but_no_grounding() -> None:
    rec = Record("q1", refused=True, contexts=CTX, citations=())
    s = score_item(answer_item(), rec)
    assert s.metrics["answered_rate"] is False
    assert "recall_at_k" in s.metrics
    assert "grounding" not in s.metrics


def test_refuse_item_only_scores_refusal() -> None:
    s = score_item(refuse_item(), Record("r1", refused=True, contexts=CTX))
    assert s.metrics == {"refusal_rate": True}
    s = score_item(refuse_item(), Record("r1", "a confident wrong answer"))
    assert s.metrics == {"refusal_rate": False}


def test_unreported_contexts_skip_retrieval_with_a_note() -> None:
    rec = Record("q1", "answer", contexts=None, citations=(Citation(doc_id="docA"),))
    s = score_item(answer_item(), rec)
    assert "recall_at_k" not in s.metrics
    assert s.metrics["grounding"] is True
    assert "contexts not reported: retrieval metrics skipped" in s.notes


def test_no_gold_ids_skips_retrieval_with_a_note() -> None:
    s = score_item(answer_item(gold_passage_ids=()), Record("q1", contexts=CTX, citations=()))
    assert "recall_at_k" not in s.metrics
    assert "no gold_passage_ids: retrieval metrics skipped" in s.notes


def test_unreported_citations_skip_grounding() -> None:
    s = score_item(answer_item(), Record("q1", "answer", contexts=CTX, citations=None))
    assert "grounding" not in s.metrics
    assert "citations not reported: grounding skipped" in s.notes


def test_answer_without_citations_is_not_grounded() -> None:
    s = score_item(answer_item(), Record("q1", "answer", contexts=CTX, citations=()))
    assert s.metrics["grounding"] is False


def test_citation_doc_is_resolved_from_contexts() -> None:
    rec = Record("q1", "answer", contexts=CTX, citations=(Citation(passage_id="p2"),))
    s = score_item(answer_item(), rec)
    assert s.metrics["grounding"] is True
    assert s.cited_docs == ("docA",)


def test_unresolvable_citation_skips_grounding() -> None:
    rec = Record("q1", "answer", contexts=CTX, citations=(Citation(passage_id="zz"),))
    s = score_item(answer_item(), rec)
    assert "grounding" not in s.metrics
    assert "citation 'zz' has no doc_id and is not in the contexts: grounding skipped" in s.notes


def test_citing_another_document_fails_grounding() -> None:
    rec = Record("q1", "answer", contexts=CTX, citations=(Citation("p9", "docB"),))
    s = score_item(answer_item(), rec)
    assert s.metrics["grounding"] is False
    assert s.cited_docs == ("docB",)


def test_errored_and_missing_records_are_not_scored() -> None:
    s = score_item(answer_item(), Record("q1", error="HTTP 500"))
    assert s.metrics == {} and s.error == "HTTP 500"
    s = score_item(answer_item(), None)
    assert s.metrics == {} and s.error == "no answer recorded for this item"


# corpus level


def test_aggregate_means_and_counts() -> None:
    items = [answer_item(id="a"), answer_item(id="b"), refuse_item(id="r")]
    recs = [
        Record("a", "x", contexts=CTX, citations=(Citation("p1", "docA"),)),
        Record("b", refused=True, contexts=(), citations=()),
        Record("r", refused=True),
    ]
    scores = [score_item(i, r) for i, r in zip(items, recs, strict=True)]
    agg = aggregate(scores)
    assert agg["answered_rate"] == Aggregate(0.5, 2)
    assert agg["recall_at_k"] == Aggregate(0.5, 2)
    assert agg["grounding"] == Aggregate(1.0, 1)
    assert agg["refusal_rate"] == Aggregate(1.0, 1)
    assert agg["retrieved_recall"] == Aggregate(None, 0)


def test_aggregate_counts_errors() -> None:
    scores = [score_item(answer_item(), Record("q1", error="timeout"))]
    agg = aggregate(scores)
    assert agg["errors"] == Aggregate(1.0, 1)
    assert agg["answered_rate"] == Aggregate(None, 0)


def test_by_tag_and_by_document() -> None:
    items = [
        answer_item(id="a", tags=("wrong-document",)),
        answer_item(id="b", source_doc_ids=("docB",)),
    ]
    recs = [
        Record("a", "x", contexts=CTX, citations=(Citation("p9", "docC"),)),
        Record("b", "x", contexts=CTX, citations=(Citation(doc_id="docB"),)),
    ]
    scores = [score_item(i, r) for i, r in zip(items, recs, strict=True)]
    assert by_tag(scores)["wrong-document"]["grounding"] == Aggregate(0.0, 1)
    rows = {r.doc_id: r for r in by_document(scores)}
    assert rows["docA"].questions == 1 and rows["docA"].cited_other_docs == ("docC",)
    assert rows["docB"].grounding == Aggregate(1.0, 1)


def test_unreported_refusals_skip_answered_and_refusal_rates_with_a_note() -> None:
    answered = score_item(answer_item(), Record("q1", "x", refused=None, contexts=CTX))
    assert "answered_rate" not in answered.metrics
    assert "refusals not reported: answered_rate skipped" in answered.notes
    assert answered.refused is None and "recall_at_k" in answered.metrics  # retrieval still scored

    refuse = score_item(refuse_item(), Record("r1", "x", refused=None))
    assert refuse.metrics == {} and "refusals not reported: refusal_rate skipped" in refuse.notes


def test_unreported_refusals_still_get_grounding() -> None:
    rec = Record("q1", "x", refused=None, contexts=CTX, citations=(Citation("p1", "docA"),))
    assert score_item(answer_item(), rec).metrics["grounding"] is True


def test_aggregate_has_no_value_for_unreported_refusals() -> None:
    items = [answer_item(id="a"), refuse_item(id="r")]
    recs = [Record("a", "x", refused=None, contexts=CTX), Record("r", "x", refused=None)]
    agg = aggregate([score_item(i, r) for i, r in zip(items, recs, strict=True)])
    assert agg["answered_rate"] == Aggregate(None, 0) and agg["refusal_rate"] == Aggregate(None, 0)
