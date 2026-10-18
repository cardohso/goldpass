"""Deterministic metrics: no LLM, no network, same inputs give the same numbers.

Retrieval metrics compare the ranked passage ids the generator saw with the golden set's
`gold_passage_ids`. Grounding checks that every cited document is one the question is about.
Answered and refusal rates check that the system answers what it should and declines the rest.

A metric that cannot be measured for an item (no gold ids, contexts not reported, ...) is left
out for that item, with a note saying why, rather than scored as 0.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from statistics import mean

from goldpass.golden import Expected, GoldenItem
from goldpass.record import Record

Score = float | bool


@dataclass(frozen=True)
class MetricInfo:
    label: str
    description: str
    higher_is_better: bool = True


METRICS: dict[str, MetricInfo] = {
    "faithfulness": MetricInfo(
        "Faithfulness (judge)", "share of the answer's claims supported by its contexts"
    ),
    "answer_relevancy": MetricInfo(
        "Answer relevancy (judge)",
        "1-5 rubric rating of how well the answer addresses the question, scaled to 0-1",
    ),
    "precision_at_k": MetricInfo(
        "Precision@k", "share of the top-k contexts that are gold passages"
    ),
    "recall_at_k": MetricInfo("Recall@k", "share of the gold passages found in the top-k contexts"),
    "context_precision": MetricInfo(
        "Context precision",
        "mean precision at each rank where a gold passage appears (RAGAS definition, gold ids "
        "instead of an LLM verdict); rewards gold passages ranked high",
    ),
    "mrr": MetricInfo("MRR", "1 / rank of the first gold passage in the top k, 0 if none"),
    "hit_rate": MetricInfo("Hit@k", "share of questions with at least one gold passage in top k"),
    "retrieved_recall": MetricInfo(
        "Retrieved recall",
        "recall over the wider `retrieved` list, before reranking or filtering (diagnostic)",
    ),
    "grounding": MetricInfo(
        "Grounding", "answered questions whose citations all come from the expected documents"
    ),
    "answered_rate": MetricInfo("Answered rate", "'answer' questions the system answered"),
    "refusal_rate": MetricInfo("Refusal rate", "'refuse' questions the system declined"),
    "errors": MetricInfo(
        "Errors", "count of items whose call failed or that have no recorded answer", False
    ),
}


# ranking functions (ids in rank order; duplicates count once, at their best rank)


def _top(ranked: Sequence[str], k: int | None) -> list[str]:
    return list(dict.fromkeys(ranked if k is None else ranked[:k]))


def precision_at_k(ranked: Sequence[str], gold: set[str], k: int | None = None) -> float:
    """Divides by the number of passages returned (at most k), so a system that returns fewer
    passages is not penalised for the ones it left out."""
    top = _top(ranked, k)
    return sum(pid in gold for pid in top) / len(top) if top else 0.0


def recall_at_k(ranked: Sequence[str], gold: set[str], k: int | None = None) -> float:
    return len(gold.intersection(_top(ranked, k))) / len(gold) if gold else 0.0


def context_precision(ranked: Sequence[str], gold: set[str], k: int | None = None) -> float:
    found, total = 0, 0.0
    for rank, pid in enumerate(_top(ranked, k), start=1):
        if pid in gold:
            found += 1
            total += found / rank
    return total / found if found else 0.0


def reciprocal_rank(ranked: Sequence[str], gold: set[str], k: int | None = None) -> float:
    rank = next((r for r, pid in enumerate(_top(ranked, k), start=1) if pid in gold), None)
    return 1 / rank if rank else 0.0


def hit_at_k(ranked: Sequence[str], gold: set[str], k: int | None = None) -> bool:
    return any(pid in gold for pid in _top(ranked, k))


def grounded(cited_docs: Iterable[str], source_docs: Iterable[str]) -> bool:
    """At least one citation, and every cited document is expected. Catches the right answer
    taken from the wrong document."""
    cited = set(cited_docs)
    return bool(cited) and cited <= set(source_docs)


# per item


@dataclass
class ItemScore:
    id: str
    expected: Expected
    tags: tuple[str, ...] = ()
    source_doc_ids: tuple[str, ...] = ()
    refused: bool | None = None
    cited_docs: tuple[str, ...] = ()
    metrics: dict[str, Score] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    error: str | None = None


def _cited_docs(record: Record, notes: list[str]) -> tuple[str, ...] | None:
    """Document ids of the citations; a citation without doc_id is looked up in the passages."""
    assert record.citations is not None
    doc_of = {p.id: p.doc_id for p in (record.contexts or ()) + (record.retrieved or ())}
    docs = []
    for c in record.citations:
        doc = c.doc_id or (doc_of.get(c.passage_id) if c.passage_id else None)
        if doc is None:
            notes.append(
                f"citation {c.passage_id!r} has no doc_id and is not in the contexts: "
                "grounding skipped"
            )
            return None
        docs.append(doc)
    return tuple(sorted(set(docs)))


def score_item(item: GoldenItem, record: Record | None, k: int | None = None) -> ItemScore:
    s = ItemScore(item.id, item.expected, item.tags, item.source_doc_ids)
    if record is None or record.error:
        s.error = record.error if record else "no answer recorded for this item"
        return s
    s.refused = record.refused

    if item.expected == "refuse":
        s.metrics["refusal_rate"] = record.refused
        return s

    s.metrics["answered_rate"] = not record.refused
    gold = set(item.gold_passage_ids)
    if not gold:
        s.notes.append("no gold_passage_ids: retrieval metrics skipped")
    elif record.contexts is None:
        s.notes.append("contexts not reported: retrieval metrics skipped")
    else:
        ranked = [p.id for p in record.contexts]
        s.metrics |= {
            "precision_at_k": precision_at_k(ranked, gold, k),
            "recall_at_k": recall_at_k(ranked, gold, k),
            "context_precision": context_precision(ranked, gold, k),
            "mrr": reciprocal_rank(ranked, gold, k),
            "hit_rate": hit_at_k(ranked, gold, k),
        }
        if record.retrieved is not None:
            s.metrics["retrieved_recall"] = recall_at_k([p.id for p in record.retrieved], gold)

    if record.refused:
        return s
    if not item.source_doc_ids:
        s.notes.append("no source_doc_ids: grounding skipped")
    elif record.citations is None:
        s.notes.append("citations not reported: grounding skipped")
    elif (docs := _cited_docs(record, s.notes)) is not None:
        s.cited_docs = docs
        s.metrics["grounding"] = grounded(docs, item.source_doc_ids)
    return s


# corpus level


@dataclass(frozen=True)
class Aggregate:
    value: float | None  # mean over the items where the metric was measured
    n: int  # how many items that was


def mean_of(values: Iterable[Score | None]) -> Aggregate:
    vs = [float(v) for v in values if v is not None]
    return Aggregate(round(mean(vs), 4) if vs else None, len(vs))


def aggregate(scores: Sequence[ItemScore]) -> dict[str, Aggregate]:
    out = {
        name: mean_of(s.metrics.get(name) for s in scores) for name in METRICS if name != "errors"
    }
    errors = sum(s.error is not None for s in scores)
    out["errors"] = Aggregate(float(errors), errors)
    return out


def by_tag(scores: Sequence[ItemScore]) -> dict[str, dict[str, Aggregate]]:
    groups: dict[str, list[ItemScore]] = defaultdict(list)
    for s in scores:
        for tag in s.tags:
            groups[tag].append(s)
    return {tag: aggregate(group) for tag, group in sorted(groups.items())}


@dataclass(frozen=True)
class DocumentRow:
    doc_id: str
    questions: int
    answered: int
    grounding: Aggregate
    cited_other_docs: tuple[str, ...]


def by_document(scores: Sequence[ItemScore]) -> list[DocumentRow]:
    """Grounding per source document, over 'answer' items; shows which other documents leak in."""
    groups: dict[str, list[ItemScore]] = defaultdict(list)
    for s in scores:
        if s.expected == "answer" and s.error is None:
            for doc in s.source_doc_ids:
                groups[doc].append(s)
    rows = []
    for doc, group in sorted(groups.items()):
        answered = [s for s in group if not s.refused]
        others = {d for s in answered for d in s.cited_docs} - {doc}
        rows.append(
            DocumentRow(
                doc_id=doc,
                questions=len(group),
                answered=len(answered),
                grounding=mean_of(s.metrics.get("grounding") for s in answered),
                cited_other_docs=tuple(sorted(others)),
            )
        )
    return rows
