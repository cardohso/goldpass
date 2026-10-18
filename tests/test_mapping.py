from typing import Any

import pytest

from goldpass.mapping import (
    MISSING,
    CitationFields,
    FieldMap,
    MappingError,
    PassageFields,
    get_path,
    to_record,
)
from goldpass.record import Citation, Passage

DOC = {"data": {"answer": "65 dB", "hits": [{"chunk": "p1"}, {"chunk": "p2"}]}, "n": [10, 20]}


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("data.answer", "65 dB"),
        ("data.hits.1.chunk", "p2"),
        ("n.-1", 20),
        (".", DOC),
        ("data.nope", MISSING),
        ("n.5", MISSING),
        ("data.answer.x", MISSING),
    ],
)
def test_get_path(path: str, expected: Any) -> None:
    assert get_path(DOC, path) == expected or get_path(DOC, path) is expected


def test_native_format_needs_no_mapping() -> None:
    obj = {
        "answer": "65 dB [1]",
        "refused": False,
        "contexts": [{"id": "p1", "doc_id": "d1", "text": "t", "score": 0.9}],
        "citations": [{"passage_id": "p1", "doc_id": "d1"}],
    }
    r = to_record(obj, "q1", FieldMap(), latency_s=1.5)
    assert r.answer == "65 dB [1]" and r.refused is False and r.latency_s == 1.5
    assert r.contexts == (Passage("p1", "d1", "t", 0.9),)
    assert r.citations == (Citation("p1", "d1"),)
    assert r.retrieved is None  # absent means not reported


def test_absent_lists_are_not_reported_but_empty_lists_are_empty() -> None:
    r = to_record({"answer": "x", "contexts": []}, "q1", FieldMap())
    assert r.contexts == () and r.citations is None


def test_custom_mapping() -> None:
    obj = {
        "result": {"text": "hi", "sources": [{"chunk_id": "c1", "doc": {"name": "D"}}]},
        "cites": ["c1"],
    }
    fields = FieldMap(
        answer="result.text",
        refused=None,
        contexts="result.sources",
        citations="cites",
        passage=PassageFields(id="chunk_id", doc_id="doc.name", text=None, score=None),
        citation=CitationFields(),
    )
    r = to_record(obj, "q1", fields)
    assert r.contexts == (Passage("c1", "D"),)
    assert r.citations == (Citation(passage_id="c1"),)


def test_refusal_patterns_when_there_is_no_flag() -> None:
    fields = FieldMap(refused=None, refusal_patterns=(r"i (cannot|can't) answer",))
    assert to_record({"answer": "Sorry, I can't answer that."}, "q", fields).refused is True
    assert to_record({"answer": "65 dB"}, "q", fields).refused is False


def test_bare_id_lists_are_accepted() -> None:
    r = to_record({"answer": "x", "contexts": ["p1", "p2"]}, "q", FieldMap())
    assert r.contexts == (Passage("p1"), Passage("p2"))


def test_wrong_shapes_raise() -> None:
    with pytest.raises(MappingError, match="'contexts' should be a list, got str"):
        to_record({"answer": "x", "contexts": "p1"}, "q", FieldMap())
    with pytest.raises(MappingError, match="passage has no 'id'"):
        to_record({"answer": "x", "contexts": [{"pid": "p1"}]}, "q", FieldMap())
