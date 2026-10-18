"""Turn whatever JSON a RAG system returns into a Record, using dotted paths.

The defaults are goldpass's own record format, so a system (or a recorded run) that already
speaks it needs no mapping at all. A path that is absent from the response means "not
reported", never "empty".
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from goldpass.record import Citation, Passage, Record


class MappingError(ValueError):
    pass


class _Missing:
    def __repr__(self) -> str:
        return "MISSING"


MISSING: Any = _Missing()


@dataclass(frozen=True)
class PassageFields:
    id: str = "id"
    doc_id: str | None = "doc_id"
    text: str | None = "text"
    score: str | None = "score"


@dataclass(frozen=True)
class CitationFields:
    passage_id: str | None = "passage_id"
    doc_id: str | None = "doc_id"


@dataclass(frozen=True)
class FieldMap:
    id: str = "id"  # only used when reading recorded runs
    answer: str = "answer"
    refused: str | None = "refused"
    refusal_patterns: tuple[str, ...] = ()  # regexes on the answer, for systems with no flag
    contexts: str | None = "contexts"
    retrieved: str | None = "retrieved"
    citations: str | None = "citations"
    passage: PassageFields = field(default_factory=PassageFields)
    citation: CitationFields = field(default_factory=CitationFields)


def get_path(obj: Any, path: str) -> Any:
    """`a.b.0.c` walks dicts by key and lists by index; `.` alone is the object itself."""
    if path == ".":
        return obj
    for part in path.split("."):
        if isinstance(obj, Mapping) and part in obj:
            obj = obj[part]
        elif (
            isinstance(obj, list)
            and part.lstrip("-").isdigit()
            and -len(obj) <= int(part) < len(obj)
        ):
            obj = obj[int(part)]
        else:
            return MISSING
    return obj


def _opt(obj: Any, path: str | None) -> Any:
    return MISSING if path is None else get_path(obj, path)


def _list(obj: Any, path: str | None) -> list[Any] | None:
    value = _opt(obj, path)
    if value is MISSING or value is None:
        return None
    if not isinstance(value, list):
        raise MappingError(f"'{path}' should be a list, got {type(value).__name__}")
    return value


def _passage(p: Any, f: PassageFields, where: str) -> Passage:
    if isinstance(p, str):
        return Passage(p)  # a bare list of ids is fine
    pid = get_path(p, f.id)
    if pid is MISSING or pid is None:
        raise MappingError(f"{where}: passage has no '{f.id}'")
    doc, text, score = (_opt(p, x) for x in (f.doc_id, f.text, f.score))
    return Passage(
        str(pid),
        None if doc in (MISSING, None) else str(doc),
        None if text in (MISSING, None) else str(text),
        None if score in (MISSING, None) else float(score),
    )


def _citation(c: Any, f: CitationFields) -> Citation:
    if isinstance(c, str):
        return Citation(passage_id=c)
    pid, doc = _opt(c, f.passage_id), _opt(c, f.doc_id)
    return Citation(
        None if pid in (MISSING, None) else str(pid),
        None if doc in (MISSING, None) else str(doc),
    )


def to_record(obj: Any, item_id: str, fields: FieldMap, latency_s: float | None = None) -> Record:
    answer = get_path(obj, fields.answer)
    answer = "" if answer in (MISSING, None) else str(answer)
    flag = _opt(obj, fields.refused)
    refused = bool(flag) if flag is not MISSING else False
    if not refused and fields.refusal_patterns:
        refused = any(re.search(p, answer, re.IGNORECASE) for p in fields.refusal_patterns)

    contexts = _list(obj, fields.contexts)
    retrieved = _list(obj, fields.retrieved)
    citations = _list(obj, fields.citations)
    return Record(
        id=item_id,
        answer=answer,
        refused=refused,
        contexts=None
        if contexts is None
        else tuple(_passage(p, fields.passage, str(fields.contexts)) for p in contexts),
        retrieved=None
        if retrieved is None
        else tuple(_passage(p, fields.passage, str(fields.retrieved)) for p in retrieved),
        citations=None
        if citations is None
        else tuple(_citation(c, fields.citation) for c in citations),
        latency_s=latency_s,
    )
