"""The golden set: a JSONL file of questions, what the system should do, and the evidence.

The JSON Schema (`schemas/golden-item.schema.json`) checks each line on its own. This module
adds what a per-line schema cannot: valid JSON with line numbers, unique ids, and warnings for
items that some metrics will have to skip.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

import jsonschema

Expected = Literal["answer", "refuse"]


@dataclass(frozen=True)
class GoldenItem:
    id: str
    question: str
    expected: Expected
    reference_answer: str | None = None
    gold_passage_ids: tuple[str, ...] = ()
    source_doc_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> GoldenItem:
        return cls(
            id=d["id"],
            question=d["question"],
            expected=d["expected"],
            reference_answer=d.get("reference_answer"),
            gold_passage_ids=tuple(d.get("gold_passage_ids", ())),
            source_doc_ids=tuple(d.get("source_doc_ids", ())),
            tags=tuple(d.get("tags", ())),
            notes=d.get("notes"),
            metadata=dict(d.get("metadata", {})),
        )


@dataclass(frozen=True)
class Issue:
    path: Path
    line: int | None
    message: str
    level: Literal["error", "warning"] = "error"

    def __str__(self) -> str:
        where = f"{self.path}:{self.line}" if self.line else str(self.path)
        return f"{where}: {self.level}: {self.message}"


@dataclass
class ValidationReport:
    path: Path
    items: list[GoldenItem] = field(default_factory=list)
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, line: int | None, message: str) -> None:
        self.errors.append(Issue(self.path, line, message))

    def warn(self, line: int | None, message: str) -> None:
        self.warnings.append(Issue(self.path, line, message, "warning"))


class GoldenSetError(ValueError):
    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__("\n".join(str(e) for e in report.errors))


@cache
def schema() -> dict[str, Any]:
    text = files("goldpass").joinpath("schemas/golden-item.schema.json").read_text("utf-8")
    result: dict[str, Any] = json.loads(text)
    return result


@cache
def _validator() -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(schema())


def _schema_messages(obj: object) -> list[str]:
    out = []
    for err in sorted(_validator().iter_errors(obj), key=lambda e: list(map(str, e.path))):
        where = "/".join(map(str, err.path))
        if err.validator == "maxItems" and where == "gold_passage_ids":
            out.append("a 'refuse' item cannot have gold_passage_ids")
        else:
            out.append(f"{where}: {err.message}" if where else err.message)
    return out


def validate_golden(path: Path) -> ValidationReport:
    """Check every line and collect all problems, instead of stopping at the first one."""
    path = Path(path)
    report = ValidationReport(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as e:
        report.error(None, f"cannot read file: {e}")
        return report

    first_line: dict[str, int] = {}
    first_question: dict[str, str] = {}
    for n, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            report.error(n, f"invalid JSON: {e.msg} (column {e.colno})")
            continue
        if not isinstance(obj, dict):
            report.error(n, f"each line must be a JSON object, got {type(obj).__name__}")
            continue
        problems = _schema_messages(obj)
        for msg in problems:
            report.error(n, msg)
        item_id = obj.get("id")
        if isinstance(item_id, str):
            if item_id in first_line:
                report.error(n, f"duplicate id {item_id!r} (first on line {first_line[item_id]})")
                continue
            first_line[item_id] = n
        if problems:
            continue

        item = GoldenItem.from_dict(obj)
        report.items.append(item)
        question = " ".join(item.question.split()).casefold()
        if question in first_question:
            report.warn(n, f"{item.id}: same question as {first_question[question]!r}")
        else:
            first_question[question] = item.id
        if item.expected == "answer":
            if not item.gold_passage_ids:
                report.warn(n, f"{item.id}: no gold_passage_ids: retrieval metrics skip this item")
            if not item.source_doc_ids:
                report.warn(n, f"{item.id}: no source_doc_ids: grounding skips this item")

    if not first_line and report.ok:
        report.error(None, "no items: the file is empty")
    elif report.ok:
        kinds = {i.expected for i in report.items}
        if "answer" not in kinds:
            report.warn(None, "no 'answer' items: retrieval and grounding cannot be measured")
        if "refuse" not in kinds:
            report.warn(None, "no 'refuse' items: refusal_rate cannot be measured")
    return report


def load_golden(path: Path) -> list[GoldenItem]:
    """Load a golden set, raising GoldenSetError with every problem if any line is invalid."""
    report = validate_golden(path)
    if not report.ok:
        raise GoldenSetError(report)
    return report.items
