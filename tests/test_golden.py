import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from goldpass.cli import ExitCode, main
from goldpass.golden import GoldenItem, GoldenSetError, load_golden, schema, validate_golden

ANSWER: dict[str, Any] = {
    "id": "q01",
    "question": "What is the stationary noise limit for an EMU?",
    "expected": "answer",
    "gold_passage_ids": ["tsi-noi:Annex §4.2.1#1"],
    "source_doc_ids": ["tsi-noi"],
}
REFUSE: dict[str, Any] = {
    "id": "q02",
    "question": "What is the fare from Lisbon to Porto?",
    "expected": "refuse",
}


def write(tmp_path: Path, *lines: Any) -> Path:
    path = tmp_path / "golden.jsonl"
    text = "\n".join(x if isinstance(x, str) else json.dumps(x, ensure_ascii=False) for x in lines)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def messages(issues: list[Any]) -> str:
    return "\n".join(str(i) for i in issues)


def test_schema_is_valid_draft_2020_12() -> None:
    jsonschema.Draft202012Validator.check_schema(schema())


def test_valid_file_loads(tmp_path: Path) -> None:
    report = validate_golden(write(tmp_path, ANSWER, REFUSE))
    assert report.ok, messages(report.errors)
    assert report.warnings == []
    assert [i.id for i in report.items] == ["q01", "q02"]


def test_all_fields_parse(tmp_path: Path) -> None:
    item = ANSWER | {
        "reference_answer": "65 dB.",
        "tags": ["wrong-document"],
        "notes": "trap: a similar limit exists in another TSI",
        "metadata": {"author": "jp", "difficulty": 2},
    }
    (parsed,) = load_golden(write(tmp_path, item, REFUSE))[:1]
    assert parsed == GoldenItem(
        id="q01",
        question=ANSWER["question"],
        expected="answer",
        reference_answer="65 dB.",
        gold_passage_ids=("tsi-noi:Annex §4.2.1#1",),
        source_doc_ids=("tsi-noi",),
        tags=("wrong-document",),
        notes="trap: a similar limit exists in another TSI",
        metadata={"author": "jp", "difficulty": 2},
    )


def test_blank_lines_are_skipped_and_line_numbers_kept(tmp_path: Path) -> None:
    report = validate_golden(write(tmp_path, ANSWER, "", REFUSE | {"expected": "maybe"}))
    assert [i.line for i in report.errors] == [3]


def test_invalid_json_reports_line(tmp_path: Path) -> None:
    report = validate_golden(write(tmp_path, ANSWER, '{"id": "q02",'))
    assert len(report.errors) == 1
    assert report.errors[0].line == 2
    assert "invalid JSON" in str(report.errors[0])


def test_line_must_be_an_object(tmp_path: Path) -> None:
    report = validate_golden(write(tmp_path, ANSWER, "[1, 2]"))
    assert "must be a JSON object" in messages(report.errors)


@pytest.mark.parametrize(
    ("change", "expected_text"),
    [
        ({"question": None}, "'question' is a required property"),
        ({"expected": "maybe"}, "'maybe' is not one of ['answer', 'refuse']"),
        ({"question": ""}, "question"),
        ({"gold_passage_ids": "p1"}, "gold_passage_ids"),
        ({"gold_passage_ids": ["p1", "p1"]}, "non-unique"),
        ({"tags": ["Wrong Document"]}, "tags/0"),
        ({"id": "has space"}, "id"),
    ],
)
def test_schema_errors(tmp_path: Path, change: dict[str, Any], expected_text: str) -> None:
    item = {k: v for k, v in (ANSWER | change).items() if v is not None}
    report = validate_golden(write(tmp_path, item, REFUSE))
    assert not report.ok
    assert expected_text in messages(report.errors)


def test_unknown_field_is_named(tmp_path: Path) -> None:
    report = validate_golden(write(tmp_path, ANSWER | {"gold_passages": ["p1"]}, REFUSE))
    assert "'gold_passages' was unexpected" in messages(report.errors)


def test_duplicate_ids(tmp_path: Path) -> None:
    report = validate_golden(write(tmp_path, ANSWER, REFUSE, ANSWER))
    assert len(report.errors) == 1
    assert report.errors[0].line == 3
    assert "duplicate id 'q01' (first on line 1)" in str(report.errors[0])


def test_refuse_items_cannot_have_gold_passages(tmp_path: Path) -> None:
    report = validate_golden(write(tmp_path, ANSWER, REFUSE | {"gold_passage_ids": ["p1"]}))
    assert "a 'refuse' item cannot have gold_passage_ids" in messages(report.errors)


def test_empty_file(tmp_path: Path) -> None:
    report = validate_golden(write(tmp_path, ""))
    assert "no items" in messages(report.errors)


def test_missing_file(tmp_path: Path) -> None:
    report = validate_golden(tmp_path / "nope.jsonl")
    assert "cannot read" in messages(report.errors)


def test_warnings_for_what_metrics_will_skip(tmp_path: Path) -> None:
    bare = {"id": "q03", "question": "What is the EMU noise limit?", "expected": "answer"}
    report = validate_golden(write(tmp_path, bare))
    assert report.ok
    text = messages(report.warnings)
    assert "no gold_passage_ids: retrieval metrics skip this item" in text
    assert "no source_doc_ids: grounding skips this item" in text
    assert "no 'refuse' items: refusal_rate cannot be measured" in text


def test_duplicate_question_warns(tmp_path: Path) -> None:
    report = validate_golden(write(tmp_path, ANSWER, REFUSE, ANSWER | {"id": "q03"}))
    assert report.ok
    assert "same question as 'q01'" in messages(report.warnings)


def test_load_golden_raises_with_all_errors(tmp_path: Path) -> None:
    path = write(tmp_path, ANSWER | {"expected": "x"}, ANSWER)
    with pytest.raises(GoldenSetError) as exc:
        load_golden(path)
    assert len(exc.value.report.errors) == 2
    assert f"{path}:1:" in str(exc.value)


def test_bom_is_tolerated(tmp_path: Path) -> None:
    path = tmp_path / "bom.jsonl"
    path.write_text("﻿" + json.dumps(ANSWER) + "\n" + json.dumps(REFUSE), encoding="utf-8")
    assert validate_golden(path).ok


# CLI


def test_cli_validate_ok(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", str(write(tmp_path, ANSWER, REFUSE))]) == ExitCode.OK
    assert "2 items (1 answer, 1 refuse): valid" in capsys.readouterr().out


def test_cli_validate_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write(tmp_path, ANSWER, ANSWER)
    assert main(["validate", str(path)]) == ExitCode.USAGE
    out = capsys.readouterr().out
    assert f"{path}:2: error: duplicate id" in out
    assert "1 error" in out


def test_cli_strict_fails_on_warnings(tmp_path: Path) -> None:
    path = write(tmp_path, ANSWER)  # no refuse items -> warning
    assert main(["validate", str(path)]) == ExitCode.OK
    assert main(["validate", "--strict", str(path)]) == ExitCode.USAGE
