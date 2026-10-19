import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from goldpass.cli import ExitCode, main

RAIL = Path(__file__).parents[1] / "examples" / "rail"
RECORDS = RAIL / "recorded" / "records.json"
GOLDEN = RAIL / "golden.jsonl"


def rail_expected() -> dict[str, Any]:
    metrics: dict[str, Any] = json.loads((RAIL / "recorded" / "expected.json").read_text("utf-8"))[
        "metrics"
    ]
    return metrics


def test_replay_of_rail_run_reproduces_its_metrics(tmp_path: Path) -> None:
    out = tmp_path / "run.json"
    code = main(["run", "--golden", str(GOLDEN), "--replay", str(RECORDS),
                 "--out", str(out)])  # fmt: skip
    assert code == ExitCode.OK
    result = json.loads(out.read_text(encoding="utf-8"))
    m, exp = result["metrics"], rail_expected()
    assert m["recall_at_k"]["value"] == pytest.approx(exp["context_recall"], abs=5e-5)
    assert m["grounding"] == {"value": 1.0, "n": 15}
    assert m["refusal_rate"] == {"value": 1.0, "n": 4}
    assert result["meta"]["n_items"] == 20
    assert len(result["items"]) == 20
    assert not (tmp_path / "run.answers.json").exists()  # replays are not re-saved


FAKE = """
def ask(question):
    if "fare" in question:
        return {"answer": "I cannot answer that.", "refused": True}
    return {"answer": "65 dB [1]", "contexts": [{"id": "p1", "doc_id": "d1"}],
            "citations": [{"passage_id": "p1"}]}

def broken(question):
    raise ConnectionError("index not loaded")
"""


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    (tmp_path / "fake_rag_run.py").write_text(FAKE, encoding="utf-8")
    golden = [
        {"id": "a1", "question": "Noise limit?", "expected": "answer",
         "gold_passage_ids": ["p1"], "source_doc_ids": ["d1"]},
        {"id": "r1", "question": "What is the fare?", "expected": "refuse"},
    ]  # fmt: skip
    (tmp_path / "golden.jsonl").write_text("\n".join(json.dumps(g) for g in golden), "utf-8")
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    sys.modules.pop("fake_rag_run", None)


def test_python_run_saves_answers_that_replay_identically(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", "--golden", "golden.jsonl", "--python", "fake_rag_run:ask"]) == 0
    live = json.loads((project / "goldpass-results/run.json").read_text(encoding="utf-8"))
    assert live["metrics"]["grounding"] == {"value": 1.0, "n": 1}
    assert live["metrics"]["refusal_rate"] == {"value": 1.0, "n": 1}
    assert "Recall@k" in capsys.readouterr().out

    answers = project / "goldpass-results/run.answers.json"
    assert main(["run", "--golden", "golden.jsonl", "--replay", str(answers),
                 "--out", "replayed.json"]) == 0  # fmt: skip
    replayed = json.loads((project / "replayed.json").read_text(encoding="utf-8"))
    assert replayed["metrics"] == live["metrics"]
    assert replayed["meta"]["system"] == "python:fake_rag_run:ask (replayed from run.answers.json)"


def test_config_file(project: Path) -> None:
    (project / "goldpass.yaml").write_text(
        "golden: golden.jsonl\nout: out/r.json\nk: 1\n"
        "system: {type: python, target: 'fake_rag_run:ask'}\n",
        encoding="utf-8",
    )
    assert main(["run"]) == ExitCode.OK  # picks up ./goldpass.yaml
    result = json.loads((project / "out/r.json").read_text(encoding="utf-8"))
    assert result["meta"]["k"] == 1


def test_every_item_failing_is_a_runtime_error(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["run", "--golden", "golden.jsonl", "--python", "fake_rag_run:broken"])
    assert code == ExitCode.RUNTIME
    err = capsys.readouterr().err
    assert "error: ConnectionError: index not loaded" in err
    assert "every item failed" in err


def test_unreachable_url_is_a_runtime_error(project: Path) -> None:
    (project / "goldpass.yaml").write_text(
        "golden: golden.jsonl\n"
        "system: {type: http, url: 'http://127.0.0.1:9/ask', timeout_s: 1, retry: {attempts: 1}}\n",
        encoding="utf-8",
    )
    assert main(["run"]) == ExitCode.RUNTIME


@pytest.mark.parametrize(
    ("argv", "code", "message"),
    [
        (["run", "--golden", "golden.jsonl"], ExitCode.USAGE, "no system to test"),
        (["run", "--python", "fake_rag_run:ask"], ExitCode.USAGE, "no golden set"),
        (["run", "--golden", "nope.jsonl", "--python", "x:y"], ExitCode.USAGE, "cannot read"),
        (["run", "--golden", "golden.jsonl", "--python", "x:y", "-k", "0"], ExitCode.USAGE, "-k"),
        (["run", "--golden", "golden.jsonl", "--python", "no_such_mod:f"], ExitCode.RUNTIME,
         "ModuleNotFoundError"),
        (["run", "--golden", "golden.jsonl", "--replay", "missing.json"], ExitCode.RUNTIME,
         "cannot set up"),
    ],
)  # fmt: skip
def test_setup_errors(
    project: Path, capsys: pytest.CaptureFixture[str], argv: list[str], code: int, message: str
) -> None:
    assert main(argv) == code
    assert message in capsys.readouterr().err


def test_a_judge_free_run_leaves_no_cache_file_behind(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", "--golden", "golden.jsonl", "--python", "fake_rag_run:ask"]) == 0
    assert not (project / ".goldpass").exists()


def test_retrieval_only_system_does_not_get_an_answered_rate(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A system with `refused: null` is scored on retrieval only: no made-up 'answered 1.00'."""
    (project / "records.json").write_text(
        json.dumps(
            {
                "records": [{"id": "a1", "contexts": [{"id": "p1", "doc_id": "d1"}]}, {"id": "r1"}]
            }  # fmt: skip
        ),
        encoding="utf-8",
    )
    (project / "goldpass.yaml").write_text(
        "golden: golden.jsonl\nsystem:\n  type: replay\n  path: records.json\n"
        "  fields: {refused: null}\n",
        encoding="utf-8",
    )
    assert main(["run"]) == ExitCode.OK
    result = json.loads((project / "goldpass-results/run.json").read_text(encoding="utf-8"))
    assert result["metrics"]["recall_at_k"] == {"value": 1.0, "n": 1}
    assert result["metrics"]["answered_rate"] == {"value": None, "n": 0}
    assert result["metrics"]["refusal_rate"] == {"value": None, "n": 0}
    assert result["skipped"]["refusals not reported: answered_rate skipped"] == 1
    out = capsys.readouterr().out
    import re

    assert re.search(r"Answered rate\s+—\s+0\b", out)  # shown as not measured, never 1.000
    md = (project / "goldpass-results/run.md").read_text(encoding="utf-8")
    assert "Answered rate" not in md  # not measured and not gated: left out of the report
