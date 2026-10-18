"""Judges inside `goldpass run`, against a fake OpenAI-compatible API (no network)."""

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from goldpass import judging
from goldpass.cli import ExitCode, main

from .fake_llm import FakeLLM, default_llm

SYSTEM = """
def ask(question):
    if "fare" in question:
        return {"answer": "I cannot answer that.", "refused": True}
    return {"answer": "The stationary noise limit for EMUs is 65 dB [1]. EMUs are blue.",
            "contexts": [{"id": "p1", "doc_id": "d1",
                          "text": "The stationary noise limit for EMUs is 65 dB"}],
            "citations": [{"passage_id": "p1"}]}
"""
GOLDEN = [
    {"id": "a1", "question": "Noise limit?", "expected": "answer",
     "gold_passage_ids": ["p1"], "source_doc_ids": ["d1"]},
    {"id": "r1", "question": "What is the fare?", "expected": "refuse"},
]  # fmt: skip
JUDGES = """
golden: golden.jsonl
generator: {model: gemini-2.5-flash}
system: {type: python, target: 'fake_rag_judged:ask'}
judges:
  - {provider: groq, model: openai/gpt-oss-120b}
  - {provider: gemini, model: gemini-2.5-pro}   # same family as the generator: comparison only
judging: {retry: {attempts: 2, backoff_s: 0}}
"""


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    fake = default_llm()
    monkeypatch.setattr(judging, "TRANSPORT", httpx.MockTransport(fake))
    return fake


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    (tmp_path / "fake_rag_judged.py").write_text(SYSTEM, encoding="utf-8")
    (tmp_path / "golden.jsonl").write_text("\n".join(json.dumps(g) for g in GOLDEN), "utf-8")
    (tmp_path / "goldpass.yaml").write_text(JUDGES, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "test-groq")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    yield tmp_path
    sys.modules.pop("fake_rag_judged", None)


def result(project: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((project / "goldpass-results/run.json").read_text("utf-8"))
    return data


def test_gating_judge_feeds_main_metrics(project: Path, llm: FakeLLM) -> None:
    assert main(["run"]) == ExitCode.OK
    r = result(project)
    assert r["metrics"]["faithfulness"] == {"value": 0.5, "n": 1}  # 1 of 2 claims supported
    assert r["metrics"]["answer_relevancy"] == {"value": 1.0, "n": 1}
    first, second = r["judges"]
    assert (first["name"], first["family"], first["gating"]) == (
        "groq:openai/gpt-oss-120b", "openai", True
    )  # fmt: skip
    assert (second["family"], second["gating"]) == ("google", False)
    assert first["items"]["a1"]["details"]["faithfulness"]["claims"][1]["supported"] is False
    assert r["meta"]["generator"] == {"model": "gemini-2.5-flash", "family": "google"}
    assert r["meta"]["judge_free"] is False
    auth = {req["_auth"] for req in llm.requests}
    assert auth == {"Bearer test-groq", "Bearer test-gemini"}
    sent = json.dumps(llm.requests)
    assert "65 dB." in sent and "65 dB [1]" not in sent  # citation markers stripped
    assert all("fare" not in json.dumps(req) for req in llm.requests)  # refused: not judged


def test_same_family_gating_judge_is_refused(
    project: Path, llm: FakeLLM, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", "--generator", "openai/gpt-oss-20b"]) == ExitCode.USAGE
    assert "same model family (openai)" in capsys.readouterr().err
    assert llm.requests == []  # refused before any call
    assert main(["run", "--generator", "openai/gpt-oss-20b", "--allow-same-family"]) == 0
    assert result(project)["meta"]["allow_same_family"] is True


def test_unknown_generator_is_refused(
    project: Path, llm: FakeLLM, capsys: pytest.CaptureFixture[str]
) -> None:
    (project / "goldpass.yaml").write_text(
        JUDGES.replace("generator: {model: gemini-2.5-flash}\n", "")
    )
    assert main(["run"]) == ExitCode.USAGE
    assert "generator's model family is unknown" in capsys.readouterr().err


def test_missing_api_key(
    project: Path, llm: FakeLLM, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY")
    assert main(["run"]) == ExitCode.USAGE
    assert "GEMINI_API_KEY is not set" in capsys.readouterr().err


def test_no_judge_makes_no_llm_calls(project: Path, llm: FakeLLM) -> None:
    assert main(["run", "--no-judge"]) == ExitCode.OK
    assert llm.requests == []
    r = result(project)
    assert r["meta"]["judge_free"] is True and r["judges"] == []
    assert r["metrics"]["faithfulness"] == {"value": None, "n": 0}
    assert r["metrics"]["grounding"] == {"value": 1.0, "n": 1}


def test_cache_makes_reruns_free(project: Path, llm: FakeLLM) -> None:
    assert main(["run"]) == 0
    sent = len(llm.requests)
    assert sent == 6  # 2 judges x (claims + verdicts + relevancy) for the one answered item
    assert main(["run"]) == 0
    assert len(llm.requests) == sent
    assert main(["run", "--no-cache"]) == 0
    assert len(llm.requests) == 2 * sent


def test_gating_judge_failing_everywhere_is_a_runtime_error(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeLLM(claims=[httpx.Response(401, text="bad key")], relevancy=["{}"])
    monkeypatch.setattr(judging, "TRANSPORT", httpx.MockTransport(fake))
    assert main(["run"]) == ExitCode.RUNTIME
    assert "failed on every item" in capsys.readouterr().err


def test_replay_without_context_text_skips_faithfulness(project: Path, llm: FakeLLM) -> None:
    rail = Path(__file__).parent / "fixtures" / "rail"
    argv = ["run", "--golden", str(rail / "golden.jsonl"), "--replay", str(rail / "records.json"),
            "--generator", "gemini-3.5-flash-lite"]  # fmt: skip
    assert main(argv) == ExitCode.OK
    r = result(project)
    assert r["metrics"]["faithfulness"]["n"] == 0
    assert r["metrics"]["answer_relevancy"]["n"] == 15
    assert r["skipped"]["contexts have no text: faithfulness skipped"] == 15
