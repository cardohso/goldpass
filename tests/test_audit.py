"""goldpass judge-audit end to end, against a fake OpenAI-compatible API (no network)."""

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from goldpass import judging
from goldpass.audit import JudgeScores, summarise
from goldpass.cli import ExitCode, main
from goldpass.perturb import Perturbation

from .fake_llm import FakeLLM, chat_response, default_llm

FACTS = {
    "noise": "The stationary noise limit for EMUs is 65 dB",
    "invoice": "Invoices are due within 30 days",
    "tokens": "Access tokens expire after 12 hours",
}
SYSTEM = f"""
FACTS = {FACTS!r}
def ask(question):
    if "fare" in question:
        return {{"answer": "I cannot answer that.", "refused": True}}
    fact = FACTS[question.split()[0]]
    return {{"answer": fact + " [1].", "contexts": [{{"id": "p", "doc_id": "d", "text": fact}}],
             "citations": [{{"passage_id": "p", "doc_id": "d"}}]}}
"""
GOLDEN = [{"id": k, "question": f"{k} question?", "expected": "answer"} for k in FACTS] + [
    {"id": "r1", "question": "What is the fare?", "expected": "refuse"}
]
CONFIG = """
golden: golden.jsonl
generator: {model: gemini-2.5-flash}
system: {type: python, target: 'fake_rag_audited:ask'}
judges:
  - {provider: groq, model: openai/gpt-oss-120b}   # strict, other family
  - {provider: gemini, model: gemini-2.5-pro}      # lenient, same family as the generator
judging: {retry: {attempts: 1}}
"""


class Router:
    """The strict judge checks claims against passages; the lenient one supports everything."""

    def __init__(self) -> None:
        self.strict = default_llm()
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        if (
            body["model"] == "gemini-2.5-pro"
            and FakeLLM.kind(body["messages"][0]["content"]) == "verdicts"
        ):
            n = body["messages"][1]["content"].split("Claims:\n", 1)[1].count("\n") + 1
            verdicts = [{"claim": i, "supported": True} for i in range(1, n + 1)]
            return chat_response(json.dumps({"verdicts": verdicts}))
        return self.strict(request)


@pytest.fixture
def router(monkeypatch: pytest.MonkeyPatch) -> Router:
    r = Router()
    monkeypatch.setattr(judging, "TRANSPORT", httpx.MockTransport(r))
    return r


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    (tmp_path / "fake_rag_audited.py").write_text(SYSTEM, encoding="utf-8")
    (tmp_path / "golden.jsonl").write_text("\n".join(json.dumps(g) for g in GOLDEN), "utf-8")
    (tmp_path / "goldpass.yaml").write_text(CONFIG, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "test-groq")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    yield tmp_path
    sys.modules.pop("fake_rag_audited", None)


def audit(project: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(
        (project / "goldpass-results/judge-audit.json").read_text("utf-8")
    )
    return data


def test_audit_separates_a_strict_judge_from_a_lenient_one(
    project: Path, router: Router, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["judge-audit"]) == ExitCode.OK
    result = audit(project)
    assert result["meta"]["n_controls"] == 3 and result["meta"]["n_perturbed"] == 6
    strict = result["summary"]["per_judge"]["groq:openai/gpt-oss-120b"]
    lenient = result["summary"]["per_judge"]["gemini:gemini-2.5-pro"]
    assert strict["by_kind"]["all"]["missed"] == 0 and strict["control_mean"] == 1.0
    assert lenient["by_kind"]["all"]["missed"] == 6
    assert lenient["same_family_as_generator"] and not strict["same_family_as_generator"]
    assert lenient["by_kind"]["all"]["ci95"] == [0.6097, 1.0]
    (cmp,) = result["summary"]["comparisons"]
    assert (cmp["only_first_missed"], cmp["only_second_missed"]) == (0, 6)
    assert cmp["mcnemar_p"] == 0.0312
    out = capsys.readouterr().out
    assert "| `gemini-2.5-pro` | google (same as generator) | all | 6 | 6 | 1.00" in out
    assert "exact McNemar p = 0.03" in out
    # the refused question is never perturbed or judged
    assert all("fare" not in json.dumps(r) for r in router.requests)


def test_cached_rerun_makes_no_calls(project: Path, router: Router) -> None:
    assert main(["judge-audit"]) == 0
    sent = len(router.requests)
    assert main(["judge-audit"]) == 0
    assert len(router.requests) == sent


def test_dry_run_lists_planted_errors_without_keys_or_calls(
    project: Path, router: Router, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    monkeypatch.delenv("GROQ_API_KEY")
    assert main(["judge-audit", "--dry-run"]) == ExitCode.OK
    out = capsys.readouterr().out
    assert (
        "noise number: '65' -> '66'" in out and "invoice number: '30' -> '20'" in out
    )  # "within": moves down
    assert out.count(" sentence: ") == 3
    assert router.requests == []


def test_custom_sentences_from_config(project: Path, router: Router) -> None:
    (project / "goldpass.yaml").write_text(
        CONFIG + "audit:\n  sentences:\n    - {marker: refund, sentence: Refunds are automatic.}\n",
        encoding="utf-8",
    )
    assert main(["judge-audit", "--dry-run"]) == 0
    assert main(["judge-audit"]) == 0
    planted = {p["planted"] for p in audit(project)["perturbations"] if p["kind"] == "sentence"}
    assert planted == {"Refunds are automatic. [1]"}


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (CONFIG.split("judges:")[0], "needs at least one judge"),
        (CONFIG + "audit: {sentences: []}\n", "expected a non-empty list"),
    ],
)
def test_audit_config_errors(
    project: Path, router: Router, capsys: pytest.CaptureFixture[str], config: str, message: str
) -> None:
    (project / "goldpass.yaml").write_text(config, encoding="utf-8")
    assert main(["judge-audit"]) == ExitCode.USAGE
    assert message in capsys.readouterr().err


def test_judges_that_cannot_score_anything_are_a_runtime_error(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dead = FakeLLM(claims=[httpx.Response(401, text="bad key")])
    monkeypatch.setattr(judging, "TRANSPORT", httpx.MockTransport(dead))
    assert main(["judge-audit", "--no-cache"]) == ExitCode.RUNTIME


def test_summary_excludes_unjudged_pairs_and_uses_control_as_the_bar() -> None:
    ps = [
        Perturbation("a", "number", "1", "2", 0, "x"),
        Perturbation("a", "sentence", "", "s", 0, "y"),
        Perturbation("b", "sentence", "", "s", 0, "z"),
    ]
    # control 0.5 (judge already doubted the answer): 0.67 after planting is still a miss
    s = JudgeScores("j:m", "openai", control={"a": 0.5, "b": 1.0}, perturbed=[0.25, 0.67, None])
    j = summarise(ps, [s], "google")["per_judge"]["j:m"]
    assert j["by_kind"]["all"]["n"] == 2 and j["by_kind"]["all"]["missed"] == 1
    assert j["controls_below_1"] == 1
