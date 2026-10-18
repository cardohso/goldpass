import copy
import json
from pathlib import Path
from typing import Any

import pytest

from goldpass.cli import ExitCode, main
from goldpass.config import ConfigError
from goldpass.gate import (
    Rule,
    Thresholds,
    check_thresholds,
    compare_baseline,
    gate_section,
    parse_thresholds,
)
from goldpass.report import render

RAIL = Path(__file__).parents[1] / "examples" / "rail"
RECORDS = RAIL / "recorded" / "records.json"


@pytest.fixture(scope="module")
def rail(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Results of replaying the rail CI run (the parity fixture), as goldpass writes them."""
    out = tmp_path_factory.mktemp("rail") / "run.json"
    argv = ["run", "--golden", str(RAIL / "golden.jsonl"), "--replay", str(RECORDS),
            "--out", str(out)]  # fmt: skip
    assert main(argv) == ExitCode.OK
    data: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    return data


def regressed(result: dict[str, Any]) -> dict[str, Any]:
    """The same run, but q01 now cites the wrong document and q02 errors."""
    r = copy.deepcopy(result)
    for it in r["items"]:
        if it["id"] == "q01":
            it["metrics"]["grounding"] = False
        if it["id"] == "q02":
            it["metrics"], it["error"] = {}, "HTTP 500"
    r["metrics"]["grounding"] = {"value": 0.8667, "n": 15}
    r["metrics"]["errors"] = {"value": 1.0, "n": 1}
    return r


# thresholds file


def test_parse_thresholds() -> None:
    th = parse_thresholds(
        {
            "metrics": {"grounding": {"min": 0.9, "min_n": 10}, "errors": {"max": 0}},
            "tags": {"wrong-document": {"grounding": {"min": 1}}},
            "baseline": {"max_drop": 0.02, "metrics": {"grounding": 0}},
        }
    )
    assert th.metrics["grounding"] == Rule(min=0.9, min_n=10)
    assert th.tags["wrong-document"]["grounding"] == Rule(min=1.0)
    assert th.allowed_drop("grounding") == 0.0 and th.allowed_drop("mrr") == 0.02


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"metrics": {"context_recall": {"min": 0.7}}}, "did you mean 'recall_at_k'"),
        ({"metrics": {"groundng": {"min": 0.7}}}, "did you mean 'grounding'"),
        ({"metrics": {"grounding": 0.9}}, "expected a mapping with min, max and/or min_n"),
        ({"metrics": {"grounding": {"min": "high"}}}, "expected a number, got str"),
        ({"metrics": {"grounding": {"min": 1, "min_n": 0}}}, "positive integer"),
        ({"gates": {}}, "expected keys metrics, tags and/or baseline"),
        ({"baseline": {"max_drop": 0.1, "per_metric": {}}}, "max_drop and/or metrics"),
    ],
)
def test_thresholds_errors(data: Any, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_thresholds(data)


# absolute checks


def test_rail_run_passes_its_own_thresholds(rail: dict[str, Any]) -> None:
    th = parse_thresholds(
        {
            "metrics": {
                "context_precision": {"min": 0.5},
                "recall_at_k": {"min": 0.7},
                "grounding": {"min": 0.9},
                "answered_rate": {"min": 0.8},
                "refusal_rate": {"min": 0.75},
                "errors": {"max": 0},
            }
        }
    )
    assert all(c.ok for c in check_thresholds(rail, th))


def test_failing_check_lists_the_worst_questions(rail: dict[str, Any]) -> None:
    th = Thresholds(metrics={"answered_rate": Rule(min=1.0), "recall_at_k": Rule(min=0.95)})
    answered, recall = check_thresholds(rail, th)
    assert not answered.ok and answered.message == "0.938 < 1"
    assert [i for i, _ in answered.worst] == ["q13"]  # the one answerable question refused
    assert not recall.ok and recall.worst[0][1] < 1.0


def test_missing_values_min_n_and_tags(rail: dict[str, Any]) -> None:
    th = Thresholds(
        metrics={"faithfulness": Rule(min=0.8), "grounding": Rule(min=0.9, min_n=20)},
        tags={"wrong-document": {"grounding": Rule(min=1.0)}},
    )
    faith, grounding, tagged = check_thresholds(rail, th)
    assert faith.message == "no value (nothing could be scored)"
    assert grounding.message == "measured on 15 items, needs 20"
    assert tagged.ok and tagged.scope == "tag:wrong-document" and tagged.n == 3


def test_errors_max(rail: dict[str, Any]) -> None:
    (check,) = check_thresholds(regressed(rail), Thresholds(metrics={"errors": Rule(max=0)}))
    assert not check.ok and check.worst == (("q02", 1.0),)


# baseline


def test_same_run_has_no_regressions(rail: dict[str, Any]) -> None:
    comparison = compare_baseline(rail, rail)
    assert comparison.ok and comparison.warnings == ()
    assert {d.metric for d in comparison.deltas} >= {"grounding", "recall_at_k", "errors"}


def test_regressions_name_the_questions(rail: dict[str, Any]) -> None:
    comparison = compare_baseline(regressed(rail), rail)
    bad = {d.metric: d for d in comparison.deltas if not d.ok}
    assert set(bad) == {"grounding", "errors"}
    assert bad["grounding"].regressed == (("q01", 1.0, 0.0),)
    assert bad["errors"].regressed == (("q02", 0.0, 1.0),)
    assert bad["grounding"].drop == pytest.approx(0.1333, abs=1e-4)


def test_allowed_drop_and_improvements(rail: dict[str, Any]) -> None:
    lenient = Thresholds(max_drop=0.2, max_drop_by_metric={"errors": 1})
    assert compare_baseline(regressed(rail), rail, lenient).ok
    assert compare_baseline(rail, regressed(rail)).ok  # getting better never fails


def test_judge_change_and_golden_change_are_flagged(rail: dict[str, Any]) -> None:
    now, then = copy.deepcopy(rail), copy.deepcopy(rail)
    now["judges"] = [{"name": "groq:a", "gating": True}]
    then["judges"] = [{"name": "gemini:b", "gating": True}]
    now["metrics"]["faithfulness"] = {"value": 0.5, "n": 15}
    then["metrics"]["faithfulness"] = {"value": 1.0, "n": 15}
    then["items"] = then["items"][:-1]
    comparison = compare_baseline(now, then)
    assert comparison.ok  # faithfulness not compared across different judges
    assert any("judge metrics not compared" in w for w in comparison.warnings)
    assert any("1 items added" in w for w in comparison.warnings)


# report


def test_report_for_a_failed_gate(rail: dict[str, Any]) -> None:
    now = regressed(rail)
    th = Thresholds(metrics={"grounding": Rule(min=0.9), "errors": Rule(max=0)})
    now["gate"] = gate_section(check_thresholds(now, th), compare_baseline(now, rail, th))
    md = render(now)
    assert md.startswith("## goldpass: FAILED ❌")
    assert "| Grounding | 0.87 | 15 | ≥ 0.9 | 1.00 | -0.13 | ❌ |" in md
    assert "### What failed" in md
    assert "questions that got worse: `q01` (1.00 → 0.00)" in md
    assert "`q02`: HTTP 500" in md
    assert "### Grounding per source document" in md
    assert "| `wrong-document` |" in md


def test_report_without_gate_and_with_same_family_override(rail: dict[str, Any]) -> None:
    r = copy.deepcopy(rail)
    r["meta"]["allow_same_family"] = True

    def judge(name: str, family: str, gating: bool, f: float, rel: float) -> dict[str, Any]:
        metrics = {"faithfulness": {"value": f, "n": 2}, "answer_relevancy": {"value": rel, "n": 2}}
        return {"name": name, "family": family, "gating": gating, "metrics": metrics}

    r["judges"] = [
        judge("gemini:x", "google", True, 1.0, 1.0),
        judge("groq:y", "openai", False, 0.5, 0.75),
    ]
    md = render(r)
    assert md.startswith("## goldpass\n")
    assert "self-preference" in md
    assert "### Same answers, different judges" in md
    assert "| Faithfulness (judge) | 1.00 (n=2) | 0.50 (n=2) |" in md


# CLI


THRESHOLDS = "metrics:\n  grounding: {min: 0.9}\n  answered_rate: {min: 0.8}\n"


def test_run_gate_pass_and_fail(tmp_path: Path) -> None:
    th = tmp_path / "t.yaml"
    th.write_text(THRESHOLDS, encoding="utf-8")
    summary = tmp_path / "summary.md"
    base = ["run", "--golden", str(RAIL / "golden.jsonl"), "--replay", str(RECORDS),
            "--thresholds", str(th), "--summary", str(summary)]  # fmt: skip
    assert main([*base, "--out", str(tmp_path / "a.json")]) == ExitCode.OK
    assert (tmp_path / "a.md").read_text("utf-8").startswith("## goldpass: PASSED ✅")
    th.write_text(THRESHOLDS.replace("0.8", "0.99"), encoding="utf-8")
    assert main([*base, "--out", str(tmp_path / "b.json")]) == ExitCode.GATE_FAILED
    gate = json.loads((tmp_path / "b.json").read_text("utf-8"))["gate"]
    assert gate["failures"] == ["answered_rate (all): 0.938 < 0.99"]
    assert summary.read_text("utf-8").count("## goldpass") == 2  # appended, not overwritten


def test_missing_baseline_is_skipped_with_a_note(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "run",
        "--golden",
        str(RAIL / "golden.jsonl"),
        "--replay",
        str(RECORDS),
    ]
    argv += ["--baseline", str(tmp_path / "none.json"), "--out", str(tmp_path / "r.json")]
    assert main(argv) == ExitCode.OK
    assert "comparison skipped" in capsys.readouterr().out
    assert "No baseline found" in (tmp_path / "r.md").read_text("utf-8")


def test_compare_command(tmp_path: Path, rail: dict[str, Any]) -> None:
    (tmp_path / "base.json").write_text(json.dumps(rail), encoding="utf-8")
    (tmp_path / "now.json").write_text(json.dumps(regressed(rail)), encoding="utf-8")
    assert main(["compare", str(tmp_path / "base.json"), str(tmp_path / "base.json")]) == 0
    code = main(["compare", str(tmp_path / "now.json"), str(tmp_path / "base.json")])
    assert code == ExitCode.GATE_FAILED
    out = json.loads((tmp_path / "now.compared.json").read_text("utf-8"))
    assert out["gate"]["failures"][0].startswith("grounding: dropped 0.133")
    (tmp_path / "junk.json").write_text("[]", encoding="utf-8")
    assert main(["compare", str(tmp_path / "junk.json"), str(tmp_path / "base.json")]) == 2
