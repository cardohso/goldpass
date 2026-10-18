"""The shipped examples work: the fake service passes its gate in good mode and fails it in
regressed mode, over real HTTP on localhost; the recorded rail run passes against its baseline."""

import json
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from goldpass.cli import ExitCode, main

EXAMPLES = Path(__file__).parents[1] / "examples"
FAKE = EXAMPLES / "fake_rag"


@pytest.fixture(scope="module")
def servers() -> Iterator[dict[str, str]]:
    sys.path.insert(0, str(FAKE))
    from server import serve  # type: ignore[import-not-found]

    urls, running = {}, []
    for mode in ("good", "regressed"):
        srv = serve(0, mode)  # port 0: any free port
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        urls[mode] = f"http://127.0.0.1:{srv.server_address[1]}/ask"
        running.append(srv)
    yield urls
    for srv in running:
        srv.shutdown()
    sys.path.remove(str(FAKE))


def run(tmp_path: Path, url: str) -> tuple[int, dict[str, Any]]:
    out = tmp_path / "run.json"
    code = main(["run", "--config", str(FAKE / "goldpass.yaml"), "--url", url, "--out", str(out)])
    return code, json.loads(out.read_text(encoding="utf-8"))


def test_fake_service_good_mode_passes(tmp_path: Path, servers: dict[str, str]) -> None:
    code, result = run(tmp_path, servers["good"])
    assert code == ExitCode.OK and result["gate"]["passed"]
    assert result["metrics"]["recall_at_k"]["value"] == 1.0
    assert result["metrics"]["refusal_rate"] == {"value": 1.0, "n": 2}


def test_fake_service_regressed_mode_fails_and_names_the_questions(
    tmp_path: Path, servers: dict[str, str]
) -> None:
    code, result = run(tmp_path, servers["regressed"])
    assert code == ExitCode.GATE_FAILED
    assert result["gate"]["failures"] == [
        "grounding (all): 0.889 < 0.9",
        "refusal_rate (all): 0.000 < 1",
        "grounding (tag:wrong-document): 0.000 < 1",
    ]
    worst = {c["metric"] + c["scope"]: c["worst"] for c in result["gate"]["checks"]}
    assert worst["groundingall"] == [["g08", 0.0]]
    assert sorted(i for i, _ in worst["refusal_rateall"]) == ["r01", "r02"]


def test_rail_recorded_example_passes_against_its_baseline(tmp_path: Path) -> None:
    out = tmp_path / "rail.json"
    assert (
        main(["run", "--config", str(EXAMPLES / "rail" / "goldpass.yaml"), "--out", str(out)]) == 0
    )
    gate = json.loads(out.read_text(encoding="utf-8"))["gate"]
    assert gate["passed"] and gate["baseline"]["deltas"]
    assert all(d["change"] == 0 for d in gate["baseline"]["deltas"])


@pytest.mark.parametrize("golden", ["fake_rag/golden.jsonl", "rail/golden.jsonl"])
def test_example_golden_sets_are_valid(golden: str) -> None:
    assert main(["validate", "--strict", str(EXAMPLES / golden)]) == ExitCode.OK
