"""The Action's judge-cache detection: run the step's own shell script against sample configs."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ACTION = Path(__file__).parents[1] / "action.yml"
BASH = shutil.which("bash")

CASES = [
    ("judges:\n  - {provider: groq, model: m}\n", "auto", True),
    ("judges:   # the first one gates\n  - provider: groq\n", "auto", True),
    ("judges: [{provider: groq, model: m}]\n", "auto", True),
    ("judges: [ {provider: groq} ]\n", "auto", True),
    ("judges: []\n", "auto", False),
    ("judges: [ ]\n", "auto", False),
    ("golden: g.jsonl\nsystem: {type: replay, path: a}\n", "auto", False),
    ("system:\n  judges: [x]\n", "auto", False),  # an indented key is something else
    ("# judges:\n#  - x\n", "auto", False),
    ("golden: g\n", "true", True),  # forced on
    ("judges:\n  - x\n", "false", False),  # forced off
]


def detection_script() -> str:
    steps = yaml.safe_load(ACTION.read_text(encoding="utf-8"))["runs"]["steps"]
    script: str = next(s for s in steps if s.get("id") == "judges")["run"]
    return script


def run_step(cwd: Path, cache: str, config: str) -> bool:
    out = cwd / "github_output"
    env = dict(os.environ, GP_CACHE=cache, GP_CONFIG=config, GITHUB_OUTPUT=str(out))
    subprocess.run([str(BASH), "-c", detection_script()], cwd=cwd, env=env, check=True)
    return "enabled=true" in out.read_text(encoding="utf-8")


@pytest.mark.skipif(BASH is None, reason="needs bash")
@pytest.mark.parametrize(("config", "mode", "expected"), CASES)
def test_judge_cache_is_kept_only_when_judges_are_configured(
    tmp_path: Path, config: str, mode: str, expected: bool
) -> None:
    (tmp_path / "goldpass.yaml").write_text(config, encoding="utf-8")
    assert run_step(tmp_path, mode, "goldpass.yaml") is expected


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_no_config_input_means_no_judges(tmp_path: Path) -> None:
    assert run_step(tmp_path, "auto", "") is False
    assert run_step(tmp_path, "auto", "missing.yaml") is False


def test_cache_step_follows_the_detection_step_and_defaults_to_auto() -> None:
    action = yaml.safe_load(ACTION.read_text(encoding="utf-8"))
    assert action["inputs"]["cache-judge"]["default"] == "auto"
    ids = [s.get("id") or s.get("uses", "") for s in action["runs"]["steps"]]
    assert ids.index("judges") < next(i for i, x in enumerate(ids) if "actions/cache" in x)
    cache = next(s for s in action["runs"]["steps"] if "actions/cache" in s.get("uses", ""))
    assert cache["if"] == "steps.judges.outputs.enabled == 'true'"
