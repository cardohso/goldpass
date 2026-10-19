"""Secrets must never reach results, reports, answers files or logs."""

import json
from pathlib import Path

import pytest

from goldpass.adapters import display_url
from goldpass.cli import ExitCode, main

SECRET = "s3cr3t-token-value"


@pytest.mark.parametrize(
    ("url", "shown"),
    [
        ("https://api.example.com/ask?key=ABC&x=1", "https://api.example.com/ask"),
        ("https://user:pw@api.example.com:8443/v1/ask#frag", "https://api.example.com:8443/v1/ask"),
        ("http://127.0.0.1:8000/ask", "http://127.0.0.1:8000/ask"),
    ],
)
def test_display_url_drops_credentials_query_and_fragment(url: str, shown: str) -> None:
    assert display_url(url) == shown


def test_expanded_env_secrets_do_not_reach_any_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("RAG_KEY", SECRET)
    (tmp_path / "golden.jsonl").write_text(
        json.dumps({"id": "q1", "question": "hello there?", "expected": "answer"}), "utf-8"
    )
    (tmp_path / "goldpass.yaml").write_text(
        "golden: golden.jsonl\nout: out/run.json\n"
        "system:\n  type: http\n  url: 'http://127.0.0.1:9/ask?key=${RAG_KEY}'\n"
        "  headers: {Authorization: 'Bearer ${RAG_KEY}'}\n"
        "  timeout_s: 1\n  retry: {attempts: 1}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    assert main(["run"]) == ExitCode.RUNTIME  # nothing listens on port 9
    seen = capsys.readouterr()
    texts = [seen.out.encode(), seen.err.encode()]
    texts += [p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()]  # incl. any cache
    # the user's own goldpass.yaml holds ${RAG_KEY}, never the value
    assert [t for t in texts if SECRET.encode() in t] == []
