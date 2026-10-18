from pathlib import Path

import pytest

from goldpass.config import ConfigError, HttpSystem, PythonSystem, ReplaySystem, load_config
from goldpass.mapping import FieldMap
from goldpass.retry import RetryPolicy


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "goldpass.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_http_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_TOKEN", "t0k")
    cfg = load_config(
        write(
            tmp_path,
            """
golden: eval/golden.jsonl
k: 5
system:
  type: http
  url: http://localhost:8000/ask
  headers: {Authorization: "Bearer ${RAG_TOKEN}"}
  body: {query: "{question}", top_k: 5}
  fields:
    answer: data.text
    contexts: data.sources
    passage: {id: chunk_id}
    refusal_patterns: ["cannot answer"]
  retry: {attempts: 3, backoff_s: 0.5}
""",
        )
    )
    assert cfg.golden == tmp_path / "eval/golden.jsonl"
    assert cfg.out == tmp_path / "goldpass-results/run.json"
    assert cfg.k == 5
    s = cfg.system
    assert isinstance(s, HttpSystem)
    assert s.headers == {"Authorization": "Bearer t0k"}
    assert s.body == {"query": "{question}", "top_k": 5}
    assert s.fields.answer == "data.text" and s.fields.passage.id == "chunk_id"
    assert s.fields.passage.doc_id == "doc_id"  # unspecified fields keep their defaults
    assert s.fields.citations == "citations"
    assert s.fields.refusal_patterns == ("cannot answer",)
    assert s.retry == RetryPolicy(attempts=3, backoff_s=0.5)


def test_python_and_replay(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path, "system: {type: python, target: 'my_rag.app:ask'}"))
    assert cfg.system == PythonSystem("my_rag.app:ask", search_path=tmp_path)
    cfg = load_config(write(tmp_path, "system: {type: replay, path: runs/a.json}"))
    assert cfg.system == ReplaySystem(tmp_path / "runs/a.json", FieldMap())


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("golden: x.jsonl", "needs a 'system' section"),
        ("system: {type: grpc}", "expected http, python or replay, got 'grpc'"),
        ("system: {type: http}", "system.url: required"),
        ("system: {type: http, url: u, timeout: 5}", "unknown key 'timeout'"),
        ("system: {type: http, url: u, fields: {answr: a}}", "system.fields: unknown key 'answr'"),
        (
            "system: {type: http, url: u, timeout_s: fast}",
            "timeout_s: expected int or float, got str",
        ),
        ("system: {type: http, url: u, retry: {attempts: 0}}", "attempts: must be at least 1"),
        ("system: {type: http, url: '${GOLDPASS_NOPE}'}", "GOLDPASS_NOPE is not set"),
        ("system: {type: http, url: u, fields: {refusal_patterns: ['(']}}", "not a valid regex"),
        ("system: {type: python, target: my_rag.ask}", "expected 'module:function'"),
        ("system: {type: replay, path: a.json}\nk: 0", "k: must be at least 1"),
        ("system: [", "invalid YAML"),
    ],
)
def test_config_errors(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message.replace("(", r"\(").replace("$", r"\$")):
        load_config(write(tmp_path, text))


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "nope.yaml")
