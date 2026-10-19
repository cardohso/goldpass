"""Concurrency end to end: real threads against real localhost HTTP servers."""

import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest

from goldpass import judging
from goldpass.cli import ExitCode, main
from goldpass.config import ConfigError, load_config

from .fake_llm import default_llm

DELAY = 0.2
N = 8


class Slow(BaseHTTPRequestHandler):
    """Answers after DELAY seconds; counts how many requests are in flight at once."""

    lock = threading.Lock()
    running = 0
    peak = 0
    starts: list[float] = []
    fail_first = 0

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        cls = type(self)
        with cls.lock:
            cls.running += 1
            cls.peak = max(cls.peak, cls.running)
            cls.starts.append(time.monotonic())
            cls.fail_first -= 1
            reject = cls.fail_first >= 0
        time.sleep(DELAY)
        with cls.lock:
            cls.running -= 1
        if reject:
            self.send_response(429)
            self.send_header("Retry-After", "1")
            self.end_headers()
            return
        data = json.dumps({"answer": "echo " + body["question"]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def slow_server() -> Iterator[tuple[str, type[Slow]]]:
    class Handler(Slow):
        lock = threading.Lock()
        running = peak = 0
        starts: list[float] = []
        fail_first = 0

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/ask", Handler
    srv.shutdown()


@pytest.fixture
def project(tmp_path: Path, slow_server: tuple[str, type[Slow]]) -> Path:
    golden = [{"id": f"q{i}", "question": f"question {i}?", "expected": "answer"} for i in range(N)]
    (tmp_path / "golden.jsonl").write_text("\n".join(json.dumps(g) for g in golden), "utf-8")
    return tmp_path


def write_config(project: Path, url: str, system_extra: str = "") -> Path:
    cfg = project / "goldpass.yaml"
    cfg.write_text(
        f"golden: golden.jsonl\nout: out/run.json\n"
        f"system:\n  type: http\n  url: {url}\n"
        f"  retry: {{attempts: 3, backoff_s: 0}}\n{system_extra}",
        encoding="utf-8",
    )
    return cfg


def test_workers_overlap_requests_and_keep_input_order(
    project: Path, slow_server: tuple[str, type[Slow]]
) -> None:
    url, server = slow_server
    cfg = write_config(project, url)
    t0 = time.monotonic()
    assert main(["run", "--config", str(cfg), "--workers", "4"]) == ExitCode.OK
    elapsed = time.monotonic() - t0
    assert server.peak == 4
    assert elapsed < N * DELAY * 0.75  # sequential would take N * DELAY = 1.6 s
    answers = json.loads((project / "out/run.answers.json").read_text("utf-8"))["records"]
    assert [r["id"] for r in answers] == [f"q{i}" for i in range(N)]
    assert all(r["answer"] == f"echo question {i}?" for i, r in enumerate(answers))


def test_default_is_sequential(project: Path, slow_server: tuple[str, type[Slow]]) -> None:
    url, server = slow_server
    assert main(["run", "--config", str(write_config(project, url))]) == ExitCode.OK
    assert server.peak == 1


def test_rate_limit_holds_however_many_workers(
    project: Path, slow_server: tuple[str, type[Slow]]
) -> None:
    url, server = slow_server
    cfg = write_config(project, url, "  concurrency: 8\n  min_interval_s: 0.15\n")
    assert main(["run", "--config", str(cfg)]) == ExitCode.OK
    gaps = [b - a for a, b in zip(sorted(server.starts), sorted(server.starts)[1:], strict=False)]
    assert min(gaps) > 0.1  # 8 workers, but request starts are 0.15 s apart
    assert server.peak > 1  # and they still overlap in flight


def test_retry_after_pauses_every_worker(
    project: Path, slow_server: tuple[str, type[Slow]]
) -> None:
    url, server = slow_server
    server.fail_first = 1  # the first request gets 429 + Retry-After: 1
    cfg = write_config(project, url, "  concurrency: 4\n")
    assert main(["run", "--config", str(cfg)]) == ExitCode.OK
    first, *rest = sorted(server.starts)
    # while one worker was told to wait 1 s, no other worker started a new request
    # (the 3 others started with the first; later requests are all >= ~1 s after it)
    late = [s for s in rest if s - first > 0.5]
    assert len(server.starts) == N + 1
    assert all(s - first >= 0.95 for s in late) and len(late) >= N - 3 - 1


# judges


def test_concurrent_judging_matches_sequential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = default_llm()
    monkeypatch.setattr(judging, "TRANSPORT", httpx.MockTransport(fake))
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("GEMINI_API_KEY", "k")

    def record(i: int) -> dict[str, Any]:
        fact = f"The limit is {i} dB"
        return {
            "id": f"q{i}",
            "answer": fact + ".",
            "contexts": [{"id": "p", "text": fact}],
            "citations": [{"passage_id": "p", "doc_id": "d"}],
        }

    answers = {"system": "fake", "records": [record(i) for i in range(N)]}
    (tmp_path / "answers.json").write_text(json.dumps(answers), "utf-8")
    golden = [{"id": f"q{i}", "question": "?", "expected": "answer"} for i in range(N)]
    (tmp_path / "golden.jsonl").write_text("\n".join(json.dumps(g) for g in golden), "utf-8")
    (tmp_path / "goldpass.yaml").write_text(
        "golden: golden.jsonl\ngenerator: {model: gemini-2.5-flash}\n"
        "system: {type: replay, path: answers.json}\n"
        "judges: [{provider: groq, model: openai/gpt-oss-120b}]\n"
        "judging: {cache_dir: cache, concurrency: 4}\n", encoding="utf-8")  # fmt: skip

    def run(extra: list[str], out: str) -> dict[str, Any]:
        argv = ["run", "--config", str(tmp_path / "goldpass.yaml"), "--out", str(tmp_path / out)]
        assert main([*argv, *extra]) == ExitCode.OK
        data: dict[str, Any] = json.loads((tmp_path / out).read_text("utf-8"))
        return data

    parallel = run([], "par.json")
    sequential = run(["--workers", "1", "--no-cache"], "seq.json")
    assert parallel["judges"][0]["items"] == sequential["judges"][0]["items"]
    assert parallel["metrics"] == sequential["metrics"]
    assert list(parallel["judges"][0]["items"]) == [f"q{i}" for i in range(N)]


# config


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("system: {type: http, url: u, concurrency: 0}", "concurrency: must be at least 1"),
        ("system: {type: http, url: u, concurrency: many}", "expected int"),
        ("system: {type: http, url: u, min_interval_s: 1, requests_per_minute: 5}", "not both"),
        ("system: {type: http, url: u, requests_per_minute: 0}", "must be positive"),
        ("system: {type: http, url: u, min_interval_s: -1}", "must not be negative"),
        ("system: {type: python, target: 'a:b', concurrency: 0}", "at least 1"),
        ("system: {type: replay, path: a}\njudging: {concurrency: 0}", "judging.concurrency"),
    ],
)
def test_concurrency_config_errors(tmp_path: Path, text: str, message: str) -> None:
    cfg = tmp_path / "goldpass.yaml"
    cfg.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(cfg)


def test_requests_per_minute_becomes_an_interval(tmp_path: Path) -> None:
    cfg = tmp_path / "goldpass.yaml"
    cfg.write_text(
        "system: {type: http, url: u, requests_per_minute: 5, concurrency: 3}\n"
        "judges: [{provider: groq, model: m, requests_per_minute: 30}]\n"
        "judging: {concurrency: 2}\n", encoding="utf-8")  # fmt: skip
    c = load_config(cfg)
    assert (c.system.min_interval_s, c.system.concurrency) == (12.0, 3)  # type: ignore[union-attr]
    assert c.judging.judges[0].min_interval_s == 2.0 and c.judging.concurrency == 2
