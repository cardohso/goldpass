import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from goldpass.adapters import HttpAdapter, PythonAdapter, ReplayAdapter, render
from goldpass.config import HttpSystem, PythonSystem, ReplaySystem
from goldpass.golden import GoldenItem
from goldpass.mapping import FieldMap, PassageFields
from goldpass.record import Citation, Passage, Record
from goldpass.retry import RetryPolicy

ITEM = GoldenItem("q1", 'What is "EMU" noise?', "answer")
OK_BODY = {
    "answer": "65 dB",
    "refused": False,
    "contexts": [{"id": "p1", "doc_id": "d1"}],
    "citations": [{"passage_id": "p1", "doc_id": "d1"}],
}


def test_render_fills_placeholders_without_format_syntax() -> None:
    template = {"q": "{question}", "meta": ["{id}", 3, {"x": "id={id} {not_a_field}"}]}
    assert render(template, ITEM) == {
        "q": 'What is "EMU" noise?',
        "meta": ["q1", 3, {"x": "id=q1 {not_a_field}"}],
    }


class Server:
    """Scripted responses for httpx.MockTransport; records every request."""

    def __init__(self, *responses: httpx.Response | Exception):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def adapter(server: Server, **kw: Any) -> tuple[HttpAdapter, list[float]]:
    waits: list[float] = []
    system = HttpSystem(url="http://rag.test/ask", **kw)
    return HttpAdapter(system, httpx.MockTransport(server), sleep=waits.append), waits


def test_http_success() -> None:
    server = Server(httpx.Response(200, json=OK_BODY))
    a, _ = adapter(server, headers={"Authorization": "Bearer x"}, body={"query": "{question}"})
    r = a.ask(ITEM)
    assert r.error is None
    assert r.answer == "65 dB"
    assert r.contexts == (Passage("p1", "d1"),)
    assert r.citations == (Citation("p1", "d1"),)
    assert r.latency_s is not None
    req = server.requests[0]
    assert req.method == "POST" and req.headers["Authorization"] == "Bearer x"
    assert json.loads(req.content) == {"query": 'What is "EMU" noise?'}


def test_http_get_sends_query_params() -> None:
    server = Server(httpx.Response(200, json=OK_BODY))
    a, _ = adapter(server, method="GET", body={"q": "{question}"})
    a.ask(ITEM)
    assert server.requests[0].url.params["q"] == 'What is "EMU" noise?'


def test_http_429_honours_retry_after() -> None:
    server = Server(
        httpx.Response(429, headers={"Retry-After": "7"}),
        httpx.Response(503),
        httpx.Response(200, json=OK_BODY),
    )
    a, waits = adapter(server, retry=RetryPolicy(attempts=5, backoff_s=1.0))
    assert a.ask(ITEM).error is None
    assert waits == [7.0, 2.0]  # server's wait first, then backoff for attempt 2


def test_http_connection_errors_are_retried_then_reported() -> None:
    boom = httpx.ConnectError("connection refused")
    a, waits = adapter(Server(boom, boom, boom), retry=RetryPolicy(attempts=3))
    r = a.ask(ITEM)
    assert r.error == "gave up after 3 attempts: ConnectError: connection refused"
    assert len(waits) == 2


def test_http_client_errors_are_not_retried() -> None:
    server = Server(httpx.Response(422, text="question too short"))
    a, waits = adapter(server)
    assert a.ask(ITEM).error == "HTTP 422: question too short"
    assert waits == []


def test_http_bad_bodies() -> None:
    a, _ = adapter(Server(httpx.Response(200, text="<html>")))
    assert "response is not JSON" in str(a.ask(ITEM).error)
    a, _ = adapter(Server(httpx.Response(200, json={"answer": "x", "contexts": "p1"})))
    assert "cannot read response: 'contexts' should be a list" in str(a.ask(ITEM).error)


# python


def module(tmp_path: Path, body: str) -> PythonSystem:
    (tmp_path / "fake_rag_mod.py").write_text(body, encoding="utf-8")
    return PythonSystem("fake_rag_mod:ask", search_path=tmp_path)


@pytest.fixture(autouse=True)
def _forget_fake_module() -> Any:
    import sys

    yield
    sys.modules.pop("fake_rag_mod", None)


def test_python_returns_dict(tmp_path: Path) -> None:
    a = PythonAdapter(
        module(tmp_path, "def ask(q):\n    return {'answer': q.upper(), 'contexts': ['p1']}\n")
    )
    r = a.ask(ITEM)
    assert r.answer == 'WHAT IS "EMU" NOISE?' and r.contexts == (Passage("p1"),)
    assert a.name == "python:fake_rag_mod:ask"


def test_python_returns_record_or_string(tmp_path: Path) -> None:
    src = (
        "from goldpass.record import Record\n"
        "def ask(q):\n    return Record('ignored', answer='a', refused=True)\n"
        "def plain(q):\n    return 'just text'\n"
    )
    system = module(tmp_path, src)
    r = PythonAdapter(system).ask(ITEM)
    assert r.id == "q1" and r.refused is True
    r = PythonAdapter(PythonSystem("fake_rag_mod:plain", tmp_path)).ask(ITEM)
    assert r.answer == "just text" and r.contexts is None


def test_python_exception_becomes_error(tmp_path: Path) -> None:
    a = PythonAdapter(module(tmp_path, "def ask(q):\n    raise RuntimeError('index missing')\n"))
    assert a.ask(ITEM).error == "RuntimeError: index missing"


# replay


def test_replay_native_and_missing(tmp_path: Path) -> None:
    path = tmp_path / "run.answers.json"
    records = [OK_BODY | {"id": "q1", "latency_s": 2.0}, {"id": "q2", "error": "HTTP 500"}]
    path.write_text(json.dumps({"system": "http:POST x", "records": records}), encoding="utf-8")
    a = ReplayAdapter(ReplaySystem(path))
    assert a.system_name == "http:POST x"
    r = a.ask(ITEM)
    assert r.answer == "65 dB" and r.latency_s == 2.0
    assert a.ask(GoldenItem("q2", "?", "answer")).error == "HTTP 500"
    assert a.ask(GoldenItem("q9", "?", "answer")).error == "not in the replay file"


def test_replay_with_mapping(tmp_path: Path) -> None:
    path = tmp_path / "rail.json"
    rows = [{"qid": "q1", "answer": "x", "hits": [{"passage_id": "p1"}]}]
    path.write_text(json.dumps(rows), encoding="utf-8")
    fields = FieldMap(id="qid", contexts="hits", passage=PassageFields(id="passage_id"))
    r = ReplayAdapter(ReplaySystem(path, fields)).ask(ITEM)
    assert r == Record("q1", "x", contexts=(Passage("p1"),))
