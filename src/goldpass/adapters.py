"""Adapters: ask the system under test one golden question, get a Record back.

A failed call never raises; it returns a Record with `error` set, so one bad item does not
lose the rest of the run.
"""

from __future__ import annotations

import importlib
import json
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

import httpx

from goldpass.config import HttpSystem, PythonSystem, ReplaySystem, System
from goldpass.golden import GoldenItem
from goldpass.mapping import FieldMap, MappingError, get_path, to_record
from goldpass.record import Record
from goldpass.retry import GaveUp, RetryableError, call_with_retry, parse_retry_after

RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}

Log = Callable[[str], None]


class Adapter(Protocol):
    name: str

    def ask(self, item: GoldenItem) -> Record: ...


def render(template: Any, item: GoldenItem) -> Any:
    """Fill `{question}` and `{id}` anywhere in a JSON body template. A string that is exactly a
    placeholder is replaced by the value itself; no other brace syntax is interpreted."""
    if isinstance(template, str):
        return template.replace("{question}", item.question).replace("{id}", item.id)
    if isinstance(template, dict):
        return {k: render(v, item) for k, v in template.items()}
    if isinstance(template, list):
        return [render(v, item) for v in template]
    return template


class HttpAdapter:
    def __init__(
        self,
        system: HttpSystem,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        log: Log | None = None,
    ):
        self.system = system
        self.name = f"http:{system.method} {system.url}"
        self._client = httpx.Client(
            headers=dict(system.headers), timeout=system.timeout_s, transport=transport
        )
        self._sleep = sleep
        self._log = log
        self._clock = time.monotonic
        self._last: float | None = None

    def _pace(self) -> None:
        """Keep at least min_interval_s between request starts (e.g. 12 s for 5/minute)."""
        if self._last is not None and self.system.min_interval_s > 0:
            wait = self._last + self.system.min_interval_s - self._clock()
            if wait > 0:
                self._sleep(wait)
        self._last = self._clock()

    def _call(self, body: Any) -> httpx.Response:
        try:
            if self.system.method == "GET":
                resp = self._client.get(self.system.url, params=body)
            else:
                resp = self._client.request(self.system.method, self.system.url, json=body)
        except httpx.TransportError as e:
            raise RetryableError(f"{type(e).__name__}: {e}") from e
        if resp.status_code in RETRY_STATUS:
            raise RetryableError(
                f"HTTP {resp.status_code}", parse_retry_after(resp.headers.get("Retry-After"))
            )
        return resp

    def ask(self, item: GoldenItem) -> Record:
        body = render(self.system.body, item)

        def on_retry(attempt: int, wait: float, e: RetryableError) -> None:
            if self._log:
                self._log(f"  {item.id}: {e}, retrying in {wait:.1f}s (attempt {attempt + 1})")

        self._pace()  # before the clock starts: pacing is ours, not the system's latency
        t0 = time.perf_counter()
        try:
            resp = call_with_retry(
                lambda: self._call(body), self.system.retry, self._sleep, on_retry
            )
        except GaveUp as e:
            return Record(item.id, error=str(e))
        latency = round(time.perf_counter() - t0, 3)
        if resp.status_code >= 400:
            return Record(item.id, error=f"HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            data = resp.json()
        except ValueError:
            return Record(item.id, error=f"response is not JSON: {resp.text[:200]!r}")
        try:
            return to_record(data, item.id, self.system.fields, latency)
        except MappingError as e:
            return Record(item.id, error=f"cannot read response: {e}")

    def close(self) -> None:
        self._client.close()


class PythonAdapter:
    """Calls `fn(question)`. It may return a Record, a dict in goldpass's record format, or a
    plain string (an answer with nothing else reported)."""

    def __init__(self, system: PythonSystem):
        self.name = f"python:{system.target}"
        module_name, attr = system.target.split(":")
        if system.search_path:
            where = str(Path(system.search_path).resolve())
            if where not in sys.path:
                sys.path.insert(0, where)
        importlib.invalidate_caches()  # the module may have been written after startup
        obj: Any = importlib.import_module(module_name)
        for part in attr.split("."):
            obj = getattr(obj, part)
        if not callable(obj):
            raise TypeError(f"{system.target} is not callable")
        self._fn: Callable[[str], Any] = obj

    def ask(self, item: GoldenItem) -> Record:
        t0 = time.perf_counter()
        try:
            out = self._fn(item.question)
        except Exception as e:  # the system under test may raise anything
            return Record(item.id, error=f"{type(e).__name__}: {e}")
        latency = round(time.perf_counter() - t0, 3)
        if isinstance(out, Record):
            return replace(out, id=item.id, latency_s=out.latency_s or latency)
        if isinstance(out, str):
            return Record(item.id, answer=out, latency_s=latency)
        try:
            return to_record(out, item.id, FieldMap(), latency)
        except MappingError as e:
            return Record(item.id, error=f"cannot read return value: {e}")


class ReplayAdapter:
    """Answers from a recorded run: `{"records": [...]}` or a bare list, one object per item."""

    def __init__(self, system: ReplaySystem):
        self.name = f"replay:{system.path.name}"
        data = json.loads(Path(system.path).read_text(encoding="utf-8"))
        rows = data.get("records") if isinstance(data, dict) else data
        if not isinstance(rows, list):
            raise ValueError(f"{system.path}: expected a list of records or {{'records': [...]}}")
        self.system_name = data.get("system") if isinstance(data, dict) else None
        self._rows: dict[str, Any] = {}
        for n, row in enumerate(rows):
            rid = get_path(row, system.fields.id)
            if not isinstance(rid, str):
                raise ValueError(f"{system.path}: record {n} has no '{system.fields.id}'")
            self._rows[rid] = row
        self._fields = system.fields

    def ask(self, item: GoldenItem) -> Record:
        row = self._rows.get(item.id)
        if row is None:
            return Record(item.id, error="not in the replay file")
        if isinstance(row, dict) and row.get("error"):
            return Record(item.id, error=str(row["error"]))
        try:
            latency = row.get("latency_s") if isinstance(row, dict) else None
            return to_record(row, item.id, self._fields, latency)
        except MappingError as e:
            return Record(item.id, error=f"cannot read record: {e}")


def make_adapter(system: System, log: Log | None = None) -> Adapter:
    if isinstance(system, HttpSystem):
        return HttpAdapter(system, log=log)
    if isinstance(system, PythonSystem):
        return PythonAdapter(system)
    return ReplayAdapter(system)
