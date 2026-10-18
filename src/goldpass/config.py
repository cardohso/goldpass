"""`goldpass.yaml`: which system to test and how to read its answers.

Relative paths resolve against the config file's directory. `${NAME}` in the URL or headers is
replaced from the environment at load time, so secrets never live in the file.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from goldpass.mapping import CitationFields, FieldMap, PassageFields
from goldpass.retry import RetryPolicy


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class HttpSystem:
    url: str
    method: str = "POST"
    headers: Mapping[str, str] = field(default_factory=dict)
    body: Any = field(default_factory=lambda: {"question": "{question}"})
    timeout_s: float = 60.0
    fields: FieldMap = field(default_factory=FieldMap)
    retry: RetryPolicy = field(default_factory=RetryPolicy)


@dataclass(frozen=True)
class PythonSystem:
    target: str  # "package.module:function"
    search_path: Path | None = None  # added to sys.path before importing


@dataclass(frozen=True)
class ReplaySystem:
    path: Path
    fields: FieldMap = field(default_factory=FieldMap)


System = HttpSystem | PythonSystem | ReplaySystem


@dataclass(frozen=True)
class Config:
    system: System
    golden: Path | None = None
    out: Path = Path("goldpass-results/run.json")
    k: int | None = None


# small, strict readers: unknown keys and wrong types are errors with a location


def _section(data: Any, where: str, allowed: set[str]) -> dict[str, Any]:
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(
            f"{where}: unknown key '{unknown[0]}' (allowed: {', '.join(sorted(allowed))})"
        )
    return data


def _typed(
    d: dict[str, Any], key: str, where: str, types: type | tuple[type, ...], default: Any
) -> Any:
    if key not in d:
        return default
    value = d[key]
    if value is None and default is None:
        return None
    if isinstance(value, bool) and bool not in (types if isinstance(types, tuple) else (types,)):
        raise ConfigError(f"{where}.{key}: expected {_names(types)}, got bool")
    if not isinstance(value, types):
        raise ConfigError(f"{where}.{key}: expected {_names(types)}, got {type(value).__name__}")
    return value


def _names(types: type | tuple[type, ...]) -> str:
    return " or ".join(t.__name__ for t in (types if isinstance(types, tuple) else (types,)))


_ENV = re.compile(r"\$\{(\w+)\}")


def _expand(value: str, where: str) -> str:
    def sub(m: re.Match[str]) -> str:
        if m.group(1) not in os.environ:
            raise ConfigError(f"{where}: environment variable {m.group(1)} is not set")
        return os.environ[m.group(1)]

    return _ENV.sub(sub, value)


def _fields(data: Any, where: str) -> FieldMap:
    keys = {"id", "answer", "refused", "refusal_patterns", "contexts", "retrieved", "citations"}
    d = _section(data, where, keys | {"passage", "citation"})
    flat: dict[str, Any] = {}
    for key in keys - {"refusal_patterns"}:
        if key in d:
            flat[key] = _typed(d, key, where, str, None)
    patterns = _typed(d, "refusal_patterns", where, list, [])
    for i, p in enumerate(patterns):
        try:
            re.compile(p)
        except (re.error, TypeError) as e:
            raise ConfigError(f"{where}.refusal_patterns[{i}]: not a valid regex: {e}") from e
    p = _section(d.get("passage"), f"{where}.passage", {"id", "doc_id", "text", "score"})
    c = _section(d.get("citation"), f"{where}.citation", {"passage_id", "doc_id"})
    return FieldMap(
        **flat,
        refusal_patterns=tuple(patterns),
        passage=PassageFields(**{k: _typed(p, k, f"{where}.passage", str, None) for k in p}),
        citation=CitationFields(**{k: _typed(c, k, f"{where}.citation", str, None) for k in c}),
    )


def _retry(data: Any, where: str) -> RetryPolicy:
    d = _section(data, where, {"attempts", "backoff_s", "max_wait_s"})
    base = RetryPolicy()
    attempts = _typed(d, "attempts", where, int, base.attempts)
    if attempts < 1:
        raise ConfigError(f"{where}.attempts: must be at least 1")
    return RetryPolicy(
        attempts=attempts,
        backoff_s=float(_typed(d, "backoff_s", where, (int, float), base.backoff_s)),
        max_wait_s=float(_typed(d, "max_wait_s", where, (int, float), base.max_wait_s)),
    )


def parse_system(data: Any, base_dir: Path, where: str = "system") -> System:
    if not isinstance(data, dict) or "type" not in data:
        raise ConfigError(f"{where}: needs a 'type' (http, python or replay)")
    kind = data["type"]
    if kind == "http":
        d = _section(
            data,
            where,
            {"type", "url", "method", "headers", "body", "timeout_s", "fields", "retry"},
        )
        if "url" not in d:
            raise ConfigError(f"{where}.url: required for an http system")
        headers = _typed(d, "headers", where, dict, {})
        return HttpSystem(
            url=_expand(_typed(d, "url", where, str, ""), f"{where}.url"),
            method=_typed(d, "method", where, str, "POST").upper(),
            headers={str(k): _expand(str(v), f"{where}.headers.{k}") for k, v in headers.items()},
            body=d.get("body", {"question": "{question}"}),
            timeout_s=float(_typed(d, "timeout_s", where, (int, float), 60.0)),
            fields=_fields(d.get("fields"), f"{where}.fields"),
            retry=_retry(d.get("retry"), f"{where}.retry"),
        )
    if kind == "python":
        d = _section(data, where, {"type", "target"})
        target = _typed(d, "target", where, str, "")
        if not re.fullmatch(r"[\w.]+:[\w.]+", target):
            raise ConfigError(f"{where}.target: expected 'module:function', got {target!r}")
        return PythonSystem(target, search_path=base_dir)
    if kind == "replay":
        d = _section(data, where, {"type", "path", "fields"})
        if "path" not in d:
            raise ConfigError(f"{where}.path: required for a replay system")
        return ReplaySystem(
            base_dir / _typed(d, "path", where, str, ""),
            _fields(d.get("fields"), f"{where}.fields"),
        )
    raise ConfigError(f"{where}.type: expected http, python or replay, got {kind!r}")


def load_config(path: Path) -> Config:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except OSError as e:
        raise ConfigError(f"{path}: cannot read: {e}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: invalid YAML: {e}") from e
    base = Path(path).parent
    d = _section(data, str(path), {"system", "golden", "out", "k"})
    if "system" not in d:
        raise ConfigError(f"{path}: needs a 'system' section")
    golden = _typed(d, "golden", str(path), str, None)
    k = _typed(d, "k", str(path), int, None)
    if k is not None and k < 1:
        raise ConfigError(f"{path}.k: must be at least 1")
    return Config(
        system=parse_system(d["system"], base),
        golden=base / golden if golden else None,
        out=base / _typed(d, "out", str(path), str, "goldpass-results/run.json"),
        k=k,
    )
