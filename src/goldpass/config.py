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

from goldpass.judge import JudgeSpec
from goldpass.mapping import CitationFields, FieldMap, PassageFields
from goldpass.providers import PROVIDERS, model_family
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
    min_interval_s: float = 0.0  # pause between requests, for rate-limited endpoints
    concurrency: int = 1  # questions in flight at once (the pace above is shared by all)
    fields: FieldMap = field(default_factory=FieldMap)
    retry: RetryPolicy = field(default_factory=RetryPolicy)


@dataclass(frozen=True)
class PythonSystem:
    target: str  # "package.module:function"
    search_path: Path | None = None  # added to sys.path before importing
    concurrency: int = 1  # the function must be thread-safe if this is above 1


@dataclass(frozen=True)
class ReplaySystem:
    path: Path
    fields: FieldMap = field(default_factory=FieldMap)


System = HttpSystem | PythonSystem | ReplaySystem


@dataclass(frozen=True)
class Generator:
    """The model behind the system under test; only used by the same-family guard."""

    model: str | None = None
    family: str | None = None


@dataclass(frozen=True)
class Judging:
    judges: tuple[JudgeSpec, ...] = ()  # the first one is the gating judge
    concurrency: int = 1  # answers judged at once per judge (each judge's pace is shared)
    cache_dir: Path | None = Path(".goldpass/cache")
    strip_pattern: str | None = r"\s*\[\d+\]"  # citation markers removed before judging
    allow_same_family: bool = False
    retry: RetryPolicy = field(default_factory=lambda: RetryPolicy(attempts=8, backoff_s=2.0))


@dataclass(frozen=True)
class Config:
    system: System
    golden: Path | None = None
    out: Path = Path("goldpass-results/run.json")
    k: int | None = None
    generator: Generator = field(default_factory=Generator)
    judging: Judging = field(default_factory=Judging)
    thresholds: Path | None = None
    baseline: Path | None = None  # a previous results JSON; skipped with a warning if absent
    audit_sentences: tuple[tuple[str, str], ...] | None = None  # (marker, sentence) to plant


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


def _interval(d: dict[str, Any], where: str) -> float:
    """Seconds between request starts, from `min_interval_s` or `requests_per_minute`."""
    if "min_interval_s" in d and "requests_per_minute" in d:
        raise ConfigError(f"{where}: set min_interval_s or requests_per_minute, not both")
    if "requests_per_minute" in d:
        rpm = _typed(d, "requests_per_minute", where, (int, float), 0)
        if rpm <= 0:
            raise ConfigError(f"{where}.requests_per_minute: must be positive")
        return 60.0 / float(rpm)
    seconds = float(_typed(d, "min_interval_s", where, (int, float), 0.0))
    if seconds < 0:
        raise ConfigError(f"{where}.min_interval_s: must not be negative")
    return seconds


def _concurrency(d: dict[str, Any], where: str) -> int:
    n = _typed(d, "concurrency", where, int, 1)
    if n < 1:
        raise ConfigError(f"{where}.concurrency: must be at least 1")
    return int(n)


def parse_generator(data: Any, where: str = "generator") -> Generator:
    d = _section(data, where, {"model", "family"})
    model = _typed(d, "model", where, str, None)
    return Generator(model, _typed(d, "family", where, str, None) or model_family(model))


def parse_judge(data: Any, where: str) -> JudgeSpec:
    keys = {"provider", "model", "base_url", "api_key_env", "family", "json_mode", "max_tokens",
            "timeout_s", "extra_body", "min_interval_s", "requests_per_minute"}  # fmt: skip
    d = _section(data, where, keys)
    for key in ("provider", "model"):
        if key not in d:
            raise ConfigError(f"{where}.{key}: required")
    provider, model = _typed(d, "provider", where, str, ""), _typed(d, "model", where, str, "")
    known = PROVIDERS.get(provider)
    if known is None and "base_url" not in d:
        raise ConfigError(
            f"{where}.provider: unknown provider {provider!r} (known: {', '.join(PROVIDERS)}); "
            "set base_url for any other OpenAI-compatible endpoint"
        )
    return JudgeSpec(
        provider=provider,
        model=model,
        base_url=_typed(d, "base_url", where, str, known.base_url if known else ""),
        api_key_env=_typed(d, "api_key_env", where, str, known.api_key_env if known else None),
        family=_typed(d, "family", where, str, None) or model_family(model),
        json_mode=_typed(d, "json_mode", where, bool, True),
        max_tokens=_typed(d, "max_tokens", where, int, 4096),
        timeout_s=float(_typed(d, "timeout_s", where, (int, float), 120.0)),
        min_interval_s=_interval(d, where),
        extra_body=_typed(d, "extra_body", where, dict, {}),
    )


def parse_judging(judges: Any, options: Any, base_dir: Path) -> Judging:
    if judges is None:
        judges = []
    if not isinstance(judges, list):
        raise ConfigError("judges: expected a list")
    d = _section(
        options,
        "judging",
        {"cache_dir", "strip_pattern", "allow_same_family", "retry", "concurrency"},
    )
    cache_dir = _typed(d, "cache_dir", "judging", str, ".goldpass/cache")
    strip = _typed(d, "strip_pattern", "judging", str, r"\s*\[\d+\]")
    if strip is not None:
        try:
            re.compile(strip)
        except re.error as e:
            raise ConfigError(f"judging.strip_pattern: not a valid regex: {e}") from e
    return Judging(
        judges=tuple(parse_judge(j, f"judges[{i}]") for i, j in enumerate(judges)),
        cache_dir=None if cache_dir is None else base_dir / cache_dir,
        strip_pattern=strip,
        allow_same_family=_typed(d, "allow_same_family", "judging", bool, False),
        concurrency=_concurrency(d, "judging"),
        retry=_retry(d["retry"], "judging.retry") if "retry" in d else Judging().retry,
    )


def parse_system(data: Any, base_dir: Path, where: str = "system") -> System:
    if not isinstance(data, dict) or "type" not in data:
        raise ConfigError(f"{where}: needs a 'type' (http, python or replay)")
    kind = data["type"]
    if kind == "http":
        d = _section(
            data,
            where,
            {
                "type",
                "url",
                "method",
                "headers",
                "body",
                "timeout_s",
                "min_interval_s",
                "requests_per_minute",
                "concurrency",
                "fields",
                "retry",
            },
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
            min_interval_s=_interval(d, where),
            concurrency=_concurrency(d, where),
            fields=_fields(d.get("fields"), f"{where}.fields"),
            retry=_retry(d.get("retry"), f"{where}.retry"),
        )
    if kind == "python":
        d = _section(data, where, {"type", "target", "concurrency"})
        target = _typed(d, "target", where, str, "")
        if not re.fullmatch(r"[\w.]+:[\w.]+", target):
            raise ConfigError(f"{where}.target: expected 'module:function', got {target!r}")
        return PythonSystem(target, search_path=base_dir, concurrency=_concurrency(d, where))
    if kind == "replay":
        d = _section(data, where, {"type", "path", "fields"})
        if "path" not in d:
            raise ConfigError(f"{where}.path: required for a replay system")
        return ReplaySystem(
            base_dir / _typed(d, "path", where, str, ""),
            _fields(d.get("fields"), f"{where}.fields"),
        )
    raise ConfigError(f"{where}.type: expected http, python or replay, got {kind!r}")


CONFIG_KEYS = {"system", "golden", "out", "k", "generator", "judges", "judging", "thresholds",
               "baseline", "audit"}  # fmt: skip


def load_config(path: Path) -> Config:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except OSError as e:
        raise ConfigError(f"{path}: cannot read: {e}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: invalid YAML: {e}") from e
    base = Path(path).parent
    d = _section(data, str(path), CONFIG_KEYS)
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
        generator=parse_generator(d.get("generator")),
        judging=parse_judging(d.get("judges"), d.get("judging"), base),
        thresholds=_path(d, "thresholds", str(path), base),
        baseline=_path(d, "baseline", str(path), base),
        audit_sentences=_sentences(d.get("audit"), "audit"),
    )


def _sentences(data: Any, where: str) -> tuple[tuple[str, str], ...] | None:
    d = _section(data, where, {"sentences"})
    if "sentences" not in d:
        return None
    items = d["sentences"]
    if not isinstance(items, list) or not items:
        raise ConfigError(f"{where}.sentences: expected a non-empty list")
    out = []
    for i, it in enumerate(items):
        e = _section(it, f"{where}.sentences[{i}]", {"marker", "sentence"})
        marker = _typed(e, "marker", f"{where}.sentences[{i}]", str, "")
        sentence = _typed(e, "sentence", f"{where}.sentences[{i}]", str, "")
        if not marker or not sentence:
            raise ConfigError(f"{where}.sentences[{i}]: needs a marker and a sentence")
        out.append((marker, sentence))
    return tuple(out)


def _path(d: dict[str, Any], key: str, where: str, base: Path) -> Path | None:
    value = _typed(d, key, where, str, None)
    return None if value is None else base / value
