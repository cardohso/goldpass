"""Command-line entry point: `goldpass run`, `goldpass validate`, `goldpass compare`."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from enum import IntEnum
from pathlib import Path
from typing import Any

from goldpass import __version__
from goldpass.adapters import HttpAdapter, ReplayAdapter, make_adapter
from goldpass.config import (
    Config,
    ConfigError,
    HttpSystem,
    ReplaySystem,
    System,
    load_config,
    parse_system,
)
from goldpass.golden import GoldenSetError, load_golden, validate_golden
from goldpass.metrics import METRICS
from goldpass.runner import collect, results, write_answers


class ExitCode(IntEnum):
    OK = 0  # gate passed / file valid
    GATE_FAILED = 1  # a metric is below its threshold or dropped past the baseline
    USAGE = 2  # bad arguments or an invalid golden set, thresholds or config file
    RUNTIME = 3  # the system under test or a judge could not be reached


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _err(message: str) -> None:
    print(f"goldpass: error: {message}", file=sys.stderr)


def cmd_validate(args: argparse.Namespace) -> int:
    report = validate_golden(args.golden)
    for issue in report.errors + report.warnings:
        print(issue)
    if not report.ok:
        print(f"{report.path}: {_plural(len(report.errors), 'error')}, "
              f"{_plural(len(report.warnings), 'warning')}")  # fmt: skip
        return ExitCode.USAGE
    kinds = Counter(i.expected for i in report.items)
    n_warn = len(report.warnings)
    status = f"valid, {_plural(n_warn, 'warning')}" if n_warn else "valid"
    print(f"{report.path}: {_plural(len(report.items), 'item')} "
          f"({kinds['answer']} answer, {kinds['refuse']} refuse): {status}")  # fmt: skip
    if args.strict and report.warnings:
        print("--strict: warnings count as errors")
        return ExitCode.USAGE
    return ExitCode.OK


def _config(args: argparse.Namespace) -> Config:
    """The config file, if any, with command-line flags taking precedence."""
    path = args.config
    if path is None and Path("goldpass.yaml").exists():
        path = Path("goldpass.yaml")
    cfg = load_config(path) if path else None
    system: System | None = cfg.system if cfg else None
    if args.replay:
        system = ReplaySystem(args.replay)
    elif args.python:
        system = parse_system({"type": "python", "target": args.python}, Path.cwd(), "--python")
    elif args.url:
        system = HttpSystem(url=args.url)
    if system is None:
        raise ConfigError("no system to test: pass --config, --url, --python or --replay")
    if args.k is not None and args.k < 1:
        raise ConfigError("-k: must be at least 1")
    golden = args.golden or (cfg.golden if cfg else None)
    if golden is None:
        raise ConfigError("no golden set: pass --golden or set 'golden' in the config")
    return Config(
        system=system,
        golden=golden,
        out=args.out or (cfg.out if cfg else Config(system).out),
        k=args.k if args.k is not None else (cfg.k if cfg else None),
    )


def _print_metrics(result: dict[str, Any]) -> None:
    rows = [
        (METRICS[name].label, m["value"], m["n"])
        for name, m in result["metrics"].items()
        if name != "errors"
    ]
    width = max(len(r[0]) for r in rows)
    print(f"\n{'metric':<{width}}  value   n")
    for label, value, n in rows:
        shown = "—" if value is None else f"{value:.3f}"
        print(f"{label:<{width}}  {shown:>5}  {n:>2}")
    errors = result["metrics"]["errors"]["n"]
    if errors:
        print(f"\n{_plural(errors, 'item')} failed and were not scored (see items[].error)")
    for note, count in result["skipped"].items():
        print(f"skipped on {_plural(count, 'item')}: {note}")


def cmd_run(args: argparse.Namespace) -> int:
    try:
        cfg = _config(args)
        assert cfg.golden is not None
        items = load_golden(cfg.golden)
    except (ConfigError, GoldenSetError) as e:
        _err(str(e))
        return ExitCode.USAGE

    def log(msg: str) -> None:
        print(msg, file=sys.stderr)

    try:
        adapter = make_adapter(cfg.system, log=log)
    except (ImportError, AttributeError, TypeError, OSError, ValueError) as e:
        _err(f"cannot set up the system under test: {type(e).__name__}: {e}")
        return ExitCode.RUNTIME
    system_name = adapter.name
    if isinstance(adapter, ReplayAdapter) and adapter.system_name:
        assert isinstance(cfg.system, ReplaySystem)
        system_name = f"{adapter.system_name} (replayed from {cfg.system.path.name})"

    log(f"goldpass {__version__}: {len(items)} questions -> {system_name}")
    try:
        records = collect(items, adapter, log)
    finally:
        if isinstance(adapter, HttpAdapter):
            adapter.close()

    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(cfg.system, ReplaySystem):
        answers = cfg.out.with_name(cfg.out.stem + ".answers.json")
        write_answers(answers, system_name, records)
        log(f"answers saved to {answers}")
    result = results(items, records, system_name, cfg.golden, cfg.k)
    cfg.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"results saved to {cfg.out}")
    _print_metrics(result)

    if all(r.error for r in records):
        _err("every item failed: is the system running and reachable?")
        return ExitCode.RUNTIME
    return ExitCode.OK


def cmd_stub(args: argparse.Namespace) -> int:
    print(f"goldpass {args.command}: not implemented yet", file=sys.stderr)
    return ExitCode.USAGE


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="goldpass", description=__doc__)
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the golden set against a RAG system")
    run.add_argument("--config", type=Path, help="goldpass.yaml (default: ./goldpass.yaml)")
    run.add_argument("--golden", type=Path, help="golden-set JSONL (overrides the config)")
    which = run.add_mutually_exclusive_group()
    which.add_argument("--url", help="HTTP endpoint that speaks goldpass's record format")
    which.add_argument("--python", metavar="MODULE:FUNCTION", help="callable taking a question")
    which.add_argument("--replay", type=Path, help="recorded answers file")
    run.add_argument("--out", type=Path, help="results JSON (default: goldpass-results/run.json)")
    run.add_argument("-k", type=int, dest="k", help="score only the top k contexts")
    run.set_defaults(func=cmd_run)

    validate = sub.add_parser("validate", help="check a golden-set file against the schema")
    validate.add_argument("golden", type=Path, help="golden-set JSONL file")
    validate.add_argument("--strict", action="store_true", help="fail on warnings too")
    validate.set_defaults(func=cmd_validate)

    compare = sub.add_parser("compare", help="compare a results file with a saved baseline")
    compare.set_defaults(func=cmd_stub)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code
