"""Command-line entry point: `goldpass run`, `goldpass validate`, `goldpass compare`."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from enum import IntEnum
from pathlib import Path

from goldpass import __version__
from goldpass.golden import validate_golden


class ExitCode(IntEnum):
    OK = 0  # gate passed / file valid
    GATE_FAILED = 1  # a metric is below its threshold or dropped past the baseline
    USAGE = 2  # bad arguments or an invalid golden set, thresholds or config file
    RUNTIME = 3  # the system under test or a judge could not be reached


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


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


def cmd_stub(args: argparse.Namespace) -> int:
    print(f"goldpass {args.command}: not implemented yet", file=sys.stderr)
    return ExitCode.USAGE


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="goldpass", description=__doc__)
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the golden set against a RAG system and gate")
    run.set_defaults(func=cmd_stub)

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
