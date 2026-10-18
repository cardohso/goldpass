"""Command-line entry point: `goldpass run`, `goldpass validate`, `goldpass compare`."""

from __future__ import annotations

import argparse
import sys
from enum import IntEnum

from goldpass import __version__


class ExitCode(IntEnum):
    OK = 0  # gate passed / file valid
    GATE_FAILED = 1  # a metric is below its threshold or dropped past the baseline
    USAGE = 2  # bad arguments or an invalid golden set, thresholds or config file
    RUNTIME = 3  # the system under test or a judge could not be reached


COMMANDS = {
    "run": "run the golden set against a RAG system and gate on thresholds",
    "validate": "check a golden-set file against the schema",
    "compare": "compare a results file with a saved baseline",
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="goldpass", description=__doc__)
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)
    for name, help_text in COMMANDS.items():
        sub.add_parser(name, help=help_text, description=help_text)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"goldpass {args.command}: not implemented yet", file=sys.stderr)
    return ExitCode.USAGE
