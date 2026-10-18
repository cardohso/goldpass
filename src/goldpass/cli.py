"""Command-line entry point: `goldpass run`, `goldpass validate`, `goldpass compare`."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import replace
from enum import IntEnum
from pathlib import Path
from typing import Any

from goldpass import __version__
from goldpass.adapters import HttpAdapter, ReplayAdapter, make_adapter
from goldpass.config import (
    Config,
    ConfigError,
    Generator,
    HttpSystem,
    ReplaySystem,
    System,
    load_config,
    parse_system,
)
from goldpass.gate import (
    Thresholds,
    check_thresholds,
    compare_baseline,
    gate_section,
    load_thresholds,
)
from goldpass.golden import GoldenSetError, load_golden, validate_golden
from goldpass.judging import JudgeRun, judge_records, make_judges
from goldpass.metrics import METRICS
from goldpass.providers import SameFamilyError, check_judge_family, model_family
from goldpass.report import render
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
    base = cfg or Config(system)
    generator = base.generator
    if args.generator:
        generator = Generator(args.generator, model_family(args.generator))
    judging = replace(
        base.judging,
        judges=() if args.no_judge else base.judging.judges,
        allow_same_family=args.allow_same_family or base.judging.allow_same_family,
    )
    return Config(
        system=system,
        golden=golden,
        out=args.out or base.out,
        k=args.k if args.k is not None else base.k,
        generator=generator,
        judging=judging,
        thresholds=args.thresholds or base.thresholds,
        baseline=args.baseline or base.baseline,
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
    gate = result.get("gate")
    if gate:
        print(f"\ngate: {'PASSED' if gate['passed'] else 'FAILED'}")
        for failure in gate["failures"]:
            print(f"  - {failure}")
        if gate.get("baseline_missing"):
            print(f"  (no baseline at {gate['baseline_missing']}: comparison skipped)")


def _load_json(path: Path, what: str) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ConfigError(f"{what} {path}: cannot read: {e}") from e
    if not isinstance(data, dict) or not {"meta", "metrics", "items"} <= set(data):
        raise ConfigError(f"{what} {path}: not a goldpass results file")
    return data


def finish(
    result: dict[str, Any],
    thresholds: Thresholds | None,
    baseline: Path | None,
    out: Path,
    summary: Path | None,
    json_paths: Path | None = None,
) -> int:
    """Gate, write results JSON + Markdown, append to the job summary, return the exit code."""
    gated = thresholds is not None or baseline is not None
    comparison, missing = None, None
    if baseline is not None:
        if baseline.exists():
            comparison = compare_baseline(result, _load_json(baseline, "baseline"), thresholds)
        else:
            missing = str(baseline)
    if gated:
        checks = check_thresholds(result, thresholds) if thresholds else []
        result["gate"] = gate_section(checks, comparison) | {"baseline_missing": missing}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown = render(result)
    if missing:
        markdown += f"\n_No baseline found at `{missing}`: the comparison was skipped._\n"
    out.with_suffix(".md").write_text(markdown, encoding="utf-8")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as f:
            f.write(markdown)
    if json_paths:
        paths = {"results": str(out.resolve()), "markdown": str(out.with_suffix(".md").resolve())}
        Path(json_paths).write_text(json.dumps(paths), encoding="utf-8")
    _print_metrics(result)
    print(f"\nresults: {out} (markdown: {out.with_suffix('.md')})")
    if gated and not result["gate"]["passed"]:
        return ExitCode.GATE_FAILED
    return ExitCode.OK


def cmd_run(args: argparse.Namespace) -> int:
    try:
        cfg = _config(args)
        assert cfg.golden is not None
        items = load_golden(cfg.golden)
        thresholds = load_thresholds(cfg.thresholds) if cfg.thresholds else None
    except (ConfigError, GoldenSetError) as e:
        _err(str(e))
        return ExitCode.USAGE

    def log(msg: str) -> None:
        print(msg, file=sys.stderr)

    judging = cfg.judging
    try:
        if judging.judges:
            first = judging.judges[0]
            check_judge_family(
                first.name, first.family, cfg.generator.family, judging.allow_same_family
            )
        judges = make_judges(judging, use_cache=not args.no_cache, log=log)
    except (SameFamilyError, ConfigError) as e:
        _err(str(e))
        return ExitCode.USAGE
    if judging.judges and judging.allow_same_family:
        log("warning: --allow-same-family: the gating judge was not checked against the generator")

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

    runs: list[JudgeRun] = []
    for judge in judges:
        log(f"judging with {judge.name} ({judge.family or 'unknown family'})")
        runs.append(judge_records(judge, items, records, log))
        judge.close()
    if judges:
        cache = judges[0].cache
        log(f"judge cache: {cache.hits} hits, {cache.misses} misses")
    elif not args.no_judge:
        log("no judges configured: judge-free run (deterministic metrics only)")

    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(cfg.system, ReplaySystem):
        answers = cfg.out.with_name(cfg.out.stem + ".answers.json")
        write_answers(answers, system_name, records)
        log(f"answers saved to {answers}")
    result = results(
        items, records, system_name, cfg.golden, cfg.k, runs, cfg.generator,
        judging.allow_same_family,
    )  # fmt: skip
    try:
        code = finish(result, thresholds, cfg.baseline, cfg.out, args.summary, args.json_paths)
    except ConfigError as e:
        _err(str(e))
        return ExitCode.USAGE

    if all(r.error for r in records):
        _err("every item failed: is the system running and reachable?")
        return ExitCode.RUNTIME
    if runs and runs[0].attempted and runs[0].failed == runs[0].attempted:
        _err(f"the gating judge {runs[0].name} failed on every item it judged")
        return ExitCode.RUNTIME
    return code


def cmd_compare(args: argparse.Namespace) -> int:
    """Re-gate a saved results file against a baseline, without calling anything."""
    try:
        result = _load_json(args.results, "results")
        _load_json(args.baseline, "baseline")
        thresholds = load_thresholds(args.thresholds) if args.thresholds else None
    except ConfigError as e:
        _err(str(e))
        return ExitCode.USAGE
    result.pop("gate", None)
    out = args.out or args.results.with_name(args.results.stem + ".compared.json")
    return finish(result, thresholds, args.baseline, out, args.summary, args.json_paths)


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
    run.add_argument("--generator", metavar="MODEL", help="the system's model, for the judge guard")
    run.add_argument("--no-judge", action="store_true", help="skip LLM judges: no API calls")
    run.add_argument("--no-cache", action="store_true", help="do not read or write the judge cache")
    run.add_argument("--allow-same-family", action="store_true",
                     help="let the gating judge share the generator's model family")  # fmt: skip
    _gate_args(run)
    run.set_defaults(func=cmd_run)

    validate = sub.add_parser("validate", help="check a golden-set file against the schema")
    validate.add_argument("golden", type=Path, help="golden-set JSONL file")
    validate.add_argument("--strict", action="store_true", help="fail on warnings too")
    validate.set_defaults(func=cmd_validate)

    compare = sub.add_parser("compare", help="gate a saved results file against a baseline")
    compare.add_argument("results", type=Path, help="results JSON from goldpass run")
    compare.add_argument("baseline", type=Path, help="baseline results JSON")
    compare.add_argument("--out", type=Path, help="where to write the compared results")
    compare.add_argument("--thresholds", type=Path, help="thresholds YAML")
    compare.add_argument("--summary", type=Path, help="append the Markdown report here")
    compare.add_argument("--json-paths", type=Path, help="write the result and report paths here")
    compare.set_defaults(func=cmd_compare)
    return ap


def _gate_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--thresholds", type=Path, help="thresholds YAML: fail (exit 1) below them")
    p.add_argument("--baseline", type=Path, help="previous results JSON: fail on regressions")
    p.add_argument("--summary", type=Path, help="append the Markdown report here, e.g. "
                   "$GITHUB_STEP_SUMMARY")  # fmt: skip
    p.add_argument("--json-paths", type=Path, help="write the result and report paths to this "
                   "JSON file, for CI wrappers")  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code
