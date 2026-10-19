"""`goldpass judge-audit`: does each judge notice an answer that is no longer faithful?

Every answered question is judged as it is (the control), then once per planted error
(goldpass.perturb). A judge **misses** a planted error when the perturbed answer's faithfulness
is not lower than the control's: the unsupported claim cost it nothing. Comparing judges on the
same planted errors shows leniency differences, and labelling each judge's model family next to
the generator's shows whether a same-family judge is the lenient one (self-preference).

A comparison on correct answers cannot show this when faithfulness sits at its ceiling; planted
errors give the judges something to forgive.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from itertools import combinations
from statistics import mean
from typing import Any

from goldpass import __version__
from goldpass.golden import GoldenItem
from goldpass.judge import Judge, JudgeError
from goldpass.perturb import Perturbation, context_texts
from goldpass.record import Record
from goldpass.stats import mcnemar_exact, wilson

Log = Callable[[str], None]
DEFINITION = (
    "a miss is a planted error whose perturbed answer scores at least the control's faithfulness"
)


@dataclass
class JudgeScores:
    name: str
    family: str | None
    control: dict[str, float] = field(default_factory=dict)  # item id -> faithfulness
    perturbed: list[float | None] = field(default_factory=list)  # aligned with perturbations
    failures: list[str] = field(default_factory=list)


def score_judge(
    judge: Judge,
    items: Sequence[GoldenItem],
    records: Sequence[Record],
    perturbations: Sequence[Perturbation],
    log: Log | None = None,
) -> JudgeScores:
    """Faithfulness of every control and every perturbed copy. One call at a time."""
    question = {i.id: i.question for i in items}
    by_id = {r.id: r for r in records}
    out = JudgeScores(judge.name, judge.family)
    for item_id in dict.fromkeys(p.id for p in perturbations):
        r = by_id[item_id]
        texts = context_texts(r) or []
        try:
            f = judge.faithfulness(question[item_id], r.answer, texts)
        except JudgeError as e:
            out.failures.append(f"{item_id} control: {e}")
            continue
        if f is not None:
            out.control[item_id] = f.value
            if log:
                log(f"  {judge.name} {item_id} control: {f.value:.2f}")
    for p in perturbations:
        value = None
        if p.id in out.control:
            try:
                f = judge.faithfulness(question[p.id], p.answer, context_texts(by_id[p.id]) or [])
                value = None if f is None else f.value
            except JudgeError as e:
                out.failures.append(f"{p.id} {p.kind}: {e}")
        out.perturbed.append(value)
        if log and value is not None:
            log(f"  {judge.name} {p.id} {p.kind}: {value:.2f}")
    return out


def _rows(perturbations: Sequence[Perturbation], s: JudgeScores) -> list[dict[str, Any]]:
    rows = []
    for p, f in zip(perturbations, s.perturbed, strict=True):
        control = s.control.get(p.id)
        if control is None or f is None:
            continue  # not judged: excluded, never counted as caught or missed
        rows.append(
            {
                "id": p.id,
                "kind": p.kind,
                "control": control,
                "perturbed": f,
                "drop": round(control - f, 4),
                "missed": f >= control - 1e-9,
            }
        )
    return rows


def _rate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    k, n = sum(r["missed"] for r in rows), len(rows)
    ci = wilson(k / n, n) if n else None
    return {
        "n": n,
        "missed": k,
        "miss_rate": round(k / n, 4) if n else None,
        "ci95": None if ci is None else [round(ci[0], 4), round(ci[1], 4)],
        "mean_drop": round(mean(r["drop"] for r in rows), 4) if rows else None,
    }


def summarise(
    perturbations: Sequence[Perturbation],
    scores: Sequence[JudgeScores],
    generator_family: str | None,
) -> dict[str, Any]:
    per_judge = {}
    rows_by_judge = {}
    for s in scores:
        rows = _rows(perturbations, s)
        rows_by_judge[s.name] = {(r["id"], r["kind"]): r for r in rows}
        kinds = sorted({r["kind"] for r in rows})
        per_judge[s.name] = {
            "family": s.family,
            "same_family_as_generator": generator_family is not None
            and s.family == generator_family,
            "controls": len(s.control),
            "control_mean": round(mean(s.control.values()), 4) if s.control else None,
            "controls_below_1": sum(v < 1.0 for v in s.control.values()),
            "by_kind": {k: _rate([r for r in rows if r["kind"] == k]) for k in kinds}
            | {"all": _rate(rows)},
            "failures": s.failures,
            "rows": rows,
        }
    comparisons = []
    for a, b in combinations(per_judge, 2):
        shared = rows_by_judge[a].keys() & rows_by_judge[b].keys()
        ra = [rows_by_judge[a][key] for key in sorted(shared)]
        rb = [rows_by_judge[b][key] for key in sorted(shared)]
        only_a = sum(x["missed"] and not y["missed"] for x, y in zip(ra, rb, strict=True))
        only_b = sum(y["missed"] and not x["missed"] for x, y in zip(ra, rb, strict=True))
        both = sum(x["missed"] and y["missed"] for x, y in zip(ra, rb, strict=True))
        comparisons.append(
            {
                "judges": [a, b],
                "pairs": len(shared),
                "only_first_missed": only_a,
                "only_second_missed": only_b,
                "both_missed": both,
                "mcnemar_p": round(mcnemar_exact(only_a, only_b), 4),
            }
        )
    return {"per_judge": per_judge, "comparisons": comparisons}


def audit_result(
    system: str,
    generator: Mapping[str, Any],
    records: Sequence[Record],
    perturbations: Sequence[Perturbation],
    scores: Sequence[JudgeScores],
) -> dict[str, Any]:
    by_id = {r.id: r for r in records}
    return {
        "goldpass_version": __version__,
        "meta": {
            "system": system,
            "generator": dict(generator),
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "n_controls": len({p.id for p in perturbations}),
            "n_perturbed": len(perturbations),
            "definition": DEFINITION,
        },
        # keep exactly what was judged: the source answers file may change later
        "controls": {i: by_id[i].answer for i in dict.fromkeys(p.id for p in perturbations)},
        "perturbations": [asdict(p) for p in perturbations],
        "summary": summarise(perturbations, scores, generator.get("family")),
    }


def _short(name: str) -> str:
    return name.split(":", 1)[-1].removeprefix("openai/")


def render(result: Mapping[str, Any]) -> str:
    meta, summary = result["meta"], result["summary"]
    gen = meta["generator"]
    per_judge = summary["per_judge"]
    lines = [
        "## goldpass judge audit: do the judges catch planted errors?",
        "",
        f"System `{meta['system']}` · generator `{gen.get('model') or '?'}` "
        f"({gen.get('family') or 'unknown family'}) · {meta['n_controls']} answers as controls, "
        f"{meta['n_perturbed']} copies with one planted unsupported claim each · "
        f"{meta['timestamp']}",
        "",
        f"A **miss** is {DEFINITION.removeprefix('a miss is ')}. **Drop** is control minus "
        "perturbed faithfulness, paired per answer. Lower miss rate is better.",
        "",
        "| Judge | Family | Planted | n | Missed | Miss rate (95% CI) | Mean drop |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, j in per_judge.items():
        fam = (j["family"] or "?") + (
            " (same as generator)" if j["same_family_as_generator"] else ""
        )
        for kind, s in j["by_kind"].items():
            if not s["n"]:
                continue
            lo, hi = s["ci95"]
            lines.append(
                f"| `{_short(name)}` | {fam} | {kind} | {s['n']} | {s['missed']} | "
                f"{s['miss_rate']:.2f} ({lo:.2f}–{hi:.2f}) | {s['mean_drop']:.2f} |"
            )
    lines += [
        "",
        "| Judge | Controls | Mean control faithfulness | Controls below 1.00 | Failed calls |",
        "|---|---|---|---|---|",
    ]
    for name, j in per_judge.items():
        cm = "—" if j["control_mean"] is None else f"{j['control_mean']:.2f}"
        lines.append(
            f"| `{_short(name)}` | {j['controls']} | {cm} | {j['controls_below_1']} | "
            f"{len(j['failures'])} |"
        )
    for c in summary["comparisons"]:
        a, b = (_short(x) for x in c["judges"])
        lines += [
            "",
            f"**`{a}` vs `{b}`** on {c['pairs']} shared planted errors: missed by both "
            f"{c['both_missed']}, only `{a}` {c['only_first_missed']}, only `{b}` "
            f"{c['only_second_missed']} (exact McNemar p = {c['mcnemar_p']:.2f}).",
        ]
    lines += [
        "",
        "Controls below 1.00 are correct answers the judge still marked partly unsupported: a "
        "judge that is strict everywhere catches more planted errors for the wrong reason, so "
        "read the two tables together.",
        "",
        "<details><summary>Every planted error and its scores</summary>",
        "",
        "| Item | Planted | Change | "
        + " | ".join(f"`{_short(n)}` control → perturbed" for n in per_judge)
        + " |",
        "|---|---|---|" + "---|" * len(per_judge),
    ]
    rows = {n: {(r["id"], r["kind"]): r for r in j["rows"]} for n, j in per_judge.items()}
    for p in result["perturbations"]:
        change = (
            f"{p['original']} → {p['planted']}" if p["kind"] == "number" else f"+ “{p['planted']}”"
        )
        cells = []
        for n in per_judge:
            r = rows[n].get((p["id"], p["kind"]))
            cells.append(
                "not judged"
                if r is None
                else f"{r['control']:.2f} → {r['perturbed']:.2f}"
                + (" (missed)" if r["missed"] else "")
            )
        lines.append(f"| `{p['id']}` | {p['kind']} | {change} | " + " | ".join(cells) + " |")
    lines += ["", "</details>", ""]
    return "\n".join(lines)
