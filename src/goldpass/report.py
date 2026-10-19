"""Markdown for the GitHub job summary (and anyone reading results by hand)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from goldpass.metrics import METRICS
from goldpass.stats import wilson

YES_NO = {"grounding", "answered_rate", "refusal_rate", "hit_rate", "errors"}  # 0/1 per item


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _count(v: Any) -> str:
    return "—" if v is None else str(int(v))


def _interval(value: Any, n: int) -> str:
    ci = wilson(float(value), n) if value is not None and n else None
    return "—" if ci is None else f"{ci[0]:.2f}–{ci[1]:.2f}"


def _changed(d: Mapping[str, Any]) -> str:
    if not d["worse"] and not d["better"]:
        return "—"
    return f"{d['worse']} worse, {d['better']} better (p {d['p_value']:.2f})"


def _signed(v: float) -> str:
    return f"{v:+.2f}" if abs(v) >= 0.005 else "±0.00"


def _who(meta: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    gen = meta.get("generator") or {}
    parts = [f"System `{meta['system']}`"]
    if gen.get("model") or gen.get("family"):
        parts.append(
            f"generator `{gen.get('model') or '?'}` ({gen.get('family') or 'unknown family'})"
        )
    gating = next((j for j in result.get("judges", []) if j["gating"]), None)
    if gating:
        parts.append(f"judge `{gating['name']}` ({gating['family'] or 'unknown family'})")
    else:
        parts.append("no LLM judge")
    parts.append(f"{meta['n_items']} golden items")
    if meta.get("k"):
        parts.append(f"k={meta['k']}")
    if meta.get("commit"):
        parts.append(f"commit `{meta['commit']}`")
    parts.append(meta["timestamp"])
    return " · ".join(parts)


def render(result: Mapping[str, Any]) -> str:
    meta, metrics = result["meta"], result["metrics"]
    gate = result.get("gate")
    title = "goldpass"
    if gate is not None:
        title += ": PASSED ✅" if gate["passed"] else ": FAILED ❌"
    lines = [f"## {title}", "", _who(meta, result), ""]
    if meta.get("allow_same_family") and result.get("judges"):
        lines += [
            "> ⚠️ Gated with `--allow-same-family`: the judge was not checked against the "
            "generator's model family, so judge scores may be inflated by self-preference.",
            "",
        ]

    checks = {c["metric"]: c for c in (gate or {}).get("checks", []) if c["scope"] == "all"}
    base = (gate or {}).get("baseline")
    deltas = {d["metric"]: d for d in base["deltas"]} if base else {}
    head = "| Metric | Value | 95% CI | n | Threshold |"
    head += (" Baseline | Δ | Questions changed |" if base else "") + " Status |"
    lines += [head, "|---|---|---|---|---|" + ("---|---|---|" if base else "") + "---|"]
    for name, info in METRICS.items():
        m = metrics.get(name, {})
        c, d = checks.get(name), deltas.get(name)
        if m.get("value") is None and c is None:
            continue  # not measured and not gated: leave it out
        ok = (c is None or c["ok"]) and (d is None or d["ok"])
        status = "✅" if ok else "❌"
        if c is None and d is None:
            status = ""
        show = _count if name == "errors" else _fmt
        n = "" if name == "errors" else m.get("n", 0)
        ci = "" if name == "errors" else _interval(m.get("value"), m.get("n", 0))
        rule = c["rule"] if c else "—"
        row = f"| {info.label} | {show(m.get('value'))} | {ci} | {n} | {rule} |"
        if base:
            row += f" {show(d['baseline']) if d else '—'} | {_signed(d['change']) if d else '—'} |"
            row += f" {_changed(d) if d else '—'} |"
        lines.append(row + f" {status} |")
    lines += [
        "",
        "_95% CI: Wilson score interval over n questions (conservative for scores that are not "
        "yes/no). With few questions it is wide: one question moves a rate by 1/n._",
    ]
    fragile = [c for c in (gate or {}).get("checks", []) if c.get("fragile")]
    if fragile:
        names = ", ".join(_check_label(c) for c in fragile)
        lines += ["", f"> ⚠️ **Fragile:** passes, but a single question could fail it: "
                      f"{names}."]  # fmt: skip

    lines += _failures(result, gate)
    lines += _within_noise(gate)
    if base and base["warnings"]:
        lines += ["", "**Baseline comparison caveats:**", ""] + [f"- {w}" for w in base["warnings"]]
    lines += _judges(result)
    lines += _tags(result, gate)
    lines += _documents(result)
    lines += _skipped(result)
    lines += _items(result)
    return "\n".join(lines) + "\n"


def _check_label(c: Mapping[str, Any]) -> str:
    return METRICS[c["metric"]].label + ("" if c["scope"] == "all" else f" [{c['scope']}]")


def _failures(result: Mapping[str, Any], gate: Mapping[str, Any] | None) -> list[str]:
    if not gate or gate["passed"]:
        return []
    out = ["", "### What failed", ""]
    for c in gate["checks"]:
        if c["ok"]:
            continue
        margin = c.get("margin")
        away = f"; {margin} question{'s' if margin != 1 else ''} from passing" if margin else ""
        out.append(f"- **{_check_label(c)}**: {c['message']} (needs {c['rule']}{away})")
        if c["worst"] and c["metric"] in YES_NO:
            out.append("  - failing questions: " + ", ".join(f"`{i}`" for i, _ in c["worst"]))
        elif c["worst"]:
            shown = ", ".join(f"`{i}` ({_fmt(v)})" for i, v in c["worst"])
            out.append(f"  - lowest-scoring questions: {shown}")
    base = gate.get("baseline") or {}
    for d in base.get("deltas", []):
        if d["ok"]:
            continue
        if d["metric"] == "errors":
            out.append(
                f"- **Errors**: {_count(d['baseline'])} → {_count(d['current'])} since the "
                f"baseline (allowed increase {d['allowed_drop']:g})"
            )
        else:
            out.append(
                f"- **{METRICS[d['metric']].label}**: {_fmt(d['baseline'])} → "
                f"{_fmt(d['current'])}, dropped {d['drop']:.2f} from the baseline (allowed "
                f"{d['allowed_drop']:g}); {_changed(d)}"
            )
        if d["regressed"]:
            shown = ", ".join(f"`{i}` ({_fmt(b)} → {_fmt(a)})" for i, b, a in d["regressed"])
            out.append(f"  - questions that got worse: {shown}")
        if base.get("alpha") is None and d["p_value"] > 0.05 and d["metric"] != "errors":
            out.append(
                f"  - p = {d['p_value']:.2f}: a change this size could be noise. Set "
                "`baseline.alpha` to fail only on significant drops."
            )
    errors = [it for it in result["items"] if it.get("error")]
    if errors:
        out.append(
            f"- {len(errors)} {'item was' if len(errors) == 1 else 'items were'} not scored "
            "because the call failed, e.g. "
            f"`{errors[0]['id']}`: {errors[0]['error'][:120]}"
        )
    return out


def _within_noise(gate: Mapping[str, Any] | None) -> list[str]:
    base = (gate or {}).get("baseline") or {}
    noisy = [d for d in base.get("deltas", []) if d.get("within_noise")]
    if not noisy:
        return []
    out = ["", f"### Dropped, but within noise (alpha {base['alpha']:g})", ""]
    for d in noisy:
        out.append(
            f"- **{METRICS[d['metric']].label}** dropped {d['drop']:.2f} (allowed "
            f"{d['allowed_drop']:g}), but {_changed(d)}: not failed"
        )
    return out


def _judges(result: Mapping[str, Any]) -> list[str]:
    judges = result.get("judges") or []
    if len(judges) < 2:
        return []
    out = [
        "",
        "### Same answers, different judges",
        "",
        "| Metric | " + " | ".join(f"`{j['name']}` ({j['family'] or '?'})" for j in judges) + " |",
        "|---|" + "---|" * len(judges),
    ]
    for m in ("faithfulness", "answer_relevancy"):
        out.append(
            f"| {METRICS[m].label} | "
            + " | ".join(
                f"{_fmt(j['metrics'][m]['value'])} (n={j['metrics'][m]['n']})" for j in judges
            )
            + " |"
        )
    out += ["", "Only the first judge gates. The others are shown for comparison."]
    return out


def _tags(result: Mapping[str, Any], gate: Mapping[str, Any] | None) -> list[str]:
    by_tag = result.get("by_tag") or {}
    if not by_tag:
        return []
    shown = ("answered_rate", "grounding", "recall_at_k", "faithfulness")
    out = [
        "",
        "### By tag",
        "",
        "| Tag | " + " | ".join(METRICS[m].label for m in shown) + " |",
        "|---|" + "---|" * len(shown),
    ]
    for tag, agg in by_tag.items():
        cells = [
            f"{_fmt(agg[m]['value'])} (n={agg[m]['n']})" if agg[m]["n"] else "—" for m in shown
        ]
        out.append(f"| `{tag}` | " + " | ".join(cells) + " |")
    return out


def _documents(result: Mapping[str, Any]) -> list[str]:
    rows = result.get("by_document") or []
    if not rows:
        return []
    out = [
        "",
        "### Grounding per source document",
        "",
        "| Document | Questions | Answered | Grounding | Also cited |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| `{r['doc_id']}` | {r['questions']} | {r['answered']} | "
            f"{_fmt(r['grounding']['value'])} | {', '.join(r['cited_other_docs']) or '—'} |"
        )
    return out


def _skipped(result: Mapping[str, Any]) -> list[str]:
    skipped = result.get("skipped") or {}
    if not skipped:
        return []
    out = ["", "<details><summary>Metrics skipped on some items</summary>", ""]
    out += [f"- {count} × {note}" for note, count in skipped.items()]
    return out + ["", "</details>"]


def _items(result: Mapping[str, Any]) -> list[str]:
    cols = ["recall_at_k", "grounding", "faithfulness", "answer_relevancy"]
    out = [
        "",
        "<details><summary>Per-item scores</summary>",
        "",
        "| Item | Expected | Refused | " + " | ".join(METRICS[c].label for c in cols) + " | Note |",
        "|---|---|---|" + "---|" * len(cols) + "---|",
    ]
    for it in result["items"]:
        note = it.get("error") or ""
        cells = " | ".join(_fmt(it["metrics"].get(c)) for c in cols)
        refused = _fmt(it.get("refused"))
        out.append(f"| `{it['id']}` | {it['expected']} | {refused} | {cells} | {note[:80]} |")
    return out + ["", "</details>"]
