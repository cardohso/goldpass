"""Markdown for the GitHub job summary (and anyone reading results by hand)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from goldpass.metrics import METRICS

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
    head = "| Metric | Value | n | Threshold |" + (" Baseline | Δ |" if base else "") + " Status |"
    lines += [head, "|---|---|---|---|" + ("---|---|" if base else "") + "---|"]
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
        row = f"| {info.label} | {show(m.get('value'))} | {n} | {c['rule'] if c else '—'} |"
        if base:
            row += f" {show(d['baseline']) if d else '—'} | {_signed(d['change']) if d else '—'} |"
        lines.append(row + f" {status} |")

    lines += _failures(result, gate)
    if base and base["warnings"]:
        lines += ["", "**Baseline comparison caveats:**", ""] + [f"- {w}" for w in base["warnings"]]
    lines += _judges(result)
    lines += _tags(result, gate)
    lines += _documents(result)
    lines += _skipped(result)
    lines += _items(result)
    return "\n".join(lines) + "\n"


def _failures(result: Mapping[str, Any], gate: Mapping[str, Any] | None) -> list[str]:
    if not gate or gate["passed"]:
        return []
    out = ["", "### What failed", ""]
    for c in gate["checks"]:
        if c["ok"]:
            continue
        label = METRICS[c["metric"]].label + ("" if c["scope"] == "all" else f" [{c['scope']}]")
        out.append(f"- **{label}**: {c['message']} (needs {c['rule']})")
        if c["worst"] and c["metric"] in YES_NO:
            out.append("  - failing questions: " + ", ".join(f"`{i}`" for i, _ in c["worst"]))
        elif c["worst"]:
            shown = ", ".join(f"`{i}` ({_fmt(v)})" for i, v in c["worst"])
            out.append(f"  - lowest-scoring questions: {shown}")
    for d in (gate.get("baseline") or {}).get("deltas", []):
        if d["ok"]:
            continue
        drop = d["drop"]
        out.append(
            f"- **{METRICS[d['metric']].label}**: {_fmt(d['baseline'])} → {_fmt(d['current'])}, "
            f"dropped {drop:.2f} from the baseline (allowed {d['allowed_drop']:g})"
        )
        if d["regressed"]:
            shown = ", ".join(f"`{i}` ({_fmt(b)} → {_fmt(a)})" for i, b, a in d["regressed"])
            out.append(f"  - questions that got worse: {shown}")
    errors = [it for it in result["items"] if it.get("error")]
    if errors:
        out.append(
            f"- {len(errors)} items were not scored because the call failed, e.g. "
            f"`{errors[0]['id']}`: {errors[0]['error'][:120]}"
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
