"""A tiny, fake RAG system for trying goldpass. Standard library only: no network, no keys.

The corpus is SYNTHETIC: documentation for "Nimbus Storage", a product that does not exist.
Retrieval is keyword overlap and the "generator" quotes the best passage, so every run gives
the same answers.

Two modes:
- `good`: filters out the legacy document, refuses when no passage matches well enough.
- `regressed`: the kind of change that slips through without a gate. The legacy filter is gone
  (so the wrong-document trap cites an outdated page) and the refusal threshold is removed
  (so it answers questions the corpus cannot support).
"""

from __future__ import annotations

import re
from typing import Any

PASSAGES = [
    ("billing#1", "billing", "Nimbus Storage bills storage at 0.02 EUR per gigabyte per month, "
     "measured as the daily average."),
    ("billing#2", "billing", "Invoices are issued on the first day of each month and are due "
     "within 30 days."),
    ("billing#3", "billing", "Accounts with unpaid invoices after 60 days are suspended until "
     "payment is received."),
    ("security#1", "security", "All objects are encrypted at rest with AES-256, using keys "
     "managed by Nimbus."),
    ("security#2", "security", "Customer-managed encryption keys are available on the "
     "Enterprise plan."),
    ("security#3", "security", "Access tokens expire after 12 hours and must be refreshed with "
     "the refresh token."),
    ("limits#1", "limits", "The API allows 100 requests per second per account; extra requests "
     "receive HTTP 429."),
    ("limits#2", "limits", "A single object upload is limited to 5 gigabytes; larger files must "
     "use multipart upload."),
    ("limits#3", "limits", "Multipart uploads support up to 10000 parts of at least 5 megabytes "
     "each."),
    ("legacy-v1#1", "legacy-v1", "Legacy v1 API: a single object upload is limited to 2 "
     "gigabytes."),
]  # fmt: skip
LEGACY = {"legacy-v1"}
# fmt: off
STOP = {  # common words, plus the product name (it is in every passage)
    "a", "an", "the", "is", "are", "of", "to", "in", "on", "for", "and", "or", "with", "by",
    "at", "how", "what", "when", "which", "who", "does", "do", "can", "many", "much", "long",
    "there", "their", "my", "our", "you", "your", "per", "nimbus", "storage",
}
# fmt: on
TOP_K, RETRIEVE_K, MIN_SCORE = 3, 6, 0.34


def tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words} - STOP


def ask(question: str, mode: str = "good") -> dict[str, Any]:
    q = tokens(question)
    order = list(enumerate(PASSAGES))
    if mode == "regressed":
        order.reverse()  # ties now go to later passages, i.e. the legacy page
    else:
        order = [(i, p) for i, p in order if p[1] not in LEGACY]
    scored = [(len(q & tokens(text)) / max(len(q), 1), i, pid, doc, text)
              for i, (pid, doc, text) in order]  # fmt: skip
    ranked = sorted(scored, key=lambda s: -s[0])  # stable: keeps `order` on ties
    retrieved = [s for s in ranked if s[0] > 0][:RETRIEVE_K]
    contexts = retrieved[:TOP_K]

    def passage(s: tuple[float, int, str, str, str]) -> dict[str, Any]:
        return {"id": s[2], "doc_id": s[3], "text": s[4], "score": round(s[0], 3)}

    weak = not contexts or contexts[0][0] < MIN_SCORE
    if weak and mode != "regressed":
        return {"answer": "I cannot answer that from the documentation.", "refused": True,
                "contexts": [passage(s) for s in contexts],
                "retrieved": [passage(s) for s in retrieved], "citations": []}  # fmt: skip
    if not contexts:  # regressed: answers anyway, from whatever ranks first
        contexts = ranked[:1]
    best = contexts[0]
    return {
        "answer": f"{best[4]} [1]",
        "refused": False,
        "contexts": [passage(s) for s in contexts],
        "retrieved": [passage(s) for s in retrieved],
        "citations": [{"passage_id": best[2], "doc_id": best[3]}],
    }
