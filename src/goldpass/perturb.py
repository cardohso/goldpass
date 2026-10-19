"""Planted errors for auditing an LLM judge: does it notice an answer that is no longer faithful?

Each answered question yields up to two copies of its answer, each with exactly one claim its
contexts do not support:

- `number`: one number the contexts support is changed to the nearest value that appears
  nowhere in them. Range endpoints are skipped and upper bounds move down ("within four
  months" -> "within three months"), so the new claim is never implied by the old one.
- `sentence`: one unsupported sentence is appended, with a citation marker, as a
  hallucinating generator would write it. A template is skipped if its marker word appears in
  the contexts, so a planted sentence is never one the passages might back.

Everything is deterministic (no randomness, no LLM). Adapted from the perturbation test in
rail-regulation-rag, with domain-neutral default sentences.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from goldpass.record import Record

CITATION = re.compile(r"\[\d+\]")
# "1 000" (spaced thousands), "0,225" / "2.25" (decimals) and "4.2.3" (section references)
NUMBER = re.compile(r"(?<![\w.,])(\d{1,3}(?: \d{3})+|\d+(?:[.,]\d+)*)(?![\w]|[.,]\d)")
WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve"]  # fmt: skip
WORD_NUMBER = re.compile(r"\b(" + "|".join(WORDS) + r")\b", re.IGNORECASE)
RANGE_BEFORE = re.compile(r"(?:\d\s*(?:to|-|–)|\bbetween)\s*$", re.IGNORECASE)
RANGE_AFTER = re.compile(r"^\s*(?:to|-|–)\s*\d", re.IGNORECASE)
# After a lower bound used as a condition ("speed greater than 250 km/h"), raising the number
# narrows the condition, so the new claim follows from the old one. Such numbers are skipped.
LOWER_BOUND = re.compile(
    r"\b(?:greater than|more than|higher than|faster than|longer than|above|over|exceeding)\s+$",
    re.IGNORECASE,
)
UPPER_BOUND = re.compile(
    r"\b(?:within|up to|at most|maximum|not more than|no more than|less than|below)\s+$",
    re.IGNORECASE,
)

# (marker, sentence): plausible-sounding claims about almost any documented rule or product,
# none of which a passage supports unless it mentions the marker.
DEFAULT_SENTENCES: tuple[tuple[str, str], ...] = (
    ("2017", "This was last revised in 2017 and replaced the earlier version."),
    ("written request", "Exceptions can be granted on written request."),
    ("auditor", "Compliance must be re-checked by an independent auditor every year."),
    ("fewer than 50", "Organisations with fewer than 50 employees are exempt."),
    ("pilot", "It started as a voluntary pilot and only later became mandatory."),
    ("weekend", "A reduced version applies at weekends and on public holidays."),
)


@dataclass(frozen=True)
class Perturbation:
    id: str  # golden item id
    kind: str  # "number" or "sentence"
    original: str  # the span replaced ("" for an appended sentence)
    planted: str  # what was put in its place
    position: int  # where `planted` starts in `answer`
    answer: str  # the full perturbed answer


def _key(token: str) -> str:
    """Normalise a number so '0,8', '0.80' and '0.8' compare equal; references stay strings."""
    t = token.replace(" ", "").replace(",", ".")
    if t.count(".") > 1:
        return t
    try:
        return format(Decimal(t).normalize(), "f")
    except InvalidOperation:
        return t


def numbers_in(text: str) -> set[str]:
    keys = {_key(m.group(1)) for m in NUMBER.finditer(text)}
    keys |= {str(WORDS.index(m.group(1).lower()) + 1) for m in WORD_NUMBER.finditer(text)}
    return keys


def _digit_candidates(token: str, down_first: bool = False) -> Iterator[str]:
    """Change the last non-zero digit by +1, +2, ... then -1, -2, ... (or down first), never
    carrying, so the format stays the same: 65 -> 66, 2.25 -> 2.26, 1 000 -> 2 000."""
    digits = [i for i, ch in enumerate(token) if ch.isdigit()]
    pos = ([i for i in digits if token[i] != "0"] or digits)[-1]
    d = int(token[pos])
    leading = not any(ch.isdigit() for ch in token[:pos])
    up, down = [*range(d + 1, 10)], [*range(d - 1, 0 if leading else -1, -1)]
    for new in down + up if down_first else up + down:
        yield token[:pos] + str(new) + token[pos + 1 :]


def _word_candidates(token: str, down_first: bool = False) -> Iterator[str]:
    n = WORDS.index(token.lower())
    up, down = [*range(n + 1, len(WORDS))], [*range(n - 1, -1, -1)]
    for i in down + up if down_first else up + down:
        yield WORDS[i].capitalize() if token[0].isupper() else WORDS[i]


def perturb_number(item_id: str, answer: str, contexts: Sequence[str]) -> Perturbation | None:
    """Replace the first number the contexts support; None if no number qualifies."""
    in_contexts = set().union(*(numbers_in(c) for c in contexts)) if contexts else set()
    markers = [m.span() for m in CITATION.finditer(answer)]
    spans: list[tuple[int, int, str, Callable[[str, bool], Iterator[str]], str]] = [
        (m.start(1), m.end(1), m.group(1), _digit_candidates, _key(m.group(1)))
        for m in NUMBER.finditer(answer)
        if not any(a <= m.start() < b for a, b in markers)
    ]
    spans += [
        (m.start(1), m.end(1), m.group(1), _word_candidates,
         str(WORDS.index(m.group(1).lower()) + 1))
        for m in WORD_NUMBER.finditer(answer)
    ]  # fmt: skip
    for start, end, token, candidates, key in sorted(spans, key=lambda s: s[0]):
        before, after = answer[:start], answer[end:]
        if key not in in_contexts or RANGE_BEFORE.search(before) or RANGE_AFTER.search(after):
            continue
        if LOWER_BOUND.search(before):
            continue
        for new in candidates(token, bool(UPPER_BOUND.search(before))):
            if numbers_in(new) & in_contexts:
                continue
            return Perturbation(item_id, "number", token, new, start, before + new + after)
    return None


def add_sentence(
    item_id: str,
    answer: str,
    contexts: Sequence[str],
    sentences: Sequence[tuple[str, str]] = DEFAULT_SENTENCES,
) -> Perturbation:
    """Append one unsupported sentence, picked by a hash of the item id."""
    text = " ".join(contexts).lower()
    start = int(hashlib.sha256(item_id.encode()).hexdigest()[:8], 16) % len(sentences)
    for i in range(len(sentences)):
        marker, sentence = sentences[(start + i) % len(sentences)]
        if marker.lower() not in text:
            planted = f"{sentence} [1]"
            kept = answer.rstrip()
            return Perturbation(
                item_id, "sentence", "", planted, len(kept) + 1, f"{kept} {planted}"
            )
    raise ValueError(f"every planted sentence is supported by the contexts of {item_id}")


def context_texts(record: Record) -> list[str] | None:
    """The context texts, or None if any is missing (the judge needs them all)."""
    texts = [p.text for p in record.contexts or ()]
    if not texts or any(t is None for t in texts):
        return None
    return [t for t in texts if t is not None]


def build(
    records: Sequence[Record], sentences: Sequence[tuple[str, str]] = DEFAULT_SENTENCES
) -> list[Perturbation]:
    """Perturbations for every answered record whose contexts have text."""
    out = []
    for r in records:
        texts = context_texts(r)
        if r.refused or r.error or texts is None:
            continue
        if number := perturb_number(r.id, r.answer, texts):
            out.append(number)
        out.append(add_sentence(r.id, r.answer, texts, sentences))
    return out
