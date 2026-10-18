"""What the system under test returned for one golden item. Adapters produce these.

`None` and empty mean different things: `contexts=None` says the system did not report its
contexts (metrics that need them are skipped), `contexts=()` says it retrieved nothing (they
score 0). The same holds for `citations` and `retrieved`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Passage:
    id: str
    doc_id: str | None = None
    text: str | None = None
    score: float | None = None


@dataclass(frozen=True)
class Citation:
    passage_id: str | None = None
    doc_id: str | None = None


@dataclass(frozen=True)
class Record:
    id: str
    answer: str = ""
    refused: bool = False
    contexts: tuple[Passage, ...] | None = None  # ranked passages the generator saw
    retrieved: tuple[Passage, ...] | None = None  # optional wider list, before rerank/filter
    citations: tuple[Citation, ...] | None = None
    latency_s: float | None = None
    error: str | None = None  # set when the call failed; the item is then not scored
    extra: Mapping[str, Any] = field(default_factory=dict)

    @staticmethod
    def passages(items: Sequence[Passage | Mapping[str, Any]] | None) -> tuple[Passage, ...] | None:
        if items is None:
            return None
        return tuple(p if isinstance(p, Passage) else Passage(**p) for p in items)

    @staticmethod
    def cites(items: Sequence[Citation | Mapping[str, Any]] | None) -> tuple[Citation, ...] | None:
        if items is None:
            return None
        return tuple(c if isinstance(c, Citation) else Citation(**c) for c in items)
