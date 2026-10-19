"""Optional LLM-judge metrics over any OpenAI-compatible chat API. No SDKs, no RAGAS.

- faithfulness: the answer is split into atomic claims, then each claim is checked against the
  contexts the generator saw; score = supported claims / claims (the RAGAS definition).
- answer_relevancy: a 1-5 rubric rating of how directly and completely the answer addresses
  the question, scaled to 0-1. This is NOT RAGAS' embedding-based answer relevancy; it needs no
  embedding model, and its numbers are not comparable with RAGAS'.

Every call is cached on disk and retried on rate limits. A judge never sees the reference
answer, gold ids or the generator's identity.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from goldpass.cache import Cache, NoCache, cache_key
from goldpass.retry import GaveUp, RetryableError, RetryPolicy, call_with_retry, parse_retry_after

PROMPT_VERSION = "2026-10-19.1"  # bump when prompts change: invalidates cached verdicts
RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}
MALFORMED_RETRIES = 2

CLAIMS_SYSTEM = """You split an answer into atomic factual claims.
A claim is one short, self-contained statement that can be checked on its own: resolve
pronouns, keep numbers, units and conditions, and drop citation markers, hedges and filler.
Do not add facts that are not in the answer. If the answer states no facts (for example it
only declines to answer), return an empty list.
Reply with JSON only: {"claims": ["...", "..."]}"""

CLAIMS_USER = """Question:
{question}

Answer:
{answer}"""

VERDICTS_SYSTEM = """You check claims against source passages.
A claim is supported only if the passages state it or it follows directly from them. Claims
that rely on outside knowledge, even if true, are not supported. Judge every claim.
Reply with JSON only:
{"verdicts": [{"claim": 1, "supported": true, "reason": "short reason"}, ...]}"""

VERDICTS_USER = """Passages:
{passages}

Claims:
{claims}"""

RELEVANCY_SYSTEM = """You rate how well an answer addresses a question, ignoring whether it is
true. Use this scale:
5 = answers the question directly and completely
4 = answers it, with minor gaps or some irrelevant content
3 = partly answers it, or answers a related question
2 = mostly off-topic or evasive, with a little relevant content
1 = does not address the question, or only declines
Reply with JSON only: {"score": 1-5, "reason": "short reason"}"""

RELEVANCY_USER = """Question:
{question}

Answer:
{answer}"""


class JudgeError(RuntimeError):
    pass


@dataclass(frozen=True)
class JudgeScore:
    value: float
    detail: Mapping[str, Any] = field(default_factory=dict)


class Judge(Protocol):
    name: str
    family: str | None

    def faithfulness(
        self, question: str, answer: str, contexts: Sequence[str]
    ) -> JudgeScore | None: ...

    def answer_relevancy(self, question: str, answer: str) -> JudgeScore: ...


@dataclass(frozen=True)
class JudgeSpec:
    provider: str
    model: str
    base_url: str
    api_key_env: str | None = None
    family: str | None = None
    json_mode: bool = True  # send response_format=json_object; turn off if a provider rejects it
    max_tokens: int = 4096  # reasoning models spend part of this on hidden reasoning
    timeout_s: float = 120.0
    extra_body: Mapping[str, Any] = field(default_factory=dict)  # e.g. {reasoning_effort: low}

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"


def parse_json(text: str) -> Any:
    """JSON from a model reply, tolerating code fences and text around one object."""
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


class LLMJudge:
    def __init__(
        self,
        spec: JudgeSpec,
        api_key: str | None,
        cache: Cache | None = None,
        retry: RetryPolicy | None = None,
        strip_pattern: str | None = r"\s*\[\d+\]",
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] | None = None,
        offline: bool = False,
    ):
        self.offline = offline  # answer only from the cache; a miss is an error, never a call
        self.spec = spec
        self.name = spec.name
        self.family = spec.family
        self.cache: Cache = cache or NoCache()
        self.retry = retry or RetryPolicy(attempts=8, backoff_s=2.0)
        self._strip = re.compile(strip_pattern) if strip_pattern else None
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(headers=headers, timeout=spec.timeout_s, transport=transport)
        self._sleep = sleep
        self._log = log
        self.calls = 0  # requests actually sent (cache misses)

    def close(self) -> None:
        self._client.close()

    def _clean(self, answer: str) -> str:
        """Citation markers are for readers; they only add noise to claim extraction."""
        return self._strip.sub("", answer).strip() if self._strip else answer.strip()

    def _post(self, payload: dict[str, Any]) -> str:
        url = self.spec.base_url.rstrip("/") + "/chat/completions"

        def call() -> httpx.Response:
            try:
                resp = self._client.post(url, json=payload)
            except httpx.TransportError as e:
                raise RetryableError(f"{type(e).__name__}: {e}") from e
            if resp.status_code in RETRY_STATUS:
                raise RetryableError(
                    f"HTTP {resp.status_code}", parse_retry_after(resp.headers.get("Retry-After"))
                )
            return resp

        def on_retry(attempt: int, wait: float, e: RetryableError) -> None:
            if self._log:
                self._log(f"  {self.name}: {e}, retrying in {wait:.1f}s")

        try:
            resp = call_with_retry(call, self.retry, self._sleep, on_retry)
        except GaveUp as e:
            raise JudgeError(str(e)) from e
        self.calls += 1
        if resp.status_code >= 400:
            raise JudgeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise JudgeError(f"unexpected response shape: {resp.text[:300]}") from e
        if not isinstance(content, str):
            raise JudgeError(f"no text content in response: {resp.text[:300]}")
        return content

    def chat(self, system: str, user: str) -> Any:
        """One JSON-returning chat call, cached by its full request."""
        payload: dict[str, Any] = {
            "model": self.spec.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": self.spec.max_tokens,
            **self.spec.extra_body,
        }
        if self.spec.json_mode:
            payload["response_format"] = {"type": "json_object"}
        key = cache_key({"v": PROMPT_VERSION, "base_url": self.spec.base_url, **payload})
        if (hit := self.cache.get(key)) is not None:
            return parse_json(hit)
        if self.offline:
            raise JudgeError("not in the judge cache (offline: no API call made)")
        for attempt in range(MALFORMED_RETRIES + 1):
            content = self._post(payload)
            try:
                parsed = parse_json(content)
            except json.JSONDecodeError:
                if attempt == MALFORMED_RETRIES:
                    raise JudgeError(f"reply is not JSON: {content[:200]!r}") from None
                continue
            self.cache.set(key, content)
            return parsed
        raise AssertionError("unreachable")

    def faithfulness(
        self, question: str, answer: str, contexts: Sequence[str]
    ) -> JudgeScore | None:
        """None when the answer makes no factual claims (nothing to be faithful about)."""
        answer = self._clean(answer)
        got = self.chat(CLAIMS_SYSTEM, CLAIMS_USER.format(question=question, answer=answer))
        claims = (
            got if isinstance(got, list) else got.get("claims") if isinstance(got, dict) else None
        )
        if not isinstance(claims, list) or not all(isinstance(c, str) for c in claims):
            raise JudgeError(f"claims reply has no list of strings: {got!r:.200}")
        claims = [c.strip() for c in claims if c.strip()]
        if not claims:
            return None
        passages = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, start=1))
        numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, start=1))
        got = self.chat(VERDICTS_SYSTEM, VERDICTS_USER.format(passages=passages, claims=numbered))
        # some models return the bare list instead of {"verdicts": [...]}
        verdicts = (
            got if isinstance(got, list) else got.get("verdicts") if isinstance(got, dict) else None
        )
        if not isinstance(verdicts, list):
            raise JudgeError(f"verdicts reply has no list: {got!r:.200}")
        supported: dict[int, bool] = {}
        for v in verdicts:
            if isinstance(v, dict) and isinstance(v.get("claim"), int):
                supported[v["claim"]] = v.get("supported") is True
        # a claim the judge skipped counts as unsupported: it was not shown to be grounded
        flags = [supported.get(i, False) for i in range(1, len(claims) + 1)]
        detail = {
            "claims": [{"claim": c, "supported": f} for c, f in zip(claims, flags, strict=True)],
            "unjudged": len(claims) - len(set(supported) & set(range(1, len(claims) + 1))),
        }
        return JudgeScore(sum(flags) / len(claims), detail)

    def answer_relevancy(self, question: str, answer: str) -> JudgeScore:
        answer = self._clean(answer)
        got = self.chat(RELEVANCY_SYSTEM, RELEVANCY_USER.format(question=question, answer=answer))
        score = got.get("score") if isinstance(got, dict) else None
        if isinstance(score, str) and score.strip().isdigit():
            score = int(score)
        if isinstance(score, bool) or not isinstance(score, int | float) or not 1 <= score <= 5:
            raise JudgeError(f"relevancy reply has no score from 1 to 5: {got!r:.200}")
        return JudgeScore((float(score) - 1) / 4, {"rating": score, "reason": got.get("reason")})
