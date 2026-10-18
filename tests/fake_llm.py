"""A fake OpenAI-compatible chat API for offline tests. It answers by prompt kind."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx

Reply = str | httpx.Response | Exception


def chat_response(content: str) -> httpx.Response:
    return httpx.Response(
        200, json={"choices": [{"message": {"role": "assistant", "content": content}}]}
    )


class FakeLLM:
    """`replies` maps a prompt kind (claims, verdicts, relevancy) to a list of replies used in
    order (the last one repeats), or to a function of the user message."""

    def __init__(self, **replies: list[Reply] | Callable[[str], Reply]):
        self.replies = replies
        self.requests: list[dict[str, Any]] = []

    @staticmethod
    def kind(system: str) -> str:
        if "atomic factual claims" in system:
            return "claims"
        if "check claims" in system:
            return "verdicts"
        return "relevancy"

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body | {"_auth": request.headers.get("Authorization")})
        system, user = (m["content"] for m in body["messages"])
        script = self.replies[self.kind(system)]
        reply = (
            script(user) if callable(script) else (script.pop(0) if len(script) > 1 else script[0])
        )
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, httpx.Response) else chat_response(reply)

    def count(self, kind: str) -> int:
        return sum(self.kind(r["messages"][0]["content"]) == kind for r in self.requests)


def default_llm() -> FakeLLM:
    """Supports every claim whose text appears in the passages; rates every answer 5."""

    def claims(user: str) -> str:
        answer = user.split("Answer:\n", 1)[1]
        return json.dumps({"claims": [s.strip() for s in answer.split(".") if s.strip()]})

    def verdicts(user: str) -> str:
        passages, claims = user.split("Claims:\n", 1)
        out = []
        for line in claims.splitlines():
            n, claim = line.split(". ", 1)
            out.append({"claim": int(n), "supported": claim in passages, "reason": ""})
        return json.dumps({"verdicts": out})

    return FakeLLM(claims=claims, verdicts=verdicts, relevancy=[json.dumps({"score": 5})])
