import json
from pathlib import Path

import httpx
import pytest

from goldpass.cache import DiskCache
from goldpass.judge import JudgeError, JudgeSpec, LLMJudge, parse_json
from goldpass.retry import RetryPolicy

from .fake_llm import FakeLLM, chat_response, default_llm

SPEC = JudgeSpec("groq", "openai/gpt-oss-120b", "https://judge.test/v1", family="openai")
CONTEXTS = ["The stationary noise limit for EMUs is 65 dB", "Measured at 7,5 m from the track"]


def judge(llm: FakeLLM, tmp_path: Path | None = None, **kw: object) -> tuple[LLMJudge, list[float]]:
    waits: list[float] = []
    cache = DiskCache(tmp_path / "c.sqlite") if tmp_path else None
    j = LLMJudge(
        SPEC,
        "k3y",
        cache=cache,
        transport=httpx.MockTransport(llm),
        sleep=waits.append,
        **kw,  # type: ignore[arg-type]
    )
    return j, waits


@pytest.mark.parametrize(
    "text",
    ['{"score": 4}', '```json\n{"score": 4}\n```', 'Sure! {"score": 4} Hope that helps.'],
)
def test_parse_json(text: str) -> None:
    assert parse_json(text) == {"score": 4}


def test_faithfulness_counts_supported_claims() -> None:
    j, _ = judge(default_llm())
    answer = "The stationary noise limit for EMUs is 65 dB [1]. EMUs are painted blue"
    s = j.faithfulness("What is the EMU noise limit?", answer, CONTEXTS)
    assert s is not None and s.value == 0.5
    assert [c["supported"] for c in s.detail["claims"]] == [True, False]


def test_citation_markers_are_stripped_before_judging() -> None:
    llm = default_llm()
    j, _ = judge(llm)
    j.faithfulness("q", "The stationary noise limit for EMUs is 65 dB [1]", CONTEXTS)
    assert "[1]" not in llm.requests[0]["messages"][1]["content"]


def test_request_shape() -> None:
    llm = default_llm()
    spec = JudgeSpec("groq", "m", "https://judge.test/v1", extra_body={"reasoning_effort": "low"})
    LLMJudge(spec, "k3y", transport=httpx.MockTransport(llm)).answer_relevancy("q", "a")
    body = llm.requests[0]
    assert body["_auth"] == "Bearer k3y"
    assert body["temperature"] == 0 and body["max_tokens"] == 4096
    assert body["response_format"] == {"type": "json_object"}
    assert body["reasoning_effort"] == "low"


def test_no_claims_means_no_score() -> None:
    llm = FakeLLM(claims=['{"claims": []}'])
    j, _ = judge(llm)
    assert j.faithfulness("q", "I cannot answer that.", CONTEXTS) is None
    assert llm.count("verdicts") == 0


def test_skipped_claims_count_as_unsupported() -> None:
    llm = FakeLLM(
        claims=['{"claims": ["a", "b", "c"]}'],
        verdicts=['{"verdicts": [{"claim": 1, "supported": true}]}'],
    )
    s = judge(llm)[0].faithfulness("q", "a. b. c", CONTEXTS)
    assert s is not None and s.value == pytest.approx(1 / 3) and s.detail["unjudged"] == 2


@pytest.mark.parametrize(("rating", "value"), [(5, 1.0), (3, 0.5), (1, 0.0), ("4", 0.75)])
def test_relevancy_scale(rating: object, value: float) -> None:
    llm = FakeLLM(relevancy=[json.dumps({"score": rating, "reason": "r"})])
    assert judge(llm)[0].answer_relevancy("q", "a").value == value


@pytest.mark.parametrize("bad", ['{"score": 7}', '{"score": true}', '{"rating": 5}', "[]"])
def test_relevancy_rejects_bad_scores(bad: str) -> None:
    with pytest.raises(JudgeError, match="no score from 1 to 5"):
        judge(FakeLLM(relevancy=[bad]))[0].answer_relevancy("q", "a")


def test_malformed_json_is_retried_then_fails() -> None:
    llm = FakeLLM(relevancy=["not json", '{"score": 5}'])
    assert judge(llm)[0].answer_relevancy("q", "a").value == 1.0
    assert llm.count("relevancy") == 2
    with pytest.raises(JudgeError, match="reply is not JSON"):
        judge(FakeLLM(relevancy=["nope"]))[0].answer_relevancy("q", "a")


def test_rate_limits_honour_retry_after() -> None:
    llm = FakeLLM(
        relevancy=[
            httpx.Response(429, headers={"Retry-After": "12"}),
            httpx.Response(503),
            chat_response('{"score": 5}'),
        ]
    )
    j, waits = judge(llm)
    assert j.answer_relevancy("q", "a").value == 1.0
    assert waits == [12.0, 4.0]  # Retry-After, then backoff 2 s x 2^(2-1)


def test_auth_errors_fail_fast() -> None:
    j, waits = judge(FakeLLM(relevancy=[httpx.Response(401, text="invalid api key")]))
    with pytest.raises(JudgeError, match="HTTP 401: invalid api key"):
        j.answer_relevancy("q", "a")
    assert waits == []


def test_gives_up_after_retries() -> None:
    llm = FakeLLM(relevancy=[httpx.Response(429)])
    j, _ = judge(llm, retry=RetryPolicy(attempts=3))
    with pytest.raises(JudgeError, match="gave up after 3 attempts: HTTP 429"):
        j.answer_relevancy("q", "a")


def test_cache_avoids_repeat_calls_and_survives_restarts(tmp_path: Path) -> None:
    llm = default_llm()
    j, _ = judge(llm, tmp_path)
    first = j.faithfulness("q", "The stationary noise limit for EMUs is 65 dB", CONTEXTS)
    sent = len(llm.requests)
    j2, _ = judge(llm, tmp_path)  # a new process, same cache file
    assert j2.faithfulness("q", "The stationary noise limit for EMUs is 65 dB", CONTEXTS) == first
    assert len(llm.requests) == sent and j2.calls == 0
    j2.faithfulness("q", "a different answer", CONTEXTS)
    assert len(llm.requests) > sent


def test_malformed_replies_are_not_cached(tmp_path: Path) -> None:
    llm = FakeLLM(relevancy=["not json", "still not", "nope", '{"score": 5}'])
    with pytest.raises(JudgeError):
        judge(llm, tmp_path)[0].answer_relevancy("q", "a")
    assert judge(llm, tmp_path)[0].answer_relevancy("q", "a").value == 1.0
