# LLM judges

Judges are optional. Without a `judges:` section, or with `--no-judge`, goldpass makes no LLM
calls at all.

```yaml
generator: {model: gemini-2.5-flash}       # the model behind your system (for the guard)
judges:
  - provider: groq                          # gating judge: feeds the metrics and the gate
    model: openai/gpt-oss-120b
    extra_body: {reasoning_effort: low}     # passed through to the API
  - provider: gemini                        # extra judges: reported side by side, never gate
    model: gemini-2.5-pro
judging:
  cache_dir: .goldpass/cache
```

Providers with presets are `groq`, `gemini`, `openai`, `openrouter`, `mistral` and `ollama`. Any
other OpenAI-compatible endpoint works with `base_url` (and `api_key_env` if it needs a key).
Keys are read from the environment (`GROQ_API_KEY`, `GEMINI_API_KEY`, …) and checked before the
system under test is called, so a missing key fails fast.

## What the judge measures

- **faithfulness**: the answer is split into atomic claims, then each claim is checked against
  the contexts the generator saw. The score is supported claims ÷ claims (the RAGAS definition).
  A claim the judge leaves out counts as unsupported. An answer with no factual claims is
  skipped, not scored.
- **answer_relevancy**: a 1–5 rubric rating of how directly and completely the answer
  addresses the question, scaled to 0–1. This is **not** RAGAS' embedding-based answer
  relevancy, and its numbers are not comparable with RAGAS'. It needs no embedding model.

The judge sees the question, the answer (with citation markers like `[1]` stripped) and, for
faithfulness, the context texts. It never sees the reference answer, the gold ids or which model
generated the answer. Prompts are in `src/goldpass/judge.py`. Temperature is 0, and JSON mode is
used unless `json_mode: false`.

## The same-family guard

LLM judges tend to rate outputs from their own model family higher (self-preference bias). A
judge from the generator's family can inflate exactly the scores you gate on. So goldpass
refuses to run (exit 2, before any call) when:

- the gating judge's family equals the generator's, or
- either family is unknown, because the check cannot be done.

Families are about who trained the model, not who serves it: `openai/gpt-oss-120b` on Groq is
`openai`, and a Llama judge on OpenRouter shares a family with a Llama generator on Groq. They
are inferred from the model name (`src/goldpass/providers.py`) or set with `family:` on the
judge or `generator`. Distills and fine-tunes are a judgement call; set `family` when it matters.

`--allow-same-family` overrides the guard. The override is recorded in the results
(`meta.allow_same_family`) and flagged at the top of the report. Extra (non-gating) judges may
share the generator's family. Comparing them with the gating judge on the same answers is
how you see the bias.

## Free tiers

- One call at a time by default. Each judged answer costs up to 3 calls (claims, verdicts,
  relevancy). For larger sets set `judging.concurrency` and a per-judge `requests_per_minute`
  (or `min_interval_s`); the pace is shared by all workers, and a `Retry-After` pauses them all.
- 408/425/429/5xx and connection errors are retried with exponential backoff, honouring
  `Retry-After`. Auth and other 4xx errors fail at once.
- Every successful reply is cached in one SQLite file, keyed by the full request plus a prompt
  version (never the API key). Re-running the same answers costs nothing, and the Action keeps
  the cache between CI runs. `--no-cache` turns it off.
- A reply that is not valid JSON is retried and never cached.
- Judging is a second stage over saved answers, so you can re-judge a recorded run with
  `--replay` without calling your system again.

## Auditing a judge: `goldpass judge-audit`

High faithfulness on correct answers says little about a judge: a judge that approves everything
scores 1.0 there too. `judge-audit` gives judges something to forgive. For each answered question
whose contexts have text, it scores the answer as it is (the control), then copies with exactly
one planted unsupported claim:

- **number**: one number the contexts support is changed to the nearest value that appears
  nowhere in them. Range endpoints are skipped, and upper bounds move down ("within 30 days" →
  "within 20 days"), so the new claim is never implied by the old one.
- **sentence**: one unsupported sentence is appended with a citation marker. The defaults are
  domain-neutral ("Exceptions can be granted on written request."); a template is skipped when
  its marker word appears in the contexts. Add your own under `audit.sentences`.

A judge **misses** a planted error when the perturbed copy scores at least the control's
faithfulness: the false claim cost it nothing. The report gives each judge's miss rate with a
95% Wilson interval, the mean paired drop, how many controls it already scored below 1.00 (a
judge strict about everything catches more for the wrong reason), and, for each pair of judges,
an exact McNemar test on the planted errors they disagree about. Each judge's family is shown
next to the generator's. The same-family guard does not apply here: measuring same-family
leniency is the point.

```bash
goldpass judge-audit --config goldpass.yaml --dry-run   # list the planted errors, no calls
goldpass judge-audit --config goldpass.yaml             # up to 2 calls per answer per judge
```

```yaml
audit:
  sentences:                          # optional; replaces the defaults
    - {marker: refund, sentence: "Refunds are processed automatically within a week."}
```

It needs answers with context texts: a live system, or a recorded `*.answers.json` replayed.
Everything is deterministic except the judges, and every judge reply is cached. `--offline`
(on `run` and `judge-audit`) answers only from the cache and needs no keys, so a committed cache
makes a judged result reproducible by anyone; `examples/rail/audit/` is one.

Limits: the planted errors are simple and mechanical, so a judge that catches them all may still
miss subtler unsupported claims; "missed" is measured on the score, not on the judge's verdict
for the planted claim itself; and a few dozen planted errors give wide intervals.
