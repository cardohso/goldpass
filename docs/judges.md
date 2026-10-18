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

- One call at a time. Each judged answer costs up to 3 calls (claims, verdicts, relevancy).
- 408/425/429/5xx and connection errors are retried with exponential backoff, honouring
  `Retry-After`. Auth and other 4xx errors fail at once.
- Every successful reply is cached in one SQLite file, keyed by the full request plus a prompt
  version (never the API key). Re-running the same answers costs nothing, and the Action keeps
  the cache between CI runs. `--no-cache` turns it off.
- A reply that is not valid JSON is retried and never cached.
- Judging is a second stage over saved answers, so you can re-judge a recorded run with
  `--replay` without calling your system again.
