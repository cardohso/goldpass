# Configuration

## Golden set

One JSON object per line. The schema is
[`src/goldpass/schemas/golden-item.schema.json`](../src/goldpass/schemas/golden-item.schema.json)
(Draft 2020-12). Point your editor at it, or run `goldpass validate golden.jsonl [--strict]`.

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | unique, stable, no spaces. Used to match answers and compare runs. |
| `question` | yes | sent to the system under test |
| `expected` | yes | `answer` or `refuse` |
| `reference_answer` | | a correct answer, for humans (never shown to judges) |
| `gold_passage_ids` | | passages that support the answer. Retrieval metrics need them. Must be empty for `refuse`. |
| `source_doc_ids` | | documents the question is about. Grounding needs them. |
| `tags` | | labels for slicing and per-tag thresholds, lowercase (`wrong-document`) |
| `notes` | | for reviewers; ignored |
| `metadata` | | anything else; ignored |

Unknown fields are errors, so a typo like `gold_passages` is caught instead of silently
disabling a metric. `goldpass validate` reports every problem with its line number and warns
about questions that some metric will have to skip.

## goldpass.yaml

Relative paths resolve from the config file's folder. `${NAME}` in `url` and `headers` is read
from the environment, so secrets stay out of the file. Unknown keys are errors.

```yaml
golden: golden.jsonl
thresholds: thresholds.yaml          # optional: gate
baseline: baseline.json              # optional: regression check (skipped if missing)
out: goldpass-results/run.json       # Markdown report is written next to it
k: 5                                 # optional: score only the top 5 contexts

system:                              # one of: http, python, replay
  type: http
  url: https://my-rag.example.com/ask
  method: POST                       # GET sends the body as query parameters
  headers: {Authorization: "Bearer ${RAG_TOKEN}"}
  body: {query: "{question}", top_k: 5}    # {question} and {id} are filled in
  timeout_s: 60
  min_interval_s: 0                  # pause between requests for rate-limited endpoints
  retry: {attempts: 5, backoff_s: 1, max_wait_s: 120}
  fields:                            # dotted paths into the response; defaults shown
    answer: answer
    refused: refused
    refusal_patterns: []             # regexes on the answer, for systems without a flag
    contexts: contexts               # ranked passages the generator saw
    retrieved: retrieved             # optional wider list, before reranking/filtering
    citations: citations
    passage: {id: id, doc_id: doc_id, text: text, score: score}
    citation: {passage_id: passage_id, doc_id: doc_id}

generator: {model: gemini-2.5-flash} # or {family: google}; only used by the judge guard
judges: []                           # see judges.md
```

Paths like `data.sources.0.id` walk into objects and lists. A path that is missing from the
response means "not reported", and the metrics that need it are skipped. A list of bare strings
is accepted for `contexts` (passage ids) and `citations` (passage ids).

**Python system.** `fn(question)` may return a `goldpass.record.Record`, a dict in the format
above, or a plain string. Exceptions become per-question errors.

```yaml
system: {type: python, target: "my_rag.app:ask"}
```

**Replay system.** Scores recorded answers (`{"records": [...]}` or a list) with the same
`fields` mapping, plus `fields.id`. Every live run saves `<out>.answers.json` in this format.

```yaml
system: {type: replay, path: goldpass-results/run.answers.json}
```

## Thresholds

```yaml
metrics:
  recall_at_k:  {min: 0.70}
  grounding:    {min: 0.90, min_n: 10}   # and measured on at least 10 questions
  errors:       {max: 0}
tags:
  wrong-document:
    grounding:  {min: 0.90}
baseline:
  max_drop: 0.05                         # largest allowed drop for any metric
  metrics: {grounding: 0}                # per-metric override
```

Metric names are listed in [metrics.md](metrics.md). Names from RAGAS or similar tools get a
suggestion (`context_recall` → `recall_at_k`).

A baseline is just an earlier results JSON. Commit one, and later runs fail on any drop larger
than allowed, with the questions that got worse listed. goldpass warns when the golden set or
`k` changed, and does not compare judge metrics when the gating judge changed.

## CLI

```text
goldpass validate GOLDEN [--strict]
goldpass run [--config F] [--golden F] [--url U | --python M:F | --replay F] [--out F] [-k N]
             [--thresholds F] [--baseline F] [--summary F] [--json-paths F]
             [--generator MODEL] [--no-judge] [--no-cache] [--allow-same-family]
goldpass compare RESULTS BASELINE [--thresholds F] [--summary F] [--out F]
```

| Exit code | Meaning |
|---|---|
| 0 | gate passed (or nothing gated) / file valid |
| 1 | a threshold failed or a metric dropped past the baseline |
| 2 | bad arguments or an invalid golden set, config or thresholds file; same-family guard |
| 3 | the system could not be set up, every question failed, or the gating judge failed everywhere |

## GitHub Action

```yaml
- uses: cardohso/goldpass@v0
  with:
    config: eval/goldpass.yaml
  env:
    GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}   # only if you use judges
```

Inputs: `config`, `golden`, `url`, `replay`, `thresholds`, `baseline`, `out`, `k`, `generator`,
`no-judge`, `allow-same-family`, `working-directory`, `python-version` (3.12), `cache-judge`
(true), `cache-dir` (`.goldpass/cache`), `summary` (true). Outputs: `passed`, `exit-code`,
`results`, `markdown`. The report goes to the job summary. See `.github/workflows/e2e.yml` for a
working example, including how to assert that a regressed system fails.
