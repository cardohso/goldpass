# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Small-sample statistics (`goldpass.stats`, standard library only):
  - a 95% Wilson interval for every rate in the report (exact for yes/no metrics, conservative
    for means of 0–1 scores);
  - a flip margin on every threshold check ("1 question from passing"), and a **Fragile**
    warning for passing checks that a single question could fail (zero-tolerance rules are
    not flagged);
  - against a baseline: questions worse and better, with an exact one-sided sign test.
- `baseline.alpha` in the thresholds file: a drop past `max_drop` fails only if it is also
  significant. Off by default. New errors are never excused.
- When a failing drop has p > 0.05, the report says it could be noise and points to `alpha`.
- `goldpass judge-audit`: plants one unsupported claim per copy of each answer (a changed
  number, or an appended sentence; deterministic, checked against the contexts) and measures
  how often each judge misses it, with Wilson intervals, mean paired drop, controls already
  scored below 1, and an exact McNemar test between judges. Each judge's model family is shown
  next to the generator's. `--dry-run` lists the planted errors without calls; domain
  sentences go under `audit.sentences`.
- `--offline` for `run` and `judge-audit`: judge only from the cache, with no API keys; a cache
  miss is an error, never a call.
- `examples/rail/audit/`: a real audit of Groq `openai/gpt-oss-120b` and Gemini
  `gemini-3.5-flash-lite` on recorded rail answers, with the judge cache, reproducible offline.

- Concurrency, opt-in: `system.concurrency`, `judging.concurrency` and `--workers` overlap
  questions and judge calls on threads. A shared `RateLimiter` (`min_interval_s` or
  `requests_per_minute`, on systems and on each judge) keeps the request rate legal however
  many workers there are, and a `Retry-After` from one worker pauses all of them. Results stay
  in golden-set order. The judge cache is thread-safe. The default is still one at a time.

### Fixed

- "1 items were not scored" in the report.
- Judges that return a bare JSON list for claims or verdicts are accepted.
- Planted number errors skip numbers after lower-bound conditions ("greater than 250 km/h"),
  where a changed value can still be implied by the passage. Found by the first real audit.

## [0.1.0] - 2026-10-19

First release. Alpha: the interface may still change before 1.0.

### Added

- `goldpass` CLI with `run`, `validate` and `compare`, documented exit codes (0 pass, 1 gate
  failed, 2 bad input, 3 system or judge unreachable). CI on Python 3.11 and 3.12.
- Golden-set format: JSONL with a JSON Schema (Draft 2020-12) shipped in the package, fields
  `id`, `question`, `expected` (`answer` | `refuse`), `reference_answer`, `gold_passage_ids`,
  `source_doc_ids`, `tags`, `notes`, `metadata`.
- `goldpass validate [--strict]`: reports every problem with its line number (invalid JSON,
  schema errors, unknown fields, duplicate ids, gold passages on `refuse` items) and warns about
  items that metrics will skip.
- `Record`: what the system under test returned (answer, refusal, ranked contexts, optional
  wider `retrieved` list, citations). `None` means "not reported" and skips the metrics that
  need it; empty means "nothing" and scores 0.
- Deterministic metrics: precision@k, recall@k, context precision (RAGAS definition with gold
  ids), MRR, hit@k, retrieved recall, grounding, answered rate, refusal rate, error count;
  slices by tag and by source document. Skipped metrics carry a note saying why.
- Parity test: reproduces the judge-free metrics of a recorded rail-regulation-rag CI run
  exactly.
- Adapters: `http` (any endpoint; JSON body template with `{question}`/`{id}`, `${ENV}` in URL
  and headers, dotted-path field mapping, refusal regexes for systems without a flag),
  `python` (`module:function`), and `replay` (recorded answers, same field mapping).
- Retries with exponential backoff on 408/425/429/5xx and connection errors, honouring
  `Retry-After` (seconds or HTTP date), with a cap on any single wait.
- `goldpass run` with `goldpass.yaml` or `--url` / `--python` / `--replay`, `--golden`, `-k`,
  `--out`. Live runs save `<out>.answers.json` so the same answers can be replayed. Exit code 3
  when the system cannot be set up or every item fails.
- Optional LLM judges over any OpenAI-compatible API (presets: groq, gemini, openai,
  openrouter, mistral, ollama; or any `base_url`), no SDK and no RAGAS dependency:
  faithfulness (claims extracted, then checked against the contexts) and answer relevancy
  (1-5 rubric scaled to 0-1; not RAGAS' embedding-based metric). Citation markers are stripped
  before judging. Several judges can score the same answers; only the first feeds the metrics.
- Same-family guard: refuses to run when the gating judge's model family (inferred from the
  model name or set explicitly) matches the generator's, or when either is unknown, unless
  `--allow-same-family` is passed (recorded in the results).
- Disk cache for judge calls (SQLite, keyed by the full request and a prompt version, never the
  API key); `--no-cache`. Judge calls are retried like system calls; malformed JSON replies
  are retried and never cached.
- `--no-judge` runs without any LLM; a run with no judges configured is judge-free too.
- Thresholds file (YAML): `min`, `max` and `min_n` per metric, per-tag rules, and baseline
  drop limits (global `max_drop` plus per-metric overrides). Unknown metric names are errors
  with a suggestion (`context_recall` -> `recall_at_k`).
- Baseline comparison: fails when a metric drops more than allowed from a saved results file,
  and names the questions that got worse. Warns when the golden set or `k` changed, and does
  not compare judge metrics across different gating judges. A missing baseline is skipped
  with a note, so the first run can create it.
- Reports: Markdown (gate table with thresholds, baseline and deltas; "What failed" with the
  lowest-scoring or regressed questions; judge comparison; tags; grounding per document;
  per-item scores) written next to the results JSON and appended to `--summary`
  (e.g. `$GITHUB_STEP_SUMMARY`). The results JSON gains a `gate` section.
- `goldpass run --thresholds/--baseline/--summary` (or `thresholds:`/`baseline:` in the
  config) exits 1 when the gate fails. `goldpass compare RESULTS BASELINE` re-gates saved
  runs without calling anything.
- `min_interval_s` for HTTP systems: paces requests for rate-limited endpoints. The pause is
  not counted in `latency_s`.
- Composite GitHub Action (`action.yml`): runs goldpass from the action's own source with uv,
  writes the report to the job summary, persists the judge cache with actions/cache, and
  exposes `passed`, `exit-code`, `results` and `markdown` outputs. Inputs reach the script
  only through environment variables. `--json-paths` tells wrappers where results were written.
- Examples: a synthetic fake RAG service (standard library only) with a `regressed` mode that
  the gate must catch, and the rail-regulation-rag recorded run with its thresholds and a
  baseline, plus a manual-only workflow that gates the live rail demo.
- e2e workflow: runs the Action against the fake service (good must pass, regressed must fail
  with exit 1) and on the recorded rail run.
- README and docs: `docs/configuration.md` (golden set, `goldpass.yaml`, thresholds, CLI, exit
  codes, Action), `docs/metrics.md` (definitions and skip rules), `docs/judges.md` (judge
  metrics, same-family guard, free-tier behaviour).

[Unreleased]: https://github.com/cardohso/goldpass/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/cardohso/goldpass/releases/tag/v0.1.0
