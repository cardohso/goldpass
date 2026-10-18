# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Project skeleton: package layout, `goldpass` CLI stub with `run`, `validate` and `compare`
  subcommands, documented exit codes, CI on Python 3.11 and 3.12.
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
