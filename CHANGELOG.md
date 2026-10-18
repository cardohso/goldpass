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
