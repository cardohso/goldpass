# goldpass

Regression tests for RAG systems: a golden set, deterministic retrieval and grounding metrics,
and a CI gate.

> **Status: pre-alpha, under construction.** Nothing here works yet beyond the CLI skeleton.
> This README will be rewritten before the first release.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Gate passed, or the file is valid |
| 1 | A metric is below its threshold, or dropped past the baseline |
| 2 | Bad arguments, or an invalid golden set, thresholds or config file |
| 3 | The system under test or a judge could not be reached |

## Licence

MIT
