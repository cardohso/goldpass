# rail-regulation-rag

goldpass was extracted from the evaluation gate of
[rail-regulation-rag](https://github.com/cardohso/rail-regulation-rag), question answering over
EU railway regulations. This folder runs goldpass on that system, two ways.

## Recorded run (offline, runs in CI)

`recorded/records.json` is what the rail pipeline returned in its CI run
[38838060201](https://github.com/cardohso/rail-regulation-rag/actions/runs/38838060201)
(commit `097bfba`): for each of the 20 golden questions, the passage and document ids it
retrieved, their scores, its citations and whether it refused. Answer and passage texts are
left out, so the LLM-judge metrics are not available here.

```bash
goldpass run --config examples/rail/goldpass.yaml
```

goldpass reproduces the rail repo's own judge-free numbers for that run exactly
(`recorded/expected.json`; checked by `tests/test_rail_parity.py`). `baseline.json` is that run's
goldpass results, so the gate also checks for regressions against it.

| File | What it is |
|---|---|
| `golden.jsonl` | the rail golden set, converted to the goldpass format (16 answer, 4 refuse, 3 tagged `wrong-document`) |
| `thresholds.yaml` | the rail repo's thresholds, minus the two judge metrics |
| `baseline.json` | goldpass results of the recorded run |
| `recorded/expected.json` | the metrics the rail repo computed for that run |

## Live demo (manual)

`live.yaml` points at the public demo. Its `/ask` endpoint returns answers, refusals and
citations but not the retrieved passages, so retrieval metrics are skipped and only grounding,
answered rate and refusal rate are gated (`thresholds-live.yaml`). The demo allows 5 questions
per minute, so goldpass waits 13 s between requests and a run takes about 5 minutes.

```bash
goldpass run --config examples/rail/live.yaml
```

Or run the `rail-live` workflow from the Actions tab. It is never triggered automatically.
