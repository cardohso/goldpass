# Rail parity fixture

A recorded run of the public [rail-regulation-rag](https://github.com/cardohso/rail-regulation-rag)
eval gate (`eval/results/latest.json`, CI run 38838060201, commit 097bfba), reduced to ids:

- `golden.jsonl`: the rail golden set converted to the goldpass format.
- `records.json`: what the pipeline returned per question (passage and document ids, scores,
  citations, refusals). Answer and passage texts are removed.
- `expected.json`: the judge-free metrics the rail repo computed for that run.

`tests/test_rail_parity.py` checks that goldpass reproduces those numbers exactly.
