# Fake RAG service

A tiny RAG system to try goldpass on, with no network, no keys and no dependencies beyond the
Python standard library. **Everything here is synthetic**: the corpus documents "Nimbus Storage",
a product that does not exist, retrieval is keyword overlap, and the "generator" quotes the
best passage. It is a test target, not a RAG implementation.

```bash
python examples/fake_rag/server.py --port 8765          # terminal 1
goldpass run --config examples/fake_rag/goldpass.yaml   # terminal 2: gate PASSED, exit 0
```

`--mode regressed` simulates two changes that look harmless in a code review:

- the filter that hides the outdated `legacy-v1` page is removed, so `g08` (tagged
  `wrong-document`) cites the old 2 GB upload limit instead of the current 5 GB one;
- the refusal threshold is removed, so it answers `r01` and `r02`, which the documentation
  cannot answer.

```bash
python examples/fake_rag/server.py --port 8766 --mode regressed
goldpass run --config examples/fake_rag/goldpass.yaml --url http://127.0.0.1:8766/ask
# gate FAILED, exit 1: grounding, refusal_rate and wrong-document grounding, naming g08, r01, r02
```

Files: `rag.py` (corpus and logic), `server.py` (HTTP), `golden.jsonl` (11 questions: 9 to
answer, 2 to refuse), `thresholds.yaml`, `goldpass.yaml`. CI runs both modes through the
Action in `.github/workflows/e2e.yml`.
