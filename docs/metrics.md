# Metrics

All metrics except the two judge metrics are deterministic: no LLM, no network, and the same
inputs always give the same numbers. Each one is computed per question, then averaged over the
questions where it could be measured. Every corpus-level value is reported with that count `n`,
because "0.95 over 3 questions" and "0.95 over 300" are different claims.

## Retrieval

These compare the ranked passage ids the generator saw (`contexts`) with the question's
`gold_passage_ids`. They are computed for `answer` questions only, including ones the system
refused: a refusal can be caused by bad retrieval, so retrieval is still worth measuring.
Use `-k` / `k:` to score only the top k contexts. A passage that appears twice counts once.

| Metric | Per question |
|---|---|
| `precision_at_k` | gold passages in the top k ÷ passages returned (at most k). A system that returns fewer than k passages is not penalised for the missing ones. |
| `recall_at_k` | gold passages in the top k ÷ gold passages |
| `context_precision` | mean of precision@i over the ranks i where a gold passage appears (the RAGAS definition, with relevance taken from gold ids instead of an LLM verdict). 1.0 when every gold passage found ranks above every non-gold one. |
| `mrr` | 1 ÷ rank of the first gold passage, 0 if none is in the top k |
| `hit_rate` | 1 if any gold passage is in the top k |
| `retrieved_recall` | recall over the optional wider `retrieved` list (before reranking or filtering). A diagnostic: low `recall_at_k` with high `retrieved_recall` points at the reranker or filter, not the retriever. |

## Grounding

`grounding` is 1 for an answered `answer` question if it cites at least one passage and every
cited document is in `source_doc_ids`. It catches the right-sounding answer taken from the wrong
document, which retrieval metrics and LLM judges both tend to miss. Tag such questions (for
example `wrong-document`) and gate on the tag slice too.

A citation that has only a `passage_id` gets its document from the contexts. If it cannot be
resolved, grounding is skipped for that question, with a note, rather than guessed.

## Answering and refusing

| Metric | Over | Per question |
|---|---|---|
| `answered_rate` | `answer` questions | 1 if the system did not refuse |
| `refusal_rate` | `refuse` questions | 1 if the system refused |
| `errors` | all | a count, not a rate: questions whose call failed or that have no answer |

A system without a refusal flag can be read with `refusal_patterns` (regexes on the answer).

## LLM judge (optional)

`faithfulness` and `answer_relevancy` are averaged over answered `answer` questions, with the
first (gating) judge. See [judges.md](judges.md).

## When a metric is skipped

A metric that cannot be measured for a question is left out for that question, never scored 0.
Each skip is recorded with its reason, counted in the report and in `skipped` in the results:

- `no gold_passage_ids: retrieval metrics skipped`
- `contexts not reported: retrieval metrics skipped`. The system's response had no contexts
  field. This is different from an empty list, which scores 0.
- `no source_doc_ids: grounding skipped`
- `citations not reported: grounding skipped`
- `citation '…' has no doc_id and is not in the contexts: grounding skipped`
- `contexts have no text: faithfulness skipped` and `answer makes no factual claims: faithfulness skipped`

Use `min_n` in the thresholds so a gate cannot pass because almost nothing was measured.
