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

## Small samples

A golden set of 20 questions is small. One question moves a rate by 1/n: 0.05 at n = 20, and
0.25 when there are only 4 `refuse` questions. goldpass puts that next to every number instead
of hiding it:

- **95% interval.** Every rate in the report has a Wilson score interval. It is exact (in the
  binomial sense) for yes/no metrics such as grounding and refusal rate. For means of 0–1 scores
  (recall, faithfulness, …) it is conservative, because their variance is at most p(1 − p)/n.
  A perfect 20/20 still only shows the true rate is above about 0.84.
- **Flip margin.** Each threshold check reports how few questions would have to change for it to
  flip: "1 question from passing" on a failure, and a **Fragile** warning on a pass that a single
  question could fail. Rules with no tolerance by design (`min: 1`, `errors: {max: 0}`) are not
  flagged. For scores that are not yes/no, the margin is a lower bound.
- **Questions changed.** Against a baseline, each metric shows how many of the questions scored
  in both runs got worse and how many got better, with an exact one-sided sign test (McNemar's
  exact test for yes/no metrics): the chance of at least that many getting worse if nothing
  really changed. One question worse and none better gives p = 0.50, a coin flip.

By default, a drop larger than `max_drop` fails the gate whatever its p-value, so a single
regressed question can fail it. That is the safe default for a gate. To fail only when a drop is
also unlikely to be noise, set `baseline.alpha`:

```yaml
baseline:
  max_drop: 0.05
  alpha: 0.05   # a drop past max_drop fails only if p <= 0.05
```

Whatever the size of the set, that needs at least 5 questions worse and none better (p = 1/32;
4 and none is p = 1/16). That is the
trade-off: `alpha` stops noisy failures and also stops single-question regressions from
failing the build. New errors are never excused by `alpha`; gate them with `errors: {max: 0}`.
