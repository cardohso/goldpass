# Judge audit on a recorded rail-regulation-rag run

Reproduce the numbers in the main README, with no API keys and no network:

```bash
goldpass judge-audit --config examples/rail/audit/goldpass.yaml --offline
```

| File | What it is |
|---|---|
| `answers.json` | the 20 answers of rail-regulation-rag CI run 38831471610, with their retrieved passages and citations (generator `gemini-3.5-flash-lite`). Trace ids, rewritten queries and latencies removed. Passage texts are from EU railway legislation. |
| `cache/judge.sqlite` | every judge reply from the original run (Groq `openai/gpt-oss-120b`, Gemini `gemini-3.5-flash-lite`), keyed by request; no API keys |
| `judge-audit.json`, `judge-audit.md` | the published results |
| `goldpass.yaml` | the config used |

`--offline` answers every judge call from the cache and fails rather than call an API. Without
it, and with `GROQ_API_KEY` and `GEMINI_API_KEY` set, the same command re-judges for real; the
judges are not deterministic, so numbers may then differ slightly.
