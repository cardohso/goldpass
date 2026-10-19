# Contributing

Thanks for looking. goldpass is small on purpose; the bar for a change is that it is tested,
documented, and keeps the default behaviour free, deterministic and offline.

## Set up

```bash
git clone https://github.com/cardohso/goldpass && cd goldpass
uv sync                      # Python 3.11 or 3.12
uv run pytest                # offline: fake HTTP servers and a fake LLM API, no keys
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

CI runs the same on Linux (3.11 and 3.12), Windows and macOS (3.12), and requires 90% coverage.
The tests never call a real LLM API, and new ones must not either.

## Changes

1. Open an issue first for anything bigger than a bug fix, so we agree on the shape.
2. Write the test first and watch it fail. Prefer real behaviour (a local HTTP server, real
   threads) over mocks; `tests/fake_llm.py` and `examples/fake_rag/` show how.
3. Update `docs/` and add a line under *Unreleased* in `CHANGELOG.md`.
4. Every number in the README comes from a command in this repo. If you change behaviour that
   a README number depends on, regenerate it and say how.

## What fits

- New metrics must be defined in `docs/metrics.md`, say what they skip and why, and be
  deterministic unless they are judge metrics.
- New adapters must turn failures into per-question errors instead of raising.
- New judge prompts must bump `PROMPT_VERSION` in `src/goldpass/judge.py`, so cached verdicts
  are not reused.

## Conduct

Be kind and be specific. Disagree with the code, not the person.
