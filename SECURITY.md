# Security

goldpass sends your questions and your system's answers to the endpoints and LLM APIs you
configure, and reads API keys from environment variables. Keys are not written to results,
answers files, caches or reports: judge cache keys are hashes of the request and exclude the key,
and system names are saved without URL credentials or query strings (`tests/test_secrets.py`
checks this against every file a run writes). Your own `goldpass.yaml` is yours: keep keys
out of it with `${ENV_VAR}`.

If you find a vulnerability (for example a way for a config file or a response to run code,
leak a key, or write outside the working directory), please report it privately through
[GitHub's private vulnerability reporting](https://github.com/cardohso/goldpass/security/advisories/new)
rather than a public issue. I will reply within a week.

Supported: the latest release.
