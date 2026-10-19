## goldpass judge audit: do the judges catch planted errors?

System `replay:answers.json` · generator `gemini-3.5-flash-lite` (google) · 15 answers as controls, 28 copies with one planted unsupported claim each · 2026-10-19T11:46:48Z

A **miss** is a planted error whose perturbed answer scores at least the control's faithfulness. **Drop** is control minus perturbed faithfulness, paired per answer. Lower miss rate is better.

| Judge | Family | Planted | n | Missed | Miss rate (95% CI) | Mean drop |
|---|---|---|---|---|---|---|
| `gpt-oss-120b` | openai | number | 13 | 0 | 0.00 (0.00–0.23) | 0.60 |
| `gpt-oss-120b` | openai | sentence | 15 | 0 | 0.00 (0.00–0.20) | 0.40 |
| `gpt-oss-120b` | openai | all | 28 | 0 | 0.00 (0.00–0.12) | 0.49 |
| `gemini-3.5-flash-lite` | google (same as generator) | number | 13 | 0 | 0.00 (0.00–0.23) | 0.52 |
| `gemini-3.5-flash-lite` | google (same as generator) | sentence | 15 | 1 | 0.07 (0.01–0.30) | 0.35 |
| `gemini-3.5-flash-lite` | google (same as generator) | all | 28 | 1 | 0.04 (0.01–0.18) | 0.43 |

| Judge | Controls | Mean control faithfulness | Controls below 1.00 | Failed calls |
|---|---|---|---|---|
| `gpt-oss-120b` | 15 | 1.00 | 0 | 0 |
| `gemini-3.5-flash-lite` | 15 | 1.00 | 0 | 0 |

**`gpt-oss-120b` vs `gemini-3.5-flash-lite`** on 28 shared planted errors: missed by both 0, only `gpt-oss-120b` 0, only `gemini-3.5-flash-lite` 1 (exact McNemar p = 1.00).

Controls below 1.00 are correct answers the judge still marked partly unsupported: a judge that is strict everywhere catches more planted errors for the wrong reason, so read the two tables together.

<details><summary>Every planted error and its scores</summary>

| Item | Planted | Change | `gpt-oss-120b` control → perturbed | `gemini-3.5-flash-lite` control → perturbed |
|---|---|---|---|---|
| `q01` | number | 65 → 66 | 1.00 → 0.00 | 1.00 → 0.00 |
| `q01` | sentence | + “Compliance must be re-checked by an independent auditor every year. [1]” | 1.00 → 0.50 | 1.00 → 0.50 |
| `q02` | number | 0,225 → 0,226 | 1.00 → 0.00 | 1.00 → 0.50 |
| `q02` | sentence | + “Exceptions can be granted on written request. [1]” | 1.00 → 0.50 | 1.00 → 0.67 |
| `q03` | number | 0.8 → 0.9 | 1.00 → 0.50 | 1.00 → 0.50 |
| `q03` | sentence | + “Organisations with fewer than 50 employees are exempt. [1]” | 1.00 → 0.67 | 1.00 → 0.67 |
| `q04` | sentence | + “Exceptions can be granted on written request. [1]” | 1.00 → 0.67 | 1.00 → 0.75 |
| `q05` | number | 5 → 6 | 1.00 → 0.75 | 1.00 → 0.71 |
| `q05` | sentence | + “Organisations with fewer than 50 employees are exempt. [1]” | 1.00 → 0.80 | 1.00 → 0.83 |
| `q06` | number | 3 → 6 | 1.00 → 0.00 | 1.00 → 0.50 |
| `q06` | sentence | + “It started as a voluntary pilot and only later became mandatory. [1]” | 1.00 → 0.50 | 1.00 → 0.33 |
| `q07` | number | 800 → 900 | 1.00 → 0.50 | 1.00 → 0.50 |
| `q07` | sentence | + “It started as a voluntary pilot and only later became mandatory. [1]” | 1.00 → 0.67 | 1.00 → 1.00 (missed) |
| `q08` | number | one → two | 1.00 → 0.00 | 1.00 → 0.00 |
| `q08` | sentence | + “This was last revised in 2017 and replaced the earlier version. [1]” | 1.00 → 0.33 | 1.00 → 0.33 |
| `q09` | number | two → six | 1.00 → 0.67 | 1.00 → 0.75 |
| `q09` | sentence | + “Compliance must be re-checked by an independent auditor every year. [1]” | 1.00 → 0.75 | 1.00 → 0.80 |
| `q10` | number | 2 → 3 | 1.00 → 0.67 | 1.00 → 0.60 |
| `q10` | sentence | + “A reduced version applies at weekends and on public holidays. [1]” | 1.00 → 0.75 | 1.00 → 0.83 |
| `q11` | number | 1 → 3 | 1.00 → 0.50 | 1.00 → 0.50 |
| `q11` | sentence | + “A reduced version applies at weekends and on public holidays. [1]” | 1.00 → 0.67 | 1.00 → 0.67 |
| `q12` | number | four → two | 1.00 → 0.50 | 1.00 → 0.50 |
| `q12` | sentence | + “It started as a voluntary pilot and only later became mandatory. [1]” | 1.00 → 0.50 | 1.00 → 0.50 |
| `q14` | number | 1,4 → 1,6 | 1.00 → 0.50 | 1.00 → 0.50 |
| `q14` | sentence | + “Compliance must be re-checked by an independent auditor every year. [1]” | 1.00 → 0.50 | 1.00 → 0.67 |
| `q15` | number | 10 → 20 | 1.00 → 0.67 | 1.00 → 0.67 |
| `q15` | sentence | + “It started as a voluntary pilot and only later became mandatory. [1]” | 1.00 → 0.75 | 1.00 → 0.75 |
| `q16` | sentence | + “Compliance must be re-checked by an independent auditor every year. [1]” | 1.00 → 0.50 | 1.00 → 0.50 |

</details>
