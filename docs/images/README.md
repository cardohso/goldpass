# Images

- `how-it-works.svg`: hand-written diagram; light and dark mode via `prefers-color-scheme`.
- `failed-report.png`: the Markdown report goldpass writes for the fake service in `regressed`
  mode, rendered to HTML with GitHub's Markdown API and screenshotted at 2x in headless Chrome.
  It is not a capture of GitHub's job-summary page, which does not show summaries to
  anonymous visitors.

To regenerate (needs the `gh` CLI, Chrome and, for the crop, macOS `sips`), follow the commands in
the docstring of [`render_report.py`](render_report.py): run goldpass against the `regressed`
fake service, render the report to HTML, screenshot it at 2x, and crop the empty space below the
card.
