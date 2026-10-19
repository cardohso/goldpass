"""Render goldpass's Markdown report to the HTML page used for docs/images/failed-report.png.

    goldpass run ... --out out/run.json          # writes out/run.md
    python docs/images/render_report.py out/run.md report-page.html "fake RAG service, regressed mode"
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --hide-scrollbars \
        --force-device-scale-factor=2 --window-size=922,1000 --screenshot=report.png file://$PWD/report-page.html

Needs the `gh` CLI (GitHub's Markdown API renders the Markdown) and Chrome. Only the report down
to "What failed" is kept. Crop the bottom of the PNG to the card's edge afterwards.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CSS = """
:root { color-scheme: light; }
body { margin: 0; background: #fff; font: 16px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; color: #1f2328; }
.card { box-sizing: border-box; width: 920px; margin: 0; padding: 24px 32px 28px; border: 1px solid #d1d9e0; border-radius: 6px; }
.bar { display:flex; align-items:center; gap:8px; margin:-8px 0 14px; padding-bottom:10px; border-bottom:1px solid #d1d9e0; font-size:13px; color:#59636e; }
.dot { width:10px; height:10px; border-radius:50%; background:#d1242f; display:inline-block; }
h2 { font-size: 1.5em; margin: 0 0 12px; padding-bottom: .3em; border-bottom: 1px solid #d1d9e0; }
h3 { font-size: 1.15em; margin: 22px 0 8px; }
code { font: 85% ui-monospace, SFMono-Regular, Menlo, monospace; background: #818b981f; padding: .2em .4em; border-radius: 6px; }
table { border-collapse: collapse; margin: 12px 0; width: 100%; font-size: 14px; }
th, td { border: 1px solid #d1d9e0; padding: 5px 12px; text-align: left; }
tr:nth-child(2n) td { background: #f6f8fa; } th { background: #f6f8fa; font-weight: 600; }
blockquote { margin: 12px 0; padding: 4px 16px; border-left: 4px solid #d1d9e0; color: #59636e; }
ul { padding-left: 2em; } li { margin: 2px 0; } p { margin: 8px 0; } em { color: #59636e; font-size: 13px; }
"""


def main(md_path: str, out_path: str, caption: str) -> None:
    md = Path(md_path).read_text(encoding="utf-8").split("### By tag")[0].rstrip() + "\n"
    body = subprocess.run(
        ["gh", "api", "markdown", "-f", "mode=gfm", "-f", "context=cardohso/goldpass", "-F", "text=@-"],
        input=md, capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    html = (
        f"<!doctype html><meta charset=utf-8><style>{CSS}</style>"
        f"<div class=card><div class=bar><span class=dot></span><b>goldpass</b> job summary · "
        f"{caption}</div>{body}</div>"
    )
    Path(out_path).write_text(html, encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
