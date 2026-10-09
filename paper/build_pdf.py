"""Render paper/flexcast_paper.md to PDF (Microsoft Edge headless + MathJax).

    python paper/build_pdf.py            # needs `pip install markdown` and Edge (Windows) or Chrome
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
SRC, DST = HERE / "flexcast_paper.md", HERE / "flexcast_paper.pdf"
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            shutil.which("google-chrome") or "", shutil.which("chromium") or ""]

CSS = """
@page { size: A4; margin: 18mm 17mm; }
body { font-family: 'Cambria', 'Georgia', serif; font-size: 10.3pt; line-height: 1.45; color: #111; }
h1 { font-size: 17pt; text-align: center; margin: 0 0 6px; line-height: 1.25; }
.authors { text-align: center; font-size: 10.5pt; margin-bottom: 14px; }
h2 { font-size: 12.5pt; margin: 18px 0 6px; }
h3 { font-size: 11pt; margin: 12px 0 4px; }
p { text-align: justify; margin: 5px 0 7px; }
table { border-collapse: collapse; margin: 8px auto 10px; font-size: 9pt; }
th, td { border-top: 1px solid #999; border-bottom: 1px solid #999; padding: 3px 8px; text-align: left; }
th { background: #f2f2f2; }
code { font-family: Consolas, monospace; font-size: 8.8pt; }
img { display: block; max-width: 92%; margin: 8px auto 2px; }
.caption { text-align: center; font-size: 9pt; color: #333; margin-bottom: 10px; }
blockquote { margin: 6px 20px; color: #333; }
ul, ol { margin: 4px 0 8px 20px; padding: 0; }
li { margin: 2px 0; }
ul { list-style-type: "•  "; }
.abstract { margin: 0 28px 10px; font-size: 9.8pt; }
"""
MATHJAX = """
<script>window.MathJax={tex:{inlineMath:[['$','$']],displayMath:[['$$','$$']]},svg:{fontCache:'global'}};</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js"></script>
"""


def protect_math(text):
    store = []

    def keep(m):
        store.append(m.group(0))
        return f"@@MATH{len(store) - 1}@@"

    text = re.sub(r"\$\$.+?\$\$", keep, text, flags=re.S)
    text = re.sub(r"(?<!\\)\$[^$\n]+?\$", keep, text)
    return text, store


def main():
    text, store = protect_math(SRC.read_text(encoding="utf-8"))
    html = markdown.markdown(text, extensions=["tables", "fenced_code", "md_in_html"])
    for i, m in enumerate(store):
        html = html.replace(f"@@MATH{i}@@", m.replace("<", "&lt;").replace(">", "&gt;"))
    doc = (f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style>{MATHJAX}</head>"
           f"<body>{html}</body></html>")
    browser = next(b for b in BROWSERS if b and Path(b).exists())
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8", dir=HERE) as f:
        f.write(doc)
        tmp = Path(f.name)
    try:
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        "--virtual-time-budget=20000", f"--print-to-pdf={DST}", tmp.as_uri()],
                       check=True, timeout=180, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        tmp.unlink(missing_ok=True)
    print("wrote", DST, DST.stat().st_size, "bytes")


if __name__ == "__main__":
    sys.exit(main())
