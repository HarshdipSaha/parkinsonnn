"""Render HOWTO.md (approach C) into a one-page HOWTO.pdf with reportlab.  python kaggle/run6c_features_ssl/make_howto_pdf.py"""
from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer

HERE = Path(__file__).resolve().parent
md = (HERE / "HOWTO.md").read_text(encoding="utf-8").splitlines()
ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=12.5, leading=15, spaceAfter=3)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=10, leading=12, spaceBefore=4, spaceAfter=2)
P = ParagraphStyle("p", parent=ss["BodyText"], fontSize=7.4, leading=9.2, spaceAfter=2)
CODE = ParagraphStyle("code", parent=ss["Code"], fontSize=6.8, leading=8.4, leftIndent=4, spaceAfter=3)


def inline(s: str) -> str:
    """Markdown inline -> reportlab markup; bold/italic are applied outside code spans only (code may contain '*')."""
    parts = s.split("`")
    out = []
    for k, seg in enumerate(parts):
        seg = seg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if k % 2 == 1:
            out.append(f"<font face='Courier'>{seg}</font>")
        else:
            seg = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", seg)
            seg = re.sub(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"<i>\1</i>", seg)
            out.append(seg)
    return "".join(out)


story, buf, code, in_code = [], [], [], False
def flush():
    global buf
    if buf:
        story.append(Paragraph(inline(" ".join(buf)), P)); buf = []
for line in md:
    if line.strip().startswith("```"):
        if in_code:
            story.append(Preformatted("\n".join(code), CODE)); code = []
        else:
            flush()
        in_code = not in_code; continue
    if in_code:
        code.append(line); continue
    if line.startswith("# "):
        flush(); story.append(Paragraph(inline(line[2:]), H1))
    elif line.startswith("## "):
        flush(); story.append(Paragraph(inline(line[3:]), H2))
    elif not line.strip():
        flush()
    else:
        buf.append(line.strip())
flush()
doc = SimpleDocTemplate(str(HERE / "HOWTO.pdf"), pagesize=A4, leftMargin=12 * mm, rightMargin=12 * mm, topMargin=9 * mm, bottomMargin=9 * mm,
                        title="Run 6c (approach C) — how to run on Kaggle", author="parkinsonnn")
doc.build(story)
print("wrote", HERE / "HOWTO.pdf")
