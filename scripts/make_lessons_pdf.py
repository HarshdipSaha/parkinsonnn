"""Render a markdown file into a styled PDF (reportlab).

    python scripts/make_lessons_pdf.py                       # LESSONS.md  -> LESSONS.pdf
    python scripts/make_lessons_pdf.py RUNBOOK.md RUNBOOK.pdf
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (HRFlowable, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "LESSONS.md")
OUT = ROOT / (sys.argv[2] if len(sys.argv) > 2 else SRC.stem + ".pdf")
INK, ACCENT, ACCENT2, SOFT, CODEBG, MUTED = (colors.HexColor("#1f2933"), colors.HexColor("#24476b"), colors.HexColor("#b0562f"),
                                             colors.HexColor("#eaf0f6"), colors.HexColor("#f4f6f8"), colors.HexColor("#5b6773"))
W, H = A4

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=21, leading=25, textColor=colors.white, alignment=TA_LEFT, spaceAfter=0),
    "sub": ParagraphStyle("s", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13, textColor=colors.white),
    "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=ACCENT, spaceBefore=12, spaceAfter=5),
    "h3": ParagraphStyle("h3", parent=ss["Heading3"], fontName="Helvetica-Bold", fontSize=10.4, leading=13, textColor=ACCENT2, spaceBefore=9, spaceAfter=3),
    "body": ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.3, leading=12.6, textColor=INK, spaceAfter=4),
    "step": ParagraphStyle("st", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.3, leading=12.6, textColor=INK, leftIndent=16, spaceAfter=3),
    "bullet": ParagraphStyle("bu", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.3, leading=12.6, textColor=INK, leftIndent=13, bulletIndent=3, spaceAfter=2),
    "code": ParagraphStyle("c", parent=ss["Code"], fontName="Courier", fontSize=7.9, leading=9.8, textColor=INK, backColor=CODEBG,
                           borderColor=colors.HexColor("#d5dbe0"), borderWidth=0.6, borderPadding=5, leftIndent=6, spaceBefore=3, spaceAfter=7),
    "cell": ParagraphStyle("ce", parent=ss["BodyText"], fontName="Helvetica", fontSize=8.3, leading=10.5, textColor=INK),
    "cellh": ParagraphStyle("ch", parent=ss["BodyText"], fontName="Helvetica-Bold", fontSize=8.3, leading=10.5, textColor=colors.white),
}


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline(t: str) -> str:
    """Code spans are stashed before bold/italic run, so a literal * inside `oof_*.csv` cannot open a tag."""
    spans: list[str] = []

    def stash(m):
        spans.append(m.group(1))
        return f"\x00{len(spans) - 1}\x00"

    t = re.sub(r"`([^`]+)`", stash, t)
    t = _esc(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*(?!\s)([^*]+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", t)
    return re.sub(r"\x00(\d+)\x00", lambda m: f'<font face="Courier" size="8" color="#24476b">{_esc(spans[int(m.group(1))])}</font>', t)


def header_block(title: str, sub: str):
    t = Table([[Paragraph(title, S["title"])], [Paragraph(sub, S["sub"])]], colWidths=[W - 30 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), ACCENT), ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                           ("TOPPADDING", (0, 0), (-1, 0), 11), ("BOTTOMPADDING", (0, -1), (-1, -1), 11)]))
    return t


def section(title: str):
    bar = Table([[Paragraph(inline(title), S["h2"])]], colWidths=[W - 30 * mm])
    bar.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 1.1, ACCENT), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return bar


def md_table(rows):
    cells = [r.strip().strip("|").split("|") for r in rows if not set(r.replace("|", "").strip()) <= set("-: ")]
    n = max(len(r) for r in cells)
    data = [[Paragraph(inline(c.strip()), S["cellh"] if i == 0 else S["cell"]) for c in (r + [""] * (n - len(r)))] for i, r in enumerate(cells)]
    widths = [(W - 30 * mm) * f for f in ([0.09, 0.55, 0.20, 0.16] if n == 4 else [1 / n] * n)]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), ACCENT), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, SOFT]),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d3dc")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return t


FOOT = ""


def footer(canvas, doc):
    canvas.saveState(); canvas.setFont("Helvetica", 7.5); canvas.setFillColor(MUTED)
    canvas.drawString(15 * mm, 9 * mm, FOOT[:110])
    canvas.drawRightString(W - 15 * mm, 9 * mm, f"page {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#c9d3dc")); canvas.line(15 * mm, 12 * mm, W - 15 * mm, 12 * mm); canvas.restoreState()


lines = SRC.read_text(encoding="utf-8").split("\n")
title = lines[0][2:] if lines[0].startswith("# ") else SRC.stem
intro = next((l for l in lines[1:] if l.strip()), "")
FOOT = re.sub(r"[*`]", "", title)
flow = [header_block(title, inline(intro)), Spacer(1, 8)]

buf, tbl, in_code, seen_intro = [], [], False, False
for line in lines[1:]:
    if not seen_intro:
        if line.strip() == intro.strip():
            seen_intro = True
        continue
    if line.strip().startswith("```"):
        if in_code:
            flow.append(Preformatted("\n".join(buf), S["code"])); buf = []
        in_code = not in_code; continue
    if in_code:
        buf.append(line); continue
    if line.startswith("|"):
        tbl.append(line); continue
    if tbl:
        flow.append(md_table(tbl)); flow.append(Spacer(1, 7)); tbl = []
    if line.strip() == "---":
        flow.append(Spacer(1, 5)); flow.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#c9d3dc"))); flow.append(Spacer(1, 3))
    elif line.startswith("### "):
        flow.append(Paragraph(inline(line[4:]), S["h3"]))
    elif line.startswith("## "):
        flow.append(section(line[3:]))
    elif re.match(r"^\d+\.\s", line):
        num, rest = line.split(".", 1)
        flow.append(Paragraph(f'<font color="#b0562f"><b>{num}.</b></font> {inline(rest.strip())}', S["step"]))
    elif line.startswith("- "):
        flow.append(Paragraph(inline(line[2:]), S["bullet"], bulletText="•"))
    elif line.strip():
        flow.append(Paragraph(inline(line), S["body"]))
    else:
        flow.append(Spacer(1, 3))
if tbl:
    flow.append(md_table(tbl))

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm, bottomMargin=16 * mm,
                        title="What I learned running an ML competition", author="Harshdip Saha")
doc.build(flow, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT, OUT.stat().st_size, "bytes")
