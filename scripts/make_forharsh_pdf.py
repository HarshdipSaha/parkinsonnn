"""Render forharsh.md into a styled forharsh.pdf (reportlab). python scripts/make_forharsh_pdf.py"""
from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
SRC, OUT = ROOT / "forharsh.md", ROOT / "forharsh.pdf"
INK, ACCENT, ACCENT2, SOFT, CODEBG, MUTED = colors.HexColor("#1f2933"), colors.HexColor("#0b6e4f"), colors.HexColor("#c8553d"), colors.HexColor("#e6f2ee"), colors.HexColor("#f4f6f8"), colors.HexColor("#5b6773")
W, H = A4

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=colors.white, alignment=TA_LEFT, spaceAfter=0),
    "sub": ParagraphStyle("s", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.5, leading=12.5, textColor=colors.white),
    "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, leading=15, textColor=ACCENT, spaceBefore=10, spaceAfter=4),
    "body": ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.2, leading=12.2, textColor=INK, spaceAfter=3),
    "step": ParagraphStyle("st", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.2, leading=12.2, textColor=INK, leftIndent=16, spaceAfter=3),
    "bullet": ParagraphStyle("bu", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.2, leading=12.2, textColor=INK, leftIndent=12, bulletIndent=2, spaceAfter=2),
    "code": ParagraphStyle("c", parent=ss["Code"], fontName="Courier", fontSize=7.9, leading=9.8, textColor=INK, backColor=CODEBG, borderColor=colors.HexColor("#d5dbe0"), borderWidth=0.6, borderPadding=5, leftIndent=6, spaceBefore=3, spaceAfter=7),
    "cell": ParagraphStyle("ce", parent=ss["BodyText"], fontName="Helvetica", fontSize=8.4, leading=10.6, textColor=INK),
    "cellh": ParagraphStyle("ch", parent=ss["BodyText"], fontName="Helvetica-Bold", fontSize=8.4, leading=10.6, textColor=colors.white),
    "kicker": ParagraphStyle("k", parent=ss["BodyText"], fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=ACCENT2, spaceAfter=1),
}


def inline(t: str) -> str:
    t = t.replace("⋯", "...").replace("→", "-&gt;").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("-&amp;gt;", "&rarr;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`([^`]+)`", r'<font face="Courier" size="8" color="#0b6e4f">\1</font>', t)
    return t


def wrap_code(line: str, width: int = 98) -> str:
    """Soft-wrap long command lines at spaces; continuation lines are indented (the .md keeps the full line)."""
    out, cur = [], ""
    for tok in line.split(" "):
        if cur and len(cur) + 1 + len(tok) > width:
            out.append(cur); cur = "    " + tok
        else:
            cur = (cur + " " + tok) if cur else tok
    out.append(cur)
    return "\n".join(out)


def header_block(title: str, sub: str):
    t = Table([[Paragraph(title, S["title"])], [Paragraph(sub, S["sub"])]], colWidths=[W - 30 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), ACCENT), ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                           ("TOPPADDING", (0, 0), (-1, 0), 10), ("BOTTOMPADDING", (0, -1), (-1, -1), 10)]))
    return t


def section(title: str):
    bar = Table([[Paragraph(inline(title), S["h2"])]], colWidths=[W - 30 * mm])
    bar.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 1.2, ACCENT), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return bar


def md_table(rows):
    cells = [r.strip().strip("|").split("|") for r in rows if not set(r.replace("|", "").strip()) <= set("-: ")]
    n = max(len(r) for r in cells)
    data = [[Paragraph(inline(c.strip()), S["cellh"] if i == 0 else S["cell"]) for c in (r + [""] * (n - len(r)))] for i, r in enumerate(cells)]
    widths = [(W - 30 * mm) * f for f in ([0.22, 0.48, 0.30] if n == 3 else [1 / n] * n)]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), ACCENT), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, SOFT]),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d3dc")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return t


def footer(canvas, doc):
    canvas.saveState(); canvas.setFont("Helvetica", 7.5); canvas.setFillColor(MUTED)
    canvas.drawString(15 * mm, 9 * mm, "forharsh — DaT Parkinson's Challenge playbook until Sept 16 2026 · repo H:\\augsepthacks\\parkinsonnn")
    canvas.drawRightString(W - 15 * mm, 9 * mm, f"page {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#c9d3dc")); canvas.line(15 * mm, 12 * mm, W - 15 * mm, 12 * mm); canvas.restoreState()


lines = SRC.read_text(encoding="utf-8").split("\n")
flow, buf, tbl, in_code = [], [], [], False
title = lines[0][2:] if lines[0].startswith("# ") else "forharsh"
# subtitle = first paragraph after the title
intro = next((l for l in lines[1:] if l.strip()), "")
flow.append(header_block(title.split(" — ")[0] if " — " in title else title, inline(title.split(" — ", 1)[1] if " — " in title else "")))
flow.append(Spacer(1, 6))
flow.append(Paragraph(inline(intro), S["body"])); flow.append(Spacer(1, 4))
skip_intro = True
for line in lines[1:]:
    if skip_intro:
        if line.strip() == intro.strip():
            skip_intro = False
        continue
    if line.strip().startswith("```"):
        if in_code:
            ind = min((len(x) - len(x.lstrip()) for x in buf if x.strip()), default=0)
            flow.append(Preformatted("\n".join(wrap_code(x[ind:]) for x in buf), S["code"])); buf = []
        in_code = not in_code; continue
    if in_code:
        buf.append(line); continue
    if line.startswith("|"):
        tbl.append(line); continue
    if tbl:
        flow.append(md_table(tbl)); flow.append(Spacer(1, 6)); tbl = []
    if line.startswith("## "):
        flow.append(section(line[3:]))
    elif re.match(r"^\d+\.\s", line):
        num, rest = line.split(".", 1)
        flow.append(Paragraph(f'<font color="#c8553d"><b>{num}.</b></font> {inline(rest.strip())}', S["step"]))
    elif line.startswith("- "):
        flow.append(Paragraph(inline(line[2:]), S["bullet"], bulletText="•"))
    elif line.strip():
        flow.append(Paragraph(inline(line), S["body"]))
    else:
        flow.append(Spacer(1, 3))
if tbl:
    flow.append(md_table(tbl))
doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm, bottomMargin=16 * mm, title="forharsh", author="Claude + Harshdip")
doc.build(flow, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT, OUT.stat().st_size, "bytes")
