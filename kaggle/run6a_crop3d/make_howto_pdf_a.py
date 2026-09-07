"""Render HOWTO.md (one page) to HOWTO.pdf with reportlab.  python kaggle/run6a_crop3d/make_howto_pdf_a.py"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer

HERE = Path(__file__).resolve().parent
md = (HERE / "HOWTO.md").read_text(encoding="utf-8").splitlines()
ss = getSampleStyleSheet()
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=13, spaceAfter=4, spaceBefore=2)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=10.5, spaceAfter=2, spaceBefore=5)
body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=8.6, leading=10.8, spaceAfter=2)
code = ParagraphStyle("c", parent=ss["Code"], fontSize=7.6, leading=9.2, leftIndent=6, backColor="#f3f3f3", spaceAfter=3)
doc = SimpleDocTemplate(str(HERE / "HOWTO.pdf"), pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm, topMargin=11 * mm, bottomMargin=10 * mm)
flow, buf = [], []


def flush():
    if buf:
        flow.append(Preformatted("\n".join(buf), code)); buf.clear()


in_code = False
for line in md:
    if line.startswith("```"):
        in_code = not in_code
        if not in_code:
            flush()
        continue
    if in_code:
        buf.append(line); continue
    if line.startswith("# "):
        flow.append(Paragraph(line[2:], h1))
    elif line.startswith("## "):
        flow.append(Paragraph(line[3:], h2))
    elif line.strip():
        t = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        t = t.replace("**", "<b>", 1).replace("**", "</b>", 1) if t.count("**") >= 2 else t
        while "`" in t and t.count("`") >= 2:
            t = t.replace("`", "<font face='Courier'>", 1).replace("`", "</font>", 1)
        flow.append(Paragraph(("• " + t[2:]) if t.startswith("- ") else t, body))
    else:
        flow.append(Spacer(1, 2))
flush()
doc.build(flow)
print("wrote", HERE / "HOWTO.pdf")
