"""Render HOWTO.md into a one-page HOWTO.pdf (reportlab). python kaggle/run7_hemisphere/make_howto_pdf.py"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer

HERE = Path(__file__).resolve().parent
md = (HERE / "HOWTO.md").read_text(encoding="utf-8").split("\n")
ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=8.6, leading=10.6, spaceAfter=3)
code = ParagraphStyle("c", parent=ss["Code"], fontSize=7.4, leading=9, leftIndent=4, backColor="#f3f3f3", spaceAfter=3)
h1 = ParagraphStyle("h", parent=ss["Heading2"], fontSize=12, spaceAfter=4)
flow, buf, in_code = [], [], False
def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("**", "<b>", 1).replace("**", "</b>", 1) if s.count("**") >= 2 else s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
for line in md:
    if line.startswith("```"):
        if in_code:
            flow.append(Preformatted("\n".join(buf), code)); buf = []
        in_code = not in_code; continue
    if in_code:
        buf.append(line); continue
    if line.startswith("# "):
        flow.append(Paragraph(esc(line[2:]), h1))
    elif line.strip():
        t = line
        while t.count("**") >= 2:
            t = t.replace("**", "<b>", 1).replace("**", "</b>", 1)
        t = t.replace("&", "&amp;").replace("<b>", "\x00").replace("</b>", "\x01").replace("<", "&lt;").replace(">", "&gt;").replace("\x00", "<b>").replace("\x01", "</b>")
        t = t.replace("`", "")
        flow.append(Paragraph(t, body))
    else:
        flow.append(Spacer(1, 2))
doc = SimpleDocTemplate(str(HERE / "HOWTO.pdf"), pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm, topMargin=12 * mm, bottomMargin=12 * mm, title="HOWTO run 7")
doc.build(flow)
print("wrote", HERE / "HOWTO.pdf", (HERE / "HOWTO.pdf").stat().st_size, "bytes")
