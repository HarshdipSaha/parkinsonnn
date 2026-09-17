# -*- coding: utf-8 -*-
"""Update LaTeX files with clean modular figures and correct bibliography tags."""
import re
from pathlib import Path

# -------------------------------------------------------------------
# 1. Update isbi2027.tex
# -------------------------------------------------------------------
isbi_path = Path("paper_v2/isbi2027.tex")
isbi_text = isbi_path.read_text(encoding="utf-8")

# Replace inline tikz with input
isbi_text = re.sub(
    r"\\begin\{tikzpicture\}\[y=13cm,x=0\.9cm\].*?\\end\{tikzpicture\}",
    r"\\input{figures/fig_shiftgap_isbi.tex}",
    isbi_text,
    flags=re.DOTALL
)
isbi_path.write_text(isbi_text, encoding="utf-8")
print("Updated isbi2027.tex")

# -------------------------------------------------------------------
# 2. Update ejnmmi_submission.tex
# -------------------------------------------------------------------
ejnmmi_path = Path("paper_v2/ejnmmi_submission.tex")
ejnmmi_text = ejnmmi_path.read_text(encoding="utf-8")

ejnmmi_text = re.sub(
    r"\\begin\{tikzpicture\}\[y=20cm\].*?\\end\{tikzpicture\}",
    r"\\input{figures/fig_shiftgap.tex}",
    ejnmmi_text,
    flags=re.DOTALL
)
ejnmmi_path.write_text(ejnmmi_text, encoding="utf-8")
print("Updated ejnmmi_submission.tex")

# -------------------------------------------------------------------
# 3. Update main.tex
# -------------------------------------------------------------------
main_path = Path("paper_v2/main.tex")
main_text = main_path.read_text(encoding="utf-8")

# Replace Fig 1
main_text = re.sub(
    r"\\begin\{tikzpicture\}\[scale=1\].*?\\end\{tikzpicture\}",
    r"\\input{figures/fig_schematic.tex}",
    main_text,
    flags=re.DOTALL
)

# Replace Fig 2 (resizebox wrapping tikz)
main_text = re.sub(
    r"\\resizebox\{\\textwidth\}\{!\}%?\s*\\begin\{tikzpicture\}\[\s*node distance=8mm and 10mm.*?\\end\{tikzpicture\}%?\s*\}",
    r"\\resizebox{\\textwidth}{!}{\\input{figures/fig_preprocessing.tex}}",
    main_text,
    flags=re.DOTALL
)

# Replace Fig 3 (Hemi arm - was 0.86\textheight!)
main_text = re.sub(
    r"\\resizebox\{!\}\{0\.86\\textheight\}%?\s*\\begin\{tikzpicture\}\[\s*font=\\scriptsize.*?\\end\{tikzpicture\}%?\s*\}",
    r"\\centering\\input{figures/fig_hemi.tex}",
    main_text,
    flags=re.DOTALL
)

# Replace Fig 4 (Ensemble)
main_text = re.sub(
    r"\\resizebox\{\\textwidth\}\{!\}%?\s*\\begin\{tikzpicture\}\[\s*node distance=7mm and 18mm.*?\\end\{tikzpicture\}%?\s*\}",
    r"\\centering\\input{figures/fig_ensemble.tex}",
    main_text,
    flags=re.DOTALL
)

# Replace Fig 5 (Shift gap)
main_text = re.sub(
    r"\\begin\{tikzpicture\}\[y=20cm\].*?\\end\{tikzpicture\}",
    r"\\input{figures/fig_shiftgap.tex}",
    main_text,
    flags=re.DOTALL
)

# Fix natbib bibitems in main.tex
main_text = main_text.replace(
    r"\bibitem{ensemblefusion2022}",
    r"\bibitem[{Diagnostics}(2022)]{ensemblefusion2022}"
)
main_text = main_text.replace(
    r"\bibitem{radiomics2022}",
    r"\bibitem[{EJNMMI}(2022)]{radiomics2022}"
)
main_text = main_text.replace(
    r"\bibitem{trainingprotocol2026}",
    r"\bibitem[{Electronics}(2026)]{trainingprotocol2026}"
)

main_path.write_text(main_text, encoding="utf-8")
print("Updated main.tex")

# -------------------------------------------------------------------
# 4. Update tmlr_submission.tex
# -------------------------------------------------------------------
tmlr_path = Path("paper_v2/tmlr_submission.tex")
tmlr_text = tmlr_path.read_text(encoding="utf-8")

# Replace Fig 1
tmlr_text = re.sub(
    r"\\begin\{tikzpicture\}\[scale=1\].*?\\end\{tikzpicture\}",
    r"\\input{figures/fig_schematic.tex}",
    tmlr_text,
    flags=re.DOTALL
)

# Replace Fig 2
tmlr_text = re.sub(
    r"\\resizebox\{\\textwidth\}\{!\}%?\s*\\begin\{tikzpicture\}\[\s*node distance=8mm and 10mm.*?\\end\{tikzpicture\}%?\s*\}",
    r"\\resizebox{\\textwidth}{!}{\\input{figures/fig_preprocessing.tex}}",
    tmlr_text,
    flags=re.DOTALL
)

# Replace Fig 3
tmlr_text = re.sub(
    r"\\resizebox\{!\}\{0\.86\\textheight\}%?\s*\\begin\{tikzpicture\}\[\s*font=\\scriptsize.*?\\end\{tikzpicture\}%?\s*\}",
    r"\\centering\\input{figures/fig_hemi.tex}",
    tmlr_text,
    flags=re.DOTALL
)

# Replace Fig 4
tmlr_text = re.sub(
    r"\\resizebox\{\\textwidth\}\{!\}%?\s*\\begin\{tikzpicture\}\[\s*node distance=7mm and 18mm.*?\\end\{tikzpicture\}%?\s*\}",
    r"\\centering\\input{figures/fig_ensemble.tex}",
    tmlr_text,
    flags=re.DOTALL
)

# Replace Fig 5
tmlr_text = re.sub(
    r"\\begin\{tikzpicture\}\[y=20cm\].*?\\end\{tikzpicture\}",
    r"\\input{figures/fig_shiftgap.tex}",
    tmlr_text,
    flags=re.DOTALL
)

tmlr_path.write_text(tmlr_text, encoding="utf-8")
print("Updated tmlr_submission.tex")
