"""Rebuild a submission zip with the repo's current code but the artifacts (manifest + checkpoints + template) of an
existing zip. Used to repair a Kaggle-built zip when only the inference code needs to change.
    python scripts/repack_v3.py <in.zip> <out.zip>"""
from __future__ import annotations

import sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
with zipfile.ZipFile(src) as zi, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zo:
    zo.write(ROOT / "main_v3.py", "main.py")
    zo.write(ROOT / "tests_synth.py", "tests_synth.py")
    for p in sorted((ROOT / "datsc").glob("*.py")):
        zo.write(p, f"datsc/{p.name}")
    n = 0
    for info in zi.infolist():
        if info.filename.startswith("artifacts/"):
            zo.writestr(info, zi.read(info.filename)); n += 1
    code = [i.filename for i in zi.infolist() if not i.filename.startswith("artifacts/")]
print(f"{dst}: fresh code ({len(code)} files replaced) + {n} artifact files from {src.name}; {dst.stat().st_size/1e6:.0f} MB")
