"""Build submission.zip: main.py, tests_synth.py, datsc/*.py, artifacts/{lr.json,calib.json,cnn_f*_s*.pt}."""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="submission.zip")
    a = ap.parse_args()
    art = Path(a.artifacts)
    need = [art / "lr.json", art / "calib.json"]
    weights = sorted(p for p in art.glob("cnn_f*_s*.pt") if "_com" not in p.name)
    for p in need:
        assert p.exists(), f"missing {p}"
    with zipfile.ZipFile(ROOT / a.out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(ROOT / "main.py", "main.py")
        z.write(ROOT / "tests_synth.py", "tests_synth.py")
        for p in sorted((ROOT / "datsc").glob("*.py")):
            z.write(p, f"datsc/{p.name}")
        for p in need + weights:
            z.write(p, f"artifacts/{p.name}")
    print(f"wrote {a.out} with {len(weights)} checkpoints")
