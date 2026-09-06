"""Add histogram-standardisation landmarks (reference-region percentiles of the study template) to template.npz.
python scripts/template_landmarks.py [--template artifacts/template.npz] [--also artifacts_v3/template.npz data/ckpt_upload_run3/template.npz]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from datsc.register import HIST_PCTS, hist_landmarks

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="artifacts/template.npz")
    ap.add_argument("--also", nargs="*", default=["artifacts_v3/template.npz"])
    a = ap.parse_args()
    z = dict(np.load(a.template))
    lm = hist_landmarks(z["box"], z["ref_mask"].astype(bool))     # reference region (brain minus striatum), never the striatum itself
    z["hist"] = lm
    np.savez(a.template, **z)
    print("landmarks", dict(zip(HIST_PCTS, np.round(lm, 4))))
    for dst in a.also:
        Path(dst).parent.mkdir(parents=True, exist_ok=True); shutil.copy(a.template, dst); print("copied ->", dst)
