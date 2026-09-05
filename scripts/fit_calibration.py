"""Average CNN OOF over seeds, blend with LR OOF, fit w & T, write artifacts/calib.json.
lam and pi are inputs: lam from scripts/shift_diagnostic.py (default 0.02), pi from the prior-only LB (default 0.548)."""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd

from datsc import calibrate as cb
from datsc.config import DEFAULT_PRIOR

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--lam", type=float, default=0.02)
    ap.add_argument("--pi", type=float, default=DEFAULT_PRIOR)
    a = ap.parse_args()
    art = Path(a.artifacts)
    files = [f for f in glob.glob(str(art / "oof_f*_s*.csv")) if "_com" not in f]
    assert files, "no CNN OOF files found"
    cnn = pd.concat([pd.read_csv(f, dtype={"uid": str}) for f in files]).groupby("uid").agg(y=("y", "first"), p_cnn=("p", "mean")).reset_index()
    lr = pd.read_csv(Path(a.cache) / "oof_lr.csv", dtype={"uid": str})[["uid", "p_lr"]]
    df = cnn.merge(lr, on="uid")
    y = df["y"].values.astype(float)
    print(f"n={len(df)}  seeds/fold~{len(files)//5}  logloss cnn={cb.logloss(y, df.p_cnn):.4f}  lr={cb.logloss(y, df.p_lr):.4f}")
    c = cb.fit(df["p_cnn"].values, df["p_lr"].values, y, lam=a.lam, pi=a.pi)
    final = c.apply(df["p_cnn"].values, df["p_lr"].values)
    print(f"w={c.w:.2f} T={c.T:.3f} lam={c.lam} pi={c.pi:.3f}  OOF logloss after calibration={cb.logloss(y, final):.4f}")
    c.save(art / "calib.json")
