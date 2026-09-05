"""Leave-one-spacing-cluster-out on the LR arm (cheap proxy for site shift).
Rule (precommitted): worse cluster inflates log loss by <50% -> lam=0.02, else lam=0.05."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd

from datsc.calibrate import logloss
from datsc.lrmodel import LRModel, feature_matrix

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--C", type=float, default=0.3)
    a = ap.parse_args()
    f = pd.read_csv(Path(a.cache) / "folds.csv", dtype={"uid": str})
    X = feature_matrix(Path(a.cache), f["uid"]); y = f["is_pathologic"].values.astype(int)
    base = logloss(y, pd.read_csv(Path(a.cache) / "oof_lr.csv", dtype={"uid": str})["p_lr"].values)
    worst = 0.0
    for lo, hi, name in ((2.43, 2.52, "2.46mm"), (3.85, 3.95, "3.90mm")):   # 2.43 excludes the separate 2.40mm cluster
        held = (f["native_spacing_mm"] >= lo) & (f["native_spacing_mm"] < hi)
        p = LRModel(C=a.C).fit(X[~held], y[~held]).predict_proba(X[held])
        ll = logloss(y[held], p); infl = ll / base - 1
        worst = max(worst, infl)
        print(f"hold out {name}: n={int(held.sum())} logloss={ll:.4f} (random-CV={base:.4f}, inflation={infl:+.0%})")
    lam = 0.02 if worst < 0.5 else 0.05
    print(f"RULE -> lam={lam}")
