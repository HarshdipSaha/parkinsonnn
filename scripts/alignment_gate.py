"""OOF logistic regression on hotspot radiomics + scalars. PASS iff OOF AUROC > 0.90.
Also writes artifacts/lr.json (fit on all data) and cache/oof_lr.csv."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from datsc.lrmodel import LRModel, feature_matrix

GATE_AUROC = 0.90

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--C", type=float, default=0.3)
    a = ap.parse_args()
    cache, art = Path(a.cache), Path(a.artifacts)
    art.mkdir(exist_ok=True)
    f = pd.read_csv(cache / "folds.csv", dtype={"uid": str})
    X = feature_matrix(cache, f["uid"]); y = f["is_pathologic"].values.astype(int)
    oof = np.zeros(len(y))
    for k in sorted(f["fold"].unique()):
        tr, va = (f["fold"] != k).values, (f["fold"] == k).values
        oof[va] = LRModel(C=a.C).fit(X[tr], y[tr]).predict_proba(X[va])
    auc, ll = roc_auc_score(y, oof), log_loss(y, np.clip(oof, 1e-4, 1 - 1e-4))
    pd.DataFrame({"uid": f["uid"], "y": y, "p_lr": oof}).to_csv(cache / "oof_lr.csv", index=False)
    LRModel(C=a.C).fit(X, y).save(art / "lr.json")
    print(f"OOF AUROC={auc:.4f}  OOF logloss={ll:.4f}  (prior-only logloss={log_loss(y, np.full(len(y), y.mean())):.4f})")
    print("GATE:", "PASS" if auc > GATE_AUROC else "FAIL - fix the anchor before training any CNN")
