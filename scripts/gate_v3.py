"""Registration gate G1 + the run-3 LR arm: registered-ROI striatal features -> L2 logistic regression.
Reports OOF AUROC/log loss on the random folds AND leave-pseudo-site-out folds, plus per-site bias.
Writes <cache>/features_v3.npy, <cache>/oof_lr_v3.csv and (if --save) artifacts_v3/lr_v3.json.

python scripts/gate_v3.py --cache cache_v4 --template artifacts/template.npz [--save]
GATE: random-fold OOF AUROC >= 0.92 and registration fallback < 3%.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from datsc.features_v3 import FEATURE_NAMES, striatal_features
from datsc.lrmodel import LRModel
from datsc.register import Template


def oof_lr(X, y, folds, C=0.3):
    p = np.zeros(len(y))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        m = LRModel(C=C).fit(X[tr], y[tr])
        p[va] = m.predict_proba(X[va])
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache_v4")
    ap.add_argument("--template", default="artifacts/template.npz")
    ap.add_argument("--out", default="artifacts_v3")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--C", type=float, default=0.3)
    a = ap.parse_args()
    cache = Path(a.cache); T = Template.load(a.template)
    vols = np.load(cache / "vols.npy", mmap_mode="r"); meta = pd.read_csv(cache / "meta.csv", dtype={"uid": str})
    fp = cache / "features_v3.npy"
    if fp.exists() and np.load(fp).shape[0] == len(meta):
        X = np.load(fp)
    else:
        X = np.stack([striatal_features(np.asarray(vols[i], np.float32), T) for i in range(len(meta))]); np.save(fp, X)
    y = meta["is_pathologic"].values.astype(float)
    folds = pd.read_csv(cache / "folds.csv", dtype={"uid": str}).set_index("uid").loc[meta.uid, "fold"].values
    p = oof_lr(X, y, folds, a.C)
    auc, ll = roc_auc_score(y, p), log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))
    print(f"random folds : OOF AUROC={auc:.4f} logloss={ll:.4f}")
    if (cache / "folds_site.csv").exists():
        fs = pd.read_csv(cache / "folds_site.csv", dtype={"uid": str}).set_index("uid").loc[meta.uid, "fold"].values
        ps = oof_lr(X, y, fs, a.C)
        print(f"site folds   : OOF AUROC={roc_auc_score(y, ps):.4f} logloss={log_loss(y, np.clip(ps,1e-6,1-1e-6)):.4f}")
        sites = pd.read_csv(cache / "sites.csv", dtype={"uid": str}).set_index("uid").loc[meta.uid, "site"].values
        for s in pd.Series(sites).value_counts().index[:8]:
            m = sites == s
            print(f"  site {s:<22} n={m.sum():4d} prev={y[m].mean():.2f} mean_p(site)={ps[m].mean():.2f} mean_p(rand)={p[m].mean():.2f} "
                  f"AUROC site={roc_auc_score(y[m], ps[m]) if 0<y[m].mean()<1 else float('nan'):.3f} rand={roc_auc_score(y[m], p[m]) if 0<y[m].mean()<1 else float('nan'):.3f}")
    fb = 1 - meta["reg_ok"].mean() if "reg_ok" in meta else float("nan")
    print(f"registration fallback: {100*fb:.1f}%  (normals {100*(1-meta.loc[y==0,'reg_ok'].mean()):.1f}%, abnormals {100*(1-meta.loc[y==1,'reg_ok'].mean()):.1f}%)" if "reg_ok" in meta else "")
    # univariate AUROCs for insight
    print("top single features:", sorted([(round(max(roc_auc_score(y, X[:, i]), 1 - roc_auc_score(y, X[:, i])), 3), n) for i, n in enumerate(FEATURE_NAMES)], reverse=True)[:6])
    print(f"GATE G1: {'PASS' if auc >= 0.92 and fb < 0.03 else 'FAIL'} (AUROC {auc:.3f} vs 0.92; fallback {100*fb:.1f}% vs 3%)")
    pd.DataFrame({"uid": meta.uid, "y": y, "p": p}).to_csv(cache / "oof_lr_v3.csv", index=False)
    if a.save:
        Path(a.out).mkdir(exist_ok=True)
        LRModel(C=a.C).fit(X, y).save(Path(a.out) / "lr_v3.json")
        print(f"saved {Path(a.out)/'lr_v3.json'}")
