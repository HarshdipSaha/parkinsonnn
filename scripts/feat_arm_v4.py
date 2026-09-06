"""Fit the feature-engineered arm ("feat") on the registered cache: OOF on random folds and pseudo-site folds, then a
full-data model. Writes artifacts_v3/feat_v4.npz, oof_featarm.csv, oof_featarm_site.csv, featarm.json (same contract as
the LR arm from gate_v3.py, so fit_stack_v3.py can take `--arm feat featarm --site-arm feat featarm_site`).

python scripts/feat_arm_v4.py --cache cache_v4 --template artifacts/template.npz --out artifacts_v3 [--grid]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from datsc.featmodel import FeatModel
from datsc.features_v4 import FEATURE_NAMES, N_PATCH, feat_vector, mirror, striatal_patch
from datsc.register import Template


def build(cache: Path, T, force=False):
    """F[N,2,n_feat], P[N,2,n_patch]: index 1 of axis 1 is the mirrored orientation."""
    fp, pp = cache / "features_v4.npy", cache / "patch_v4.npy"
    meta = pd.read_csv(cache / "meta.csv", dtype={"uid": str})
    if fp.exists() and pp.exists() and not force and np.load(fp, mmap_mode="r").shape[0] == len(meta):
        return np.load(fp), np.load(pp), meta
    vols = np.load(cache / "vols.npy", mmap_mode="r")
    F = np.zeros((len(meta), 2, len(FEATURE_NAMES)), np.float32); P = np.zeros((len(meta), 2, N_PATCH), np.float32)
    t0 = time.time()
    for i in range(len(meta)):
        v = np.asarray(vols[i], np.float32)
        for o, vv in enumerate((v, mirror(v))):
            F[i, o] = feat_vector(vv, T); P[i, o] = striatal_patch(vv)
        if i % 200 == 0:
            print(f"features {i}/{len(meta)} ({time.time()-t0:.0f}s)", flush=True)
    np.save(fp, F); np.save(pp, P)
    return F, P, meta


def oof(F, P, y, folds, n_comp, C, with_d=False):
    p = np.zeros(len(y)); d = np.zeros(len(y))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        m = FeatModel(n_comp, C).fit(np.concatenate([F[tr, 0], F[tr, 1]]), np.concatenate([P[tr, 0], P[tr, 1]]), np.concatenate([y[tr], y[tr]]))
        z = 0.5 * (m.decision(F[va, 0], P[va, 0]) + m.decision(F[va, 1], P[va, 1]))      # average over both orientations
        p[va] = 1 / (1 + np.exp(-z)); d[va] = m.typicality(P[va, 0])
    return (p, d) if with_d else p


def ll(y, p):
    return log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache_v4")
    ap.add_argument("--template", default="artifacts/template.npz")
    ap.add_argument("--out", default="artifacts_v3")
    ap.add_argument("--n-comp", type=int, default=40)
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--grid", action="store_true", help="evaluate a small (n_comp, C) grid on the random folds first")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cache, out = Path(a.cache), Path(a.out); out.mkdir(exist_ok=True)
    T = Template.load(a.template)
    F, P, meta = build(cache, T, a.force)
    y = meta["is_pathologic"].values.astype(float)
    folds = pd.read_csv(cache / "folds.csv", dtype={"uid": str}).set_index("uid").loc[meta.uid, "fold"].values
    fs = pd.read_csv(cache / "folds_site.csv", dtype={"uid": str}).set_index("uid").loc[meta.uid, "fold"].values if (cache / "folds_site.csv").exists() else None
    n_comp, C = a.n_comp, a.C
    if a.grid:
        rows = []
        for nc in (20, 40, 80):
            for c in (0.03, 0.1, 0.3):
                p = oof(F, P, y, folds, nc, c); rows.append((ll(y, p), roc_auc_score(y, p), nc, c))
                print(f"  n_comp={nc:3d} C={c:<5} OOF logloss={rows[-1][0]:.4f} AUROC={rows[-1][1]:.4f}", flush=True)
        best = min(rows); n_comp, C = best[2], best[3]
        print(f"grid best: n_comp={n_comp} C={C} (logloss {best[0]:.4f})")
    p, d = oof(F, P, y, folds, n_comp, C, with_d=True)
    auc, l = roc_auc_score(y, p), ll(y, p)
    print(f"feat arm random folds: OOF AUROC={auc:.4f} logloss={l:.4f}  (run-3 LR arm: 0.902 / 0.4015); typicality d: median {np.median(d):.2f} p90 {np.percentile(d, 90):.2f} max {d.max():.2f}")
    pd.DataFrame({"uid": meta.uid, "y": y, "p": p, "d": d}).to_csv(out / "oof_featarm.csv", index=False)
    if fs is not None:
        ps, ds = oof(F, P, y, fs, n_comp, C, with_d=True)
        print(f"feat arm site folds  : OOF AUROC={roc_auc_score(y, ps):.4f} logloss={ll(y, ps):.4f}; typicality d (site folds): median {np.median(ds):.2f} p90 {np.percentile(ds, 90):.2f}")
        pd.DataFrame({"uid": meta.uid, "y": y, "p": ps, "d": ds}).to_csv(out / "oof_featarm_site.csv", index=False)
        sites = pd.read_csv(cache / "sites.csv", dtype={"uid": str}).set_index("uid").loc[meta.uid, "site"].values
        for s in pd.Series(sites).value_counts().index[:8]:
            m = sites == s
            print(f"  site {s:<24} n={m.sum():4d} prev={y[m].mean():.2f} mean_p={ps[m].mean():.2f} logloss={ll(y[m], ps[m]):.4f} "
                  f"AUROC={roc_auc_score(y[m], ps[m]) if 0 < y[m].mean() < 1 else float('nan'):.3f}")
    # univariate view of the engineered block
    from datsc.features_v4 import ENG_NAMES
    off = len(FEATURE_NAMES) - len(ENG_NAMES)
    uni = sorted([(round(max(roc_auc_score(y, F[:, 0, off + i]), 1 - roc_auc_score(y, F[:, 0, off + i])), 3), n) for i, n in enumerate(ENG_NAMES)], reverse=True)[:8]
    print("top engineered features (AUROC):", uni)
    full = FeatModel(n_comp, C).fit(np.concatenate([F[:, 0], F[:, 1]]), np.concatenate([P[:, 0], P[:, 1]]), np.concatenate([y, y]))
    full.save(out / "feat_v4.npz")
    (out / "featarm.json").write_text(json.dumps({"tag": "featarm", "spec": {"arm": "feat"}, "model_kw": {"n_comp": n_comp, "C": C}, "norm": "ref",
                                                  "checkpoints": ["artifacts/feat_v4.npz"], "val_auroc": float(auc), "val_logloss": float(l)}))
    print(f"saved {out/'feat_v4.npz'} ({(out/'feat_v4.npz').stat().st_size/1e6:.1f} MB) + oof_featarm.csv (+ _site) + featarm.json")
