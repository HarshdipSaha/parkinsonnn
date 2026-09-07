"""Approach C: fit the richer feature arm ("featc") on the registered cache -- OOF on random folds and pseudo-site folds
over a small model grid (kernel-RFF logistic / MLP / linear), mirror-averaged, typicality kept -- then a full-data model.
Writes <out>/featc_v5.npz, oof_featcarm.csv, oof_featcarm_site.csv, featcarm.json (same contract as the run-4 feat arm).

python scripts/feat_arm_c.py --cache cache_v4 --template artifacts/template.npz --out artifacts_v3 [--grid] [--force]
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

from datsc.featmodel_c import FeatModelC
from datsc.features_c import FEATURE_NAMES_C, C_NAMES, feat_vector_c
from datsc.features_v4 import N_PATCH, mirror, striatal_patch
from datsc.register import Template


def build(cache: Path, T, force=False):
    """F[N,2,n_feat] (index 1 of axis 1 = mirrored), P[N,2,n_patch]."""
    fp, pp = cache / "features_c.npy", cache / "patch_v4.npy"
    meta = pd.read_csv(cache / "meta.csv", dtype={"uid": str})
    have_f = fp.exists() and not force and np.load(fp, mmap_mode="r").shape[:2] == (len(meta), 2) and np.load(fp, mmap_mode="r").shape[2] == len(FEATURE_NAMES_C)
    have_p = pp.exists() and np.load(pp, mmap_mode="r").shape[0] == len(meta)
    vols = np.load(cache / "vols.npy", mmap_mode="r")
    F = np.load(fp) if have_f else np.zeros((len(meta), 2, len(FEATURE_NAMES_C)), np.float32)
    P = np.load(pp) if have_p else np.zeros((len(meta), 2, N_PATCH), np.float32)
    if have_f and have_p:
        return F, P, meta
    t0 = time.time()
    for i in range(len(meta)):
        v = np.asarray(vols[i], np.float32)
        for o, vv in enumerate((v, mirror(v))):
            if not have_f:
                F[i, o] = feat_vector_c(vv, T)
            if not have_p:
                P[i, o] = striatal_patch(vv)
        if i % 200 == 0:
            print(f"features {i}/{len(meta)} ({time.time()-t0:.0f}s)", flush=True)
    np.save(fp, F)
    if not have_p:
        np.save(pp, P)
    return F, P, meta


def oof(F, P, y, folds, kw, with_d=False):
    p = np.zeros(len(y)); d = np.zeros(len(y))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        m = FeatModelC(**kw).fit(np.concatenate([F[tr, 0], F[tr, 1]]), np.concatenate([P[tr, 0], P[tr, 1]]), np.concatenate([y[tr], y[tr]]))
        z = 0.5 * (m.decision(F[va, 0], P[va, 0]) + m.decision(F[va, 1], P[va, 1]))
        p[va] = 1 / (1 + np.exp(-z)); d[va] = m.typicality(P[va, 0])
    return (p, d) if with_d else p


def ll(y, p):
    return log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))


GRID = [
    dict(kind="linear", n_comp=40, C=0.3),
    dict(kind="linear", n_comp=40, C=0.1),
    dict(kind="rff", n_comp=40, C=0.3, gamma=0.25, n_rff=512),
    dict(kind="rff", n_comp=40, C=1.0, gamma=0.25, n_rff=512),
    dict(kind="rff", n_comp=40, C=0.3, gamma=0.5, n_rff=1024),
    dict(kind="rff", n_comp=20, C=0.3, gamma=0.5, n_rff=512),
    dict(kind="mlp", n_comp=40, hidden=(64, 32), alpha=1e-1),
    dict(kind="mlp", n_comp=40, hidden=(64, 32), alpha=1.0),
    dict(kind="mlp", n_comp=20, hidden=(32,), alpha=1.0),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache_v4")
    ap.add_argument("--template", default="artifacts/template.npz")
    ap.add_argument("--out", default="artifacts_v3")
    ap.add_argument("--grid", action="store_true", help="evaluate the model grid on the random folds first (else: linear C=0.3)")
    ap.add_argument("--kind", default=None, help="force a model kind: linear | rff | mlp (overrides --grid)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cache, out = Path(a.cache), Path(a.out); out.mkdir(exist_ok=True)
    T = Template.load(a.template)
    F, P, meta = build(cache, T, a.force)
    y = meta["is_pathologic"].values.astype(float)
    folds = pd.read_csv(cache / "folds.csv", dtype={"uid": str}).set_index("uid").loc[meta.uid, "fold"].values
    fs = pd.read_csv(cache / "folds_site.csv", dtype={"uid": str}).set_index("uid").loc[meta.uid, "fold"].values if (cache / "folds_site.csv").exists() else None
    print(f"features: {F.shape[2]} ({len(C_NAMES)} new in the C block); patch {P.shape[2]}")
    kw = dict(kind="linear", n_comp=40, C=0.3)
    if a.grid:
        rows = []
        for g in GRID:
            if a.kind and g["kind"] != a.kind:
                continue
            t0 = time.time(); p = oof(F, P, y, folds, g); rows.append((ll(y, p), roc_auc_score(y, p), g))
            print(f"  {json.dumps(g):<70} OOF logloss={rows[-1][0]:.4f} AUROC={rows[-1][1]:.4f} ({time.time()-t0:.0f}s)", flush=True)
        best = min(rows, key=lambda r: r[0]); kw = best[2]
        print(f"grid best: {json.dumps(kw)} (logloss {best[0]:.4f})")
    elif a.kind:
        kw = next(g for g in GRID if g["kind"] == a.kind)
    p, d = oof(F, P, y, folds, kw, with_d=True)
    auc, l = roc_auc_score(y, p), ll(y, p)
    print(f"featc random folds: OOF AUROC={auc:.4f} logloss={l:.4f}  (run-4 feat arm: 0.9398 / 0.3120; LR arm 0.902 / 0.4015); d median {np.median(d):.2f} p90 {np.percentile(d, 90):.2f}")
    pd.DataFrame({"uid": meta.uid, "y": y, "p": p, "d": d}).to_csv(out / "oof_featcarm.csv", index=False)
    if fs is not None:
        ps, ds = oof(F, P, y, fs, kw, with_d=True)
        print(f"featc site folds  : OOF AUROC={roc_auc_score(y, ps):.4f} logloss={ll(y, ps):.4f}  (run-4 feat arm site folds: 0.9277 / 0.3548)")
        pd.DataFrame({"uid": meta.uid, "y": y, "p": ps, "d": ds}).to_csv(out / "oof_featcarm_site.csv", index=False)
        sites = pd.read_csv(cache / "sites.csv", dtype={"uid": str}).set_index("uid").loc[meta.uid, "site"].values
        for s in pd.Series(sites).value_counts().index[:9]:
            m = sites == s
            print(f"  site {s:<24} n={m.sum():4d} prev={y[m].mean():.2f} mean_p={ps[m].mean():.2f} logloss={ll(y[m], ps[m]):.4f} "
                  f"AUROC={roc_auc_score(y[m], ps[m]) if 0 < y[m].mean() < 1 else float('nan'):.3f}")
    off = len(FEATURE_NAMES_C) - len(C_NAMES)
    uni = sorted([(round(max(roc_auc_score(y, F[:, 0, off + i]), 1 - roc_auc_score(y, F[:, 0, off + i])), 3), n) for i, n in enumerate(C_NAMES)], reverse=True)[:10]
    print("top C-block features (univariate AUROC):", uni)
    full = FeatModelC(**kw).fit(np.concatenate([F[:, 0], F[:, 1]]), np.concatenate([P[:, 0], P[:, 1]]), np.concatenate([y, y]))
    full.save(out / "featc_v5.npz")
    (out / "featcarm.json").write_text(json.dumps({"tag": "featcarm", "spec": {"arm": "featc"}, "model_kw": kw, "norm": "ref",
                                                   "checkpoints": ["artifacts/featc_v5.npz"], "val_auroc": float(auc), "val_logloss": float(l)}))
    print(f"saved {out/'featc_v5.npz'} ({(out/'featc_v5.npz').stat().st_size/1e6:.1f} MB) + oof_featcarm.csv (+ _site) + featcarm.json")
