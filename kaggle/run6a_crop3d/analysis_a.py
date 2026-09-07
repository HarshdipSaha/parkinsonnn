"""Approach-A analysis: how much room the striatal-crop 3D arm has (per-site OOF, diversity vs the 2D arms, training curves).
python kaggle/run6a_crop3d/analysis_a.py [--art <dir with oof_*.csv + *.json>] [--cache <cache dir>]
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--art", default="H:/augsepthacks/parkinsonnn/data/ckpt_ds_v4/artifacts_v3")
ap.add_argument("--cache", default="H:/augsepthacks/parkinsonnn/cache_v4")
a = ap.parse_args()
sites = pd.read_csv(f"{a.cache}/sites.csv", dtype={"uid": str}).set_index("uid").site


def arm(pat):
    fs = sorted(glob.glob(f"{a.art}/oof_{pat}.csv"))
    df = pd.concat([pd.read_csv(f, dtype={"uid": str}) for f in fs])
    return df.groupby("uid").agg(y=("y", "first"), p=("p", "mean")), len(fs)


res = {}
for name, pat in [("crop3d", "r3d_crop48x48x32_f?_s?"), ("3d64", "r3d_64x64x48_f?_s?"), ("v2s", "r2d_efficientnet_v2_s_f?_s?"),
                  ("cnx", "r2d_convnext_tiny_f?_s?"), ("b0cor", "r2d_efficientnet_b0_cor_f?_s?"), ("b0", "r2d_efficientnet_b0_f?_s?"),
                  ("b0crop64", "r2d_effb0_crop64_f?_s?")]:
    o, n = arm(pat); y, p = o.y.values, o.p.clip(1e-6, 1 - 1e-6).values; res[name] = o
    print(f"{name:8s} n_ckpt={n:2d} logloss={log_loss(y, p):.4f} auroc={roc_auc_score(y, p):.4f}")
o = res["crop3d"]; y, p = o.y.values, o.p.clip(1e-6, 1 - 1e-6).values; s = sites.loc[o.index]
print("crop3d per pseudo-site:")
for st in s.value_counts().index[:9]:
    m = (s == st).values
    print(f"    {st:<28} n={m.sum():4d} prev={y[m].mean():.2f} mean_p={p[m].mean():.2f} ll={log_loss(y[m], p[m]):.4f} "
          f"auc={roc_auc_score(y[m], p[m]) if 0 < y[m].mean() < 1 else float('nan'):.3f}")
lg = lambda q: np.log(q.clip(1e-6, 1 - 1e-6) / (1 - q.clip(1e-6, 1 - 1e-6)))
A = lg(res["crop3d"].p); B = lg(res["v2s"].p.loc[A.index]); C = lg(res["cnx"].p.loc[A.index]); D = lg(res["3d64"].p.loc[A.index])
print("logit correlation crop3d~v2s %.3f crop3d~cnx %.3f v2s~cnx %.3f crop3d~3d64 %.3f" % (np.corrcoef(A, B)[0, 1], np.corrcoef(A, C)[0, 1], np.corrcoef(B, C)[0, 1], np.corrcoef(A, D)[0, 1]))
# seed variance proxy: per-fold losses of the single crop3d seed
for f in sorted(glob.glob(f"{a.art}/r3d_crop48x48x32_f?_s0.json")):
    j = json.load(open(f)); h = j["history"]
    ep = [(i, round(r["val_logloss"], 3)) for i, r in enumerate(h) if i in (9, 19, 29, 39, 49)]
    print(f.replace(chr(92), "/").split("/")[-1], "final", round(j["val_logloss"], 4), "best", round(j["best_val_logloss"], 4), "epochs", len(h), ep,
          "raw last", round(h[-1].get("val_raw_logloss", float("nan")), 4), "sec", int(j["sec"]), "params", j["params"], "lr", j["lr"], "mixup", j["mixup"], "ema", j["ema_decay"])
# what a 2-seed average of crop3d would give: bootstrap from fold losses is not possible with one seed; report the b0 3-seed gain as the analogue
for name in ("b0",):
    fs = sorted(glob.glob(f"{a.art}/oof_r2d_efficientnet_b0_f?_s?.csv"))
    per_seed = {}
    for f in fs:
        seed = f[-6:-4]; per_seed.setdefault(seed, []).append(pd.read_csv(f, dtype={"uid": str}))
    singles = [log_loss(pd.concat(v).y, pd.concat(v).p.clip(1e-6, 1 - 1e-6)) for v in per_seed.values()]
    print("b0: single-seed OOF mean %.4f -> 3-seed mean-prob OOF %.4f (seed averaging gain %.4f)" % (np.mean(singles), log_loss(res["b0"].y, res["b0"].p.clip(1e-6, 1 - 1e-6)), np.mean(singles) - log_loss(res["b0"].y, res["b0"].p.clip(1e-6, 1 - 1e-6))))
