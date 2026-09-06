"""Fit the run-3 ensemble combiner on OOF predictions and write artifacts_v3/manifest.json.

Each arm = a glob of run tags sharing one InputSpec/model_kw (read from the first run's json). Arm-level OOF =
mean over its checkpoints. Combiner = weighted logit average (weights on a simplex grid), temperature T, prior mix.
T is fitted on the random folds (--calib random) or on leave-pseudo-site-out predictions (--calib site) and can be
floored (--t-floor) so a shifted test set is never met with sharpened probabilities.

python scripts/fit_stack_v3.py --out artifacts_v3 --arm 3d_a "r3d_64x64x48_f*_s*" --arm 2d_b0 "r2d_efficientnet_b0_f*_s*" [--lam 0.02] [--pi 0.548] [--t-floor 1.0]
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from datsc.calibrate import fit_temperature, logit, logloss, sigmoid
from datsc.config import DEFAULT_PRIOR


def arm_oof(art: Path, pattern: str):
    files = sorted(glob.glob(str(art / f"oof_{pattern}.csv")))
    assert files, f"no OOF files for {pattern}"
    df = pd.concat([pd.read_csv(f, dtype={"uid": str}) for f in files])
    oof = df.groupby("uid").agg(y=("y", "first"), p=("p", "mean"))
    tags = [Path(f).name[4:-4] for f in files]
    meta = json.loads((art / f"{tags[0]}.json").read_text())
    return oof, tags, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts_v3")
    ap.add_argument("--arm", nargs=2, action="append", metavar=("NAME", "PATTERN"), required=True)
    ap.add_argument("--site-arm", nargs=2, action="append", metavar=("NAME", "PATTERN"), default=[],
                    help="leave-pseudo-site-out OOF pattern per arm; when every arm has one, T is fitted on the site pool")
    ap.add_argument("--lam", type=float, default=0.02)
    ap.add_argument("--pi", type=float, default=DEFAULT_PRIOR)
    ap.add_argument("--t-floor", type=float, default=1.0)
    ap.add_argument("--grid", type=int, default=10, help="simplex resolution for arm weights")
    a = ap.parse_args()
    art = Path(a.out)
    arms, oofs = [], {}
    for name, pattern in a.arm:
        oof, tags, meta = arm_oof(art, pattern)
        oofs[name] = oof
        arms.append({"name": name, "arm": meta["spec"]["arm"], "spec": meta["spec"], "model_kw": meta["model_kw"],
                     "norm": meta.get("norm", "p75"), "checkpoints": [f"artifacts/{t}.pt" for t in tags]})
        print(f"{name:<10} ckpts={len(tags):2d} AUROC={roc_auc_score(oof.y, oof.p):.4f} logloss={logloss(oof.y.values, oof.p.values):.4f}")
    uids = sorted(set.intersection(*[set(o.index) for o in oofs.values()]))
    y = oofs[arms[0]["name"]].loc[uids, "y"].values.astype(float)
    Z = np.stack([logit(oofs[arm["name"]].loc[uids, "p"].values) for arm in arms])
    best = (float("inf"), None)
    for comb in itertools.product(range(a.grid + 1), repeat=len(arms)):
        if sum(comb) != a.grid:
            continue
        w = np.array(comb, float) / a.grid
        ll = logloss(y, sigmoid((w[:, None] * Z).sum(0)))
        if ll < best[0]:
            best = (ll, w)
    w = best[1]
    z = (w[:, None] * Z).sum(0)
    T_rand = fit_temperature(z, y)
    T_site = None
    site = dict(a.site_arm)
    if site and all(x["name"] in site for x in arms):
        oofs_s = {n: arm_oof(art, site[n])[0] for n in site}
        uids_s = sorted(set.intersection(*[set(o.index) for o in oofs_s.values()]))
        y_s = oofs_s[arms[0]["name"]].loc[uids_s, "y"].values.astype(float)
        Z_s = np.stack([logit(oofs_s[x["name"]].loc[uids_s, "p"].values) for x in arms])
        z_s = (w[:, None] * Z_s).sum(0)
        T_site = fit_temperature(z_s, y_s)
        print(f"site-fold pool: n={len(y_s)} AUROC={roc_auc_score(y_s, sigmoid(z_s)):.4f} logloss(T=1)={logloss(y_s, sigmoid(z_s)):.4f} "
              f"logloss(T_site)={logloss(y_s, sigmoid(z_s / T_site)):.4f} T_site={T_site:.3f} (random-fold T={T_rand:.3f})")
    T = max(T_site if T_site is not None else T_rand, a.t_floor)
    p = sigmoid(z / T); final = (1 - a.lam) * p + a.lam * a.pi
    print(f"weights={dict(zip([x['name'] for x in arms], np.round(w, 2)))}  T={T:.3f} (random {T_rand:.3f}, site {T_site}, floor {a.t_floor})  lam={a.lam}  pi={a.pi:.3f}")
    print(f"stack OOF: AUROC={roc_auc_score(y, final):.4f} logloss={logloss(y, final):.4f}  (equal-weight logloss={logloss(y, sigmoid(Z.mean(0))):.4f})")
    man = {"pi": a.pi, "T": float(T), "T_random": float(T_rand), "T_site": (float(T_site) if T_site is not None else None), "lam": a.lam, "weights": {x["name"]: float(wi) for x, wi in zip(arms, w)}, "arms": arms,
           "oof_logloss": logloss(y, final), "oof_auroc": float(roc_auc_score(y, final)), "n": int(len(y))}
    (art / "manifest.json").write_text(json.dumps(man, indent=1))
    pd.DataFrame({"uid": uids, "y": y, "p": final}).to_csv(art / "oof_stack.csv", index=False)
    print(f"wrote {art/'manifest.json'}")
