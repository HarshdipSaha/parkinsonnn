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
    mp = art / f"{tags[0]}.json"
    meta = json.loads(mp.read_text()) if mp.exists() else None    # site-fold OOFs only need the predictions
    return oof, tags, meta


def compositions(total: int, parts: int):
    """All non-negative integer vectors of length `parts` summing to `total` (stars and bars): C(total+parts-1, parts-1)
    of them, e.g. 43,758 for grid 8 x 11 arms -- NOT the 9^11 product that run 4a's report cell got stuck on."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(total - first, parts - 1):
            yield (first,) + rest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts_v3")
    ap.add_argument("--arm", nargs=2, action="append", metavar=("NAME", "PATTERN"), required=True)
    ap.add_argument("--site-arm", nargs=2, action="append", metavar=("NAME", "PATTERN"), default=[],
                    help="leave-pseudo-site-out OOF pattern per arm; when every arm has one, T is fitted on the site pool")
    ap.add_argument("--extra", nargs=2, action="append", metavar=("NAME", "GLOB"), default=[],
                    help="extra checkpoint tags (e.g. full-data models 'r2d_efficientnet_b0_fall_s*') appended to an arm")
    ap.add_argument("--lam", type=float, default=0.02)
    ap.add_argument("--pi", type=float, default=DEFAULT_PRIOR)
    ap.add_argument("--t-floor", type=float, default=1.0)
    ap.add_argument("--grid", type=int, default=10, help="simplex resolution for arm weights")
    ap.add_argument("--cache", default="cache_v4", help="for folds.csv (nested estimate) and sites.csv (per-site report)")
    ap.add_argument("--nested", action="store_true", help="print the nested-CV estimate of the whole combiner and select clip/hedge by it")
    ap.add_argument("--clip", type=float, nargs="*", default=[0.0, 0.01, 0.02, 0.03], help="candidate tail clips (0 = none)")
    ap.add_argument("--hedge-q", type=float, nargs="*", default=[0.0, 0.85, 0.95], help="candidate disagreement quantiles (0 = no hedge)")
    ap.add_argument("--hedge-lam", type=float, nargs="*", default=[0.15, 0.3])
    a = ap.parse_args()
    art = Path(a.out)
    arms, oofs = [], {}
    for name, pattern in a.arm:
        oof, tags, meta = arm_oof(art, pattern)
        oofs[name] = oof
        arms.append({"name": name, "arm": meta["spec"]["arm"], "spec": meta["spec"], "model_kw": {k: v for k, v in meta.get("model_kw", {}).items() if k != "weights_path"},
                     "norm": meta.get("norm", "p75"), "checkpoints": meta.get("checkpoints") or [f"artifacts/{t}.pt" for t in tags]})
        print(f"{name:<10} ckpts={len(tags):2d} AUROC={roc_auc_score(oof.y, oof.p):.4f} logloss={logloss(oof.y.values, oof.p.values):.4f}")
    for name, pat in a.extra:
        arm = next(x for x in arms if x["name"] == name)
        extra = sorted(Path(f).name[:-3] for f in glob.glob(str(art / f"{pat}.pt")))
        arm["checkpoints"] += [f"artifacts/{t}.pt" for t in extra]
        print(f"{name}: +{len(extra)} extra checkpoints ({pat})")
    n_max = max(len(o) for o in oofs.values())
    short = [x["name"] for x in arms if len(oofs[x["name"]]) < 0.98 * n_max]
    if short:
        print(f"dropping arms with incomplete OOF coverage (< 98% of {n_max} uids): {short}")
        arms = [x for x in arms if x["name"] not in short]; oofs = {k: v for k, v in oofs.items() if k not in short}
    uids = sorted(set.intersection(*[set(o.index) for o in oofs.values()]))
    y = oofs[arms[0]["name"]].loc[uids, "y"].values.astype(float)
    Z = np.stack([logit(oofs[arm["name"]].loc[uids, "p"].values) for arm in arms])
    best = (float("inf"), None)
    for comb in compositions(a.grid, len(arms)):
        w = np.array(comb, float) / a.grid
        ll = logloss(y, sigmoid((w[:, None] * Z).sum(0)))
        if ll < best[0]:
            best = (ll, w)
    w = best[1]
    z = (w[:, None] * Z).sum(0)
    T_rand = fit_temperature(z, y)

    # ---- nested estimate of the full combiner + clip/hedge selection (weights & T refitted on 4 folds, applied to the 5th) ----
    def fit_wt(Zs, ys):
        bw, bl = None, float("inf")
        for comb in compositions(a.grid, Zs.shape[0]):
            ww = np.array(comb, float) / a.grid
            ll = logloss(ys, sigmoid((ww[:, None] * Zs).sum(0)))
            if ll < bl:
                bl, bw = ll, ww
        zz = (bw[:, None] * Zs).sum(0)
        return bw, max(fit_temperature(zz, ys), a.t_floor)

    def apply(Zs, ww, TT, clip, hedge):
        zz = (ww[:, None] * Zs).sum(0); pp = sigmoid(zz / TT)
        lam_i = np.full(pp.shape, a.lam)
        if hedge is not None and Zs.shape[0] >= 2:
            spread = Zs.max(0) - Zs.min(0); lam_i = np.where(spread > hedge[0], hedge[1], lam_i)
        pp = (1 - lam_i) * pp + lam_i * a.pi
        return np.clip(pp, clip, 1 - clip) if clip > 0 else pp

    chosen_clip, chosen_hedge = 0.0, None
    if a.nested:
        folds = pd.read_csv(Path(a.cache) / "folds.csv", dtype={"uid": str}).set_index("uid").loc[uids, "fold"].values
        spread_all = Z.max(0) - Z.min(0)
        cands = []
        for clip in a.clip:
            for q in a.hedge_q:
                for hl in (a.hedge_lam if q > 0 else [0.0]):
                    hedge = None if q <= 0 else (float(np.quantile(spread_all, q)), hl)
                    pn = np.zeros(len(y))
                    for k in np.unique(folds):
                        tr, va = folds != k, folds == k
                        ww, TT = fit_wt(Z[:, tr], y[tr])
                        pn[va] = apply(Z[:, va], ww, TT, clip, hedge)
                    cands.append((logloss(y, pn), clip, q, hl, hedge))
        cands.sort(key=lambda c: c[0])
        base = next(c for c in cands if c[1] == 0.0 and c[2] == 0.0)
        print(f"nested combiner estimate (no clip, no hedge): {base[0]:.4f}")
        for c in cands[:6]:
            print(f"  nested {c[0]:.4f}  clip={c[1]}  hedge_q={c[2]} hedge_lam={c[3]}")
        bestc = cands[0]
        if bestc[0] < base[0] - 0.002:
            chosen_clip, chosen_hedge = bestc[1], bestc[4]
            print(f"selected clip={chosen_clip} hedge={chosen_hedge} (nested gain {base[0]-bestc[0]:.4f})")
        else:
            print("clip/hedge give no reliable nested gain; keeping plain combiner")
        # per-site report
        sp = Path(a.cache) / "sites.csv"
        if sp.exists():
            sites = pd.read_csv(sp, dtype={"uid": str}).set_index("uid").loc[uids, "site"].values
            pn = np.zeros(len(y))
            for k in np.unique(folds):
                tr, va = folds != k, folds == k
                ww, TT = fit_wt(Z[:, tr], y[tr]); pn[va] = apply(Z[:, va], ww, TT, chosen_clip, chosen_hedge)
            print("per-site nested: site n prev mean_p logloss auroc")
            for st in pd.Series(sites).value_counts().index[:9]:
                m = sites == st
                au = roc_auc_score(y[m], pn[m]) if 0 < y[m].mean() < 1 else float("nan")
                print(f"  {st:<26} {m.sum():4d} {y[m].mean():.2f} {pn[m].mean():.2f} {logloss(y[m], pn[m]):.4f} {au:.3f}")
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
    final = apply(Z, w, T, chosen_clip, chosen_hedge)
    print(f"weights={dict(zip([x['name'] for x in arms], np.round(w, 2)))}  T={T:.3f} (random {T_rand:.3f}, site {T_site}, floor {a.t_floor})  lam={a.lam}  pi={a.pi:.3f}")
    print(f"stack OOF: AUROC={roc_auc_score(y, final):.4f} logloss={logloss(y, final):.4f}  (equal-weight logloss={logloss(y, sigmoid(Z.mean(0))):.4f})")
    man = {"pi": a.pi, "T": float(T), "T_random": float(T_rand), "T_site": (float(T_site) if T_site is not None else None), "lam": a.lam,
           "clip": ([chosen_clip, 1 - chosen_clip] if chosen_clip > 0 else None),
           "hedge": ({"thresh": float(chosen_hedge[0]), "lam": float(chosen_hedge[1])} if chosen_hedge else None), "weights": {x["name"]: float(wi) for x, wi in zip(arms, w)}, "arms": arms,
           "oof_logloss": logloss(y, final), "oof_auroc": float(roc_auc_score(y, final)), "n": int(len(y))}
    (art / "manifest.json").write_text(json.dumps(man, indent=1))
    pd.DataFrame({"uid": uids, "y": y, "p": final}).to_csv(art / "oof_stack.csv", index=False)
    print(f"wrote {art/'manifest.json'}")
