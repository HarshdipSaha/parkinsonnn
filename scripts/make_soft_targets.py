"""Run 8: the soft-target table = Master v2's NESTED out-of-fold predictions (equal weights over the 5 pillars, temperature refit per
fold with floor 0.9, prior mix 0.02) for every training scan -> kaggle/run8_distill/soft_master_v2.csv (uid,y,p,loss,fold). No scan data inside.
python scripts/make_soft_targets.py [--src data/master_pool/artifacts_v3 ...] [--out kaggle/run8_distill/soft_master_v2.csv] [--T-floor 0.9]"""
import argparse, glob, os, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
src = open(Path(__file__).resolve().parent / "fit_stack_v3.py", encoding="utf-8").read().split('if __name__ == "__main__":')[0]
ns = {"__name__": "fs", "__file__": str((Path(__file__).resolve().parent / "fit_stack_v3.py"))}; exec(src, ns)
logloss, sigmoid, logit, fit_temperature = ns["logloss"], ns["sigmoid"], ns["logit"], ns["fit_temperature"]
PILLARS = {"a3d_hemi": "r3d_hemi48_f?_s?", "a3d_ssl": "r3d_ssl_f?_s?", "featc": "featcarm", "a2d_cnx_cor": "r2d_convnext_tiny_cor_f?_s?", "a2d_v2s": "r2d_efficientnet_v2_s_f?_s[01]"}

def load_arm(srcs, pat):
    fs = sorted(set(f for d in srcs for f in glob.glob(str(Path(d) / f"oof_{pat}.csv"))))
    if not fs:
        return None
    df = pd.concat([pd.read_csv(f, dtype={"uid": str}) for f in fs])
    return df.groupby("uid").agg(y=("y", "first"), p=("p", "mean"))

def nested_predictions(srcs, cache="cache_v4", T_floor=0.9, lam=0.02, pi=0.548, arms=PILLARS):
    oofs = {n: load_arm(srcs, p) for n, p in arms.items()}
    missing = [n for n, o in oofs.items() if o is None or len(o) < 1300]
    assert not missing, f"missing/incomplete OOF for {missing}"
    uids = sorted(set.intersection(*[set(o.index) for o in oofs.values()]))
    y = oofs[next(iter(arms))].loc[uids, "y"].values.astype(float)
    Z = np.stack([logit(oofs[n].loc[uids, "p"].values) for n in arms])
    folds = pd.read_csv(Path(cache) / "folds.csv", dtype={"uid": str}).set_index("uid").loc[uids, "fold"].values
    w = np.ones(len(arms)) / len(arms); pn = np.zeros(len(y))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        T = max(fit_temperature((w[:, None] * Z[:, tr]).sum(0), y[tr]), T_floor)
        pn[va] = (1 - lam) * sigmoid((w[:, None] * Z[:, va]).sum(0) / T) + lam * pi
    loss = -(y * np.log(pn) + (1 - y) * np.log(1 - pn))
    return pd.DataFrame({"uid": uids, "y": y, "p": pn, "loss": loss, "fold": folds})

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--src", action="append", default=None); ap.add_argument("--out", default="kaggle/run8_distill/soft_master_v2.csv")
    ap.add_argument("--cache", default="cache_v4"); ap.add_argument("--T-floor", type=float, default=0.9); a = ap.parse_args()
    srcs = a.src or ["data/master_pool/artifacts_v3"]
    df = nested_predictions(srcs, a.cache, a.T_floor); Path(a.out).parent.mkdir(parents=True, exist_ok=True); df.to_csv(a.out, index=False)
    o = df.sort_values("loss", ascending=False)
    print(f"wrote {a.out}: {len(df)} scans; nested log loss {df.loss.mean():.4f}; top-100 scans carry {o.loss.iloc[:100].sum()/df.loss.sum()*100:.1f}% of the loss; "
          f"{int((df.loss > 1).sum())} scans > 1 nat; confident mistakes {int(((df.p > .8) & (df.y == 0)).sum() + ((df.p < .2) & (df.y == 1)).sum())}")
