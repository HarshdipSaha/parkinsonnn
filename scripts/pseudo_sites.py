"""Pseudo-site labels from acquisition fingerprints (native spacing x native shape x intensity regime),
a leave-pseudo-site-out fold assignment for shift-aware validation/calibration, and reuse of the run-2
random folds so run-3 OOF predictions stay comparable/stackable with run-2 ones.

python scripts/pseudo_sites.py --cache cache_v3 [--run2-folds cache/folds.csv]
writes cache_v3/sites.csv (uid, site, site_id), cache_v3/folds.csv (copied run-2 folds), cache_v3/folds_site.csv
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd


def site_key(row) -> str:
    sp = f"{row.sp_x:.2f}"
    shape = f"{int(row.shape_x)}x{int(row.shape_y)}x{int(row.shape_z)}"
    regime = "ultralow" if row.p99 < 20 else ("low" if row.p99 < 2000 else "high")
    return f"{sp}|{shape}|{regime}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache_v3")
    ap.add_argument("--run2-folds", default="cache/folds.csv")
    ap.add_argument("--min-site", type=int, default=25, help="smaller fingerprints are merged into 'other'")
    ap.add_argument("--n-site-folds", type=int, default=5)
    a = ap.parse_args()
    cache = Path(a.cache)
    meta = pd.read_csv(cache / "meta.csv", dtype={"uid": str})
    meta["site"] = meta.apply(site_key, axis=1)
    counts = meta["site"].value_counts()
    small = counts[counts < a.min_site].index
    meta.loc[meta["site"].isin(small), "site"] = "other"
    tab = meta.groupby("site").agg(n=("uid", "size"), abnormal=("is_pathologic", "mean")).sort_values("n", ascending=False)
    print(tab.to_string())
    sites = {s: i for i, s in enumerate(tab.index)}
    meta["site_id"] = meta["site"].map(sites)
    meta[["uid", "site", "site_id"]].to_csv(cache / "sites.csv", index=False)

    # leave-pseudo-site-out folds: greedy balance sites into n folds by size ('other' spread by hashing)
    order = tab.drop(index="other", errors="ignore").sort_values("n", ascending=False).index.tolist()
    load = np.zeros(a.n_site_folds); site_fold = {}
    for s in order:
        k = int(np.argmin(load)); site_fold[s] = k; load[k] += tab.loc[s, "n"]
    fold = meta["site"].map(site_fold)
    rng = np.random.RandomState(0)
    other = meta["site"] == "other"
    fold[other] = rng.randint(0, a.n_site_folds, other.sum())
    fs = meta[["uid", "is_pathologic"]].copy(); fs["fold"] = fold.astype(int).values
    fs.to_csv(cache / "folds_site.csv", index=False)
    print("\nsite folds:\n", fs.groupby("fold").agg(n=("uid", "size"), abnormal=("is_pathologic", "mean")).to_string())

    r2 = Path(a.run2_folds)
    if r2.exists():
        f2 = pd.read_csv(r2, dtype={"uid": str})
        assert set(f2["uid"]) == set(meta["uid"]), "run-2 folds do not cover the same uids"
        shutil.copy(r2, cache / "folds.csv")
        print(f"\ncopied run-2 folds {r2} -> {cache/'folds.csv'}")
    else:
        print(f"\nWARNING: {r2} not found; build folds with scripts/make_folds.py")
