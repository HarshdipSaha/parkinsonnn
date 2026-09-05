"""cache/meta.csv + cache/*.npz -> cache/folds.csv (uid, is_pathologic, group, fold)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd

from datsc.folds import assign_folds, embed, group_near_duplicates

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--thresh", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cache = Path(a.cache)
    meta = pd.read_csv(cache / "meta.csv", dtype={"uid": str})
    meta = meta[meta["ok"]].reset_index(drop=True)
    crops = np.stack([np.load(cache / f"{u}.npz")["crop"] for u in meta["uid"]])
    groups = group_near_duplicates(embed(crops), a.thresh)
    n_dup = len(groups) - len(np.unique(groups))
    print(f"near-duplicate pairs collapsed into groups: {n_dup}")
    meta["group"] = groups
    meta["fold"] = assign_folds(meta["is_pathologic"].values.astype(int), groups, 5, a.seed)
    meta[["uid", "is_pathologic", "group", "fold", "native_spacing_mm", "fallback"]].to_csv(cache / "folds.csv", index=False)
    print(meta.groupby("fold")["is_pathologic"].agg(["count", "mean"]))
