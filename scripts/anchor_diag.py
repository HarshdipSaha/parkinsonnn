"""Why does find_anchor fall back? Per-scan hotspot component stats on a random sample.

python scripts/anchor_diag.py --n 400 --workers 10
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from datsc import config as C
from datsc.config import find_data_dir, find_nifti_path
from datsc.io import load_iso
from datsc.preprocess import _largest_component, brain_mask, normalize_p75


def _one(args):
    uid, path, y = args
    try:
        raw, sp = load_iso(path)
        mask = brain_mask(raw)
        norm, p75 = normalize_p75(raw, mask)
        s = C.SPACING_MM
        brain_c = np.array(ndi.center_of_mass(mask))
        sm = ndi.gaussian_filter(norm * mask, sigma=C.SMOOTH_MM / s)
        peak = float(sm.max())
        res = {"uid": uid, "y": y, "spacing": float(np.mean(sp)), "peak": peak, "shape_z": raw.shape[2]}
        for thr in (0.5, 0.6, 0.7, 0.8):
            comp = _largest_component(sm > thr * peak)
            ml = comp.sum() * s ** 3 / 1000
            hc = np.array(ndi.center_of_mass(sm, labels=comp.astype(int), index=1))
            d = float(np.linalg.norm((hc - brain_c) * s))
            res[f"ml@{thr}"] = ml; res[f"dist@{thr}"] = d
            res[f"dz@{thr}"] = float((hc[2] - brain_c[2]) * s)
        return res
    except Exception as e:  # noqa: BLE001
        return {"uid": uid, "y": y, "err": repr(e)[:120]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    d = find_data_dir()
    labels = pd.read_csv(d.labels_csv, dtype={"uid": str}).sample(n=a.n, random_state=0)
    jobs = [(u, find_nifti_path(d.niftis_dir, u), y) for u, y in zip(labels["uid"], labels["is_pathologic"])]
    jobs = [j for j in jobs if j[1] is not None]
    with get_context("spawn").Pool(a.workers) as pool:
        rows = list(pool.imap_unordered(_one, jobs, chunksize=4))
    df = pd.DataFrame(rows)
    print("errors:", df["err"].notna().sum() if "err" in df else 0)
    df = df[df.get("err").isna()] if "err" in df else df
    pd.set_option("display.width", 200)
    for thr in (0.5, 0.6, 0.7, 0.8):
        ml, dist = df[f"ml@{thr}"], df[f"dist@{thr}"]
        rej = (ml < C.ANCHOR_MIN_ML) | (ml > C.ANCHOR_MAX_ML) | (dist > C.ANCHOR_MAX_DIST_MM)
        print(f"\nthr={thr}: ml p10/50/90 = {ml.quantile(.1):.1f}/{ml.median():.1f}/{ml.quantile(.9):.1f} mL | "
              f"dist p50/90 = {dist.median():.1f}/{dist.quantile(.9):.1f} mm | dz p50 = {df[f'dz@{thr}'].median():+.1f} mm | "
              f"reject(current rule)={rej.mean():.1%}  (too big {(ml > C.ANCHOR_MAX_ML).mean():.1%}, too small {(ml < C.ANCHOR_MIN_ML).mean():.1%}, too far {(dist > C.ANCHOR_MAX_DIST_MM).mean():.1%})")
        print("   by label:", df.assign(rej=rej).groupby("y")["rej"].mean().round(3).to_dict(),
              "| by spacing:", df.assign(rej=rej, sp=df.spacing.round(1)).groupby("sp")["rej"].mean().round(2).to_dict())
    df.to_csv("cache_anchor_diag.csv", index=False)
