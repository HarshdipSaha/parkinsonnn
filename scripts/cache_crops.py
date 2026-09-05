"""Preprocess every training volume once -> cache/{uid}.npz + cache/meta.csv.

Usage: python scripts/cache_crops.py [--out cache] [--workers 10] [--limit N]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from datsc.config import find_data_dir, find_nifti_path
from datsc.features import hotspot_radiomics
from datsc.preprocess import preprocess


def _one(args):
    uid, path, out_dir = args
    try:
        r = preprocess(path)
        rad = hotspot_radiomics(r["crop"].astype(np.float32))
        np.savez_compressed(out_dir / f"{uid}.npz", crop=r["crop"], crop_com=r["crop_com"],
                            scalars=r["scalars"], radiomics=rad, anchor=r["anchor"])
        return {"uid": uid, "ok": True, "fallback": r["fallback"], "p75": r["p75"],
                "native_spacing_mm": float(r["scalars"][5]), "z_extent_mm": float(r["scalars"][4]),
                "n_levels": float(np.exp(r["scalars"][6])), "err": ""}
    except Exception as e:  # noqa: BLE001
        return {"uid": uid, "ok": False, "fallback": True, "p75": np.nan, "native_spacing_mm": np.nan,
                "z_extent_mm": np.nan, "n_levels": np.nan, "err": repr(e)[:200]}


def build_cache(labels_csv: Path, niftis_dir: Path, out_dir: Path, n_workers: int = 8, limit: int | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(labels_csv, dtype={"uid": str})
    if limit:
        labels = labels.head(limit)
    jobs = []
    for uid in labels["uid"]:
        p = find_nifti_path(niftis_dir, uid)
        if p is None:
            print(f"missing file for uid (skipping): {uid}")
            continue
        jobs.append((uid, p, out_dir))
    t0 = time.time()
    if n_workers <= 1:
        results = [_one(j) for j in jobs]
    else:
        with get_context("spawn").Pool(n_workers) as pool:
            results = list(pool.imap_unordered(_one, jobs, chunksize=4))
    meta = pd.DataFrame(results).merge(labels, on="uid", how="left")
    meta.to_csv(out_dir / "meta.csv", index=False)
    print(f"cached {meta['ok'].sum()}/{len(meta)} in {time.time()-t0:.0f}s; "
          f"fallback anchors: {int(meta['fallback'].sum())}; failures: {int((~meta['ok']).sum())}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cache")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    d = find_data_dir()
    print(f"data: {d.root}\nlabels: {d.labels_csv}\nniftis: {d.niftis_dir}")
    build_cache(d.labels_csv, d.niftis_dir, Path(a.out), a.workers, a.limit)
