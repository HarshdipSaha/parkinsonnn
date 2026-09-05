"""Build the run-3 whole-brain cache: one memmap-able shard set instead of per-uid npz files.

Outputs in --out:
  vols.npy    float16 [N, 96, 96, 72]   p75-normalised 2 mm boxes (see datsc/volume.py)
  masks.npy   uint8   [N, 96, 96, 72]   head mask (0/1)
  scalars.npy float32 [N, 8]            the run-2 scalar contract (datsc.config.SCALAR_NAMES)
  stats.npy   float32 [N, 15]           raw-intensity + geometry statistics (datsc.volume.STAT_NAMES)
  meta.csv    uid, is_pathologic, ok, err, + stats columns   (row order == array order)

Usage: python scripts/cache_volumes.py [--out cache_v3] [--workers 8] [--limit N]
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

from datsc.config import N_SCALARS, find_data_dir, find_nifti_path
from datsc.volume import N_STATS, STAT_NAMES, VOL_SHAPE, prepare_volume


def _one(args):
    i, uid, path = args
    try:
        r = prepare_volume(path)
        return i, uid, r, ""
    except Exception as e:  # noqa: BLE001
        return i, uid, None, repr(e)[:200]


def build(labels_csv: Path, niftis_dir: Path, out: Path, n_workers: int, limit: int | None):
    out.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(labels_csv, dtype={"uid": str})
    if limit:
        labels = labels.head(limit)
    jobs = []
    for uid in labels["uid"]:
        p = find_nifti_path(niftis_dir, uid)
        if p is None:
            print(f"missing file for uid (skipping): {uid}")
            continue
        jobs.append((len(jobs), uid, p))
    N = len(jobs)
    vols = np.lib.format.open_memmap(out / "vols.npy", mode="w+", dtype=np.float16, shape=(N,) + VOL_SHAPE)
    masks = np.lib.format.open_memmap(out / "masks.npy", mode="w+", dtype=np.uint8, shape=(N,) + VOL_SHAPE)
    scalars = np.zeros((N, N_SCALARS), np.float32)
    stats = np.zeros((N, N_STATS), np.float32)
    ok = np.zeros(N, bool); errs = [""] * N; uids = [j[1] for j in jobs]
    t0 = time.time(); done = 0
    it = map(_one, jobs) if n_workers <= 1 else get_context("spawn").Pool(n_workers).imap_unordered(_one, jobs, chunksize=2)
    for i, uid, r, err in it:
        done += 1
        if r is not None:
            vols[i] = r["vol"]; masks[i] = r["mask"].astype(np.uint8)
            scalars[i] = r["scalars"]; stats[i] = r["stats"]; ok[i] = True
        else:
            errs[i] = err
        if done % 100 == 0 or done == N:
            print(f"{done}/{N} in {time.time()-t0:.0f}s", flush=True)
    vols.flush(); masks.flush()
    np.save(out / "scalars.npy", scalars); np.save(out / "stats.npy", stats)
    meta = pd.DataFrame({"uid": uids, "ok": ok, "err": errs})
    for k, name in enumerate(STAT_NAMES):
        meta[name] = stats[:, k]
    meta = meta.merge(labels, on="uid", how="left")
    meta.to_csv(out / "meta.csv", index=False)
    print(f"cached {int(ok.sum())}/{N} volumes in {time.time()-t0:.0f}s; failures: {int((~ok).sum())}")
    if (~ok).any():
        print(meta.loc[~ok, ["uid", "err"]].head(20).to_string(index=False))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cache_v3")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    d = find_data_dir()
    print(f"data: {d.root}\nlabels: {d.labels_csv}\nniftis: {d.niftis_dir}")
    build(d.labels_csv, d.niftis_dir, Path(a.out), a.workers, a.limit)
