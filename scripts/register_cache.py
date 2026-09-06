"""cache_v3 (CoM boxes, p75 units) -> cache_v4 (registered to the study template, reference-region units).

Uses exactly datsc.register.register_and_normalise, i.e. the same step main.py applies to a test scan after
prepare_volume's CoM box. Output layout == cache_v3 plus the registration stats appended to stats.npy and meta.csv.

python scripts/register_cache.py --src cache_v3 --out cache_v4 --template artifacts/template.npz [--workers 10]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd

from datsc.register import N_REG_STATS, REG_STAT_NAMES, Template, register_and_normalise
from datsc.volume import VOL_SHAPE

_T = None


def _init(path):
    global _T
    _T = Template.load(path)


def _one(args):
    i, box = args
    try:
        vol, st, _ = register_and_normalise(box, _T)
        return i, vol.astype(np.float16), st, ""
    except Exception as e:  # noqa: BLE001
        return i, box.astype(np.float16), np.zeros(N_REG_STATS, np.float32), repr(e)[:120]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="cache_v3")
    ap.add_argument("--out", default="cache_v4")
    ap.add_argument("--template", default="artifacts/template.npz")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    vols = np.load(src / "vols.npy", mmap_mode="r"); stats = np.load(src / "stats.npy"); meta = pd.read_csv(src / "meta.csv", dtype={"uid": str})
    N = len(meta)
    vout = np.lib.format.open_memmap(out / "vols.npy", mode="w+", dtype=np.float16, shape=(N,) + VOL_SHAPE)
    rstats = np.zeros((N, N_REG_STATS), np.float32); errs = [""] * N
    t0 = time.time()
    with get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.template,)) as pool:
        for k, (i, vol, st, err) in enumerate(pool.imap_unordered(_one, ((i, np.asarray(vols[i], np.float32)) for i in range(N)), chunksize=4)):
            vout[i] = vol; rstats[i] = st; errs[i] = err
            if (k + 1) % 200 == 0 or k + 1 == N:
                print(f"{k+1}/{N} in {time.time()-t0:.0f}s", flush=True)
    vout.flush()
    # stats: cache_v3 may predate the n_levels_native column; pad with nan so the layout matches prepare_volume(template=...)
    if stats.shape[1] == 15:
        stats = np.concatenate([stats, np.full((N, 1), np.nan, np.float32)], axis=1)
    np.save(out / "stats.npy", np.concatenate([stats, rstats], axis=1).astype(np.float32))
    shutil.copy(src / "scalars.npy", out / "scalars.npy")
    for k, name in enumerate(REG_STAT_NAMES):
        meta[name] = rstats[:, k]
    meta["reg_err"] = errs
    meta.to_csv(out / "meta.csv", index=False)
    for f in ("folds.csv", "folds_run2.csv", "folds_site.csv", "sites.csv"):
        if (src / f).exists():
            shutil.copy(src / f, out / f)
    ok = rstats[:, 2] > 0.5
    print(f"registered {int(ok.sum())}/{N} ({100*ok.mean():.1f}%) in {time.time()-t0:.0f}s; fallback {int((~ok).sum())}; errors {sum(1 for e in errs if e)}")
    print(f"ncc_reg p2/10/50/90 = {np.percentile(rstats[:,1],[2,10,50,90]).round(3)} | scale p10/50/90 = {np.percentile(rstats[:,3],[10,50,90]).round(3)} | "
          f"ref_mean p10/50/90 = {np.percentile(rstats[:,6],[10,50,90]).round(3)}")
    lab = meta["is_pathologic"].values
    print(f"fallback rate: normals {100*(~ok[lab==0]).mean():.1f}%  abnormals {100*(~ok[lab==1]).mean():.1f}%")
