"""Build the label-blind study template from cached CoM boxes (cache_v3), iteratively:
  T0 = mean of all CoM boxes -> register every box to T_k -> T_{k+1} = mean of registered boxes  (x n_iter)
Then derive masks (brain / striatal / reference), set the NCC QC threshold from the training distribution and save
artifacts/template.npz. Also writes <cache>/template_ncc.csv for QC.

python scripts/build_template.py --cache cache_v3 --out artifacts/template.npz [--n-iter 2] [--subset 400] [--workers 8]
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

from datsc.register import Template, derive_masks, ncc, prep_for_registration, register
from datsc.volume import VOL_SHAPE

_T = None


def _init(template_path):
    global _T
    _T = np.load(template_path)


def _reg_one(args):
    i, box = args
    try:
        reg, p = register(box, _T, affine_refine=False)
        return i, reg.astype(np.float16), p["scale"], p["rot_deg"], p["trans_mm"], ""
    except Exception as e:  # noqa: BLE001
        return i, box.astype(np.float16), 1.0, 0.0, 0.0, repr(e)[:100]


def register_all(vols, idx, template, workers, tmp):
    np.save(tmp, template.astype(np.float32))
    out = np.zeros((len(idx),) + VOL_SHAPE, np.float16); stats = np.zeros((len(idx), 3), np.float32); errs = 0
    t0 = time.time()
    with get_context("spawn").Pool(workers, initializer=_init, initargs=(tmp,)) as pool:
        for k, (i, reg, s, r, t, err) in enumerate(pool.imap_unordered(_reg_one, ((j, np.asarray(vols[i], np.float32)) for j, i in enumerate(idx)), chunksize=4)):
            out[i] = reg; stats[i] = (s, r, t); errs += bool(err)
            if (k + 1) % 200 == 0:
                print(f"  {k+1}/{len(idx)} registered in {time.time()-t0:.0f}s", flush=True)
    print(f"  registered {len(idx)} boxes in {time.time()-t0:.0f}s; errors {errs}; scale p10/50/90 = "
          f"{np.percentile(stats[:,0],[10,50,90]).round(3)} rot p50/90 = {np.percentile(stats[:,1],[50,90]).round(1)} deg", flush=True)
    return out, stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache_v3")
    ap.add_argument("--out", default="artifacts/template.npz")
    ap.add_argument("--n-iter", type=int, default=2)
    ap.add_argument("--subset", type=int, default=400, help="boxes used for template building (label-blind random subset)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cache = Path(a.cache)
    vols = np.load(cache / "vols.npy", mmap_mode="r")
    meta = pd.read_csv(cache / "meta.csv", dtype={"uid": str})
    rng = np.random.RandomState(a.seed)
    idx = np.sort(rng.choice(np.where(meta.ok.values)[0], size=min(a.subset, int(meta.ok.sum())), replace=False))
    print(f"template subset: {len(idx)} boxes (abnormal fraction {meta.is_pathologic.values[idx].mean():.3f})")
    T = np.mean([np.asarray(vols[i], np.float32) for i in idx], axis=0)
    tmp = str(cache / "_template_tmp.npy")
    for it in range(a.n_iter):
        print(f"iteration {it+1}/{a.n_iter}")
        reg, st = register_all(vols, idx, T, a.workers, tmp)
        T = reg.astype(np.float32).mean(axis=0)     # boxes are already CoM-centred; no re-centring (it drifted onto the striatum)
    from scipy import ndimage as ndi
    from datsc.register import symmetrize
    T, x_mid, delta = symmetrize(T)
    print(f"symmetrised: mirror midline found at x={x_mid:.1f} vox, rolled by {delta:+d} so the midline is the box centre")
    brain, striatal, ref, s_center = derive_masks(T)
    assert brain.sum() * 8 / 1000 > 1000 and striatal.sum() * 8 / 1000 > 3, (brain.sum(), striatal.sum())
    xs = np.arange(T.shape[0])[:, None, None]
    print(f"striatal voxels left/right of the midline: {(striatal & (xs < s_center[0])).sum()} / {(striatal & (xs > s_center[0])).sum()}")
    print(f"template masks: brain {brain.sum()*8/1000:.0f} mL, striatal {striatal.sum()*8/1000:.1f} mL, reference {ref.sum()*8/1000:.0f} mL, "
          f"striatal centre (vox) {np.round(s_center, 1)}")
    # QC threshold: register the subset once more to the final template and take the 2nd percentile of NCC
    reg, st = register_all(vols, idx, T, a.workers, tmp)
    qc = ~ndi.binary_erosion(~brain, iterations=1) & ndi.binary_erosion(brain, iterations=2)
    tp = prep_for_registration(T)
    nccs = np.array([ncc(prep_for_registration(np.asarray(reg[k], np.float32)), tp, qc) for k in range(len(idx))])
    ncc0 = np.array([ncc(prep_for_registration(np.asarray(vols[i], np.float32)), tp, qc) for i in idx])
    thresh = float(np.percentile(nccs, 2))
    print(f"NCC to template: unregistered p10/50/90 = {np.percentile(ncc0,[10,50,90]).round(3)}; registered p2/10/50/90 = "
          f"{np.percentile(nccs,[2,10,50,90]).round(3)} -> ncc_thresh {thresh:.3f}")
    Template(T, brain, striatal, ref, s_center, thresh).save(a.out)
    pd.DataFrame({"uid": meta.uid.values[idx], "ncc_init": ncc0, "ncc_reg": nccs, "scale": st[:, 0], "rot_deg": st[:, 1], "trans_mm": st[:, 2]}).to_csv(cache / "template_ncc.csv", index=False)
    os.remove(tmp)
    print(f"wrote {a.out}")
