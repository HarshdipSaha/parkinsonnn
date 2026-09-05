"""Inference stages for main.py. Stage-wise: prior -> LR -> CNN ensemble, each overwriting submission.csv."""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutTimeout
from pathlib import Path

import numpy as np
import pandas as pd

from .calibrate import EPS
from .config import CROP_SHAPE, N_SCALARS, find_nifti_path
from .features import N_RADIOMICS, hotspot_radiomics
from .lrmodel import LRModel
from .preprocess import preprocess

PER_VOLUME_TIMEOUT_S = 90
LADDER_SKIP_CNN_S = 130 * 60      # past this, stop preprocessing / loading more CNN checkpoints; keep what is written


def _worker_init():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
        os.environ[k] = "1"


def _prep(path):
    r = preprocess(path)
    return r["crop"], r["scalars"], hotspot_radiomics(r["crop"].astype(np.float32))


def write_submission(uids, probs, out_csv):
    pd.DataFrame({"uid": uids, "is_pathologic": np.clip(probs, EPS, 1 - EPS)}).to_csv(out_csv, index=False)


def preprocess_all(uids, nifti_dir: Path, n_workers: int, deadline_s: float, t0: float):
    """Returns (crops[N,X,Y,Z] fp16, scalars[N,8], radiomics[N,14], ok[N] bool)."""
    N = len(uids)
    crops = np.zeros((N,) + CROP_SHAPE, np.float16); sc = np.zeros((N, N_SCALARS), np.float32)
    rad = np.zeros((N, N_RADIOMICS), np.float32); ok = np.zeros(N, bool)
    paths = [find_nifti_path(nifti_dir, u) for u in uids]
    ex = ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init)
    futs = {ex.submit(_prep, p): i for i, p in enumerate(paths) if p is not None}
    for fut, i in futs.items():
        remaining = deadline_s - (time.time() - t0)
        if remaining <= 0:
            print("ladder: preprocessing deadline hit; cancelling remaining volumes")
            break
        try:
            c, s, r = fut.result(timeout=min(PER_VOLUME_TIMEOUT_S, remaining))
            crops[i], sc[i], rad[i], ok[i] = c, s, r, True
        except (Exception, FutTimeout):  # noqa: BLE001
            ok[i] = False
    # never block on a hung worker: drop queued work, then hard-terminate every worker process.
    # (Idle workers would otherwise outlive us holding the stdout pipe, which hangs the caller.)
    procs = list((getattr(ex, "_processes", None) or {}).values())   # snapshot: shutdown() nulls this
    ex.shutdown(wait=False, cancel_futures=True)
    for proc in procs:
        try:
            proc.terminate()
        except Exception:  # noqa: BLE001
            pass
    for proc in procs:
        try:
            proc.join(timeout=5)
        except Exception:  # noqa: BLE001
            pass
    print(f"preprocessed {int(ok.sum())}/{N} volumes in {time.time()-t0:.0f}s")
    return crops, sc, rad, ok


def run_lr(lr_json: Path, rad, sc, ok, pi):
    m = LRModel.load(lr_json)
    p = np.full(len(ok), pi)
    if ok.any():
        p[ok] = m.predict_proba(np.concatenate([rad[ok], sc[ok]], axis=1))
    return p


def run_cnn(weights: list, crops, sc, ok, t0: float, batch_size: int = 32):
    import torch
    from .model import SmallResNet3D
    device = "cuda" if torch.cuda.is_available() else "cpu"
    idx = np.where(ok)[0]
    acc = np.zeros(len(idx)); used = 0
    x_all = torch.from_numpy(crops[idx].astype(np.float32))[:, None]
    s_all = torch.from_numpy(sc[idx])
    for wpath in weights:
        if time.time() - t0 > LADDER_SKIP_CNN_S:
            print("ladder: skipping remaining checkpoints"); break
        model = SmallResNet3D(n_scalars=N_SCALARS).to(device)
        model.load_state_dict(torch.load(wpath, map_location=device)); model.eval()
        out = []
        with torch.no_grad():
            for b in range(0, len(idx), batch_size):
                x = x_all[b:b + batch_size].to(device); s = s_all[b:b + batch_size].to(device)
                z = 0.5 * (model(x, s) + model(x.flip(dims=[2]), s))
                out.append(torch.sigmoid(z).float().cpu().numpy())
        acc += np.concatenate(out); used += 1
    print(f"cnn checkpoints used: {used}/{len(weights)}")
    if used == 0:
        return None
    p = np.full(len(ok), np.nan); p[idx] = acc / used
    return p
