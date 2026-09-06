"""Run-3 inference: manifest-driven multi-arm ensemble over the whole-brain cache representation.

manifest.json = {"pi": float, "T": float, "lam": float, "weights": {arm_name: w}, "arms": [
    {"name": str, "arm": "3d"|"2d", "spec": InputSpec dict, "model_kw": dict, "norm": str, "checkpoints": [relative paths]}]}
Stages (each overwrites submission.csv): prior -> per-arm averages combined by the fitted weights/T/lam.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutTimeout
from pathlib import Path

import numpy as np
import pandas as pd

from .calibrate import EPS, logit, sigmoid
from .config import N_SCALARS, find_nifti_path
from .register import N_REG_STATS, load_template
from .volume import N_STATS, VOL_SHAPE, prepare_volume, renormalise

PER_VOLUME_TIMEOUT_S = 120
LADDER_STOP_S = 150 * 60           # stop starting new checkpoints after this (3-h limit)
_TEMPLATE = None


def _worker_init(template_path=None):
    global _TEMPLATE
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
        os.environ[k] = "1"
    _TEMPLATE = load_template(template_path)


def _prep(path):
    r = prepare_volume(path, template=_TEMPLATE)
    return r["vol"], r["scalars"], r["stats"]


def write_submission(uids, probs, out_csv):
    pd.DataFrame({"uid": uids, "is_pathologic": np.clip(probs, EPS, 1 - EPS)}).to_csv(out_csv, index=False)


def preprocess_all(uids, nifti_dir: Path, n_workers: int, deadline_s: float, t0: float, template_path=None):
    N = len(uids)
    n_st = N_STATS + (N_REG_STATS if template_path and Path(template_path).exists() else 0)
    vols = np.zeros((N,) + VOL_SHAPE, np.float16); sc = np.zeros((N, N_SCALARS), np.float32)
    st = np.zeros((N, n_st), np.float32); ok = np.zeros(N, bool)
    paths = [find_nifti_path(nifti_dir, u) for u in uids]
    ex = ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init, initargs=(str(template_path) if template_path else None,))
    futs = {ex.submit(_prep, p): i for i, p in enumerate(paths) if p is not None}
    for fut, i in futs.items():
        remaining = deadline_s - (time.time() - t0)
        if remaining <= 0:
            print("ladder: preprocessing deadline hit; cancelling remaining volumes"); break
        try:
            v, s, t = fut.result(timeout=min(PER_VOLUME_TIMEOUT_S, remaining))
            vols[i], sc[i], st[i], ok[i] = v, s, t, True
        except (Exception, FutTimeout):  # noqa: BLE001
            ok[i] = False
    procs = list((getattr(ex, "_processes", None) or {}).values())
    ex.shutdown(wait=False, cancel_futures=True)
    for proc in procs:
        try: proc.terminate()
        except Exception: pass  # noqa: BLE001
    for proc in procs:
        try: proc.join(timeout=5)
        except Exception: pass  # noqa: BLE001
    print(f"preprocessed {int(ok.sum())}/{N} volumes in {time.time()-t0:.0f}s", flush=True)
    return vols, sc, st, ok


def run_lr_arm(arm_cfg: dict, root: Path, vols, st, ok):
    """Registered-ROI striatal features -> logistic regression (arm type "lr"). Needs artifacts/template.npz."""
    from .features_v3 import striatal_features
    from .lrmodel import LRModel
    tpl = load_template(root / "template.npz")
    if tpl is None:
        return None
    m = LRModel.load(root / Path(arm_cfg["checkpoints"][0]).name)
    idx = np.where(ok)[0]
    X = np.stack([striatal_features(renormalise(vols[i], st[i], arm_cfg.get("norm", "ref")), tpl) for i in idx])
    p = np.full(len(ok), np.nan); p[idx] = m.predict_proba(X)
    print(f"arm {arm_cfg['name']}: LR on {len(idx)} volumes", flush=True)
    return p


def run_feat_arm(arm_cfg: dict, root: Path, vols, st, ok):
    """Feature-engineered arm (arm type "feat"): registered-space descriptors + striatal-patch PCA -> logistic, averaged
    over both mirror orientations. Needs artifacts/template.npz and the npz model."""
    from .featmodel import FeatModel
    from .features_v4 import feat_vector, mirror, striatal_patch
    tpl = load_template(root / "template.npz")
    if tpl is None:
        return None
    m = FeatModel.load(root / Path(arm_cfg["checkpoints"][0]).name)
    idx = np.where(ok)[0]
    z = np.zeros(len(idx))
    for j, i in enumerate(idx):
        v = renormalise(vols[i], st[i], arm_cfg.get("norm", "ref")).astype(np.float32)
        zz = [float(m.decision(feat_vector(vv, tpl)[None], striatal_patch(vv)[None])[0]) for vv in (v, mirror(v))]
        z[j] = 0.5 * (zz[0] + zz[1])
    p = np.full(len(ok), np.nan); p[idx] = 1.0 / (1.0 + np.exp(-z))
    print(f"arm {arm_cfg['name']}: feat model on {len(idx)} volumes", flush=True)
    return p


def run_arm(arm_cfg: dict, root: Path, vols, sc, st, ok, t0: float, batch_size: int = 16):
    """Average of all checkpoints of one arm over the ok volumes. Returns probs[N] (nan where not ok) or None."""
    if arm_cfg["arm"] == "lr":
        return run_lr_arm(arm_cfg, root, vols, st, ok)
    if arm_cfg["arm"] == "feat":
        return run_feat_arm(arm_cfg, root, vols, st, ok)
    import torch
    from .models_v3 import build_model
    from .train_v3 import InputSpec
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = InputSpec(**arm_cfg["spec"])
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return None
    norm = arm_cfg.get("norm", "p75")
    x_all = np.stack([renormalise(vols[i], st[i], norm) for i in idx]).astype(np.float32)[:, None]
    x_all = torch.from_numpy(x_all); s_all = torch.from_numpy(sc[idx])
    acc = np.zeros(len(idx)); used = 0
    for ck in arm_cfg["checkpoints"]:
        if time.time() - t0 > LADDER_STOP_S:
            print("ladder: skipping remaining checkpoints"); break
        try:
            # weights_path is a training-time convenience (ImageNet init); the checkpoint holds the whole model and the
            # file does not exist inside the submission container (run 3 v2: every 2D arm failed with FileNotFoundError)
            model_kw = {k: v for k, v in arm_cfg.get("model_kw", {}).items() if k != "weights_path"}
            model = build_model(arm_cfg["arm"], **model_kw).to(device)
            sd = torch.load(str(root / Path(ck).name), map_location=device, weights_only=True)   # root == the artifacts dir
            model.load_state_dict(sd); model.eval()
            out = []
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
                for b in range(0, len(idx), batch_size):
                    x = x_all[b:b + batch_size].to(device); s = s_all[b:b + batch_size].to(device)
                    z = torch.stack([model(inp, s).float() for inp in spec.variants(x)]).mean(0)
                    out.append(torch.sigmoid(z).cpu().numpy())
            acc += np.concatenate(out); used += 1
        except Exception as e:  # noqa: BLE001
            print(f"checkpoint failed ({Path(ck).name}): {type(e).__name__}: {str(e)[:160]}", flush=True)
    print(f"arm {arm_cfg['name']}: checkpoints used {used}/{len(arm_cfg['checkpoints'])} at {time.time()-t0:.0f}s", flush=True)
    if used == 0:
        return None
    p = np.full(len(ok), np.nan); p[idx] = acc / used
    return p


def combine(arm_probs: dict, weights: dict, T: float, lam: float, pi: float, clip=None, hedge=None):
    """Weighted logit average over the arms that produced predictions, temperature, prior mix.
    clip: [lo, hi] final probability bounds (winners' log-loss playbook), default [EPS, 1-EPS].
    hedge: {"thresh": t, "lam": l}: scans whose arm logits disagree by more than t are mixed with the prior at
    strength l instead of lam (uncertainty hedging; only meaningful with >= 2 arms)."""
    names = [n for n in weights if n in arm_probs and arm_probs[n] is not None]
    if not names:
        return None
    w = np.array([weights[n] for n in names], float); w = w / w.sum()
    Z = np.stack([logit(arm_probs[n]) for n in names])          # [A, N] (nan where not ok)
    z = np.nansum(Z * w[:, None], axis=0) / np.nansum(np.where(np.isnan(Z), 0, w[:, None]), axis=0)
    p = sigmoid(z / T)
    lam_i = np.full(p.shape, float(lam))
    if hedge and len(names) >= 2:
        spread = np.nanmax(Z, axis=0) - np.nanmin(Z, axis=0)
        lam_i = np.where(spread > float(hedge["thresh"]), float(hedge["lam"]), lam_i)
    p = (1 - lam_i) * p + lam_i * pi
    lo, hi = (clip if clip else (EPS, 1 - EPS))
    return np.clip(p, max(lo, EPS), min(hi, 1 - EPS))


def load_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text())
