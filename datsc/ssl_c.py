"""Approach C: self-supervised pretraining of the striatal-crop 3D encoder on the TRAINING scans only (labels unused).

VICReg-style objective (invariance + variance + covariance; no negatives, so it is stable at batch 32 on a T4) between two
views of the same scan: two independent draws of our physical augmentations (flip, small affine, blur, count noise,
quantisation, reference-scale jitter) plus a random crop-centre jitter of +-2 voxels around the template striatal centre.
The encoder is the run-3 GroupNorm/SiLU ResNet3D (`features()`), so the pretrained weights load straight into the
supervised crop-3D arm (`scripts/train_v3.py --init`). Test scans are never touched (competition rule: per-sample
inference, no transductive use of the test set).

Why: contrastive / non-contrastive pretraining is the best-established way to get more out of ~1k labelled medical volumes
(Wolf et al., Sci. Rep. 2023: contrastive and masked-autoencoder pretraining for small medical datasets; VoCo, CVPR 2024:
volume contrast exploits the consistent geometry of registered anatomy -- our template frame gives that for free).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .augment import train_augment
from .data_v3 import VolDataset, VolumeCache, to_3d_crop
from .models_v3 import ResNet3D
from .train_v3 import _grad_scaler


class Projector(nn.Module):
    def __init__(self, dim: int, hidden: int = 512, out: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, out))

    def forward(self, h):
        return self.net(h)


def vicreg_loss(z1: torch.Tensor, z2: torch.Tensor, lam: float = 25.0, mu: float = 25.0, nu: float = 1.0):
    """Invariance (MSE) + variance hinge (std >= 1 per dimension) + covariance (off-diagonal -> 0)."""
    inv = F.mse_loss(z1, z2)
    def var_cov(z):
        z = z - z.mean(0)
        std = torch.sqrt(z.var(0) + 1e-4)
        v = torch.relu(1 - std).mean()
        n, d = z.shape
        cov = (z.T @ z) / max(n - 1, 1)
        off = cov - torch.diag(torch.diag(cov))
        return v, (off ** 2).sum() / d
    v1, c1 = var_cov(z1); v2, c2 = var_cov(z2)
    return lam * inv + mu * (v1 + v2) / 2 + nu * (c1 + c2), {"inv": float(inv), "var": float((v1 + v2) / 2), "cov": float(c1 + c2)}


def two_views(x: torch.Tensor, crop=(48, 48, 32), center=(47.5, 57.3, 33.5), jitter: int = 2, aug: str = "base"):
    """Two augmented striatal crops of the same batch of boxes [B,1,96,96,72]."""
    views = []
    for _ in range(2):
        xa = train_augment(x, aug)
        c = tuple(float(ci) + float(np.random.randint(-jitter, jitter + 1)) for ci in center)
        views.append(to_3d_crop(xa, crop, c))
    return views


def pretrain(cache_dir, out_path, epochs: int = 200, batch_size: int = 32, lr: float = 1e-3, weight_decay: float = 1e-4,
             widths=(24, 48, 96, 192), crop=(48, 48, 32), center=(47.5, 57.3, 33.5), jitter: int = 2, aug: str = "base",
             num_workers: int = 2, device=None, max_steps: int | None = None, seed: int = 0, norm: str = "ref", log_every: int = 1) -> dict:
    """VICReg pretraining on every scan of the cache (labels ignored). Saves {'encoder': state_dict, ...} to out_path."""
    cache_dir, out_path = Path(cache_dir), Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu"); amp = device.startswith("cuda")
    torch.manual_seed(seed); np.random.seed(seed)
    cache = VolumeCache(cache_dir)
    uids = list(cache.meta["uid"])
    dl = DataLoader(VolDataset(cache, uids, np.zeros(len(uids)), norm), batch_size=batch_size, shuffle=True, drop_last=True,
                    num_workers=num_workers, pin_memory=amp, persistent_workers=num_workers > 0)
    enc = ResNet3D(widths=tuple(widths)).to(device); proj = Projector(enc.feat_dim).to(device)
    params = list(enc.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    total = epochs * max(1, len(dl)) if max_steps is None else max_steps
    warm = max(1, min(total // 10, 5 * max(1, len(dl))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: (t + 1) / warm if t < warm else 0.5 * (1 + math.cos(math.pi * min(t - warm, total - warm) / max(1, total - warm))))
    scaler = _grad_scaler(amp)
    hist, step, t0 = [], 0, time.time()
    print(f"[ssl_c] scans={len(uids)} steps/epoch={len(dl)} epochs={epochs} encoder params={sum(p.numel() for p in enc.parameters())/1e6:.2f}M device={device}", flush=True)
    for ep in range(epochs):
        enc.train(); proj.train(); tl, n, parts = 0.0, 0, {"inv": 0.0, "var": 0.0, "cov": 0.0}
        for x, _, _, _ in dl:
            x = x.to(device, non_blocking=True)
            with torch.no_grad():
                v1, v2 = two_views(x, crop, center, jitter, aug)
            with torch.autocast(device_type="cuda", enabled=amp):
                z1 = proj(enc.features(v1)); z2 = proj(enc.features(v2))
            loss, p = vicreg_loss(z1.float(), z2.float())
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            tl += float(loss.detach()); n += 1; step += 1
            for k in parts: parts[k] += p[k]
            if max_steps is not None and step >= max_steps:
                break
        rec = {"epoch": ep, "loss": tl / max(n, 1), **{k: v / max(n, 1) for k, v in parts.items()}, "sec": time.time() - t0}
        hist.append(rec)
        if ep % log_every == 0 or ep == epochs - 1:
            print(f"[ssl_c] ep{ep:03d} loss={rec['loss']:.3f} inv={rec['inv']:.3f} var={rec['var']:.3f} cov={rec['cov']:.3f} ({rec['sec']:.0f}s)", flush=True)
        if max_steps is not None and step >= max_steps:
            break
    state = {k: v.detach().cpu() for k, v in enc.state_dict().items()}
    torch.save({"encoder": state, "widths": list(widths), "crop3d": list(crop), "center3d": list(center), "epochs": epochs, "steps": step,
                "history": hist, "aug": aug, "jitter": jitter}, out_path)
    print(f"[ssl_c] saved encoder -> {out_path} after {step} steps in {time.time()-t0:.0f}s", flush=True)
    return {"steps": step, "history": hist, "out": str(out_path)}


@torch.no_grad()
def extract_features(encoder_path, cache_dir, uids, batch_size: int = 32, device=None, norm: str = "ref", num_workers: int = 0) -> np.ndarray:
    """Frozen-encoder features (identity + mirror averaged) of the striatal crop for the given uids -> [N, feat_dim]."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(Path(encoder_path), map_location="cpu", weights_only=False)
    enc = ResNet3D(widths=tuple(ck["widths"])); enc.load_state_dict(ck["encoder"]); enc.to(device).eval()
    cache = VolumeCache(cache_dir)
    dl = DataLoader(VolDataset(cache, list(uids), np.zeros(len(uids)), norm), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    out = []
    for x, _, _, _ in dl:
        x = x.to(device)
        c = to_3d_crop(x, tuple(ck["crop3d"]), tuple(ck["center3d"]))
        h = 0.5 * (enc.features(c) + enc.features(c.flip(dims=[2])))
        out.append(h.float().cpu().numpy())
    return np.concatenate(out)


def linear_probe_oof(features: np.ndarray, y: np.ndarray, folds: np.ndarray, C: float = 0.1) -> np.ndarray:
    """Logistic-regression probe on frozen features, out-of-fold (reporting only; not shipped)."""
    from sklearn.linear_model import LogisticRegression
    p = np.zeros(len(y))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        mu, sd = features[tr].mean(0), features[tr].std(0) + 1e-6
        clf = LogisticRegression(C=C, max_iter=3000).fit((features[tr] - mu) / sd, y[tr])
        p[va] = clf.predict_proba((features[va] - mu) / sd)[:, 1]
    return p
