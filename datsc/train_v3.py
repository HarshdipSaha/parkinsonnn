"""Run-3 trainer: one (arm, fold, seed) run from the memmap cache.

Protocol (fixes the run-2 optimism): fixed epoch budget, EMA weights, OOF and checkpoint from the FINAL EMA state
(no best-epoch selection on the validation fold), linear warmup + cosine, AdamW with real weight decay, optional
mixup, torch.amp API, loss.detach(). Model input (3D resize or axial slabs) is derived on the GPU AFTER the shared
3D augmentation so both arms see identical physical perturbations. AUROC is logged next to log loss every epoch.
"""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .augment import train_augment
from .data_v3 import load_site_ids, VolDataset, VolumeCache, to_3d_input, to_slabs, to_3d_crop
from .models_v3 import SiteHead, build_model, count_params


def logloss_np(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auroc_np(y, p):
    return float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan")


class InputSpec:
    """How a [B,1,96,96,72] box becomes model input, and which test-time variants are averaged.
    tta_z: slab-centre offsets in voxels for the 2D arm (e.g. [-1, 0, 1]); every variant is also L/R-flipped."""

    def __init__(self, arm: str, size=(64, 64, 48), n_slabs: int = 3, slab_mm: float = 12.0, hw: int = 192,
                 slab_mode: str = "mean", z_center: float | None = None, plane: str = "axial", tta_z=(0,), crop=None,
                 crop3d=None, center3d=None, tta_affine: int = 0):
        self.arm, self.size, self.n_slabs, self.slab_mm, self.hw, self.slab_mode, self.z_center = \
            arm, tuple(size), n_slabs, slab_mm, hw, slab_mode, z_center
        self.plane, self.tta_z, self.crop = plane, tuple(tta_z), crop
        self.crop3d = tuple(int(v) for v in crop3d) if crop3d else None        # 3D: striatal crop before the resize
        self.center3d = tuple(float(v) for v in center3d) if center3d else None
        self.tta_affine = int(tta_affine or 0)          # K reproducible small affine perturbations of the box at test time

    def __call__(self, x, dz: int = 0):
        if self.arm == "3d":
            if self.crop3d:
                x = to_3d_crop(x, self.crop3d, self.center3d)
            return to_3d_input(x, self.size)
        zc = None if self.z_center is None else self.z_center + dz
        return to_slabs(x, self.n_slabs, self.slab_mm, self.hw, mode=self.slab_mode, z_center=zc, plane=self.plane, crop=self.crop)

    def _affine_boxes(self, x):
        """K reproducible small affine perturbations of the registered box (rotation <= 3 deg, scale <= 2 %, shift <= 1 voxel):
        the same K boxes every call (fixed generator seed), so OOF and inference average identical variants."""
        g = torch.Generator(device="cpu").manual_seed(1234)
        for _ in range(self.tta_affine):
            ang = torch.deg2rad((torch.rand(3, generator=g) * 2 - 1) * 3.0)
            sc = 1 + (torch.rand(3, generator=g) * 2 - 1) * 0.02
            sh = (torch.rand(3, generator=g) * 2 - 1) * 2.0 / torch.tensor(x.shape[2:], dtype=torch.float32)
            cx, cy, cz = torch.cos(ang); sx, sy, sz = torch.sin(ang)
            Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]); Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]); Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            theta = torch.zeros(x.shape[0], 3, 4); theta[:, :, :3] = (Rz @ Ry @ Rx) @ torch.diag(sc); theta[:, :, 3] = sh
            grid = F.affine_grid(theta.to(x.device, x.dtype), list(x.shape), align_corners=False)
            yield F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    def variants(self, x):
        """Yield every TTA input ((identity + affine boxes) x z-shift x flip). Training uses only variant 0."""
        boxes = [x] + (list(self._affine_boxes(x)) if self.tta_affine else [])
        for b in boxes:
            for dz in (self.tta_z if self.arm == "2d" else (0,)):
                inp = self(b, dz)
                yield inp
                yield self.flip(inp)

    def flip(self, inp):
        """Left-right mirror of a model input (label-preserving)."""
        return inp.flip(dims=[2]) if self.arm == "3d" else inp.flip(dims=[3])

    def to_dict(self):
        return dict(arm=self.arm, size=list(self.size), n_slabs=self.n_slabs, slab_mm=self.slab_mm, hw=self.hw,
                    slab_mode=self.slab_mode, z_center=self.z_center, plane=self.plane, tta_z=list(self.tta_z), crop=self.crop,
                    crop3d=(list(self.crop3d) if self.crop3d else None), center3d=(list(self.center3d) if self.center3d else None),
                    tta_affine=self.tta_affine)


def _seed_all(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, spec: InputSpec, device, tta_flip=True):
    """tta_flip=False: single plain pass (per-epoch monitoring). tta_flip=True: logit-average over spec.variants()."""
    model.eval(); ps, uids = [], []
    for x, s, _, u in loader:
        x, s = x.to(device, non_blocking=True), s.to(device)
        with torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            if tta_flip:
                zs = [torch.nan_to_num(model(inp, s).float(), nan=0.0, posinf=30.0, neginf=-30.0) for inp in spec.variants(x)]
                z = torch.stack(zs).mean(0)
            else:
                z = torch.nan_to_num(model(spec(x), s).float(), nan=0.0, posinf=30.0, neginf=-30.0)
        ps.append(torch.sigmoid(z).cpu().numpy()); uids += list(u)
    return np.concatenate(ps), uids


class EMA:
    """Exponential moving average of the weights with a step-adaptive decay min(decay, (1+t)/(10+t)).
    Keep the window SHORT (2-3 epochs: 0.99 for ~35 steps/epoch, 0.995 for ~70): a 0.998 window of ~15 epochs
    averaged weights from when the model was still poor and produced compressed, under-confident logits
    (measured: val log loss 0.58 at AUROC 0.87 on the 2.5D arm)."""

    def __init__(self, model, decay=0.995):
        self.m = copy.deepcopy(model).eval(); self.decay = decay; self.t = 0
        for p in self.m.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.t += 1
        d = min(self.decay, (1 + self.t) / (10 + self.t))
        msd = model.state_dict()
        for k, v in self.m.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


def _grad_scaler(amp: bool):
    if hasattr(torch.amp, "GradScaler"):                 # torch >= 2.3 (Kaggle / runtime)
        return torch.amp.GradScaler("cuda", enabled=amp)
    return torch.cuda.amp.GradScaler(enabled=amp)        # local torch 2.2 fallback


def train_run(cache_dir, fold: int, seed: int, out_dir, spec: InputSpec, model_kw: dict, tag: str,
              epochs=60, batch_size=16, lr=5e-4, backbone_lr=None, weight_decay=1e-2, warmup_epochs=3,
              mixup=0.0, ema_decay=0.995, norm="ref", folds_csv="folds.csv", num_workers=2, device=None,
              augment=True, save_fp16=True, select: str = "final", aug: str = "base", z_jitter: int = 0,
              dann: float = 0.0, dro: float = 0.0) -> dict:
    cache_dir, out_dir = Path(cache_dir), Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu"); amp = device.startswith("cuda")
    torch.backends.cudnn.benchmark = True
    _seed_all(seed)
    cache = VolumeCache(cache_dir)
    f = pd.read_csv(cache_dir / folds_csv, dtype={"uid": str})
    f = f[f["uid"].isin(cache.index)]
    full = fold < 0                                        # fold -1: train on everything, no validation split
    tr, va = (f, f.iloc[:0]) if full else (f[f["fold"] != fold], f[f["fold"] == fold])
    tpl = None
    if norm == "hist":
        from .register import load_template
        tpl = load_template(next((p for p in (Path("artifacts/template.npz"), cache_dir / "template.npz", Path("artifacts_v3/template.npz")) if p.exists()), None))
        assert tpl is not None and getattr(tpl, "hist", None) is not None, "norm=hist needs artifacts/template.npz with landmarks (scripts/template_landmarks.py)"
    dl_tr = DataLoader(VolDataset(cache, tr["uid"], tr["is_pathologic"], norm, tpl), batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, drop_last=True, pin_memory=amp, persistent_workers=num_workers > 0)
    dl_va = None if full else DataLoader(VolDataset(cache, va["uid"], va["is_pathologic"], norm, tpl), batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers, persistent_workers=num_workers > 0)
    model = build_model(spec.arm, **model_kw).to(device)
    if spec.arm == "2d" and backbone_lr is not None:
        bb = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
        rest = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
        opt = torch.optim.AdamW([{"params": bb, "lr": backbone_lr}, {"params": rest, "lr": lr}], weight_decay=weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # site-invariant training: pseudo-site ids of the training uids; DANN head (gradient reversal) and/or group-DRO weights
    site_map, n_sites = load_site_ids(cache_dir, tr["uid"], min_n=25)
    site_head = SiteHead(model.feat_dim, n_sites).to(device) if (dann > 0 and n_sites > 1) else None
    if site_head is not None:
        opt.add_param_group({"params": list(site_head.parameters()), "lr": lr, "weight_decay": weight_decay})
    q_dro = torch.ones(n_sites, device=device) / n_sites
    steps_per_epoch = max(1, len(dl_tr)); total = epochs * steps_per_epoch; warm = max(1, warmup_epochs * steps_per_epoch)

    def lr_lambda(t):
        if t < warm:
            return (t + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min(t - warm, total - warm) / max(1, total - warm)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = _grad_scaler(amp)
    ema = EMA(model, ema_decay) if ema_decay else None
    yv_lookup = va.set_index("uid")["is_pathologic"]
    best, best_state, hist, t0 = float("inf"), None, [], time.time()
    print(f"[{tag}] params={count_params(model)/1e6:.2f}M train={len(tr)} val={len(va)} norm={norm} spec={spec.to_dict()} "
          f"model_kw={ {k: (str(v) if isinstance(v, Path) else v) for k, v in model_kw.items()} }", flush=True)
    n_skipped = 0; step = 0
    for ep in range(epochs):
        model.train(); tl, n = 0.0, 0
        for x, s, y, u in dl_tr:
            x, s, y = x.to(device, non_blocking=True), s.to(device), y.to(device)
            site = torch.tensor([site_map.get(uid, 0) for uid in u], device=device)
            with torch.no_grad():
                if augment:
                    x = train_augment(x, aug)
                dz = int(np.random.randint(-z_jitter, z_jitter + 1)) if (z_jitter and spec.arm in ("2d", "tri")) else 0
                inp = spec(x, dz)                                     # random slab-centre shift == the TTA offsets
                if mixup > 0:
                    lam = float(np.random.beta(mixup, mixup)); perm = torch.randperm(inp.shape[0], device=device)
                    inp = lam * inp + (1 - lam) * inp[perm]; s = lam * s + (1 - lam) * s[perm]; y = lam * y + (1 - lam) * y[perm]
            with torch.autocast(device_type="cuda", enabled=amp):
                if site_head is None and not (dro > 0 and n_sites > 1):
                    loss = F.binary_cross_entropy_with_logits(model(inp, s).float(), y)
                else:
                    h = model.features(inp)
                    logit = model.head(model.scalars(h, s)).squeeze(1).float()
                    per = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
                    if dro > 0 and n_sites > 1:                       # group-DRO: exponentially up-weight the worst pseudo-sites
                        Lg = torch.zeros(n_sites, device=device).index_add_(0, site, per)
                        cnt = torch.zeros(n_sites, device=device).index_add_(0, site, torch.ones_like(per))
                        Lg = Lg / cnt.clamp_min(1); present = (cnt > 0).float()
                        with torch.no_grad():
                            q_dro = q_dro * torch.exp(dro * Lg.detach() * present); q_dro = q_dro / q_dro.sum()
                        loss = (q_dro * Lg * present).sum() / (q_dro * present).sum().clamp_min(1e-6)
                    else:
                        loss = per.mean()
                    if site_head is not None:                          # DANN: reversed-gradient site classification, ramped over the first third
                        lam_dann = dann * min(1.0, step / max(1, total // 3))
                        loss = loss + F.cross_entropy(site_head(h.float(), lam_dann), site)
            step += 1
            if not torch.isfinite(loss):                      # diverged batch (fp16 overflow): skip it instead of poisoning the weights
                n_skipped += 1; opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            if ema is not None:
                ema.update(model)
            tl += float(loss.detach()) * len(y); n += len(y)
        eval_model = ema.m if ema is not None else model
        if full:
            hist.append({"epoch": ep, "train_loss": tl / max(n, 1), "sec": time.time() - t0})
            print(f"[{tag}] ep{ep:02d} train={tl/max(n,1):.4f} ({time.time()-t0:.0f}s)", flush=True)
            continue
        p, uids = predict(eval_model, dl_va, spec, device, tta_flip=False)
        yv = yv_lookup.loc[uids].values.astype(float)
        vl, va_auc = logloss_np(yv, p), auroc_np(yv, p)
        raw = ""
        rec = {"epoch": ep, "train_loss": tl / max(n, 1), "val_logloss": vl, "val_auroc": va_auc, "sec": time.time() - t0}
        if ema is not None:                                   # raw-model metrics next to the EMA ones (lag / collapse detector)
            p_raw, u_raw = predict(model, dl_va, spec, device, tta_flip=False)
            y_raw = yv_lookup.loc[u_raw].values.astype(float)
            rec["val_raw_logloss"], rec["val_raw_auroc"] = logloss_np(y_raw, p_raw), auroc_np(y_raw, p_raw)
            raw = f" raw={rec['val_raw_logloss']:.4f}/{rec['val_raw_auroc']:.4f}"
        hist.append(rec)
        print(f"[{tag}] ep{ep:02d} train={tl/max(n,1):.4f} val_logloss={vl:.4f} val_auroc={va_auc:.4f}{raw} ({time.time()-t0:.0f}s)", flush=True)
        if vl < best:
            best = vl
            if select == "best":
                best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
    eval_model = ema.m if ema is not None else model
    if select == "best" and best_state is not None:
        eval_model.load_state_dict(best_state)
    state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
    if full:
        p, yv = np.zeros(0), np.zeros(0)
    else:
        p, uids = predict(eval_model, dl_va, spec, device, tta_flip=True)
        yv = yv_lookup.loc[uids].values.astype(float)
        pd.DataFrame({"uid": uids, "y": yv, "p": p}).to_csv(out_dir / f"oof_{tag}.csv", index=False)
    if save_fp16:
        state = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in state.items()}
    torch.save(state, out_dir / f"{tag}.pt")
    meta = {"tag": tag, "fold": fold, "seed": seed, "epochs": epochs, "select": select, "best_val_logloss": best,
            "val_logloss": (logloss_np(yv, p) if len(yv) else None), "val_auroc": (auroc_np(yv, p) if len(yv) else None),
            "spec": spec.to_dict(), "model_kw": {k: (str(v) if isinstance(v, Path) else v) for k, v in model_kw.items()},
            "norm": norm, "folds_csv": folds_csv, "mixup": mixup, "ema_decay": ema_decay, "lr": lr, "backbone_lr": backbone_lr,
            "aug": aug, "z_jitter": z_jitter, "skipped_batches": n_skipped, "dann": dann, "dro": dro, "n_sites": n_sites,
            "weight_decay": weight_decay, "history": hist, "params": count_params(model), "sec": time.time() - t0}
    (out_dir / f"{tag}.json").write_text(json.dumps(meta))
    if full:
        print(f"[{tag}] done (full-data model, no validation) in {time.time()-t0:.0f}s", flush=True)
    else:
        print(f"[{tag}] done final val_logloss={meta['val_logloss']:.4f} auroc={meta['val_auroc']:.4f} (best epoch {best:.4f}) in {time.time()-t0:.0f}s", flush=True)
    return meta
