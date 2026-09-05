"""Train one (fold, seed) of SmallResNet3D from the crop cache; save state_dict + OOF predictions."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .augment import train_augment
from .config import N_SCALARS
from .model import SmallResNet3D


class CropDataset(Dataset):
    def __init__(self, cache: Path, uids, labels, key: str = "crop"):
        self.cache, self.uids, self.labels, self.key = Path(cache), list(uids), np.asarray(labels, np.float32), key

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        z = np.load(self.cache / f"{self.uids[i]}.npz")
        x = torch.from_numpy(z[self.key].astype(np.float32))[None]
        s = torch.from_numpy(z["scalars"].astype(np.float32))
        return x, s, torch.tensor(self.labels[i]), self.uids[i]


def _seed_all(seed: int):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device, tta_flip: bool = True):
    model.eval(); ps, uids = [], []
    for x, s, _, u in loader:
        x, s = x.to(device), s.to(device)
        logit = model(x, s)
        if tta_flip:
            logit = 0.5 * (logit + model(x.flip(dims=[2]), s))
        ps.append(torch.sigmoid(logit).float().cpu().numpy()); uids += list(u)
    return np.concatenate(ps), uids


def train_fold(cache, fold: int, seed: int, out_dir, epochs: int = 40, batch_size: int = 16, lr: float = 1e-3,
               weight_decay: float = 1e-4, device: str | None = None, num_workers: int = 2,
               control: bool = False, amp: bool | None = None) -> dict:
    cache, out_dir = Path(cache), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    amp = (device.startswith("cuda")) if amp is None else amp
    _seed_all(seed)
    f = pd.read_csv(cache / "folds.csv", dtype={"uid": str})
    key = "crop_com" if control else "crop"
    tr, va = f[f["fold"] != fold], f[f["fold"] == fold]
    dl_tr = DataLoader(CropDataset(cache, tr["uid"], tr["is_pathologic"], key), batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, drop_last=len(tr) > batch_size, pin_memory=device.startswith("cuda"))
    dl_va = DataLoader(CropDataset(cache, va["uid"], va["is_pathologic"], key), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model = SmallResNet3D(n_scalars=N_SCALARS).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = max(1, epochs * len(dl_tr))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + math.cos(math.pi * min(t, steps) / steps)))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    tag = f"f{fold}_s{seed}" + ("_com" if control else "")
    best, best_state, hist, t0 = float("inf"), None, [], time.time()
    for ep in range(epochs):
        model.train(); tl, n = 0.0, 0
        for x, s, y, _ in dl_tr:
            x, s, y = x.to(device), s.to(device), y.to(device)
            with torch.no_grad():
                x = train_augment(x)
            with torch.autocast(device_type="cuda", enabled=amp):
                loss = F.binary_cross_entropy_with_logits(model(x, s), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += float(loss) * len(y); n += len(y)
        p, uids = predict(model, dl_va, device, tta_flip=False)
        yv = va.set_index("uid").loc[uids, "is_pathologic"].values
        vl = float(-np.mean(yv * np.log(np.clip(p, 1e-6, 1)) + (1 - yv) * np.log(np.clip(1 - p, 1e-6, 1))))
        hist.append({"epoch": ep, "train_loss": tl / max(n, 1), "val_logloss": vl, "sec": time.time() - t0})
        print(f"[{tag}] ep{ep:02d} train={tl/max(n,1):.4f} val_logloss={vl:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if vl < best:
            best, best_state = vl, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / f"cnn_{tag}.pt")
    p, uids = predict(model, dl_va, device, tta_flip=True)
    yv = va.set_index("uid").loc[uids, "is_pathologic"].values
    pd.DataFrame({"uid": uids, "y": yv, "p": p}).to_csv(out_dir / f"oof_{tag}.csv", index=False)
    meta = {"fold": fold, "seed": seed, "control": control, "epochs": epochs, "best_val_logloss": best,
            "history": hist, "device": device, "amp": amp}
    (out_dir / f"cnn_{tag}.json").write_text(json.dumps(meta))
    return meta
