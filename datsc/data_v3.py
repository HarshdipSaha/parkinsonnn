"""Run-3 data access: memmap-backed whole-brain cache + GPU-side derivation of model inputs.

Cache layout (scripts/cache_volumes.py): vols.npy fp16 [N,96,96,72], masks.npy u8, scalars.npy [N,8],
stats.npy [N,15], meta.csv (row order == array order). Inputs derived on the training device:
  * 3D: trilinear resize of the 2 mm box to `size` (64x64x48 == 3 mm effective, 96x96x72 == 2 mm)
  * 2.5D: `n_slabs` axial mean-projections of `slab_mm` each, centred on the box centre (== head CoM),
    stacked as channels and bilinearly resized to `hw` pixels for an ImageNet-pretrained backbone.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .volume import VOL_SHAPE, VOL_SPACING_MM, renormalise


class VolumeCache:
    def __init__(self, root):
        self.root = Path(root)
        self.vols = np.load(self.root / "vols.npy", mmap_mode="r")
        self.scalars = np.load(self.root / "scalars.npy")
        self.stats = np.load(self.root / "stats.npy")
        self.meta = pd.read_csv(self.root / "meta.csv", dtype={"uid": str})
        self.index = {u: i for i, u in enumerate(self.meta["uid"])}
        assert self.vols.shape[1:] == VOL_SHAPE, self.vols.shape

    def __len__(self):
        return len(self.meta)


class VolDataset(Dataset):
    """Yields (vol fp32 [1,X,Y,Z] in the requested intensity normalisation, scalars fp32 [8], y, uid)."""

    def __init__(self, cache: VolumeCache, uids, labels, norm: str = "p75"):
        self.c, self.uids, self.norm = cache, list(uids), norm
        self.labels = np.asarray(labels, np.float32)

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        k = self.c.index[self.uids[i]]
        v = renormalise(np.asarray(self.c.vols[k]), self.c.stats[k], self.norm)   # "ref" == as stored in a registered cache
        return (torch.from_numpy(np.ascontiguousarray(v))[None], torch.from_numpy(self.c.scalars[k].copy()),
                torch.tensor(self.labels[i]), self.uids[i])


def to_3d_input(x: torch.Tensor, size=(64, 64, 48)) -> torch.Tensor:
    """[B,1,96,96,72] -> [B,1,*size] (trilinear). size == VOL_SHAPE is a no-op."""
    if tuple(x.shape[2:]) == tuple(size):
        return x
    return F.interpolate(x, size=size, mode="trilinear", align_corners=False)


def to_slabs(x: torch.Tensor, n_slabs: int = 3, slab_mm: float = 12.0, hw: int = 192,
             spacing_mm: float = VOL_SPACING_MM, mode: str = "mean", z_center: float | None = None,
             plane: str = "axial", crop: int | None = None) -> torch.Tensor:
    """[B,1,X,Y,Z] -> [B,n_slabs,hw,hw]: consecutive slabs centred on voxel `z_center` along the slab axis
    (axial: z, default box centre; coronal: y). In the registered frame pass the template's striatal centre.
    Each channel is the mean (or max) over `slab_mm`. The image's last axis is always X (L/R), so a flip on
    dim 3 is the label-preserving mirror for both planes."""
    B, _, X, Y, Z = x.shape
    t = max(1, int(round(slab_mm / spacing_mm)))
    if plane == "coronal":
        x = x.transpose(3, 4)                              # [B,1,X,Z,Y]: slab axis (y) last, rows will be z
        Z = Y
    zc = Z // 2 if z_center is None else int(round(z_center))
    z0 = zc - (n_slabs * t) // 2
    chans = []
    for k in range(n_slabs):
        a, b = max(0, z0 + k * t), min(Z, z0 + (k + 1) * t)
        sl = x[:, 0, :, :, a:b]
        chans.append(sl.amax(dim=-1) if mode == "max" else sl.mean(dim=-1))
    img = torch.stack(chans, dim=1)                       # [B, n, X, rows]
    img = img.transpose(2, 3)                             # [B, n, rows, X]
    if crop:                                              # central crop (e.g. 72 voxels = 144 mm at 2 mm) before resizing
        H, W = img.shape[-2:]; c0 = max(0, (H - crop) // 2); c1 = max(0, (W - crop) // 2)
        img = img[:, :, c0:c0 + crop, c1:c1 + crop]
    if hw != img.shape[-1] or hw != img.shape[-2]:
        img = F.interpolate(img, size=(hw, hw), mode="bilinear", align_corners=False)
    return img
