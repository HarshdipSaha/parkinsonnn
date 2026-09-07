"""Run-6b inputs: sagittal slab stacks and the tri-planar view tensor.

The registered box is [B,1,X,Y,Z] (X = left/right, Y = posterior->anterior, Z = inferior->superior, 2 mm). The existing
`to_slabs` yields axial (z) and coronal (y) stacks whose last image axis is X, so an L/R mirror is a flip on that axis.
Sagittal stacks are taken along X around each hemisphere's striatal centroid; their in-plane axes are (Z rows, Y cols), so
the L/R mirror of the volume maps the left stack onto the right stack and vice versa (a swap, not an in-plane flip).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .volume import VOL_SPACING_MM


def to_slabs_sagittal(x: torch.Tensor, n_slabs: int = 3, slab_mm: float = 8.0, hw: int = 160, x_center: float = 36.8,
                      spacing_mm: float = VOL_SPACING_MM, mode: str = "mean", crop: int | None = None) -> torch.Tensor:
    """[B,1,X,Y,Z] -> [B,n_slabs,hw,hw]: consecutive slabs along X centred on voxel `x_center`; image rows = Z, cols = Y."""
    B, _, X, Y, Z = x.shape
    t = max(1, int(round(slab_mm / spacing_mm)))
    xc = int(round(x_center)); x0 = xc - (n_slabs * t) // 2
    chans = []
    for k in range(n_slabs):
        a, b = max(0, x0 + k * t), min(X, x0 + (k + 1) * t)
        sl = x[:, 0, a:b, :, :]                                        # [B, t, Y, Z]
        chans.append(sl.amax(dim=1) if mode == "max" else sl.mean(dim=1))
    img = torch.stack(chans, dim=1).transpose(2, 3)                    # [B, n, Z, Y]
    if crop:
        H, W = img.shape[-2:]; c0 = max(0, (H - crop) // 2); c1 = max(0, (W - crop) // 2)
        img = img[:, :, c0:c0 + crop, c1:c1 + crop]
    if hw != img.shape[-1] or hw != img.shape[-2]:
        img = F.interpolate(img, size=(hw, hw), mode="bilinear", align_corners=False)
    return img


def hemisphere_centres(striatal_mask: np.ndarray, mid_x: float) -> tuple[float, float]:
    """x centroids (voxels) of the left and right halves of the template striatal mask."""
    xs = np.arange(striatal_mask.shape[0])[:, None, None]
    L = np.argwhere(striatal_mask & (xs < mid_x)); R = np.argwhere(striatal_mask & (xs >= mid_x))
    if len(L) == 0 or len(R) == 0:
        return float(mid_x) - 11.0, float(mid_x) + 11.0
    return float(L[:, 0].mean()), float(R[:, 0].mean())
