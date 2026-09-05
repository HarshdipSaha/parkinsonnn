"""Hand-crafted striatal features computed in the anchored crop frame (LR arm + alignment gate)."""
from __future__ import annotations

import numpy as np

from .config import CROP_SHAPE, SPACING_MM, STRIATAL_HALF_MM

RADIOMICS_NAMES = (
    "left_mean", "right_mean", "left_max", "right_max", "left_p90", "right_p90",
    "asym_mean", "asym_p90", "min_side_mean", "min_side_p90", "max_side_p90",
    "bg_ring_mean", "striatal_to_bg", "hot_frac_gt2",
)
N_RADIOMICS = len(RADIOMICS_NAMES)


def _boxes(shape=CROP_SHAPE, spacing_mm=SPACING_MM):
    cx, cy, cz = [s // 2 for s in shape]
    hx, hy, hz = [int(round(h / spacing_mm)) for h in STRIATAL_HALF_MM]
    left = (slice(cx - hx, cx), slice(cy - hy, cy + hy), slice(cz - hz, cz + hz))
    right = (slice(cx, cx + hx), slice(cy - hy, cy + hy), slice(cz - hz, cz + hz))
    return left, right, (cx, cy, cz), (hx, hy, hz)


def _ring_mask(shape, center, inner_vox, outer_vox):
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    d = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    ring2d = (d >= inner_vox) & (d < outer_vox)
    m = np.zeros(shape, bool)
    z0, z1 = max(center[2] - 3, 0), min(center[2] + 3, shape[2])
    m[:, :, z0:z1] = ring2d[:, :, None]
    return m


def hotspot_radiomics(crop: np.ndarray) -> np.ndarray:
    """crop: [x,y,z] float32 in p75-normalised units, anchored so the striatum is at the centre."""
    c = crop.astype(np.float32)
    left, right, center, (hx, hy, hz) = _boxes(c.shape)
    L, R = c[left], c[right]
    lm, rm = float(L.mean()), float(R.mean())
    lx, rx = float(L.max()), float(R.max())
    lp, rp = float(np.percentile(L, 90)), float(np.percentile(R, 90))
    ring = _ring_mask(c.shape, center, inner_vox=max(hx, hy) + 3, outer_vox=max(hx, hy) + 9)
    bg = float(c[ring].mean()) if ring.any() else 1.0
    stri = 0.5 * (lp + rp)
    eps = 1e-6
    f = np.array([
        lm, rm, lx, rx, lp, rp,
        abs(lm - rm) / (lm + rm + eps),
        abs(lp - rp) / (lp + rp + eps),
        min(lm, rm), min(lp, rp), max(lp, rp),
        bg,
        stri / (bg + eps),
        float(((c[left] > 2.0).mean() + (c[right] > 2.0).mean()) / 2),
    ], dtype=np.float32)
    assert f.shape == (N_RADIOMICS,)
    return np.nan_to_num(f)
