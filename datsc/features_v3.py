"""Registered-ROI striatal features (run-3 LR arm and the registration gate G1).

All ROIs are fixed masks in template space (datsc.register.Template), so the features are only as good as the
registration -- which is exactly what the gate measures. Input volumes are registered boxes in reference units
(SBR+1), so `mean - 1` is a specific binding ratio.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

FEATURE_NAMES = (
    "left_mean", "right_mean", "left_p90", "right_p90", "left_max", "right_max", "min_side_mean", "max_side_mean",
    "asym_mean", "asym_p90", "left_post_ratio", "right_post_ratio", "min_post_ratio", "frac_gt15", "frac_gt20",
    "striatal_to_ref", "striatal_bg_ring",
)
N_FEATURES = len(FEATURE_NAMES)


def _split_lr(striatal_mask: np.ndarray, mid_x: float):
    xs = np.arange(striatal_mask.shape[0])[:, None, None]
    return striatal_mask & (xs < mid_x), striatal_mask & (xs >= mid_x)


def _post_ratio(v: np.ndarray, m: np.ndarray) -> float:
    """posterior-third mean / anterior-third mean inside one striatum (putamen tail loss shows here). y is P->A."""
    if m.sum() < 8:
        return 1.0
    ys = np.where(m.any(axis=(0, 2)))[0]
    y_lo, y_hi = ys.min(), ys.max()
    t = max(1, (y_hi - y_lo + 1) // 3)
    post = m.copy(); post[:, y_lo + t:, :] = False
    ant = m.copy(); ant[:, : y_hi - t + 1, :] = False
    a = float(v[ant].mean()) if ant.any() else 1.0
    return float(v[post].mean() / max(a, 1e-3))


def striatal_features(vol: np.ndarray, template) -> np.ndarray:
    v = vol.astype(np.float32)
    mid = float(template.striatal_center[0])
    L, R = _split_lr(template.striatal_mask, mid)
    if L.sum() == 0 or R.sum() == 0:
        L = R = template.striatal_mask
    lv, rv = v[L], v[R]
    lm, rm = float(lv.mean()), float(rv.mean())
    lp, rp = float(np.percentile(lv, 90)), float(np.percentile(rv, 90))
    ring = ndi.binary_dilation(template.striatal_mask, iterations=6) & ~ndi.binary_dilation(template.striatal_mask, iterations=2) & template.brain_mask
    ring_mean = float(v[ring].mean()) if ring.any() else 1.0
    s_all = v[template.striatal_mask]
    feats = np.array([
        lm, rm, lp, rp, float(lv.max()), float(rv.max()), min(lm, rm), max(lm, rm),
        abs(lm - rm) / max(lm + rm, 1e-3), abs(lp - rp) / max(lp + rp, 1e-3),
        _post_ratio(v, L), _post_ratio(v, R), min(_post_ratio(v, L), _post_ratio(v, R)),
        float((s_all > 1.5).mean()), float((s_all > 2.0).mean()),
        float(s_all.mean()), float(s_all.mean() / max(ring_mean, 1e-3)),
    ], dtype=np.float32)
    assert feats.shape == (N_FEATURES,)
    return np.nan_to_num(feats)
