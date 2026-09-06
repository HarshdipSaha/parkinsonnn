"""Feature-engineered arm ("feat"): registered-space striatal descriptors + PCA of the striatal box -> L2 logistic.

Why a second hand-crafted arm next to the CNNs: the stack rewards members whose errors are uncorrelated with the
convolutional ones, and the public-test loss is a site shift. Everything here is anatomy in template space
(no acquisition or site scalars, which fingerprinted the camera in run 2):

  * anterior->posterior uptake profile per striatum (4 bins + slope): posterior putamen loses first in PD;
  * threshold-volume ladder (fraction of the striatal mask above 1.25 ... 3.0 SBR) per side: comma vs dot shape;
  * hot-spot volume above half the side maximum and its A/P, S/I shift from the template centroid;
  * left/right asymmetry of each of the above (asymmetric onset);
  * PCA scores of the 48x48x32 striatal box block-averaged to 24x24x16 (the classical PCA + linear classifier
    for registered DaT-SPECT), fitted on the training folds only.

Volumes are registered boxes in reference units (SBR+1). Mirror orientation is a label-preserving augmentation:
the model is fitted on both orientations and predictions are averaged over both.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .features_v3 import _split_lr, striatal_features, FEATURE_NAMES as V3_NAMES

N_BINS = 4
LADDER = (1.25, 1.5, 2.0, 2.5, 3.0)
CROP = (slice(24, 72), slice(34, 82), slice(18, 50))       # 48x48x32 around the template striatal centre (47.5, 57.3, 33.5)
POOL = 2                                                  # block-average -> 24x24x16 = 9216 values
N_PATCH = (48 // POOL) * (48 // POOL) * (32 // POOL)

_side = ("L", "R")
ENG_NAMES = tuple(
    [f"{s}_bin{b}" for s in _side for b in range(N_BINS)] + [f"{s}_slope" for s in _side]
    + [f"{s}_gt{t}" for s in _side for t in LADDER]
    + [f"{s}_hotvol" for s in _side] + [f"{s}_hot_dy" for s in _side] + [f"{s}_hot_dz" for s in _side]
    + [f"asym_bin{b}" for b in range(N_BINS)] + [f"asym_gt{t}" for t in LADDER] + ["asym_hotvol", "asym_slope"]
)
FEATURE_NAMES = tuple(V3_NAMES) + ENG_NAMES
N_FEATURES = len(FEATURE_NAMES)


def _profile(v: np.ndarray, m: np.ndarray, n_bins: int = N_BINS):
    """Mean SBR in n_bins slabs along y (posterior -> anterior) inside one striatum, plus the linear slope."""
    if m.sum() < 8:
        return np.zeros(n_bins, np.float32), 0.0
    ys = np.where(m.any(axis=(0, 2)))[0]
    edges = np.linspace(ys.min(), ys.max() + 1, n_bins + 1)
    yy = np.arange(m.shape[1])[None, :, None]
    out = []
    for b in range(n_bins):
        sel = m & (yy >= edges[b]) & (yy < edges[b + 1])
        out.append(float(v[sel].mean()) if sel.any() else (out[-1] if out else 1.0))
    out = np.asarray(out, np.float32)
    x = np.arange(n_bins, dtype=np.float32)
    slope = float(np.polyfit(x, out, 1)[0]) if n_bins > 1 else 0.0
    return out, slope


def _hot(v: np.ndarray, m_dil: np.ndarray, centre_yz):
    """Volume (voxels) above half of the side maximum inside a dilated striatal ROI and the centroid offset (y, z)."""
    vals = v[m_dil]
    if vals.size == 0:
        return 0.0, 0.0, 0.0
    thr = 1.0 + 0.5 * max(float(vals.max()) - 1.0, 1e-3)
    hot = m_dil & (v > thr)
    n = float(hot.sum())
    if n == 0:
        return 0.0, 0.0, 0.0
    cy, cz = ndi.center_of_mass(hot)[1:]
    return n, float(cy - centre_yz[0]), float(cz - centre_yz[1])


def engineered_features(vol: np.ndarray, template) -> np.ndarray:
    v = vol.astype(np.float32)
    mid = float(template.striatal_center[0])
    L, R = _split_lr(template.striatal_mask, mid)
    if L.sum() == 0 or R.sum() == 0:
        L = R = template.striatal_mask
    cyz = (float(template.striatal_center[1]), float(template.striatal_center[2]))
    prof, slopes, ladder, hots = [], [], [], []
    for m in (L, R):
        p, s = _profile(v, m); prof.append(p); slopes.append(s)
        sv = v[m]; ladder.append([float((sv > t).mean()) for t in LADDER])
        hots.append(_hot(v, ndi.binary_dilation(m, iterations=3) & template.brain_mask, cyz))
    prof = np.stack(prof); ladder = np.asarray(ladder, np.float32); hots = np.asarray(hots, np.float32)
    asym = lambda a, b: abs(a - b) / max(abs(a) + abs(b), 1e-3)
    feats = np.concatenate([
        prof.ravel(), np.asarray(slopes, np.float32), ladder.ravel(),
        hots[:, 0], hots[:, 1], hots[:, 2],
        [asym(prof[0, b], prof[1, b]) for b in range(N_BINS)], [asym(ladder[0, i], ladder[1, i]) for i in range(len(LADDER))],
        [asym(hots[0, 0], hots[1, 0]), asym(slopes[0], slopes[1])],
    ]).astype(np.float32)
    assert feats.shape == (len(ENG_NAMES),), feats.shape
    return np.nan_to_num(feats)


def feat_vector(vol: np.ndarray, template) -> np.ndarray:
    """All hand-crafted features: the 17 run-3 ones + the engineered block."""
    return np.concatenate([striatal_features(vol, template), engineered_features(vol, template)]).astype(np.float32)


def striatal_patch(vol: np.ndarray) -> np.ndarray:
    """Block-averaged striatal box in SBR units (vol - 1), flattened: the PCA input."""
    v = vol[CROP].astype(np.float32) - 1.0
    a, b, c = v.shape
    v = v.reshape(a // POOL, POOL, b // POOL, POOL, c // POOL, POOL).mean(axis=(1, 3, 5))
    return v.ravel()


def mirror(vol: np.ndarray) -> np.ndarray:
    """L/R mirror in the registered frame (axis 0 is X; the template masks are symmetrised)."""
    return vol[::-1]
