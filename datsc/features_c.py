"""Approach C feature arm ("featc"): radiomic + shape descriptors of the registered striatum on top of the run-3/4 blocks.

Everything is computed in template space from the reference-normalised registered box (SBR+1 units), per hemisphere,
label-blind, with no acquisition/site scalars:

  * principal-axis profile: the striatal mask of each hemisphere is projected on its own geometric long axis (PCA of the
    mask voxel coordinates, oriented posterior -> anterior) and the mean SBR is taken in 6 bins along it (caudate head ..
    posterior putamen) with the slope; medial / lateral halves of the mask (caudate vs putamen proxy) and their ratio;
  * hot-spot shape: intensity-weighted second moments of the uptake above half of the hemisphere maximum inside a dilated
    ROI -> elongation, flatness, orientation of the major axis w.r.t. the anterior-posterior axis, centroid offsets;
  * radial profile: mean SBR in concentric shells (0-3-6-9-12-16 mm) around the hemisphere hot-spot centroid;
  * texture: grey-level co-occurrence (contrast, homogeneity, energy, correlation) of the 48x48x32 striatal box, averaged
    over the three axis offsets (numpy only, no scikit-image dependency);
  * asymmetry (|L-R| / (|L|+|R|)) of the profiles, moments and radial shells (asymmetric onset is the PD signature).

Radiomics of the striatum (texture + shape) has repeatedly matched or beaten semi-quantitative SBR indices for PD vs.
control on DAT-SPECT (EJNMMI Research 2022 radiomics signature; EJNMMI Physics 2024 radiomics + deep features).
Mirror orientation is a label-preserving augmentation: the model is fitted on both orientations and averaged at test time.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .features_v3 import FEATURE_NAMES as V3_NAMES, _split_lr, striatal_features
from .features_v4 import CROP, ENG_NAMES, engineered_features, mirror, striatal_patch  # noqa: F401  (re-exported)

N_AXIS_BINS = 6
SHELLS_MM = (0.0, 3.0, 6.0, 9.0, 12.0, 16.0)
GLCM_LEVELS = 12
GLCM_RANGE = (0.5, 4.0)          # SBR+1 units covered by the grey levels (below/above are clipped)
GLCM_NAMES = ("contrast", "homogeneity", "energy", "correlation")
SPACING_MM = 2.0

_side = ("L", "R")
C_NAMES = tuple(
    [f"{s}_ax{b}" for s in _side for b in range(N_AXIS_BINS)] + [f"{s}_ax_slope" for s in _side]
    + [f"{s}_medial" for s in _side] + [f"{s}_lateral" for s in _side] + [f"{s}_med_lat_ratio" for s in _side]
    + [f"{s}_elong" for s in _side] + [f"{s}_flat" for s in _side] + [f"{s}_orient" for s in _side]
    + [f"{s}_mom_dy" for s in _side] + [f"{s}_mom_dz" for s in _side] + [f"{s}_mom_vol" for s in _side]
    + [f"{s}_shell{k}" for s in _side for k in range(len(SHELLS_MM) - 1)]
    + [f"glcm_{n}" for n in GLCM_NAMES]
    + [f"asym_ax{b}" for b in range(N_AXIS_BINS)] + ["asym_ax_slope", "asym_medial", "asym_lateral", "asym_elong", "asym_mom_vol"]
    + [f"asym_shell{k}" for k in range(len(SHELLS_MM) - 1)]
)
FEATURE_NAMES_C = tuple(V3_NAMES) + tuple(ENG_NAMES) + C_NAMES
N_FEATURES_C = len(FEATURE_NAMES_C)


def _asym(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a) + abs(b), 1e-3)


def _axis_profile(v: np.ndarray, m: np.ndarray, n_bins: int = N_AXIS_BINS):
    """Mean SBR in n_bins along the mask's geometric long axis (oriented posterior -> anterior), plus the slope."""
    idx = np.argwhere(m)
    if len(idx) < 8:
        return np.zeros(n_bins, np.float32), 0.0
    c = idx.mean(0)
    cov = np.cov((idx - c).T)
    w, vec = np.linalg.eigh(cov)
    axis = vec[:, np.argmax(w)]
    if axis[1] < 0:                                    # y grows posterior -> anterior in this frame
        axis = -axis
    proj = (idx - c) @ axis
    edges = np.linspace(proj.min(), proj.max() + 1e-6, n_bins + 1)
    vals = v[m]
    out = []
    for b in range(n_bins):
        sel = (proj >= edges[b]) & (proj < edges[b + 1])
        out.append(float(vals[sel].mean()) if sel.any() else (out[-1] if out else 1.0))
    out = np.asarray(out, np.float32)
    slope = float(np.polyfit(np.arange(n_bins, dtype=np.float32), out, 1)[0])
    return out, slope


def _medial_lateral(v: np.ndarray, m: np.ndarray, mid_x: float):
    """Mean SBR of the medial and lateral halves of one hemisphere's striatal mask (caudate-ish vs putamen-ish)."""
    idx = np.argwhere(m)
    if len(idx) < 8:
        return 1.0, 1.0
    dist = np.abs(idx[:, 0] - mid_x)
    med = dist <= np.median(dist)
    vals = v[m]
    a, b = float(vals[med].mean()), float(vals[~med].mean()) if (~med).any() else float(vals.mean())
    return a, b


def _moments(v: np.ndarray, roi: np.ndarray, centre_yz):
    """Intensity-weighted second moments of the uptake above half the ROI maximum: elongation, flatness, orientation
    (|cos| of the major axis vs the A-P axis), centroid offset (y, z) from the template centre, and the hot volume."""
    vals = v[roi]
    if vals.size == 0:
        return (1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    thr = 1.0 + 0.5 * max(float(vals.max()) - 1.0, 1e-3)
    hot = roi & (v > thr)
    idx = np.argwhere(hot)
    if len(idx) < 6:
        return (1.0, 1.0, 0.0, 0.0, 0.0, float(len(idx)))
    wgt = np.clip(v[hot] - 1.0, 1e-3, None)
    c = (idx * wgt[:, None]).sum(0) / wgt.sum()
    d = idx - c
    cov = (d * wgt[:, None]).T @ d / wgt.sum()
    ev, vec = np.linalg.eigh(cov)
    ev = np.clip(ev, 1e-6, None)                      # ascending
    major = vec[:, 2]
    return (float(ev[2] / ev[1]), float(ev[0] / ev[2]), float(abs(major[1])), float(c[1] - centre_yz[0]), float(c[2] - centre_yz[1]), float(len(idx)))


def _radial(v: np.ndarray, roi: np.ndarray, brain: np.ndarray):
    """Mean SBR in concentric shells around the intensity-weighted centroid of the ROI's uptake."""
    vals = v[roi]
    if vals.size == 0:
        return np.ones(len(SHELLS_MM) - 1, np.float32)
    idx = np.argwhere(roi); wgt = np.clip(v[roi] - 1.0, 1e-3, None)
    c = (idx * wgt[:, None]).sum(0) / wgt.sum()
    zz, yy, xx = np.ogrid[:v.shape[2], :v.shape[1], :v.shape[0]]
    r = np.sqrt((xx - c[0]) ** 2 + (yy - c[1]) ** 2 + (zz - c[2]) ** 2).transpose(2, 1, 0) * SPACING_MM
    out = []
    for k in range(len(SHELLS_MM) - 1):
        sel = (r >= SHELLS_MM[k]) & (r < SHELLS_MM[k + 1]) & brain
        out.append(float(v[sel].mean()) if sel.any() else (out[-1] if out else 1.0))
    return np.asarray(out, np.float32)


def glcm_features(patch: np.ndarray, levels: int = GLCM_LEVELS, rng=GLCM_RANGE) -> np.ndarray:
    """Symmetric grey-level co-occurrence over the three unit axis offsets, averaged: contrast, homogeneity, energy, correlation."""
    q = np.clip((patch - rng[0]) / (rng[1] - rng[0]), 0, 1 - 1e-6)
    q = (q * levels).astype(np.int64)
    feats = np.zeros(4, np.float64)
    i_idx, j_idx = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
    for ax in range(3):
        a = np.moveaxis(q, ax, 0)
        p = np.zeros((levels, levels), np.float64)
        pairs = a[:-1].ravel() * levels + a[1:].ravel()
        np.add.at(p.ravel(), pairs, 1.0)
        p = p + p.T
        p /= max(p.sum(), 1.0)
        mu_i = (i_idx * p).sum(); mu_j = (j_idx * p).sum()
        sd_i = np.sqrt(((i_idx - mu_i) ** 2 * p).sum()); sd_j = np.sqrt(((j_idx - mu_j) ** 2 * p).sum())
        contrast = ((i_idx - j_idx) ** 2 * p).sum()
        homog = (p / (1 + (i_idx - j_idx) ** 2)).sum()
        energy = (p ** 2).sum()
        corr = (((i_idx - mu_i) * (j_idx - mu_j) * p).sum() / max(sd_i * sd_j, 1e-6))
        feats += np.array([contrast, homog, energy, corr])
    return (feats / 3).astype(np.float32)


def c_block(vol: np.ndarray, template) -> np.ndarray:
    v = vol.astype(np.float32)
    mid = float(template.striatal_center[0])
    L, R = _split_lr(template.striatal_mask, mid)
    if L.sum() == 0 or R.sum() == 0:
        L = R = template.striatal_mask
    cyz = (float(template.striatal_center[1]), float(template.striatal_center[2]))
    prof, slopes, medlat, moms, rads = [], [], [], [], []
    for m in (L, R):
        p, s = _axis_profile(v, m); prof.append(p); slopes.append(s)
        medlat.append(_medial_lateral(v, m, mid))
        roi = ndi.binary_dilation(m, iterations=3) & template.brain_mask
        moms.append(_moments(v, roi, cyz)); rads.append(_radial(v, roi, template.brain_mask))
    prof = np.stack(prof); medlat = np.asarray(medlat, np.float32); moms = np.asarray(moms, np.float32); rads = np.stack(rads)
    glcm = glcm_features(v[CROP])
    feats = np.concatenate([
        prof.ravel(), np.asarray(slopes, np.float32),
        medlat[:, 0], medlat[:, 1], medlat[:, 0] / np.maximum(medlat[:, 1], 1e-3),
        moms[:, 0], moms[:, 1], moms[:, 2], moms[:, 3], moms[:, 4], moms[:, 5],
        rads.ravel(), glcm,
        [_asym(prof[0, b], prof[1, b]) for b in range(N_AXIS_BINS)],
        [_asym(slopes[0], slopes[1]), _asym(medlat[0, 0], medlat[1, 0]), _asym(medlat[0, 1], medlat[1, 1]), _asym(moms[0, 0], moms[1, 0]), _asym(moms[0, 5], moms[1, 5])],
        [_asym(rads[0, k], rads[1, k]) for k in range(len(SHELLS_MM) - 1)],
    ]).astype(np.float32)
    assert feats.shape == (len(C_NAMES),), feats.shape
    return np.nan_to_num(feats)


def feat_vector_c(vol: np.ndarray, template) -> np.ndarray:
    """All hand-crafted features of approach C: run-3 (17) + run-4 engineered (44) + the C block."""
    return np.concatenate([striatal_features(vol, template), engineered_features(vol, template), c_block(vol, template)]).astype(np.float32)


def run_featc_arm(arm_cfg: dict, root, vols, st, ok):
    """Inference for arm type "featc": descriptors + striatal-patch PCA -> FeatModelC, averaged over both mirror orientations."""
    from pathlib import Path
    from .featmodel_c import FeatModelC
    from .register import load_template
    from .volume import renormalise
    tpl = load_template(Path(root) / "template.npz")
    if tpl is None:
        return None
    m = FeatModelC.load(Path(root) / Path(arm_cfg["checkpoints"][0]).name)
    idx = np.where(ok)[0]
    z = np.zeros(len(idx))
    for j, i in enumerate(idx):
        v = renormalise(vols[i], st[i], arm_cfg.get("norm", "ref")).astype(np.float32)
        zz = [float(m.decision(feat_vector_c(vv, tpl)[None], striatal_patch(vv)[None])[0]) for vv in (v, mirror(v))]
        z[j] = 0.5 * (zz[0] + zz[1])
    p = np.full(len(ok), np.nan); p[idx] = 1.0 / (1.0 + np.exp(-z))
    print(f"arm {arm_cfg['name']}: featc model ({m.kind}) on {len(idx)} volumes", flush=True)
    return p
