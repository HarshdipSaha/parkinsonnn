"""Shared preprocessing: brain mask, intensity normalisation, hotspot anchor, crop, scalars.

This module is imported unchanged by training scripts and by main.py.
Volumes are float32 arrays indexed [x, y, z] at config.SPACING_MM isotropic.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .config import (ANCHOR_MAX_DIST_MM, ANCHOR_MAX_ML, ANCHOR_MIN_ML, ANCHOR_THRESH,
                     CROP_SHAPE, FALLBACK_SUPERIOR_OFFSET_MM, N_SCALARS, SMOOTH_MM,
                     SPACING_MM)


def _otsu(values: np.ndarray) -> float:
    """Otsu threshold on a 1-D array (numpy only, no skimage dependency)."""
    v = values[np.isfinite(values)]
    if v.size == 0 or v.max() == v.min():
        return float(v.max()) if v.size else 0.0
    hist, edges = np.histogram(v, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    w1 = np.cumsum(hist).astype(float)
    w2 = w1[-1] - w1
    m1 = np.cumsum(hist * centers) / np.maximum(w1, 1e-9)
    m2 = (np.cumsum((hist * centers)[::-1])[::-1] / np.maximum(w2, 1e-9))
    between = w1[:-1] * w2[:-1] * (m1[:-1] - m2[1:]) ** 2
    return float(centers[int(np.argmax(between))])


def _largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n <= 1:
        return mask.astype(bool)
    sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def brain_mask(vol: np.ndarray, spacing_mm: float = SPACING_MM) -> np.ndarray:
    sm = ndi.gaussian_filter(vol, sigma=SMOOTH_MM / spacing_mm)
    thr = _otsu(sm.ravel())
    m = sm > thr
    if m.sum() < 50:                      # degenerate: fall back to "anything above 10% of max"
        m = sm > 0.1 * sm.max()
    m = _largest_component(m)
    return ndi.binary_fill_holes(m)


def normalize_p75(vol: np.ndarray, mask: np.ndarray):
    """Divide by the 75th percentile of masked voxels (template-free reference scaling)."""
    vals = vol[mask]
    p75 = float(np.percentile(vals, 75)) if vals.size else 0.0
    if p75 <= 0:
        p75 = float(np.percentile(vals, 99)) if vals.size else 0.0
    if p75 <= 0:
        p75 = float(vol.max()) if vol.max() > 0 else 1.0
    return (vol / p75).astype(np.float32), p75


def find_anchor(norm: np.ndarray, mask: np.ndarray, spacing_mm: float = SPACING_MM):
    """Hotspot anchor: x from the brain-mask midline, y/z from the brightest smoothed blob.

    Returns ((cx, cy, cz) in voxel coords, fallback_flag).
    Fallback = brain-mask centre of mass shifted superiorly (striatum lies above the head CoM).
    """
    vox_ml = spacing_mm ** 3 / 1000.0
    brain_c = np.array(ndi.center_of_mass(mask), dtype=float)
    fallback_c = brain_c.copy()
    fallback_c[2] += FALLBACK_SUPERIOR_OFFSET_MM / spacing_mm

    sm = ndi.gaussian_filter(norm * mask, sigma=SMOOTH_MM / spacing_mm)
    peak = float(sm.max())
    if not np.isfinite(peak) or peak <= 0:
        return tuple(fallback_c), True
    comp = sm > ANCHOR_THRESH * peak
    comp = _largest_component(comp)
    ml = comp.sum() * vox_ml
    if ml < ANCHOR_MIN_ML or ml > ANCHOR_MAX_ML:
        return tuple(fallback_c), True
    hot_c = np.array(ndi.center_of_mass(sm, labels=comp.astype(int), index=1), dtype=float)
    if np.linalg.norm((hot_c - brain_c) * spacing_mm) > ANCHOR_MAX_DIST_MM:
        return tuple(fallback_c), True
    # x from the mask midline so a one-sided (asymmetric) hotspot cannot shift the crop laterally
    return (float(brain_c[0]), float(hot_c[1]), float(hot_c[2])), False


def crop_at(vol: np.ndarray, center, shape=CROP_SHAPE) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float32)
    starts = [int(round(c)) - s // 2 for c, s in zip(center, shape)]
    src, dst = [], []
    for st, s, n in zip(starts, shape, vol.shape):
        lo, hi = max(st, 0), min(st + s, n)
        if hi <= lo:
            return out
        src.append(slice(lo, hi))
        dst.append(slice(lo - st, hi - st))
    out[tuple(dst)] = vol[tuple(src)]
    return out


def _mirror_ncc(norm: np.ndarray, mask: np.ndarray, cx: float) -> float:
    """Normalised cross-correlation between the volume and its L/R mirror about x=cx."""
    n = norm.shape[0]
    shift = int(round(2 * cx - (n - 1)))
    mirrored = np.flip(norm, axis=0)
    mirrored = np.roll(mirrored, shift, axis=0)
    a, b = norm[mask], mirrored[mask]
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def compute_scalars(raw_iso, norm, mask, p75, anchor, fallback, native_spacing, spacing_mm=SPACING_MM):
    vals = raw_iso[mask]
    p99 = float(np.percentile(vals, 99)) if vals.size else p75
    nv = norm[mask]
    sk = float(((nv - nv.mean()) ** 3).mean() / (nv.std() ** 3 + 1e-9)) if nv.size > 2 else 0.0
    zs = np.where(mask.any(axis=(0, 1)))[0]
    z_extent = float((zs.max() - zs.min() + 1) * spacing_mm) if zs.size else 0.0
    n_levels = int(np.unique(np.round(raw_iso[mask], 3)).size) if vals.size else 1
    s = np.array([
        np.log((p99 + 1e-6) / (p75 + 1e-6)),
        mask.sum() * spacing_mm ** 3 / 1000.0,
        sk,
        _mirror_ncc(norm, mask, anchor[0]),
        z_extent,
        float(np.mean(native_spacing)),
        np.log(n_levels),
        float(fallback),
    ], dtype=np.float32)
    assert s.shape == (N_SCALARS,)
    return np.nan_to_num(s)


def preprocess(path, spacing_mm: float = SPACING_MM) -> dict:
    """The single shared entrypoint. Returns dict with crop (fp16), crop_com (fp16), scalars, anchor, fallback."""
    from .io import load_iso  # local import keeps preprocess importable without nibabel at module load
    raw_iso, native_spacing = load_iso(path, spacing_mm)
    mask = brain_mask(raw_iso, spacing_mm)
    norm, p75 = normalize_p75(raw_iso, mask)
    anchor, fallback = find_anchor(norm, mask, spacing_mm)
    com = np.array(ndi.center_of_mass(mask), dtype=float)
    return {
        "crop": crop_at(norm, anchor).astype(np.float16),
        "crop_com": crop_at(norm, tuple(com)).astype(np.float16),
        "scalars": compute_scalars(raw_iso, norm, mask, p75, anchor, fallback, native_spacing, spacing_mm),
        "anchor": np.array(anchor, dtype=np.float32),
        "fallback": bool(fallback),
        "p75": float(p75),
    }
