"""Crop-agnostic whole-brain volume representation (run 3).

One `prepare_volume(path)` call yields a 2 mm isotropic, RAS-oriented, brain-CoM-centred box of the
p75-normalised volume plus the head mask and per-volume statistics. Every downstream input (tight
crop, whole-brain crop, axial slabs, alternative intensity normalisations) is derived from this box,
so the expensive NIfTI load + resample happens once per scan in training AND in main.py.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .config import N_SCALARS
from .preprocess import brain_mask, compute_scalars, crop_at, normalize_p75

VOL_SPACING_MM = 2.0
VOL_SHAPE = (96, 96, 72)          # [x, y, z] voxels at 2 mm = 192 x 192 x 144 mm, centred on the head CoM
STAT_NAMES = ("p75", "p50", "p99", "mean", "std", "mask_ml", "com_x", "com_y", "com_z",
              "sp_x", "sp_y", "sp_z", "shape_x", "shape_y", "shape_z", "n_levels_native")
N_STATS = len(STAT_NAMES)


def prepare_volume(path, spacing_mm: float = VOL_SPACING_MM, shape=VOL_SHAPE, template=None) -> dict:
    """Returns dict(vol fp16 [X,Y,Z], mask bool [X,Y,Z], scalars[N_SCALARS], stats[N_STATS (+N_REG_STATS)]).
    Without a template: the CoM box in p75 units. With a datsc.register.Template: the box is registered to the
    template (QC'd, CoM fallback) and expressed in reference-region units; registration stats are appended."""
    from .io import load_canonical, resample_iso
    raw, native_spacing, native_shape = load_canonical(path)
    n_levels_native = int(np.unique(raw[raw > 0]).size) if raw.size and (raw > 0).any() else 1
    raw_iso = resample_iso(raw, native_spacing, spacing_mm)
    del raw
    mask = brain_mask(raw_iso, spacing_mm)
    norm, p75 = normalize_p75(raw_iso, mask)
    vals = raw_iso[mask]
    if vals.size == 0:
        vals = raw_iso.ravel()
    com = np.array(ndi.center_of_mass(mask), dtype=float) if mask.any() else np.array(raw_iso.shape, float) / 2
    scalars = compute_scalars(raw_iso, norm, mask, p75, (float(com[0]), float(com[1]), float(com[2])), False,
                              native_spacing, spacing_mm)
    stats = np.array([
        p75, float(np.percentile(vals, 50)), float(np.percentile(vals, 99)), float(vals.mean()), float(vals.std()),
        float(mask.sum()) * spacing_mm ** 3 / 1000.0, com[0], com[1], com[2],
        native_spacing[0], native_spacing[1], native_spacing[2], native_shape[0], native_shape[1], native_shape[2],
    ], dtype=np.float32)
    box = crop_at(norm, tuple(com), shape)
    mbox = crop_at(mask.astype(np.float32), tuple(com), shape) > 0.5
    stats = np.concatenate([np.nan_to_num(stats), [n_levels_native]]).astype(np.float32)
    if template is None:
        return {"vol": box.astype(np.float16), "mask": mbox, "scalars": scalars.astype(np.float32), "stats": stats}
    from .register import register_and_normalise
    vol, reg_stats, params = register_and_normalise(box, template, spacing_mm)
    return {"vol": vol.astype(np.float16), "mask": mbox, "scalars": scalars.astype(np.float32),
            "stats": np.concatenate([stats, reg_stats]).astype(np.float32), "reg_params": params}


def renormalise(vol_p75: np.ndarray, stats: np.ndarray, mode: str = "p75", template=None) -> np.ndarray:
    """Re-derive alternative intensity normalisations from the p75-scaled box + stored raw statistics.
    mode "hist": reference-normalised box mapped onto the template's brain-mask intensity landmarks (needs `template`
    with `.brain_mask` and `.hist`, see datsc.register.hist_standardise)."""
    p75, p50, p99, mean, std = [float(stats[i]) for i in range(5)]
    x = vol_p75.astype(np.float32)
    if mode in ("p75", "ref"):           # "ref": the cache already stores reference-normalised registered boxes
        return x
    if mode == "hist":
        from .register import hist_landmarks, hist_standardise
        assert template is not None and getattr(template, "hist", None) is not None, "hist mode needs a template with landmarks"
        return hist_standardise(x, template.ref_mask, hist_landmarks(x, template.ref_mask), template.hist)   # landmarks from the reference region only: the striatal contrast is preserved
    if mode == "median":                 # background-reference (SBR-like) scaling
        return x * (p75 / max(p50, 1e-6))
    if mode == "zscore":                 # brain-mask z-score
        return (x * p75 - mean) / max(std, 1e-6)
    if mode == "p99":
        return x * (p75 / max(p99, 1e-6))
    raise ValueError(mode)


assert N_SCALARS == 8
