"""NIfTI loading: canonical RAS orientation and isotropic resampling."""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

from .config import SPACING_MM


def load_canonical(path: str | Path):
    """Load .nii/.nii.gz, reorient to RAS. Returns (vol[x,y,z] float32, spacing_mm, native_shape)."""
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    vol = np.asarray(img.dataobj, dtype=np.float32)
    if vol.ndim > 3:                      # defensive: squeeze any trailing singleton dims
        vol = vol.reshape(vol.shape[:3])
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    spacing = tuple(float(abs(s)) for s in img.header.get_zooms()[:3])
    return vol, spacing, tuple(int(s) for s in vol.shape)


def resample_iso(vol: np.ndarray, spacing, target_mm: float = SPACING_MM) -> np.ndarray:
    factors = [s / target_mm for s in spacing]
    out = zoom(vol, factors, order=1, mode="nearest", prefilter=False)
    return np.ascontiguousarray(out, dtype=np.float32)


def load_iso(path: str | Path, target_mm: float = SPACING_MM):
    """Convenience: load + resample. Returns (vol_iso, native_spacing)."""
    vol, spacing, _ = load_canonical(path)
    return resample_iso(vol, spacing, target_mm), spacing
