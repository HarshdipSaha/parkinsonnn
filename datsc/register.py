"""Spatial normalisation of CoM boxes to a study template (SimpleITK, 7-DOF similarity + optional affine refinement),
QC by normalised cross-correlation, and reference-region intensity normalisation in template space.

Everything operates on the 2 mm CoM box produced by datsc.volume.prepare_volume, so the same code runs in the cache
builder (from cached boxes) and in main.py (from a freshly loaded scan). SimpleITK is in the DrivenData runtime.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .volume import VOL_SHAPE, VOL_SPACING_MM

REG_STAT_NAMES = ("ncc_init", "ncc_reg", "reg_ok", "reg_scale", "reg_rot_deg", "reg_trans_mm", "ref_mean", "bg_level")
N_REG_STATS = len(REG_STAT_NAMES)


def _sitk():
    import SimpleITK as sitk
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    return sitk


def _to_sitk(box: np.ndarray, spacing: float = VOL_SPACING_MM):
    sitk = _sitk()
    img = sitk.GetImageFromArray(np.ascontiguousarray(box.astype(np.float32).transpose(2, 1, 0)))   # [x,y,z] -> [z,y,x]
    img.SetSpacing((spacing, spacing, spacing))
    return img


def _from_sitk(img) -> np.ndarray:
    return _sitk().GetArrayFromImage(img).transpose(2, 1, 0)


def prep_for_registration(box: np.ndarray, spacing: float = VOL_SPACING_MM, smooth_mm: float = 3.0) -> np.ndarray:
    x = ndi.gaussian_filter(box.astype(np.float32), smooth_mm / spacing)
    hi = float(np.percentile(x, 99.5))
    return np.clip(x, 0, hi) / max(hi, 1e-6)


def ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        a, b = a[mask], b[mask]
    a = a.astype(np.float64).ravel() - a.mean(); b = b.astype(np.float64).ravel() - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def register(moving_box: np.ndarray, template_box: np.ndarray, spacing: float = VOL_SPACING_MM,
             iterations: int = 100, affine_refine: bool = True, seed: int = 0, sampling: float = 0.15):
    """Register `moving_box` to `template_box` (both [X,Y,Z] at `spacing`). Returns (registered_box, params dict).
    Similarity (rotation, translation, isotropic scale) with moments initialisation, multi-resolution NCC at 8 and
    4 mm (a 10 mm-FWHM modality gains nothing from the 2 mm level); optional 12-DOF affine refinement at 4 mm."""
    sitk = _sitk()
    fixed = _to_sitk(prep_for_registration(template_box, spacing), spacing)
    moving = _to_sitk(prep_for_registration(moving_box, spacing), spacing)
    init = sitk.CenteredTransformInitializer(fixed, moving, sitk.Similarity3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)

    def run(initial, shrink, sigmas, iters, lr):
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsCorrelation()
        R.SetMetricSamplingStrategy(R.RANDOM); R.SetMetricSamplingPercentage(sampling, seed)
        R.SetInterpolator(sitk.sitkLinear)
        R.SetOptimizerAsGradientDescent(learningRate=lr, numberOfIterations=iters, convergenceMinimumValue=1e-6,
                                        convergenceWindowSize=10, estimateLearningRate=R.EachIteration)
        R.SetOptimizerScalesFromPhysicalShift()
        R.SetShrinkFactorsPerLevel(shrink); R.SetSmoothingSigmasPerLevel(sigmas); R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        R.SetInitialTransform(initial, inPlace=True)
        tx = R.Execute(fixed, moving)
        return tx, float(R.GetMetricValue())

    sim, m_sim = run(init, [4, 2], [3.0, 1.5], iterations, 1.0)
    tx = sim
    if affine_refine:
        aff = sitk.AffineTransform(3)
        aff.SetCenter(sim.GetCenter())
        aff.SetMatrix(sim.GetMatrix()); aff.SetTranslation(sim.GetTranslation())
        try:
            aff2, m_aff = run(aff, [2], [1.5], iterations // 2, 1.0)
            if m_aff <= m_sim:            # correlation metric is negative; keep the refinement only if it helped
                tx = aff2
        except RuntimeError:
            pass
    out = sitk.Resample(_to_sitk(moving_box, spacing), fixed, tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    p = np.array(sim.GetParameters(), float)     # versor(3), translation(3), scale
    rot_deg = float(np.degrees(2 * np.arcsin(min(1.0, np.linalg.norm(p[:3])))))
    params = {"scale": float(p[6]), "rot_deg": rot_deg, "trans_mm": float(np.linalg.norm(p[3:6])),
              "transform": tx.GetParameters(), "fixed_params": tx.GetFixedParameters(), "kind": tx.GetName()}
    return _from_sitk(out), params


def apply_transform(box: np.ndarray, kind: str, transform_params, fixed_params, spacing: float = VOL_SPACING_MM,
                    interpolator: str = "linear") -> np.ndarray:
    sitk = _sitk()
    tx = sitk.AffineTransform(3) if kind.startswith("Affine") else sitk.Similarity3DTransform()
    tx.SetFixedParameters(tuple(fixed_params)); tx.SetParameters(tuple(transform_params))
    ref = _to_sitk(np.zeros(VOL_SHAPE, np.float32), spacing)
    interp = sitk.sitkNearestNeighbor if interpolator == "nearest" else sitk.sitkLinear
    return _from_sitk(sitk.Resample(_to_sitk(box, spacing), ref, tx, interp, 0.0, sitk.sitkFloat32))


HIST_PCTS = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def hist_landmarks(vol: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    """Intensity percentiles inside `mask` used as landmarks for histogram standardisation. Use the reference region
    (brain minus striatum): standardising the striatum itself would erase the label signal."""
    return np.percentile(np.asarray(vol, np.float32)[brain_mask], HIST_PCTS).astype(np.float32)


def hist_standardise(vol: np.ndarray, brain_mask: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Piecewise-linear map of the scan's landmarks `src` onto the template's `dst` (Nyul-style), applied to every voxel;
    linear extrapolation beyond the outer landmarks with the outer segment's slope. Monotonic by construction."""
    src = np.maximum.accumulate(np.asarray(src, np.float64)); dst = np.asarray(dst, np.float64)
    x = np.asarray(vol, np.float32).ravel()
    out = np.interp(x, src, dst).astype(np.float32)
    lo, hi = x < src[0], x > src[-1]
    if src[1] > src[0]:
        out[lo] = dst[0] + (x[lo] - src[0]) * (dst[1] - dst[0]) / (src[1] - src[0])
    if src[-1] > src[-2]:
        out[hi] = dst[-1] + (x[hi] - src[-1]) * (dst[-1] - dst[-2]) / (src[-1] - src[-2])
    return out.reshape(vol.shape)



class Template:
    """Study template + derived masks. Stored as artifacts/template.npz (label-blind mean of registered training boxes)."""

    def __init__(self, box, brain_mask, striatal_mask, ref_mask, striatal_center, ncc_thresh: float, spacing: float = VOL_SPACING_MM, hist=None):
        self.box = box.astype(np.float32); self.brain_mask = brain_mask.astype(bool)
        self.hist = None if hist is None else np.asarray(hist, np.float32)     # histogram-standardisation landmarks (optional)
        self.striatal_mask = striatal_mask.astype(bool); self.ref_mask = ref_mask.astype(bool)
        self.striatal_center = np.asarray(striatal_center, float); self.ncc_thresh = float(ncc_thresh); self.spacing = spacing
        self.qc_mask = ndi.binary_erosion(self.brain_mask, iterations=2)

    def save(self, path):
        extra = {} if self.hist is None else {"hist": self.hist}
        np.savez_compressed(path, box=self.box.astype(np.float16), brain_mask=self.brain_mask, striatal_mask=self.striatal_mask,
                            ref_mask=self.ref_mask, striatal_center=self.striatal_center, ncc_thresh=self.ncc_thresh, spacing=self.spacing, **extra)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["box"], z["brain_mask"], z["striatal_mask"], z["ref_mask"], z["striatal_center"], float(z["ncc_thresh"]), float(z["spacing"]),
                   hist=(z["hist"] if "hist" in z.files else None))


def derive_masks(template_box: np.ndarray, spacing: float = VOL_SPACING_MM, erode_mm: float = 8.0, dilate_mm: float = 20.0):
    """Brain mask (Otsu on the smoothed template, largest component), striatal mask (brightest interior blob of the
    template, which is label-blind because the template is), reference mask = eroded brain minus dilated striatum."""
    from .preprocess import _largest_component
    sm = ndi.gaussian_filter(template_box.astype(np.float32), 4.0 / spacing)
    hi = float(np.percentile(sm, 99.5))
    # the template is a smooth mean in p75 units (cortex ~1, striatum ~3): a fixed fraction of the peak range gives
    # the head/brain envelope; Otsu would isolate the striatal blob instead
    brain = ndi.binary_fill_holes(_largest_component(sm > 0.12 * hi))
    depth = ndi.distance_transform_edt(brain, sampling=spacing)
    inner = depth > 15.0
    sm_in = np.where(inner, sm, 0.0)
    peak = float(sm_in.max())
    hot = sm_in > 0.7 * peak
    # both hemispheres: everything above threshold within 40 mm of the centroid of ALL hot voxels (on the midline for a
    # symmetric template), then mirrored so the mask is exactly L/R symmetric
    c = np.array(ndi.center_of_mass(hot), float)
    idx = np.indices(template_box.shape).astype(float)
    dist = np.sqrt(sum(((idx[i] - c[i]) * spacing) ** 2 for i in range(3)))
    striatal = (sm_in > 0.6 * peak) & (dist < 40.0)
    striatal = striatal | np.flip(striatal, axis=0)
    striatal_center = np.array(ndi.center_of_mass(striatal), float)
    striatal_center[0] = (template_box.shape[0] - 1) / 2.0   # the template is symmetrised about the box centre
    eroded = ndi.binary_erosion(brain, iterations=max(1, int(round(erode_mm / spacing))))
    dil = ndi.binary_dilation(striatal, iterations=max(1, int(round(dilate_mm / spacing))))
    ref = eroded & ~dil
    return brain, striatal, ref, striatal_center


def symmetrize(T: np.ndarray, max_shift: int = 12):
    """Find the mid-sagittal plane of a template by maximising NCC with its x-mirror over integer shifts, roll the
    template so that plane sits at the box centre ((X-1)/2), and average with its mirror. Returns (T_sym, x_mid, delta)."""
    best = (-2.0, 0)
    for sh in range(-max_shift, max_shift + 1):
        m = np.roll(np.flip(T, axis=0), sh, axis=0)
        v = ncc(T, m)
        if v > best[0]:
            best = (v, sh)
    sh = best[1]
    x_mid = ((T.shape[0] - 1) + sh) / 2.0
    delta = int(round((T.shape[0] - 1) / 2.0 - x_mid))
    Tr = np.roll(T, delta, axis=0)
    return 0.5 * (Tr + np.flip(Tr, axis=0)), x_mid, delta


def register_and_normalise(box_p75: np.ndarray, template: Template, spacing: float = VOL_SPACING_MM, do_register: bool = True):
    """The shared per-volume step: register the CoM box to the template, QC, reference-normalise.
    Returns (vol_out float32 in SBR+1 units, reg_stats[N_REG_STATS], params dict)."""
    box = box_p75.astype(np.float32)
    ncc_init = ncc(prep_for_registration(box, spacing), prep_for_registration(template.box, spacing), template.qc_mask)
    reg, params, ncc_reg = box, {"scale": 1.0, "rot_deg": 0.0, "trans_mm": 0.0}, ncc_init
    ok = False
    if do_register:
        try:
            reg_try, params_try = register(box, template.box, spacing)
            ncc_try = ncc(prep_for_registration(reg_try, spacing), prep_for_registration(template.box, spacing), template.qc_mask)
            plausible = 0.6 <= params_try["scale"] <= 1.6 and params_try["rot_deg"] <= 45 and params_try["trans_mm"] <= 60
            if plausible and ncc_try >= max(template.ncc_thresh, ncc_init - 0.05):
                reg, params, ncc_reg, ok = reg_try, params_try, ncc_try, True
        except Exception:  # noqa: BLE001 - any registration failure falls back to the CoM frame
            pass
    outside = ~ndi.binary_dilation(template.brain_mask, iterations=3)
    bg = float(np.median(reg[outside])) if outside.any() else 0.0
    ref_vals = reg[template.ref_mask]
    ref_mean = float(ref_vals.mean() - bg) if ref_vals.size else 1.0
    if not np.isfinite(ref_mean) or ref_mean <= 1e-3:
        ref_mean = max(float(np.percentile(reg[template.brain_mask], 75) - bg), 1e-3)
    vol = np.clip(reg - bg, 0, None) / ref_mean
    stats = np.array([ncc_init, ncc_reg, float(ok), params["scale"], params["rot_deg"], params["trans_mm"], ref_mean, bg], np.float32)
    return vol.astype(np.float32), np.nan_to_num(stats), params


def load_template(path: str | Path | None) -> Template | None:
    return Template.load(path) if path and Path(path).exists() else None
