# DaT Parkinson's Challenge Pipeline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a valid, calibrated `submission.zip` for DrivenData competition #311 (binary normal/abnormal classification of 3D DaT-SPECT volumes, scored by log loss) by Sept 16 2026, with a first scored submission by Sept 8.

**Architecture:** One `preprocess(path)` function (RAS → 3 mm isotropic → brain mask → p75 normalisation → hotspot-anchored 72×72×48 crop + 8 scalar QC features) is shared verbatim by training and by the submission `main.py`. Two model arms consume its output: (1) an L2 logistic regression on hotspot radiomics + scalars (also the alignment gate and the inference fallback) and (2) a ~0.9M-parameter 3D ResNet trained 5 folds × 3 seeds with blur/noise/quantisation augmentation. Predictions are probability-averaged, then calibrated with exactly three fitted numbers (blend weight `w`, temperature `T`, prior-mix `λ`). `main.py` writes a prior-only `submission.csv` first and overwrites it stage-wise under a wall-clock deadline ladder so any crash still leaves a valid file.

**Tech Stack:** Python 3.12, numpy, scipy, nibabel, scikit-learn, pandas, PyTorch (CPU locally / CUDA on Kaggle / CUDA 12.9 in the DrivenData runtime), pytest. No external pretrained weights, no registration library, no pickles in the submission (JSON + `state_dict` only).

**Spec:** `COUNCIL_VERDICT.md` § "The Recommendation" (the pipeline), `DAT_PARKINSONS_CHALLENGE.md` (runtime + rules), `EDA_REPORT.md` (data facts). Where this plan and the verdict differ, the verdict wins except for the accepted risk below.

---

## Accepted risks and decisions (read before starting)

1. **Training runs on Kaggle (T4 x2), not a rented instance.** The council flagged (verified in `DAT_PARKINSONS_CHALLENGE.md` lines 95–97) that Kaggle's ToS grants Kaggle a perpetual licence over uploaded content and DrivenData staff have not cleared Kaggle notebooks; the user decided to accept this prize-eligibility risk. Mitigations: keep the Kaggle dataset **private**, never save raw volumes as notebook output (only the fp16 crop cache and model artefacts), delete the dataset after the competition, and record the venue in the final disclosure. Post the question on forum thread 11479 anyway (5 minutes) so a "no" arrives early enough to re-run the ~5 GPU-hours elsewhere.
2. **Kaggle GPU choice: T4 x2**, running two folds concurrently (one process per GPU, `CUDA_VISIBLE_DEVICES=0` / `=1`) with fp16 autocast. A 0.9M-param 3D CNN on 72×72×48 fp16 crops is tiny; two T4s halve wall-clock versus one P100 and tensor cores make autocast fast. If the two-process pattern gives trouble, fall back to P100 single-process — the scripts are identical.
3. **No `just`** on the local box. Local container testing is done with `docker run` directly (Task 14) and, before that, by running `main.py` against a `data-demo/` folder with `DATSC_DATA_ROOT` overridden (Task 13). Docker image is ~21 GB; pull it early (Task 0, background).
4. **Registration and the 2.5D branch are cut** and do not appear in this plan. Do not add them.
5. **Test-data hygiene:** never print uids, per-scan predictions, or anything about test volumes from `main.py` (rules). Only aggregate counts and timings.

## Date map (today = Fri Sept 5)

| Date | Tasks | Milestone |
|---|---|---|
| Sept 5 (Fri) | 0–4 | `preprocess()` exists and is tested; forum question posted; Docker pull running |
| Sept 6 (Sat) | 5–8 | Crop cache built locally (12 CPUs); QC grid eyeballed; folds built; **alignment gate PASS (OOF AUROC > 0.90)**; prior-only submission uploaded |
| Sept 7 (Sun) | 9–11 | Model/augment/train code tested on CPU; Kaggle notebook launched: 5 folds × 1 seed + CoM control overnight |
| Sept 8 (Mon) | 12–14 | Calibration fitted; `main.py` + ladder; local container dry run timed; **scored submission #2** |
| Sept 9–10 | 15–16 | Seeds ×3 on Kaggle; augmentation ablation; leave-cluster-out diagnostics → λ; rebuild test |
| Sept 11–13 | — | Frozen (other sprint). Only queued Kaggle runs. |
| Sept 14 | 16 | Final 15-checkpoint ensemble; final dry run; submissions #3 (λ rule) and #4 (other λ) |
| Sept 15 | — | Buffer, README/disclosure, delete Kaggle dataset |
| Sept 16 | — | Deadline. Nothing new. |

## File structure

```
parkinsonnn/
├── main.py                      # submission entrypoint (must be at zip root) — Task 13
├── datsc/                       # the package, shipped inside submission.zip
│   ├── __init__.py
│   ├── config.py                # constants + data-path discovery (local / Kaggle / runtime) — Task 1
│   ├── io.py                    # load NIfTI (.nii or .nii.gz) → RAS → 3 mm isotropic — Task 2
│   ├── preprocess.py            # brain mask, p75 normalise, hotspot anchor, crop, scalars — Tasks 3–4
│   ├── features.py              # hotspot radiomics for the LR arm — Task 5
│   ├── folds.py                 # 16³ embedding repeat-patient audit + stratified group folds — Task 7
│   ├── model.py                 # SmallResNet3D (~0.9M params) with scalar head — Task 9
│   ├── augment.py               # tensor-side augmentations (blur, noise, quantise, jitter, affine, flip) — Task 10
│   ├── train.py                 # one fold×seed training run → state_dict + OOF preds — Task 11
│   ├── lrmodel.py               # L2 logistic regression with JSON (de)serialisation — Task 8
│   ├── calibrate.py             # fit/apply w, T, λ; JSON I/O — Task 12
│   └── infer.py                 # inference stages + deadline ladder used by main.py — Task 13
├── scripts/
│   ├── cache_crops.py           # preprocess all training volumes → cache/*.npz + meta.csv — Task 6
│   ├── qc_grid.py               # PNG grids of crops for eyeballing — Task 6
│   ├── make_folds.py            # → folds.csv — Task 7
│   ├── alignment_gate.py        # radiomics + LR OOF AUROC; PASS/FAIL — Task 8
│   ├── train_cnn.py             # CLI wrapper over datsc.train — Task 11
│   ├── fit_calibration.py       # → artifacts/calib.json — Task 12
│   ├── shift_diagnostic.py      # leave-cluster-out log-loss inflation → λ rule — Task 16
│   ├── pack_submission.py       # → submission.zip — Task 14
│   └── make_prior_submission.py # prior-only submission.zip — Task 13
├── kaggle/
│   └── train_notebook_cells.py  # cells for the Kaggle GPU notebook (`# %%` separated) — Task 15
├── tests/
│   ├── conftest.py              # synthetic NIfTI fixtures — Task 0
│   ├── test_config.py … test_infer.py
├── artifacts/                   # gitignored except .gitkeep: lr.json, calib.json, cnn_f{fold}_s{seed}.pt
├── cache/                       # gitignored: {uid}.npz, meta.csv, folds.csv
├── data/                        # gitignored (already): raw competition data
├── data-demo/                   # gitignored: smoke-test data extracted here
├── requirements-dev.txt
└── docs/superpowers/plans/2026-09-05-dat-pipeline.md   # this file
```

Conventions used throughout:
- Volumes are numpy `float32` arrays indexed `[x, y, z]` in RAS orientation after `io.load_canonical`. Crops are `[72, 72, 48]` = `[x, y, z]`. Torch tensors are `[B, 1, X, Y, Z]`; the left/right mirror flip is `flip(dims=[2])`.
- All spacing arithmetic is in **mm**; `config.SPACING_MM = 3.0` is the isotropic target.
- Every script has `if __name__ == "__main__":` and uses `multiprocessing.get_context("spawn")` (Windows + Kaggle safe).
- Commit after every task with the message given. Run `python -m pytest -q` before every commit.

---

### Task 0: Repo scaffold, dev requirements, synthetic-volume fixtures

**Files:**
- Create: `requirements-dev.txt`, `datsc/__init__.py`, `tests/__init__.py`, `tests/conftest.py`, `artifacts/.gitkeep`, `pytest.ini`
- Modify: `.gitignore`

- [ ] **Step 1: Write requirements and pytest config**

`requirements-dev.txt`:
```
numpy>=1.26,<2.1
scipy>=1.13
nibabel>=5.3
scikit-learn>=1.5
pandas>=2.2
matplotlib>=3.8
pytest>=8
```
(`torch` is already installed locally as 2.2.2+cpu; do not pin it here — Kaggle and the runtime supply their own.)

`pytest.ini`:
```ini
[pytest]
testpaths = tests
addopts = -q
filterwarnings = ignore::DeprecationWarning
```

- [ ] **Step 2: Extend .gitignore**

Append to `.gitignore`:
```
cache/
data-demo/
artifacts/*
!artifacts/.gitkeep
submission.zip
submission_*.zip
submission_*.csv
submission/
artifacts_smoke/
artifacts_noaug/
artifacts_timing/
data-timing/
__pycache__/
*.pyc
.pytest_cache/
qc/
```

- [ ] **Step 3: Create the package and the synthetic fixture**

`datsc/__init__.py`:
```python
"""DaT-SPECT classifier: shared preprocessing, models, calibration, inference."""
__version__ = "0.1.0"
```

`tests/__init__.py`: empty file.

`tests/conftest.py`:
```python
import numpy as np
import nibabel as nib
import pytest


def make_synthetic_volume(shape=(96, 96, 60), spacing=(2.46, 2.46, 2.46), levels=None,
                          seed=0, striatum_amp=4.0, asym=0.0, scale=1000.0):
    """Synthetic DaT-like head: ellipsoid background + two bright striatal blobs.

    asym in [0,1]: fraction by which the RIGHT blob is dimmed (simulates pathology).
    levels: if set, quantise to this many integer grey levels (ultra-low-count regime).
    Returns (array[x,y,z] float32, spacing).
    """
    rng = np.random.default_rng(seed)
    x, y, z = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    c = np.array(shape) / 2.0
    r = np.array(shape) / 2.6
    ell = (((x - c[0]) / r[0]) ** 2 + ((y - c[1]) / r[1]) ** 2 + ((z - c[2]) / r[2]) ** 2) <= 1.0
    vol = np.where(ell, 1.0, 0.0) + rng.normal(0, 0.08, size=shape)
    sig_vox = 6.0 / np.array(spacing)  # 6 mm blobs
    off_vox = 9.0 / spacing[0]         # blobs 9 mm left/right of midline
    for side, amp in ((-1, striatum_amp), (+1, striatum_amp * (1.0 - asym))):
        cx = c[0] + side * off_vox
        blob = np.exp(-(((x - cx) / sig_vox[0]) ** 2 + ((y - c[1]) / sig_vox[1]) ** 2
                        + ((z - c[2]) / sig_vox[2]) ** 2) / 2.0)
        vol += amp * blob
    vol = np.clip(vol, 0, None)
    if levels is not None:
        vol = np.round(vol / vol.max() * (levels - 1))
    else:
        vol = vol * scale
    return vol.astype(np.float32), tuple(float(s) for s in spacing)


def write_nifti(path, vol, spacing):
    affine = np.diag(list(spacing) + [1.0])
    nib.save(nib.Nifti1Image(vol, affine), str(path))
    return path


@pytest.fixture
def synth():
    return make_synthetic_volume


@pytest.fixture
def synth_nifti(tmp_path):
    def _make(name="abc12345", **kw):
        vol, sp = make_synthetic_volume(**kw)
        return write_nifti(tmp_path / f"{name}.nii.gz", vol, sp)
    return _make
```

- [ ] **Step 4: Install and verify the test scaffold runs**

Run: `python -m pip install -r requirements-dev.txt && python -m pytest`
Expected: `no tests ran` (exit code 5 is fine at this point).

- [ ] **Step 5: Start the Docker image pull in the background (do not wait for it)**

Run in a separate terminal: `docker pull competitionsfmnparkinsonsprodacr.azurecr.io/competition-sfmn-parkinsons-runtime:gpu-latest`
(≈21 GB. If a `cpu-latest` tag exists per the runtime README, pull that instead — it is what the local box can actually run.)

- [ ] **Step 6: Post the Kaggle compliance question**

Reply on https://community.drivendata.org/t/clarification-on-ai-coding-assistants-and-cloud-gpu-compute-with-competition-data/11479 asking whether a private Kaggle notebook with no dataset persistence is compliant. Takes five minutes; the answer has days of latency.

- [ ] **Step 7: Commit**

```bash
git add requirements-dev.txt pytest.ini .gitignore datsc/__init__.py tests/__init__.py tests/conftest.py artifacts/.gitkeep
git commit -m "chore: scaffold datsc package, dev requirements, synthetic fixtures"
```

---

### Task 1: `config.py` — constants and data-path discovery

**Files:**
- Create: `datsc/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from pathlib import Path
from datsc import config


def test_constants():
    assert config.SPACING_MM == 3.0
    assert config.CROP_SHAPE == (72, 72, 48)
    assert config.N_SCALARS == 8


def test_find_data_dir_prefers_env(tmp_path, monkeypatch):
    (tmp_path / "train_labels_x.csv").write_text("uid,is_pathologic\nabc,1.0\n")
    (tmp_path / "niftis").mkdir()
    (tmp_path / "niftis" / "abc.nii").write_bytes(b"")
    monkeypatch.setenv("DATSC_DATA_DIR", str(tmp_path))
    d = config.find_data_dir()
    assert d.root == tmp_path
    assert d.labels_csv.name == "train_labels_x.csv"
    assert d.niftis_dir == tmp_path / "niftis"


def test_find_nifti_path_handles_both_extensions(tmp_path):
    (tmp_path / "u1.nii").write_bytes(b"")
    (tmp_path / "u2.nii.gz").write_bytes(b"")
    assert config.find_nifti_path(tmp_path, "u1").name == "u1.nii"
    assert config.find_nifti_path(tmp_path, "u2").name == "u2.nii.gz"
    assert config.find_nifti_path(tmp_path, "u3") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'datsc.config'`

- [ ] **Step 3: Write the implementation**

```python
# datsc/config.py
"""Constants and data-location discovery shared by training and inference."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --- geometry -------------------------------------------------------------
SPACING_MM = 3.0                   # isotropic resampling target
CROP_SHAPE = (72, 72, 48)          # [x, y, z] voxels at SPACING_MM (216x216x144 mm)
SMOOTH_MM = 6.0                    # gaussian sigma used for mask + anchor
ANCHOR_THRESH = 0.6                # fraction of smoothed max defining the hotspot
ANCHOR_MIN_ML, ANCHOR_MAX_ML = 1.0, 30.0
ANCHOR_MAX_DIST_MM = 40.0
FALLBACK_SUPERIOR_OFFSET_MM = 15.0 # CoM fallback: striatum sits above the head CoM
STRIATAL_HALF_MM = (15.0, 15.0, 9.0)  # half-extent of the striatal box for radiomics

# --- scalars (order is a contract between preprocess, LR, CNN head) --------
SCALAR_NAMES = (
    "log_p99_over_p75", "mask_ml", "masked_skew", "mirror_ncc",
    "z_extent_mm", "native_spacing_mm", "log_n_levels", "anchor_fallback",
)
N_SCALARS = len(SCALAR_NAMES)

# --- runtime paths (DrivenData container) ----------------------------------
RUNTIME_DATA_ROOT = Path(os.environ.get("DATSC_DATA_ROOT", "/code_execution/data"))
RUNTIME_NIFTI_DIR = RUNTIME_DATA_ROOT / "niftis"
RUNTIME_SUBMISSION_FORMAT = RUNTIME_DATA_ROOT / "submission_format.csv"
RUNTIME_OUTPUT_CSV = Path(os.environ.get("DATSC_OUTPUT_CSV", "submission.csv"))

# --- training-time locations ---------------------------------------------
KAGGLE_DATA_DIR = Path("/kaggle/input/datasets/harshdipsaha/dat-parkinsons-challenge-data")
LOCAL_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_PRIOR = 747 / 1362         # training prevalence of is_pathologic == 1


@dataclass(frozen=True)
class DataDir:
    root: Path
    labels_csv: Path
    niftis_dir: Path


def _count_niftis(d: Path) -> int:
    return sum(1 for p in d.iterdir() if p.name.endswith((".nii", ".nii.gz"))) if d.is_dir() else 0


def find_data_dir() -> DataDir:
    """Locate labels csv + nifti folder. Order: $DATSC_DATA_DIR, Kaggle mount, ./data."""
    candidates = [Path(os.environ["DATSC_DATA_DIR"])] if "DATSC_DATA_DIR" in os.environ else []
    candidates += [KAGGLE_DATA_DIR, LOCAL_DATA_DIR]
    for root in candidates:
        if not root.is_dir():
            continue
        csvs = sorted(root.glob("train_labels*.csv"))
        if not csvs:
            continue
        best, best_n = None, 0
        for d in [root] + [p for p in root.iterdir() if p.is_dir()]:
            n = _count_niftis(d)
            if n > best_n:
                best, best_n = d, n
        if best is None:
            continue
        return DataDir(root=root, labels_csv=csvs[0], niftis_dir=best)
    raise FileNotFoundError("No data dir found; set DATSC_DATA_DIR")


def find_nifti_path(niftis_dir: Path, uid: str) -> Path | None:
    for ext in (".nii.gz", ".nii"):
        p = niftis_dir / f"{uid}{ext}"
        if p.exists():
            return p
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add datsc/config.py tests/test_config.py
git commit -m "feat(config): constants and data-path discovery"
```

---

### Task 2: `io.py` — load NIfTI → RAS → 3 mm isotropic

**Files:**
- Create: `datsc/io.py`
- Test: `tests/test_io.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_io.py
import numpy as np
from datsc import io


def test_load_canonical_returns_float32_and_spacing(synth_nifti):
    p = synth_nifti(shape=(64, 64, 40), spacing=(2.46, 2.46, 2.46))
    vol, spacing, native_shape = io.load_canonical(p)
    assert vol.dtype == np.float32
    assert vol.shape == (64, 64, 40)
    assert native_shape == (64, 64, 40)
    assert np.allclose(spacing, (2.46, 2.46, 2.46))


def test_resample_iso_changes_shape_by_spacing_ratio(synth):
    vol, sp = synth(shape=(60, 60, 30), spacing=(3.9, 3.9, 3.9))
    out = io.resample_iso(vol, sp, target_mm=3.0)
    assert out.shape == (78, 78, 39)
    assert out.dtype == np.float32
    assert out.max() > 0


def test_load_and_resample_end_to_end(synth_nifti):
    p = synth_nifti(shape=(48, 48, 24), spacing=(2.0, 2.0, 4.0))
    vol, sp = io.load_iso(p)
    assert vol.shape == (32, 32, 32)
    assert sp == (2.0, 2.0, 4.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_io.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# datsc/io.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_io.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add datsc/io.py tests/test_io.py
git commit -m "feat(io): canonical RAS load and isotropic resample"
```

---

### Task 3: `preprocess.py` part 1 — brain mask and p75 normalisation

**Files:**
- Create: `datsc/preprocess.py`
- Test: `tests/test_preprocess.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_preprocess.py
import numpy as np
from datsc import io, preprocess as pp


def _iso(synth, **kw):
    vol, sp = synth(**kw)
    return io.resample_iso(vol, sp), sp


def test_brain_mask_is_single_component_covering_head(synth):
    vol, _ = _iso(synth)
    m = pp.brain_mask(vol)
    assert m.dtype == bool
    frac = m.mean()
    assert 0.15 < frac < 0.6            # ellipsoid head, not everything, not nothing
    assert pp._largest_component(m).sum() == m.sum()


def test_brain_mask_on_ultra_low_count_volume(synth):
    vol, _ = _iso(synth, levels=8)      # 3-bit regime from the EDA
    m = pp.brain_mask(vol)
    assert 0.1 < m.mean() < 0.7


def test_normalize_divides_by_p75_of_mask(synth):
    vol, _ = _iso(synth)
    m = pp.brain_mask(vol)
    out, p75 = pp.normalize_p75(vol, m)
    assert p75 > 0
    assert np.isclose(np.percentile(out[m], 75), 1.0, atol=1e-4)


def test_normalize_falls_back_when_p75_is_zero():
    vol = np.zeros((20, 20, 20), np.float32)
    vol[10, 10, 10] = 5.0
    m = np.ones_like(vol, bool)
    out, p75 = pp.normalize_p75(vol, m)
    assert p75 > 0 and np.isfinite(out).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_preprocess.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation (part 1)**

```python
# datsc/preprocess.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_preprocess.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add datsc/preprocess.py tests/test_preprocess.py
git commit -m "feat(preprocess): brain mask and p75 normalisation"
```

---

### Task 4: `preprocess.py` part 2 — hotspot anchor, crop, scalars, `preprocess()`

**Files:**
- Modify: `datsc/preprocess.py`
- Test: `tests/test_preprocess.py` (append)

- [ ] **Step 1: Append the failing tests**

```python
# tests/test_preprocess.py (append)
from datsc.config import CROP_SHAPE, N_SCALARS


def test_anchor_lands_between_the_two_blobs(synth):
    vol, _ = _iso(synth, shape=(96, 96, 60), spacing=(2.46,) * 3)
    m = pp.brain_mask(vol)
    norm, _ = pp.normalize_p75(vol, m)
    (cx, cy, cz), fallback = pp.find_anchor(norm, m)
    assert not fallback
    c = np.array(norm.shape) / 2
    assert abs(cx - c[0]) < 3 and abs(cy - c[1]) < 3 and abs(cz - c[2]) < 3


def test_anchor_fallback_triggers_on_flat_volume():
    vol = np.ones((60, 60, 40), np.float32)
    m = np.ones_like(vol, bool)
    (cx, cy, cz), fallback = pp.find_anchor(vol, m)
    assert fallback
    assert 0 <= cx < 60 and 0 <= cy < 60 and 0 <= cz < 40


def test_crop_shape_and_zero_padding_when_center_near_edge():
    vol = np.ones((30, 30, 20), np.float32)
    crop = pp.crop_at(vol, (2, 2, 2))
    assert crop.shape == CROP_SHAPE
    assert crop.dtype == np.float32
    assert crop[0, 0, 0] == 0.0 and crop[36, 36, 24] == 1.0


def test_preprocess_end_to_end_all_regimes(synth_nifti):
    for kw in ({}, {"levels": 8}, {"shape": (128, 128, 33), "spacing": (3.9,) * 3},
               {"shape": (128, 128, 128), "spacing": (2.3,) * 3, "scale": 60000.0}):
        p = synth_nifti(**kw)
        out = pp.preprocess(p)
        assert out["crop"].shape == CROP_SHAPE and out["crop"].dtype == np.float16
        assert out["crop_com"].shape == CROP_SHAPE
        assert out["scalars"].shape == (N_SCALARS,) and np.isfinite(out["scalars"]).all()
        assert out["crop"].max() > 0


def test_preprocess_is_deterministic(synth_nifti):
    p = synth_nifti()
    a, b = pp.preprocess(p), pp.preprocess(p)
    assert np.array_equal(a["crop"], b["crop"]) and np.array_equal(a["scalars"], b["scalars"])
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `python -m pytest tests/test_preprocess.py -v`
Expected: 4 passed, 5 failed with `AttributeError: module 'datsc.preprocess' has no attribute 'find_anchor'`

- [ ] **Step 3: Append the implementation**

```python
# datsc/preprocess.py (append)

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_preprocess.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add datsc/preprocess.py tests/test_preprocess.py
git commit -m "feat(preprocess): hotspot anchor with fallback, crop, scalars, preprocess()"
```

---

### Task 5: `features.py` — hotspot radiomics for the LR arm

**Files:**
- Create: `datsc/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features.py
import numpy as np
from datsc import features as ft, preprocess as pp


def test_feature_vector_shape_and_finite(synth_nifti):
    out = pp.preprocess(synth_nifti())
    f = ft.hotspot_radiomics(out["crop"].astype(np.float32))
    assert f.shape == (ft.N_RADIOMICS,) and np.isfinite(f).all()


def test_asymmetric_striatum_raises_asymmetry_feature(synth_nifti):
    sym = ft.hotspot_radiomics(pp.preprocess(synth_nifti("s1"))["crop"].astype(np.float32))
    asym = ft.hotspot_radiomics(pp.preprocess(synth_nifti("s2", asym=0.6))["crop"].astype(np.float32))
    i = ft.RADIOMICS_NAMES.index("asym_mean")
    assert asym[i] > sym[i] + 0.1


def test_dim_striatum_lowers_min_side_ratio(synth_nifti):
    hi = ft.hotspot_radiomics(pp.preprocess(synth_nifti("h", striatum_amp=4.0))["crop"].astype(np.float32))
    lo = ft.hotspot_radiomics(pp.preprocess(synth_nifti("l", striatum_amp=1.5))["crop"].astype(np.float32))
    i = ft.RADIOMICS_NAMES.index("min_side_p90")
    assert lo[i] < hi[i]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# datsc/features.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_features.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add datsc/features.py tests/test_features.py
git commit -m "feat(features): hotspot radiomics in the anchored frame"
```

---

### Task 6: `scripts/cache_crops.py` + `scripts/qc_grid.py` — build the training cache and eyeball it

**Files:**
- Create: `scripts/__init__.py` (empty), `scripts/cache_crops.py`, `scripts/qc_grid.py`
- Test: `tests/test_cache_crops.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cache_crops.py
import numpy as np
import pandas as pd
from tests.conftest import make_synthetic_volume, write_nifti
from scripts.cache_crops import build_cache


def test_build_cache_writes_npz_and_meta(tmp_path):
    nd = tmp_path / "niftis"; nd.mkdir()
    rows = []
    for i, asym in enumerate((0.0, 0.5, 0.0)):
        uid = f"u{i:07d}"
        vol, sp = make_synthetic_volume(asym=asym, seed=i)
        write_nifti(nd / f"{uid}.nii.gz", vol, sp)
        rows.append({"uid": uid, "is_pathologic": float(asym > 0)})
    labels = tmp_path / "train_labels.csv"
    pd.DataFrame(rows).to_csv(labels, index=False)
    out = tmp_path / "cache"
    meta = build_cache(labels, nd, out, n_workers=1)
    assert len(meta) == 3 and (out / "meta.csv").exists()
    z = np.load(out / "u0000000.npz")
    assert z["crop"].shape == (72, 72, 48) and z["scalars"].shape == (8,) and z["radiomics"].shape[0] == 14
    assert set(meta.columns) >= {"uid", "is_pathologic", "fallback", "native_spacing_mm", "ok"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cache_crops.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts'`

- [ ] **Step 3: Write the cache script**

```python
# scripts/cache_crops.py
"""Preprocess every training volume once → cache/{uid}.npz + cache/meta.csv.

Usage: python scripts/cache_crops.py [--out cache] [--workers 10] [--limit N]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from datsc.config import find_data_dir, find_nifti_path
from datsc.features import hotspot_radiomics
from datsc.preprocess import preprocess


def _one(args):
    uid, path, out_dir = args
    try:
        r = preprocess(path)
        rad = hotspot_radiomics(r["crop"].astype(np.float32))
        np.savez_compressed(out_dir / f"{uid}.npz", crop=r["crop"], crop_com=r["crop_com"],
                            scalars=r["scalars"], radiomics=rad, anchor=r["anchor"])
        return {"uid": uid, "ok": True, "fallback": r["fallback"], "p75": r["p75"],
                "native_spacing_mm": float(r["scalars"][5]), "z_extent_mm": float(r["scalars"][4]),
                "n_levels": float(np.exp(r["scalars"][6])), "err": ""}
    except Exception as e:  # noqa: BLE001
        return {"uid": uid, "ok": False, "fallback": True, "p75": np.nan, "native_spacing_mm": np.nan,
                "z_extent_mm": np.nan, "n_levels": np.nan, "err": repr(e)[:200]}


def build_cache(labels_csv: Path, niftis_dir: Path, out_dir: Path, n_workers: int = 8, limit: int | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(labels_csv, dtype={"uid": str})
    if limit:
        labels = labels.head(limit)
    jobs = []
    for uid in labels["uid"]:
        p = find_nifti_path(niftis_dir, uid)
        if p is None:
            print(f"missing file for uid (skipping): {uid}")
            continue
        jobs.append((uid, p, out_dir))
    t0 = time.time()
    if n_workers <= 1:
        results = [_one(j) for j in jobs]
    else:
        with get_context("spawn").Pool(n_workers) as pool:
            results = list(pool.imap_unordered(_one, jobs, chunksize=4))
    meta = pd.DataFrame(results).merge(labels, on="uid", how="left")
    meta.to_csv(out_dir / "meta.csv", index=False)
    print(f"cached {meta['ok'].sum()}/{len(meta)} in {time.time()-t0:.0f}s; "
          f"fallback anchors: {int(meta['fallback'].sum())}; failures: {int((~meta['ok']).sum())}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cache")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    d = find_data_dir()
    print(f"data: {d.root}\nlabels: {d.labels_csv}\nniftis: {d.niftis_dir}")
    build_cache(d.labels_csv, d.niftis_dir, Path(a.out), a.workers, a.limit)
```

- [ ] **Step 4: Write the QC grid script**

```python
# scripts/qc_grid.py
"""PNG grids of cached crops for eyeballing. Usage: python scripts/qc_grid.py --cache cache --out qc"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def grid(cache: Path, uids, title, out_png: Path, ncol=6):
    n = len(uids)
    if n == 0:
        return
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.2 * ncol, 2.4 * nrow))
    for ax, uid in zip(np.ravel(axes), list(uids) + [None] * (nrow * ncol - n)):
        ax.axis("off")
        if uid is None:
            continue
        z = np.load(cache / f"{uid}.npz")
        crop = z["crop"].astype(np.float32)
        ax.imshow(crop[:, :, crop.shape[2] // 2].T, cmap="hot", origin="lower", vmin=0, vmax=max(crop.max(), 1e-3))
        ax.set_title(uid, fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="qc")
    a = ap.parse_args()
    cache, out = Path(a.cache), Path(a.out)
    out.mkdir(exist_ok=True)
    meta = pd.read_csv(cache / "meta.csv", dtype={"uid": str})
    ok = meta[meta["ok"]]
    rng = np.random.default_rng(0)
    grid(cache, rng.choice(ok["uid"], min(30, len(ok)), replace=False), "random", out / "random.png")
    grid(cache, ok[ok["n_levels"] < 12]["uid"], "ultra-low grey levels (<12)", out / "ultralow.png")
    grid(cache, ok[ok["z_extent_mm"] < 120]["uid"], "short z extent (<120mm)", out / "short_z.png")
    grid(cache, ok[ok["fallback"]]["uid"], "anchor fallback", out / "fallback.png")
    print(meta[["ok", "fallback"]].agg(["sum", "mean"]))
    print("ultra-low:", int((ok["n_levels"] < 12).sum()), "short-z:", int((ok["z_extent_mm"] < 120).sum()))
```

- [ ] **Step 5: Run the unit test**

Run: `python -m pytest tests/test_cache_crops.py -v`
Expected: 1 passed

- [ ] **Step 6: Extract the real data locally and build the full cache (Sat morning, ~10–20 min on 12 CPUs)**

```bash
cd /h/augsepthacks/parkinsonnn
mkdir -p data/niftis && unzip -q -o data/niftis_utCGpHE.zip -d data/niftis
ls data/niftis | head -3          # confirm files land directly here (if nested, adjust: find_data_dir picks the dir with most niftis)
python scripts/cache_crops.py --out cache --workers 10
python scripts/qc_grid.py --cache cache --out qc
```
Expected: `cached 1362/1362 … failures: 0`. Fallback anchors ideally < 5%.

- [ ] **Step 7: LOOK at the QC PNGs (non-negotiable)**

Open `qc/random.png`, `qc/ultralow.png`, `qc/short_z.png`, `qc/fallback.png`. The two striatal blobs must sit near the centre of every crop. If ultra-low or short-z crops are systematically off-centre, fix `find_anchor` (thresholds/smoothing) *before* Task 8; do not proceed with a broken anchor.

- [ ] **Step 8: Commit**

```bash
git add scripts/__init__.py scripts/cache_crops.py scripts/qc_grid.py tests/test_cache_crops.py
git commit -m "feat(scripts): crop cache builder and QC grids"
```

---

### Task 7: `folds.py` — repeat-patient embedding audit + stratified group folds

**Files:**
- Create: `datsc/folds.py`, `scripts/make_folds.py`
- Test: `tests/test_folds.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_folds.py
import numpy as np
from datsc import folds


def test_near_duplicates_get_same_group():
    rng = np.random.default_rng(0)
    base = rng.random((4, 72, 72, 48)).astype(np.float32)
    crops = np.concatenate([base, base[:1] + 1e-3 * rng.random((1, 72, 72, 48)).astype(np.float32)])
    emb = folds.embed(crops)
    assert emb.shape == (5, 16 * 16 * 16)
    g = folds.group_near_duplicates(emb, thresh=0.98)
    assert g[0] == g[4] and len(set(g[:4])) == 4


def test_stratified_group_folds_keep_groups_together_and_balance():
    y = np.array([0, 1] * 50)
    groups = np.arange(100); groups[1] = 0   # samples 0 and 1 share a group
    f = folds.assign_folds(y, groups, n_splits=5, seed=0)
    assert f.shape == (100,) and set(f) == set(range(5))
    assert f[0] == f[1]
    for k in range(5):
        assert 0.3 < y[f == k].mean() < 0.7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_folds.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation and script**

```python
# datsc/folds.py
"""Repeat-patient audit via coarse embeddings, then stratified group K-fold."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom
from sklearn.model_selection import StratifiedGroupKFold


def embed(crops: np.ndarray, size: int = 16) -> np.ndarray:
    """crops [N, X, Y, Z] → L2-normalised [N, size^3] embeddings."""
    out = []
    for c in crops:
        f = [size / s for s in c.shape]
        e = zoom(c.astype(np.float32), f, order=1).ravel()
        e = e - e.mean()
        out.append(e / (np.linalg.norm(e) + 1e-9))
    return np.stack(out)


def group_near_duplicates(emb: np.ndarray, thresh: float = 0.98) -> np.ndarray:
    n = len(emb)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    sim = emb @ emb.T
    ii, jj = np.where(np.triu(sim, k=1) > thresh)
    for i, j in zip(ii, jj):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    roots = np.array([find(i) for i in range(n)])
    _, groups = np.unique(roots, return_inverse=True)
    return groups


def assign_folds(y: np.ndarray, groups: np.ndarray, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.full(len(y), -1, dtype=int)
    for k, (_, val_idx) in enumerate(skf.split(np.zeros(len(y)), y, groups)):
        folds[val_idx] = k
    assert (folds >= 0).all()
    return folds
```

```python
# scripts/make_folds.py
"""cache/meta.csv + cache/*.npz → cache/folds.csv (uid, is_pathologic, group, fold)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd

from datsc.folds import assign_folds, embed, group_near_duplicates

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--thresh", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cache = Path(a.cache)
    meta = pd.read_csv(cache / "meta.csv", dtype={"uid": str})
    meta = meta[meta["ok"]].reset_index(drop=True)
    crops = np.stack([np.load(cache / f"{u}.npz")["crop"] for u in meta["uid"]])
    groups = group_near_duplicates(embed(crops), a.thresh)
    n_dup = len(groups) - len(np.unique(groups))
    print(f"near-duplicate pairs collapsed into groups: {n_dup}")
    meta["group"] = groups
    meta["fold"] = assign_folds(meta["is_pathologic"].values.astype(int), groups, 5, a.seed)
    meta[["uid", "is_pathologic", "group", "fold", "native_spacing_mm", "fallback"]].to_csv(cache / "folds.csv", index=False)
    print(meta.groupby("fold")["is_pathologic"].agg(["count", "mean"]))
```

- [ ] **Step 4: Run tests, then build real folds**

Run: `python -m pytest tests/test_folds.py -v` → 2 passed
Run: `python scripts/make_folds.py --cache cache`
Expected: five folds of ~272 with prevalence ≈0.55 each. Note the printed duplicate count in your log; if it is large (>30), open a few pairs in the QC style and confirm they are genuine repeats before continuing.

- [ ] **Step 5: Commit**

```bash
git add datsc/folds.py scripts/make_folds.py tests/test_folds.py
git commit -m "feat(folds): repeat-patient audit and stratified group folds"
```

---

### Task 8: `lrmodel.py` + `scripts/alignment_gate.py` — the L2 logistic arm and the go/no-go gate

**Files:**
- Create: `datsc/lrmodel.py`, `scripts/alignment_gate.py`
- Test: `tests/test_lrmodel.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lrmodel.py
import json
import numpy as np
from datsc.lrmodel import LRModel


def test_fit_predict_roundtrip_json(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 6)); y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    m = LRModel(C=1.0).fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (300,) and ((p > 0) & (p < 1)).all()
    assert ((p > 0.5) == y).mean() > 0.9
    path = tmp_path / "lr.json"
    m.save(path)
    m2 = LRModel.load(path)
    assert np.allclose(m2.predict_proba(X), p)
    assert json.load(open(path))["names"] == m.names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_lrmodel.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the model and the gate script**

```python
# datsc/lrmodel.py
"""L2 logistic regression with standardisation, serialised to JSON (no pickle in the submission)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from .config import SCALAR_NAMES
from .features import RADIOMICS_NAMES

DEFAULT_NAMES = list(RADIOMICS_NAMES) + list(SCALAR_NAMES)


class LRModel:
    def __init__(self, C: float = 1.0, names=None):
        self.C = C
        self.names = list(names) if names is not None else DEFAULT_NAMES
        self.mean = self.scale = self.coef = None
        self.intercept = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, np.float64)
        self.mean = X.mean(0); self.scale = X.std(0) + 1e-6
        clf = LogisticRegression(C=self.C, max_iter=2000).fit((X - self.mean) / self.scale, y)
        self.coef = clf.coef_.ravel(); self.intercept = float(clf.intercept_[0])
        return self

    def decision(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, np.float64) - self.mean) / self.scale
        return Z @ self.coef + self.intercept

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.decision(X)))

    def save(self, path):
        Path(path).write_text(json.dumps({"C": self.C, "names": self.names, "mean": self.mean.tolist(),
                                          "scale": self.scale.tolist(), "coef": self.coef.tolist(),
                                          "intercept": self.intercept}))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        m = cls(C=d["C"], names=d["names"])
        m.mean, m.scale = np.array(d["mean"]), np.array(d["scale"])
        m.coef, m.intercept = np.array(d["coef"]), float(d["intercept"])
        return m


def feature_matrix(cache_dir: Path, uids) -> np.ndarray:
    rows = []
    for u in uids:
        z = np.load(Path(cache_dir) / f"{u}.npz")
        rows.append(np.concatenate([z["radiomics"], z["scalars"]]))
    return np.stack(rows).astype(np.float32)
```

```python
# scripts/alignment_gate.py
"""OOF logistic regression on hotspot radiomics + scalars. PASS iff OOF AUROC > 0.90.
Also writes artifacts/lr.json (fit on all data) and cache/oof_lr.csv."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from datsc.lrmodel import LRModel, feature_matrix

GATE_AUROC = 0.90

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--C", type=float, default=0.3)
    a = ap.parse_args()
    cache, art = Path(a.cache), Path(a.artifacts)
    art.mkdir(exist_ok=True)
    f = pd.read_csv(cache / "folds.csv", dtype={"uid": str})
    X = feature_matrix(cache, f["uid"]); y = f["is_pathologic"].values.astype(int)
    oof = np.zeros(len(y))
    for k in sorted(f["fold"].unique()):
        tr, va = (f["fold"] != k).values, (f["fold"] == k).values
        oof[va] = LRModel(C=a.C).fit(X[tr], y[tr]).predict_proba(X[va])
    auc, ll = roc_auc_score(y, oof), log_loss(y, np.clip(oof, 1e-4, 1 - 1e-4))
    pd.DataFrame({"uid": f["uid"], "y": y, "p_lr": oof}).to_csv(cache / "oof_lr.csv", index=False)
    LRModel(C=a.C).fit(X, y).save(art / "lr.json")
    print(f"OOF AUROC={auc:.4f}  OOF logloss={ll:.4f}  (prior-only logloss={log_loss(y, np.full(len(y), y.mean())):.4f})")
    print("GATE:", "PASS" if auc > GATE_AUROC else "FAIL - fix the anchor before training any CNN")
```

- [ ] **Step 4: Run the unit test, then the real gate**

Run: `python -m pytest tests/test_lrmodel.py -v` → 1 passed
Run: `python scripts/alignment_gate.py --cache cache`
Expected: `GATE: PASS` with OOF AUROC > 0.90. **If FAIL, stop here.** Look at `qc/*.png`, adjust `ANCHOR_THRESH` / `SMOOTH_MM` / `STRIATAL_HALF_MM` in `config.py`, rebuild the cache, re-run. Record the final AUROC and log loss in your log; they are the denominator for every later gate.

- [ ] **Step 5: Commit**

```bash
git add datsc/lrmodel.py scripts/alignment_gate.py tests/test_lrmodel.py
git commit -m "feat(lr): JSON-serialised logistic arm and alignment gate"
```

---

### Task 9: `model.py` — SmallResNet3D with scalar head

**Files:**
- Create: `datsc/model.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py
import torch
from datsc.model import SmallResNet3D, count_params


def test_forward_shape_and_param_budget():
    m = SmallResNet3D(n_scalars=8)
    x = torch.zeros(2, 1, 72, 72, 48); s = torch.zeros(2, 8)
    out = m(x, s)
    assert out.shape == (2,)
    n = count_params(m)
    assert 0.6e6 < n < 1.3e6, n


def test_flip_invariance_is_not_assumed():
    torch.manual_seed(0)
    m = SmallResNet3D(n_scalars=8).eval()
    x = torch.randn(1, 1, 72, 72, 48); s = torch.zeros(1, 8)
    with torch.no_grad():
        a, b = m(x, s), m(x.flip(dims=[2]), s)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# datsc/model.py
"""Compact 3D ResNet (~0.9M params) with a scalar-feature head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm3d(cout)
        self.c2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm3d(cout)
        self.skip = None
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False), nn.BatchNorm3d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(y + (x if self.skip is None else self.skip(x)))


class SmallResNet3D(nn.Module):
    def __init__(self, n_scalars: int = 8, widths=(16, 32, 64, 128), dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, widths[0], 3, 1, 1, bias=False), nn.BatchNorm3d(widths[0]), nn.ReLU())
        layers, cin = [], widths[0]
        for i, w in enumerate(widths):
            layers.append(Block(cin, w, stride=1 if i == 0 else 2)); cin = w
        self.layers = nn.Sequential(*layers)
        self.scalar_bn = nn.BatchNorm1d(n_scalars)
        self.head = nn.Sequential(nn.Linear(cin + n_scalars, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x, scalars):
        h = self.layers(self.stem(x))
        h = F.adaptive_avg_pool3d(h, 1).flatten(1)
        h = torch.cat([h, self.scalar_bn(scalars)], dim=1)
        return self.head(h).squeeze(1)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_model.py -v`
Expected: 2 passed (print `count_params` once; expect ≈0.9M)

- [ ] **Step 5: Commit**

```bash
git add datsc/model.py tests/test_model.py
git commit -m "feat(model): SmallResNet3D with scalar head"
```

---

### Task 10: `augment.py` — tensor-side augmentations

**Files:**
- Create: `datsc/augment.py`
- Test: `tests/test_augment.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_augment.py
import torch
from datsc import augment as ag


def _x():
    torch.manual_seed(0)
    return torch.rand(3, 1, 72, 72, 48)


def test_each_op_preserves_shape_and_is_finite():
    x = _x()
    for fn in (ag.random_blur, ag.random_noise, ag.random_quantize, ag.random_scale_jitter, ag.random_affine, ag.random_flip):
        y = fn(x.clone())
        assert y.shape == x.shape and torch.isfinite(y).all(), fn.__name__


def test_quantize_reduces_distinct_levels():
    x = _x()
    y = ag.quantize(x, k=8)
    assert torch.unique(y[0]).numel() <= 9


def test_flip_mirrors_x_axis():
    x = _x()
    y = ag.random_flip(x, p=1.0)
    assert torch.equal(y, x.flip(dims=[2]))


def test_train_augment_composes():
    y = ag.train_augment(_x())
    assert y.shape == (3, 1, 72, 72, 48)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_augment.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# datsc/augment.py
"""Augmentations applied to [B,1,X,Y,Z] tensors on the training device (GPU-side on Kaggle).

Physically motivated for multi-centre SPECT: blur (resolution differences), noise (counts),
quantisation (the ultra-low-count regime), reference-scale jitter (p75 normalisation error),
small affine (positioning), L/R mirror (the label is symmetric under mirroring).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .config import SPACING_MM


def _gauss1d(sigma: float, device):
    r = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / max(sigma, 1e-3)) ** 2)
    return k / k.sum()


def blur(x: torch.Tensor, sigma_vox: float) -> torch.Tensor:
    if sigma_vox < 0.05:
        return x
    k = _gauss1d(sigma_vox, x.device)
    n = k.numel(); pad = n // 2
    for dim in (2, 3, 4):
        shape = [1, 1, 1, 1, 1]; shape[dim] = n
        kk = k.view(*shape)
        p = [0] * 6; p[2 * (4 - dim)] = pad; p[2 * (4 - dim) + 1] = pad
        x = F.conv3d(F.pad(x, p, mode="replicate"), kk)
    return x


def random_blur(x, max_sigma_mm: float = 4.0):
    out = torch.empty_like(x)
    for i in range(x.shape[0]):
        s = float(torch.rand(1)) * max_sigma_mm / SPACING_MM
        out[i:i + 1] = blur(x[i:i + 1], s)
    return out


def random_noise(x, max_frac: float = 0.15):
    std = x.flatten(1).std(dim=1).view(-1, 1, 1, 1, 1)
    frac = torch.rand(x.shape[0], 1, 1, 1, 1, device=x.device) * max_frac
    return (x + torch.randn_like(x) * std * frac).clamp_min(0)


def quantize(x, k: int):
    m = x.flatten(1).amax(dim=1).view(-1, 1, 1, 1, 1).clamp_min(1e-6)
    return torch.round(x / m * (k - 1)) / (k - 1) * m


def random_quantize(x, levels=(8, 16, 64, 256), p: float = 0.5):
    out = x.clone()
    for i in range(x.shape[0]):
        if float(torch.rand(1)) < p:
            out[i:i + 1] = quantize(x[i:i + 1], int(levels[int(torch.randint(len(levels), (1,)))]))
    return out


def random_scale_jitter(x, frac: float = 0.2):
    s = 1.0 + (torch.rand(x.shape[0], 1, 1, 1, 1, device=x.device) * 2 - 1) * frac
    return x * s


def random_affine(x, max_deg: float = 10.0, max_scale: float = 0.10, max_shift_vox: float = 3.0):
    B = x.shape[0]
    theta = torch.zeros(B, 3, 4, device=x.device)
    for i in range(B):
        a = torch.deg2rad((torch.rand(3) * 2 - 1) * max_deg)
        cx, cy, cz = torch.cos(a); sx, sy, sz = torch.sin(a)
        Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        S = torch.diag(1.0 + (torch.rand(3) * 2 - 1) * max_scale)
        M = Rz @ Ry @ Rx @ S
        shift = (torch.rand(3) * 2 - 1) * max_shift_vox * 2.0 / torch.tensor(x.shape[2:], dtype=torch.float32)
        theta[i, :, :3] = M.to(x.device); theta[i, :, 3] = shift.to(x.device)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def random_flip(x, p: float = 0.5):
    out = x.clone()
    for i in range(x.shape[0]):
        if float(torch.rand(1)) < p:
            out[i] = x[i].flip(dims=[1])   # dim 1 of [1,X,Y,Z] is X
    return out


def train_augment(x: torch.Tensor) -> torch.Tensor:
    x = random_flip(x)
    x = random_affine(x)
    x = random_blur(x)
    x = random_scale_jitter(x)
    x = random_quantize(x)
    x = random_noise(x)
    return x
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_augment.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add datsc/augment.py tests/test_augment.py
git commit -m "feat(augment): blur/noise/quantise/jitter/affine/flip on tensors"
```

---

### Task 11: `train.py` + `scripts/train_cnn.py` — one fold×seed run

**Files:**
- Create: `datsc/train.py`, `scripts/train_cnn.py`
- Test: `tests/test_train.py`

- [ ] **Step 1: Write the failing test (tiny synthetic cache, 1 epoch, CPU)**

```python
# tests/test_train.py
import json
import numpy as np
import pandas as pd
from datsc.train import train_fold


def _tiny_cache(tmp_path, n=12):
    cache = tmp_path / "cache"; cache.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        y = i % 2
        crop = (rng.random((72, 72, 48)) + y * 0.5).astype(np.float16)
        np.savez(cache / f"u{i}.npz", crop=crop, crop_com=crop, scalars=rng.random(8).astype(np.float32),
                 radiomics=rng.random(14).astype(np.float32), anchor=np.zeros(3, np.float32))
        rows.append({"uid": f"u{i}", "is_pathologic": float(y), "group": i, "fold": i % 3})
    pd.DataFrame(rows).to_csv(cache / "folds.csv", index=False)
    return cache


def test_train_fold_writes_weights_and_oof(tmp_path):
    cache = _tiny_cache(tmp_path)
    out = tmp_path / "art"
    res = train_fold(cache, fold=0, seed=1, out_dir=out, epochs=1, batch_size=4, device="cpu", num_workers=0)
    assert (out / "cnn_f0_s1.pt").exists()
    oof = pd.read_csv(out / "oof_f0_s1.csv")
    assert set(oof.columns) == {"uid", "y", "p"} and len(oof) == 4
    assert 0 <= res["best_val_logloss"] < 5
    meta = json.load(open(out / "cnn_f0_s1.json"))
    assert meta["fold"] == 0 and meta["seed"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the trainer**

```python
# datsc/train.py
"""Train one (fold, seed) of SmallResNet3D from the crop cache; save state_dict + OOF predictions."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .augment import train_augment
from .config import N_SCALARS
from .model import SmallResNet3D


class CropDataset(Dataset):
    def __init__(self, cache: Path, uids, labels, key: str = "crop"):
        self.cache, self.uids, self.labels, self.key = Path(cache), list(uids), np.asarray(labels, np.float32), key

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        z = np.load(self.cache / f"{self.uids[i]}.npz")
        x = torch.from_numpy(z[self.key].astype(np.float32))[None]
        s = torch.from_numpy(z["scalars"].astype(np.float32))
        return x, s, torch.tensor(self.labels[i]), self.uids[i]


def _seed_all(seed: int):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device, tta_flip: bool = True):
    model.eval(); ps, uids = [], []
    for x, s, _, u in loader:
        x, s = x.to(device), s.to(device)
        logit = model(x, s)
        if tta_flip:
            logit = 0.5 * (logit + model(x.flip(dims=[2]), s))
        ps.append(torch.sigmoid(logit).float().cpu().numpy()); uids += list(u)
    return np.concatenate(ps), uids


def train_fold(cache, fold: int, seed: int, out_dir, epochs: int = 40, batch_size: int = 16, lr: float = 1e-3,
               weight_decay: float = 1e-4, device: str | None = None, num_workers: int = 2,
               control: bool = False, amp: bool | None = None) -> dict:
    cache, out_dir = Path(cache), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    amp = (device.startswith("cuda")) if amp is None else amp
    _seed_all(seed)
    f = pd.read_csv(cache / "folds.csv", dtype={"uid": str})
    key = "crop_com" if control else "crop"
    tr, va = f[f["fold"] != fold], f[f["fold"] == fold]
    dl_tr = DataLoader(CropDataset(cache, tr["uid"], tr["is_pathologic"], key), batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, drop_last=len(tr) > batch_size, pin_memory=device.startswith("cuda"))
    dl_va = DataLoader(CropDataset(cache, va["uid"], va["is_pathologic"], key), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model = SmallResNet3D(n_scalars=N_SCALARS).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = max(1, epochs * len(dl_tr))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + math.cos(math.pi * min(t, steps) / steps)))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    tag = f"f{fold}_s{seed}" + ("_com" if control else "")
    best, best_state, hist, t0 = float("inf"), None, [], time.time()
    for ep in range(epochs):
        model.train(); tl, n = 0.0, 0
        for x, s, y, _ in dl_tr:
            x, s, y = x.to(device), s.to(device), y.to(device)
            with torch.no_grad():
                x = train_augment(x)
            with torch.autocast(device_type="cuda", enabled=amp):
                loss = F.binary_cross_entropy_with_logits(model(x, s), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += float(loss) * len(y); n += len(y)
        p, uids = predict(model, dl_va, device, tta_flip=False)
        yv = va.set_index("uid").loc[uids, "is_pathologic"].values
        vl = float(-np.mean(yv * np.log(np.clip(p, 1e-6, 1)) + (1 - yv) * np.log(np.clip(1 - p, 1e-6, 1))))
        hist.append({"epoch": ep, "train_loss": tl / max(n, 1), "val_logloss": vl, "sec": time.time() - t0})
        print(f"[{tag}] ep{ep:02d} train={tl/max(n,1):.4f} val_logloss={vl:.4f} ({time.time()-t0:.0f}s)")
        if vl < best:
            best, best_state = vl, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / f"cnn_{tag}.pt")
    p, uids = predict(model, dl_va, device, tta_flip=True)
    yv = va.set_index("uid").loc[uids, "is_pathologic"].values
    pd.DataFrame({"uid": uids, "y": yv, "p": p}).to_csv(out_dir / f"oof_{tag}.csv", index=False)
    meta = {"fold": fold, "seed": seed, "control": control, "epochs": epochs, "best_val_logloss": best,
            "history": hist, "device": device, "amp": amp}
    (out_dir / f"cnn_{tag}.json").write_text(json.dumps(meta))
    return meta
```

```python
# scripts/train_cnn.py
"""python scripts/train_cnn.py --fold 0 --seed 0 [--control] [--epochs 40] [--cache cache] [--out artifacts]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datsc.train import train_fold

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--control", action="store_true", help="train on plain CoM crops (baseline)")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    train_fold(a.cache, a.fold, a.seed, a.out, epochs=a.epochs, batch_size=a.batch_size,
               num_workers=a.workers, control=a.control)
```

- [ ] **Step 4: Run the unit test, then a 1-epoch CPU smoke on real data**

Run: `python -m pytest tests/test_train.py -v` → 1 passed
Run: `python scripts/train_cnn.py --fold 0 --seed 0 --epochs 1 --workers 0 --out artifacts_smoke`
Expected: one epoch completes on CPU (several minutes), writes `artifacts_smoke/cnn_f0_s0.pt` and `oof_f0_s0.csv`. Delete `artifacts_smoke/` afterwards. (Real training happens on Kaggle — Task 15.)

- [ ] **Step 5: Commit**

```bash
git add datsc/train.py scripts/train_cnn.py tests/test_train.py
git commit -m "feat(train): fold×seed trainer with augmentation, AMP, OOF export"
```

---

### Task 12: `calibrate.py` + `scripts/fit_calibration.py` — w, T, λ

**Files:**
- Create: `datsc/calibrate.py`, `scripts/fit_calibration.py`
- Test: `tests/test_calibrate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calibrate.py
import numpy as np
from datsc import calibrate as cb


def test_fit_temperature_undoes_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    logit = (y * 2 - 1) * 0.5 + rng.normal(0, 1.0, 2000)   # Bayes log-odds = 1.0*logit for this generator
    over = logit * 3.0                                      # over-confident by ~3x → optimal T ≈ 2.9
    T = cb.fit_temperature(over, y)
    assert 2.0 < T < 4.5
    assert cb.logloss(y, cb.sigmoid(over / T)) < cb.logloss(y, cb.sigmoid(over))


def test_fit_blend_weight_prefers_better_arm():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 1000)
    good = np.clip(y * 0.8 + 0.1 + rng.normal(0, 0.05, 1000), 0.01, 0.99)
    bad = np.clip(rng.random(1000), 0.01, 0.99)
    w = cb.fit_blend_weight(good, bad, y)
    assert w >= 0.8


def test_apply_and_json_roundtrip(tmp_path):
    c = cb.Calibration(w=0.7, T=1.5, lam=0.02, pi=0.55)
    p = c.apply(np.array([0.9, 0.1]), np.array([0.8, 0.2]))
    assert (p > 0).all() and (p < 1).all() and p[0] > p[1]
    c.save(tmp_path / "c.json")
    c2 = cb.Calibration.load(tmp_path / "c.json")
    assert c2 == c
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_calibrate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation and script**

```python
# datsc/calibrate.py
"""Exactly three fitted numbers: blend weight w, temperature T, prior-mix λ (plus the prior π)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

EPS = 1e-4


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def logloss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_blend_weight(p_cnn, p_lr, y, grid=np.linspace(0, 1, 21)):
    return float(min(grid, key=lambda w: logloss(y, w * p_cnn + (1 - w) * p_lr)))


def fit_temperature(z, y):
    res = minimize_scalar(lambda T: logloss(y, sigmoid(z / T)), bounds=(0.25, 10.0), method="bounded")
    return float(res.x)


@dataclass(frozen=True)
class Calibration:
    w: float
    T: float
    lam: float
    pi: float

    def apply(self, p_cnn, p_lr):
        p_cnn = np.asarray(p_cnn, float); p_lr = np.asarray(p_lr, float)
        blend = self.w * p_cnn + (1 - self.w) * p_lr
        cal = sigmoid(logit(blend) / self.T)
        return np.clip((1 - self.lam) * cal + self.lam * self.pi, EPS, 1 - EPS)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def fit(p_cnn, p_lr, y, lam: float, pi: float) -> Calibration:
    w = fit_blend_weight(p_cnn, p_lr, y)
    T = fit_temperature(logit(w * p_cnn + (1 - w) * p_lr), y)
    return Calibration(w=w, T=T, lam=lam, pi=pi)
```

```python
# scripts/fit_calibration.py
"""Average CNN OOF over seeds, blend with LR OOF, fit w & T, write artifacts/calib.json.
λ and π are inputs: λ from scripts/shift_diagnostic.py (default 0.02), π from the prior-only LB (default 0.548)."""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd

from datsc import calibrate as cb
from datsc.config import DEFAULT_PRIOR

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--lam", type=float, default=0.02)
    ap.add_argument("--pi", type=float, default=DEFAULT_PRIOR)
    a = ap.parse_args()
    art = Path(a.artifacts)
    files = [f for f in glob.glob(str(art / "oof_f*_s*.csv")) if "_com" not in f]
    assert files, "no CNN OOF files found"
    cnn = pd.concat([pd.read_csv(f, dtype={"uid": str}) for f in files]).groupby("uid").agg(y=("y", "first"), p_cnn=("p", "mean")).reset_index()
    lr = pd.read_csv(Path(a.cache) / "oof_lr.csv", dtype={"uid": str})[["uid", "p_lr"]]
    df = cnn.merge(lr, on="uid")
    y = df["y"].values.astype(float)
    print(f"n={len(df)}  seeds/fold~{len(files)//5}  logloss cnn={cb.logloss(y, df.p_cnn):.4f}  lr={cb.logloss(y, df.p_lr):.4f}")
    c = cb.fit(df["p_cnn"].values, df["p_lr"].values, y, lam=a.lam, pi=a.pi)
    final = c.apply(df["p_cnn"].values, df["p_lr"].values)
    print(f"w={c.w:.2f} T={c.T:.3f} lam={c.lam} pi={c.pi:.3f}  OOF logloss after calibration={cb.logloss(y, final):.4f}")
    c.save(art / "calib.json")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_calibrate.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add datsc/calibrate.py scripts/fit_calibration.py tests/test_calibrate.py
git commit -m "feat(calibrate): w/T/λ calibration with JSON artefact"
```

---

### Task 13: `infer.py` + `main.py` — stage-wise inference with a deadline ladder, `--selftest`, prior-only packer

**Files:**
- Create: `datsc/infer.py`, `main.py`, `scripts/make_prior_submission.py`
- Test: `tests/test_infer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_infer.py
import os, subprocess, sys
import numpy as np
import pandas as pd
from pathlib import Path
from tests.conftest import make_synthetic_volume, write_nifti


def _demo(tmp_path, n=4):
    d = tmp_path / "data"; (d / "niftis").mkdir(parents=True)
    uids = [f"t{i:07d}" for i in range(n)]
    for i, u in enumerate(uids):
        vol, sp = make_synthetic_volume(seed=i, levels=8 if i == 0 else None)
        write_nifti(d / "niftis" / f"{u}.nii.gz", vol, sp)
    pd.DataFrame({"uid": uids, "is_pathologic": 0.5}).to_csv(d / "submission_format.csv", index=False)
    return d, uids


def test_main_writes_prior_then_overwrites_with_predictions(tmp_path):
    d, uids = _demo(tmp_path)
    out = tmp_path / "submission.csv"
    env = {**os.environ, "DATSC_DATA_ROOT": str(d), "DATSC_OUTPUT_CSV": str(out),
           "DATSC_ARTIFACTS": str(tmp_path / "no_artifacts")}   # no models → prior-only path
    r = subprocess.run([sys.executable, "main.py"], env=env, capture_output=True, text=True, cwd=Path(__file__).parents[1])
    assert r.returncode == 0, r.stderr
    sub = pd.read_csv(out)
    assert list(sub.columns) == ["uid", "is_pathologic"] and list(sub["uid"]) == uids
    assert ((sub["is_pathologic"] > 0) & (sub["is_pathologic"] < 1)).all()
    for u in uids:
        assert u not in r.stdout   # never print test uids


def test_selftest_runs():
    r = subprocess.run([sys.executable, "main.py", "--selftest"], capture_output=True, text=True, cwd=Path(__file__).parents[1])
    assert r.returncode == 0, r.stderr
    assert "SELFTEST OK" in r.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_infer.py -v`
Expected: FAIL (`main.py` not found / non-zero exit)

- [ ] **Step 3: Write `infer.py`**

```python
# datsc/infer.py
"""Inference stages for main.py. Stage-wise: prior → LR → CNN ensemble, each overwriting submission.csv."""
from __future__ import annotations

import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutTimeout
from pathlib import Path

import numpy as np
import pandas as pd

from .calibrate import Calibration, EPS
from .config import CROP_SHAPE, N_SCALARS, find_nifti_path
from .features import N_RADIOMICS, hotspot_radiomics
from .lrmodel import LRModel
from .preprocess import preprocess

PER_VOLUME_TIMEOUT_S = 90
LADDER_SKIP_CNN_S = 130 * 60      # past this, stop preprocessing / loading more CNN checkpoints; keep what is written


def _worker_init():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
        os.environ[k] = "1"


def _prep(path):
    r = preprocess(path)
    return r["crop"], r["scalars"], hotspot_radiomics(r["crop"].astype(np.float32))


def write_submission(uids, probs, out_csv):
    pd.DataFrame({"uid": uids, "is_pathologic": np.clip(probs, EPS, 1 - EPS)}).to_csv(out_csv, index=False)


def preprocess_all(uids, nifti_dir: Path, n_workers: int, deadline_s: float, t0: float):
    """Returns (crops[N,X,Y,Z] fp16, scalars[N,8], radiomics[N,14], ok[N] bool)."""
    N = len(uids)
    crops = np.zeros((N,) + CROP_SHAPE, np.float16); sc = np.zeros((N, N_SCALARS), np.float32)
    rad = np.zeros((N, N_RADIOMICS), np.float32); ok = np.zeros(N, bool)
    paths = [find_nifti_path(nifti_dir, u) for u in uids]
    ex = ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init)
    futs = {ex.submit(_prep, p): i for i, p in enumerate(paths) if p is not None}
    for fut, i in futs.items():
        remaining = deadline_s - (time.time() - t0)
        if remaining <= 0:
            print("ladder: preprocessing deadline hit; cancelling remaining volumes")
            break
        try:
            c, s, r = fut.result(timeout=min(PER_VOLUME_TIMEOUT_S, remaining))
            crops[i], sc[i], rad[i], ok[i] = c, s, r, True
        except (Exception, FutTimeout):  # noqa: BLE001
            ok[i] = False
    # never block on a hung worker: drop queued work, don't wait for running ones (main.py ends with os._exit)
    ex.shutdown(wait=False, cancel_futures=True)
    print(f"preprocessed {int(ok.sum())}/{N} volumes in {time.time()-t0:.0f}s")
    return crops, sc, rad, ok


def run_lr(lr_json: Path, rad, sc, ok, pi):
    m = LRModel.load(lr_json)
    p = np.full(len(ok), pi)
    if ok.any():
        p[ok] = m.predict_proba(np.concatenate([rad[ok], sc[ok]], axis=1))
    return p


def run_cnn(weights: list[Path], crops, sc, ok, t0: float, batch_size: int = 32):
    import torch
    from .model import SmallResNet3D
    device = "cuda" if torch.cuda.is_available() else "cpu"
    idx = np.where(ok)[0]
    acc = np.zeros(len(idx)); used = 0
    x_all = torch.from_numpy(crops[idx].astype(np.float32))[:, None]
    s_all = torch.from_numpy(sc[idx])
    for wpath in weights:
        if time.time() - t0 > LADDER_SKIP_CNN_S:
            print("ladder: skipping remaining checkpoints"); break
        model = SmallResNet3D(n_scalars=N_SCALARS).to(device)
        model.load_state_dict(torch.load(wpath, map_location=device)); model.eval()
        out = []
        with torch.no_grad():
            for b in range(0, len(idx), batch_size):
                x = x_all[b:b + batch_size].to(device); s = s_all[b:b + batch_size].to(device)
                z = 0.5 * (model(x, s) + model(x.flip(dims=[2]), s))
                out.append(torch.sigmoid(z).float().cpu().numpy())
        acc += np.concatenate(out); used += 1
    print(f"cnn checkpoints used: {used}/{len(weights)}")
    if used == 0:
        return None
    p = np.full(len(ok), np.nan); p[idx] = acc / used
    return p
```

- [ ] **Step 4: Write `main.py` (must be at repo root; this exact file goes at the zip root)**

```python
# main.py
"""DrivenData submission entrypoint. Writes a prior-only submission.csv immediately, then overwrites it
stage-wise (LR → CNN ensemble) under a wall-clock ladder so any failure still leaves a valid file."""
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "4")
import sys
import time
from pathlib import Path

T0 = time.time()
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from datsc import config, infer
from datsc.calibrate import Calibration

ARTIFACTS = Path(os.environ.get("DATSC_ARTIFACTS", ROOT / "artifacts"))


def selftest():
    import tempfile
    from tests_synth import make_synthetic_volume, write_nifti  # bundled copy, see Step 5
    from datsc.preprocess import preprocess
    with tempfile.TemporaryDirectory() as td:
        for i, kw in enumerate(({}, {"levels": 8, "shape": (128, 128, 25), "spacing": (3.9,) * 3},
                                {"shape": (256, 256, 256), "spacing": (2.46,) * 3})):
            vol, sp = make_synthetic_volume(seed=i, **kw)
            p = write_nifti(Path(td) / f"s{i}.nii.gz", vol, sp)
            r = preprocess(p)
            assert r["crop"].shape == config.CROP_SHAPE and np.isfinite(r["scalars"]).all()
    print("SELFTEST OK")


def main():
    fmt = pd.read_csv(config.RUNTIME_SUBMISSION_FORMAT, dtype={"uid": str})
    uids = fmt["uid"].astype(str).tolist()
    calib = Calibration.load(ARTIFACTS / "calib.json") if (ARTIFACTS / "calib.json").exists() else \
        Calibration(w=0.0, T=1.0, lam=0.0, pi=config.DEFAULT_PRIOR)
    pi = calib.pi
    infer.write_submission(uids, np.full(len(uids), pi), config.RUNTIME_OUTPUT_CSV)   # stage 0: prior
    print(f"stage0 prior written for {len(uids)} rows at {time.time()-T0:.0f}s")

    n_workers = max(1, min(20, (os.cpu_count() or 4) - 2))
    crops, sc, rad, ok = infer.preprocess_all(uids, config.RUNTIME_NIFTI_DIR, n_workers,
                                              deadline_s=infer.LADDER_SKIP_CNN_S, t0=T0)
    p_lr = np.full(len(uids), pi)
    if (ARTIFACTS / "lr.json").exists():
        p_lr = infer.run_lr(ARTIFACTS / "lr.json", rad, sc, ok, pi)
        # LR arm alone: T was fitted on the CNN/LR blend, so apply only the prior-mix hedge here
        lr_only = Calibration(w=0.0, T=1.0, lam=max(calib.lam, 0.02), pi=pi)
        infer.write_submission(uids, np.where(ok, lr_only.apply(p_lr, p_lr), pi), config.RUNTIME_OUTPUT_CSV)
        print(f"stage1 LR written at {time.time()-T0:.0f}s")

    weights = sorted(p for p in ARTIFACTS.glob("cnn_f*_s*.pt") if "_com" not in p.name)
    if weights and time.time() - T0 < infer.LADDER_SKIP_CNN_S:
        p_cnn = infer.run_cnn(weights, crops, sc, ok, T0)
        if p_cnn is not None:
            final = calib.apply(np.where(np.isnan(p_cnn), p_lr, p_cnn), p_lr)
            final = np.where(ok, final, pi)
            infer.write_submission(uids, final, config.RUNTIME_OUTPUT_CSV)
            print(f"stage2 CNN ensemble written at {time.time()-T0:.0f}s")
    print(f"done in {time.time()-T0:.0f}s; failed volumes: {int((~ok).sum())}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
        sys.stdout.flush()
        os._exit(0)   # submission.csv is already on disk; never let a hung worker process keep us alive
```

- [ ] **Step 5: Make the synthetic generator importable from the zip**

Create `tests_synth.py` at the repo root containing a copy of `make_synthetic_volume` and `write_nifti` from `tests/conftest.py` (identical code, no pytest imports). This file ships in the zip so `--selftest` works inside the container.

- [ ] **Step 6: Write the prior-only packer**

```python
# scripts/make_prior_submission.py
"""Build a prior-only submission.zip (main.py + datsc + tests_synth.py + calib.json with lam=1)."""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from datsc.config import DEFAULT_PRIOR

if __name__ == "__main__":
    out = ROOT / "submission_prior.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(ROOT / "main.py", "main.py")
        z.write(ROOT / "tests_synth.py", "tests_synth.py")
        for p in sorted((ROOT / "datsc").glob("*.py")):
            z.write(p, f"datsc/{p.name}")
        z.writestr("artifacts/calib.json", json.dumps({"w": 0.0, "T": 1.0, "lam": 1.0, "pi": DEFAULT_PRIOR}))
    print("wrote", out)
```

- [ ] **Step 7: Run tests, then a local dry run on the smoke-test data**

Run: `python -m pytest tests/test_infer.py -v` → 2 passed
Run:
```bash
mkdir -p data-demo && tar -xf data/smoke_test_data_2gePzfM.tar.gz -C data-demo   # plain tar despite the .gz name
ls data-demo            # expect niftis/ submission_format.csv test_labels.csv
DATSC_DATA_ROOT=data-demo DATSC_OUTPUT_CSV=submission_demo.csv python main.py
```
Expected: `stage0 prior written`, `preprocessed 20/20`, `stage1 LR written`, `done in …s`. Time it; the smoke test must finish in 6 min inside the container including imports.

- [ ] **Step 8: Build and upload the prior-only submission (Sat)**

Run: `python scripts/make_prior_submission.py` → upload `submission_prior.zip` on the DrivenData submissions page **as a smoke test first**, then as a real submission. Record the public LB log loss `L`. Invert for π: solve `L = -[π' log p + (1-π') log(1-p)]` with `p = 0.548` → `π' = (L + log(1-p)) / (log(1-p) - log(p))`. Put π' into `scripts/fit_calibration.py --pi` later.

- [ ] **Step 9: Commit**

```bash
git add datsc/infer.py main.py tests_synth.py scripts/make_prior_submission.py tests/test_infer.py
git commit -m "feat(infer): stage-wise main.py with deadline ladder, selftest, prior-only packer"
```

---

### Task 14: `scripts/pack_submission.py` + container dry run

**Files:**
- Create: `scripts/pack_submission.py`

- [ ] **Step 1: Write the packer**

```python
# scripts/pack_submission.py
"""Build submission.zip: main.py, tests_synth.py, datsc/*.py, artifacts/{lr.json,calib.json,cnn_f*_s*.pt}."""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="submission.zip")
    a = ap.parse_args()
    art = Path(a.artifacts)
    need = [art / "lr.json", art / "calib.json"]
    weights = sorted(p for p in art.glob("cnn_f*_s*.pt") if "_com" not in p.name)
    for p in need:
        assert p.exists(), f"missing {p}"
    with zipfile.ZipFile(ROOT / a.out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(ROOT / "main.py", "main.py")
        z.write(ROOT / "tests_synth.py", "tests_synth.py")
        for p in sorted((ROOT / "datsc").glob("*.py")):
            z.write(p, f"datsc/{p.name}")
        for p in need + weights:
            z.write(p, f"artifacts/{p.name}")
    print(f"wrote {a.out} with {len(weights)} checkpoints")
```

- [ ] **Step 2: Container dry run (once the image pull from Task 0 has finished)**

```bash
python scripts/pack_submission.py
mkdir -p submission && cp submission.zip submission/submission.zip
MSYS_NO_PATHCONV=1 docker run --rm \
  -v "$(pwd)/data-demo:/code_execution/data:ro" \
  -v "$(pwd)/submission:/code_execution/submission" \
  competitionsfmnparkinsonsprodacr.azurecr.io/competition-sfmn-parkinsons-runtime:gpu-latest
```
(Git Bash: no `-it` (no TTY) and `MSYS_NO_PATHCONV=1` so `/code_execution/...` is not rewritten to a Windows path. If the image's entrypoint expects a different mount layout, read `runtime/entrypoint.sh` in the runtime repo and match it. Without an NVIDIA GPU the container runs the CPU code path — that is fine: it exercises imports, unzip, paths, and the CSV format, which is what the smoke test checks.)
Expected: `submission/submission.csv` appears with 20 rows, values in (0,1). Note total wall-clock.

- [ ] **Step 2b: 100-volume CPU timing run (the number the 3-hour extrapolation rests on)**

```bash
mkdir -p data-timing/niftis && cp $(ls data/niftis/* | head -100) data-timing/niftis/
python - <<'EOF'
import pandas as pd, pathlib
uids=[p.name.split('.')[0] for p in pathlib.Path('data-timing/niftis').iterdir()]
pd.DataFrame({'uid':uids,'is_pathologic':0.5}).to_csv('data-timing/submission_format.csv',index=False)
EOF
DATSC_DATA_ROOT=data-timing DATSC_OUTPUT_CSV=submission_timing.csv python main.py
```
Record the printed `preprocessed 100/100 in Ns` and the CNN seconds per checkpoint; add `data-timing/` and `submission_timing.csv` to `.gitignore`.

- [ ] **Step 3: Extrapolate the 3-hour budget**

From the 100-volume timing run: per-volume preprocess time × (assume 2,000 test volumes) ÷ 20 workers + CNN time per checkpoint × 15 — must be < 90 min (2× margin). If not, raise `PER_VOLUME_TIMEOUT_S`/lower checkpoint count via the ladder constants in `infer.py`.

- [ ] **Step 4: Commit**

```bash
git add scripts/pack_submission.py
git commit -m "feat(scripts): submission packer"
```

---

### Task 15: Kaggle GPU notebook — train 5 folds × 3 seeds + CoM control on T4 x2

**Files:**
- Create: `kaggle/train_notebook_cells.py`

Prerequisites: push the repo; create a Kaggle Secret named `GITHUB_TOKEN` (a fine-grained PAT with read access to `HarshdipSaha/parkinsonnn`). Notebook settings: Accelerator **GPU T4 x2**, Internet **ON** (needed only for the git clone), attach the private dataset `harshdipsaha/dat-parkinsons-challenge-data`.

- [ ] **Step 1: Write the notebook cells file**

```python
# kaggle/train_notebook_cells.py
# Paste each `# %%` block as one cell in the Kaggle notebook (GPU T4 x2, dataset attached).

# %% [1] clone the repo with a secret token
from kaggle_secrets import UserSecretsClient
import os, subprocess
tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
subprocess.run(["git", "clone", "-q", f"https://{tok}@github.com/HarshdipSaha/parkinsonnn.git", "/kaggle/working/repo"], check=True)
os.chdir("/kaggle/working/repo")
print(subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True).stdout)

# %% [2] environment check
import torch, os
print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count(), os.cpu_count())
!python -m pytest -q -x --ignore=tests/test_infer.py

# %% [3] build the crop cache (≈10–15 min on 4 CPUs; written to /kaggle/working/repo/cache — NOT a dataset)
!python scripts/cache_crops.py --out cache --workers 4
!python scripts/qc_grid.py --cache cache --out qc
!python scripts/make_folds.py --cache cache
!python scripts/alignment_gate.py --cache cache      # must print GATE: PASS

# %% [4] first fold, timing check (expect ~10–20 s/epoch on a T4 with AMP)
!python scripts/train_cnn.py --fold 0 --seed 0 --epochs 3 --workers 2 --out artifacts_timing
!rm -rf artifacts_timing

# %% [5] launch two GPUs in parallel: folds 0-4 seed 0 on GPU0/GPU1 alternating, plus the CoM control
import subprocess, itertools
jobs = [(f, 0, False) for f in range(5)] + [(f, 0, True) for f in range(5)]
procs = []
for i, (fold, seed, control) in enumerate(jobs):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(i % 2)}
    cmd = ["python", "scripts/train_cnn.py", "--fold", str(fold), "--seed", str(seed), "--workers", "2", "--out", "artifacts"]
    if control: cmd.append("--control")
    log = open(f"/kaggle/working/log_f{fold}_s{seed}{'_com' if control else ''}.txt", "w")
    procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env))
    if len(procs) % 2 == 0:              # two at a time, one per GPU
        for p in procs[-2:]: p.wait()
for p in procs: p.wait()
!ls artifacts && tail -n 2 /kaggle/working/log_*.txt

# %% [6] seeds 1 and 2 (run this cell on Sept 9 after the Sept 8 submission is in)
jobs = [(f, s) for s in (1, 2) for f in range(5)]
procs = []
for i, (fold, seed) in enumerate(jobs):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(i % 2)}
    log = open(f"/kaggle/working/log_f{fold}_s{seed}.txt", "w")
    procs.append(subprocess.Popen(["python", "scripts/train_cnn.py", "--fold", str(fold), "--seed", str(seed), "--workers", "2", "--out", "artifacts"],
                                  stdout=log, stderr=subprocess.STDOUT, env=env))
    if len(procs) % 2 == 0:
        for p in procs[-2:]: p.wait()
for p in procs: p.wait()

# %% [7] report + copy artefacts to the notebook output (weights + JSON only — no crops, no raw data).
#        Calibration is fitted LOCALLY (Task 16) over the merged artifacts/ of all sessions, not here.
import pandas as pd, glob, json
rows = [{"run": f.split("/")[-1][:-5], "best_val_logloss": round(json.load(open(f))["best_val_logloss"], 4)}
        for f in sorted(glob.glob("artifacts/cnn_f*_s*.json"))]
print(pd.DataFrame(rows))
!mkdir -p /kaggle/working/out && cp artifacts/*.pt artifacts/*.json cache/oof_lr.csv cache/folds.csv /kaggle/working/out/
!du -sh /kaggle/working/out

# %% [8] hygiene: remove the crop cache so nothing derived from raw scans persists as output
!rm -rf cache qc
```

- [ ] **Step 2: Run cells 1–5 on Kaggle (Sun evening, unattended ~1.5–2 h)**

Expected: `GATE: PASS`; 5 hotspot checkpoints + 5 CoM-control checkpoints in `artifacts/`; per-run `best_val_logloss` printed. Compare hotspot vs `_com` folds: the hotspot mean val log loss should be lower — that is the measured value of the anchor.

- [ ] **Step 3: Download `/kaggle/working/out/*` into local `artifacts/` and `cache/` (keep `oof_lr.csv`, `folds.csv` in `cache/`)**

Note for the Sept 9 session (cell 6): nothing under `/kaggle/working` persists between sessions, so re-run cells 1–3 first (clone, tests, cache + folds + gate). Seed-0 artefacts from the first session will not be present there — that is expected; download the seed-1/2 outputs and merge into the local `artifacts/`, then fit calibration locally. `make_folds.py` is seeded, so folds are identical across sessions (verify by diffing `folds.csv`).

- [ ] **Step 4: Commit the cells file**

```bash
git add kaggle/train_notebook_cells.py
git commit -m "feat(kaggle): GPU training notebook cells (T4 x2, folds×seeds, control)"
```

---

### Task 16: Shift diagnostic → λ, submissions #2–#4

**Files:**
- Create: `scripts/shift_diagnostic.py`

- [ ] **Step 1: Write the diagnostic**

```python
# scripts/shift_diagnostic.py
"""Leave-one-spacing-cluster-out on the LR arm (cheap proxy for site shift).
Rule (precommitted): worse cluster inflates log loss by <50% → lam=0.02, else lam=0.05."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd

from datsc.calibrate import logloss
from datsc.lrmodel import LRModel, feature_matrix

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--C", type=float, default=0.3)
    a = ap.parse_args()
    f = pd.read_csv(Path(a.cache) / "folds.csv", dtype={"uid": str})
    X = feature_matrix(Path(a.cache), f["uid"]); y = f["is_pathologic"].values.astype(int)
    base = logloss(y, pd.read_csv(Path(a.cache) / "oof_lr.csv", dtype={"uid": str})["p_lr"].values)
    worst = 0.0
    for lo, hi, name in ((2.43, 2.52, "2.46mm"), (3.85, 3.95, "3.90mm")):   # 2.43 excludes the separate 2.40mm cluster
        held = (f["native_spacing_mm"] >= lo) & (f["native_spacing_mm"] < hi)
        p = LRModel(C=a.C).fit(X[~held], y[~held]).predict_proba(X[held])
        ll = logloss(y[held], p); infl = ll / base - 1
        worst = max(worst, infl)
        print(f"hold out {name}: n={int(held.sum())} logloss={ll:.4f} (random-CV={base:.4f}, inflation={infl:+.0%})")
    lam = 0.02 if worst < 0.5 else 0.05
    print(f"RULE -> lam={lam}")
```

- [ ] **Step 2: Sept 8 — submission #2**

```bash
python scripts/shift_diagnostic.py --cache cache          # note lam
python scripts/fit_calibration.py --cache cache --artifacts artifacts --lam <lam> --pi <π' from Task 13 step 8>
python scripts/pack_submission.py
# container dry run as in Task 14 step 2, check wall-clock, then upload submission.zip as smoke test → then real
```

- [ ] **Step 3: Sept 9–10 — seeds and the augmentation ablation**

Run Kaggle cell [6] (seeds 1–2). For the ablation, temporarily set `p=0.0` in `random_quantize` and `frac=0.0` in `random_scale_jitter`, train 5 folds × seed 0 into `artifacts_noaug/`, compare mean `best_val_logloss`; keep whichever is ≥0.02 better (else keep the augmented default — it is the more conservative choice for unseen centres). Revert any temporary edits.

- [ ] **Step 3b: Sept 10 — honest post-calibration number (nested) and rebuild test**

Nested estimate (fit w/T on four folds' OOF, score the fifth, average) so the reported number is not fitted on itself:
```bash
python - <<'EOF'
import glob, numpy as np, pandas as pd
from datsc import calibrate as cb
files=[f for f in glob.glob('artifacts/oof_f*_s*.csv') if '_com' not in f]
cnn=pd.concat([pd.read_csv(f,dtype={'uid':str}) for f in files]).groupby('uid').agg(y=('y','first'),p_cnn=('p','mean')).reset_index()
df=cnn.merge(pd.read_csv('cache/oof_lr.csv',dtype={'uid':str})[['uid','p_lr']],on='uid').merge(pd.read_csv('cache/folds.csv',dtype={'uid':str})[['uid','fold']],on='uid')
ll=[]
for k in sorted(df.fold.unique()):
    tr,va=df[df.fold!=k],df[df.fold==k]
    c=cb.fit(tr.p_cnn.values,tr.p_lr.values,tr.y.values.astype(float),lam=0.02,pi=0.548)
    ll.append(cb.logloss(va.y.values.astype(float),c.apply(va.p_cnn.values,va.p_lr.values)))
print('nested post-calibration logloss: mean=%.4f  per-fold=%s'%(np.mean(ll),np.round(ll,4)))
EOF
```
Rebuild test: `git tag sub2-$(date +%Y%m%d)` on the commit used for submission #2; in a fresh Kaggle session at that tag re-run `train_cnn.py --fold 0 --seed 0` and confirm `best_val_logloss` matches the original within 0.01. Record both numbers in the disclosure.

- [ ] **Step 4: Sept 14 — submissions #3 and #4**

```bash
python scripts/fit_calibration.py --cache cache --artifacts artifacts --lam <rule lam> --pi <π'>
python scripts/pack_submission.py --out submission.zip          # 15 checkpoints; container dry run; submit (#3)
python scripts/fit_calibration.py --cache cache --artifacts artifacts --lam <other end: 0.05 or 0.02> --pi <π'>
python scripts/pack_submission.py --out submission_alt.zip      # submit (#4) — best private score counts automatically
```

- [ ] **Step 5: Commit**

```bash
git add scripts/shift_diagnostic.py
git commit -m "feat(scripts): leave-cluster-out shift diagnostic for λ rule"
```

---

## Definition of done

- `python -m pytest` green on the local box.
- `qc/*.png` eyeballed; alignment gate `PASS` recorded with its AUROC.
- Prior-only submission scored (π' recorded); submission #2 scored by Sept 8; #3/#4 by Sept 14.
- Container dry run timed and extrapolated under 90 min for 2,000 volumes.
- `artifacts/` contains `lr.json`, `calib.json`, 15 × `cnn_f{0-4}_s{0-2}.pt`; nothing pickled; no raw data or crops in any Kaggle output.
- Disclosure notes: training on Kaggle (private dataset, deleted after), no external weights, no external data.
