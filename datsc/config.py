"""Constants and data-location discovery shared by training and inference."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --- geometry -------------------------------------------------------------
SPACING_MM = 3.0                   # isotropic resampling target
CROP_SHAPE = (72, 72, 48)          # [x, y, z] voxels at SPACING_MM (216x216x144 mm)
SMOOTH_MM = 6.0                    # gaussian sigma used for mask + anchor
ANCHOR_THRESH = 0.7                # fraction of (interior) smoothed max defining the hotspot
ANCHOR_INTERIOR_MM = 20.0          # hotspot search restricted to voxels this deep inside the head mask
ANCHOR_MIN_ML, ANCHOR_MAX_ML = 1.0, 30.0   # informational flag bounds (anchor is used regardless)
ANCHOR_MAX_DIST_MM = 50.0
FALLBACK_SUPERIOR_OFFSET_MM = 15.0 # CoM fallback only when the interior region is empty (degenerate mask)
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
