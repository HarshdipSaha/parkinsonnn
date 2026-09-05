# EDA Report — DaT Parkinson's Challenge Training Data

- **Filename(s):** `train_labels.csv`, `niftis.zip` (1,362 `.nii`/`.nii.gz` volumes)
- **Generated:** 2026-09-05
- **Source notebook:** [dat-parkinsons-challenge-starter](https://www.kaggle.com/code/harshdipsaha/dat-parkinsons-challenge-starter/edit) (Kaggle, private)
- **Methodology note:** structured per a generic scientific-data EDA checklist (file-type detection → format-specific statistical analysis → quality/consistency checks → findings + preprocessing recommendations).

## 1. File properties & format identification

| File | Format | Role |
|---|---|---|
| `train_labels.csv` | Tabular, 2 columns (`uid` str, `is_pathologic` float) | Ground-truth labels, 1,362 rows |
| `niftis/*.nii(.gz)` | 3D medical imaging (NIfTI-1), single volume per file | DaT-SPECT reconstructions, one per examination |

Each `.nii`/`.nii.gz` is a single 3D scalar volume (no time/channel dimension) — analogous to a grayscale 3D "stack" in general imaging-EDA terms, but domain-specific to nuclear medicine (SPECT reconstruction, not microscopy/DICOM).

## 2. Tabular data quality (`train_labels.csv`)

- Shape: **1,362 rows × 2 columns**. `dtypes`: `uid`→object, `is_pathologic`→float64.
- **No duplicate uids, no missing labels.**
- `uid` format: uniformly **8-character lowercase alphanumeric** strings — 100% match a `^[a-z0-9]+$` pattern, no whitespace/casing anomalies.
- `is_pathologic` domain: strictly **{0.0, 1.0}** — no unexpected/coerced values.
- **Class balance:** 747 abnormal (54.8%) / 615 normal (45.2%) — imbalance ratio **1.21:1** (mild).

## 3. Imaging data: geometry (shape & voxel spacing)

Header-only scan across **all 1,362 volumes** (16.1s):

- `shape_z` std = **86.3** slices (range 33–256) — extreme depth variability.
- Two dominant `(x, y, z)` shape clusters: **256×256×256** (460 scans, ~34%) and **128×128×{25–128}** (roughly another third), plus a long tail of smaller z-depths.
- Voxel spacing (isotropic, rounded): **2.46mm** (528, ~39%), **3.90mm** (325, ~24%), **2.30mm** (207, ~15%), **2.40mm** (149, ~11%), plus **30+ additional clusters** with <10 scans each — consistent with 10 different hospital/scanner protocols, some with per-scan drift.
- **Dimensionality sanity check:** all 1,362 volumes are genuinely **3D** (`ndim == 3` for every file) — no stray 4D scans to special-case.

**Implication:** no fixed input shape/spacing can be assumed. A resampling step to a common isotropic spacing (+ crop/pad or full registration) is mandatory preprocessing.

## 4. Imaging data: bit depth, orientation & scaling metadata

Full-dataset NIfTI header pass (all 1,362 volumes, 8.1s):

| Field | Finding |
|---|---|
| `dtype` | 1,346 × `uint16`, 16 × `int16` — overwhelmingly consistent |
| `qform_code` | **1** for all 1,362 volumes |
| `sform_code` | **1** for all 1,362 volumes |
| `scl_slope` / `scl_inter` | **unset (None) for every volume** |
| `cal_min` / `cal_max` | **0.0 / 0.0 for every volume** (unpopulated) |

**Implication:** every center encodes spatial orientation the same way (no header-convention surprises for a registration pipeline), and dtype is essentially uniform. Critically, there is **no header field available to auto-correct intensity scale** — see next section.

## 5. Intensity statistics

Sampled 500 volumes (209 normal / 291 abnormal after file lookup), reading full pixel data:

- `intensity_max` is **strongly bimodal**: many volumes cap near **255** (8-bit-like range), others range up to **~65,000** (raw SPECT counts).
- Cross-checked against dtype: **both dtypes (`uint16`/`int16`) appear in both the low- and high-magnitude regimes** — the bimodality is *not* a storage-format artifact.
- Cross-checked against `scl_slope`/`scl_inter`: **both unset in both regimes** — the split cannot be explained (or corrected) via header scaling factors either.
- Conclusion: the bimodal intensity scale reflects a genuine difference in how each site's reconstruction software quantized counts. **Per-volume normalization (e.g., percentile clipping + rescale, or z-score) is mandatory** before any model — there is no shortcut via metadata.
- Abnormal scans showed lower median `intensity_mean`/`intensity_max` than normal in the raw sample, but this is confounded by the scale-mismatch above and shouldn't be trusted until normalization is applied.

## 6. Scanner/spacing vs. label (shortcut-risk check)

Cross-tab of rounded voxel-spacing cluster vs. `is_pathologic`:

| Spacing cluster | n | % Abnormal |
|---|---|---|
| 2.46mm | 528 | 59.7% |
| 3.90mm | 325 | 58.2% |
| 2.30mm | 207 | 50.2% |
| 2.40mm | 149 | 46.3% |

For the four dominant clusters (~89% of the data), % abnormal stays within 46–60% of the 54.8% dataset baseline — **no major scanner-driven label shortcut**. The long tail of tiny spacing clusters (<10 scans) shows 0%/100% abnormal, but this is small-sample noise, not a reliable leak. Still worth a spacing/hospital-aware stratified k-fold for an honest cross-center generalization estimate.

## 7. Naive hand-crafted feature test

Built a crude "central-peak / background-ring" uptake-ratio proxy (fixed relative-position ROI, **no registration**) on the same 500-volume sample:

- **AUROC = 0.469** — indistinguishable from chance.
- A raw max-intensity-projection (MIP) visual grid (6 normal vs. 6 abnormal) also showed no obvious, consistent visual separation to the eye.

**Interpretation:** this is *not* evidence of an unsolvable problem — it means the real diagnostic pattern is washed out by unregistered shape/spacing/orientation heterogeneity, consistent with the literature review finding that every published high-accuracy DaT-SPECT result first registers scans to a common template (e.g., MNI space) before extracting features or training a model.

## 8. Recommendations (preprocessing & approach)

1. **Registration/resampling is the top engineering priority**, ahead of model architecture: resample every volume to a common isotropic spacing and align to a common frame (minimum: center-of-mass + isotropic resample; ideally: affine registration to a template).
2. **Per-volume intensity normalization is mandatory** (e.g., clip to [p1, p99] then rescale to [0, 1]) — confirmed necessary since no header metadata can correct the scale mismatch.
3. Class imbalance (1.21:1) is mild — standard log-loss training + stratified CV is sufficient; don't over-engineer class-weighting.
4. Re-test a hand-crafted striatal-ratio feature *after* proper registration — the AUROC≈0.5 result here reflects a bad unregistered ROI heuristic, not necessarily an absence of signal (classical radiomics matched deep learning on registered PPMI data in the literature).
5. Use a spacing/hospital-aware stratified k-fold for cross-validation to mirror the real multi-center held-out test set.
6. Tabular/labels data needs no cleaning — it's already clean (no dupes, no missing values, consistent uid format, clean label domain).
