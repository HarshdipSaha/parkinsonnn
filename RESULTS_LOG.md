# Results log

All numbers are 5-fold StratifiedGroupKFold out-of-fold (OOF) on the 1,362 training scans unless stated. Metric = log loss (lower is better). Prior-only baseline = 0.6884.

## Run 2 — fixed interior anchor, 2 seeds, dual-crop ensemble (Kaggle T4 x2, 2026-09-05)

Kaggle notebook `dat-parkinsons-train`, session of ~3 h. Cache: 1362/1362 volumes in 1281 s, 161 (11.8 %) fallback anchors, 0 failures. Alignment gate (14 hotspot radiomics + L2 logreg): OOF AUROC **0.8615**, log loss 0.4661 (gate target 0.90 not met; proceeded on CNN validation loss).

Per-checkpoint best validation log loss (40 epochs, batch 16, AMP, ~16 s/epoch):

| run | best val logloss | crop |
|---|---|---|
| f0_s0 | 0.4139 | hotspot |
| f0_s1 | 0.3104 | hotspot |
| f1_s0 | 0.3713 | hotspot |
| f1_s1 | 0.3552 | hotspot |
| f2_s0 | 0.2863 | hotspot |
| f2_s1 | 0.3267 | hotspot |
| f3_s0 | 0.2951 | hotspot |
| f3_s1 | 0.3189 | hotspot |
| f4_s0 | 0.2847 | hotspot |
| f4_s1 | 0.2778 | hotspot |
| f0_s0_com | 0.2661 | centre-of-mass (run 1) |
| f1_s0_com | 0.3082 | centre-of-mass (run 1) |
| f2_s0_com | 0.2811 | centre-of-mass (run 1) |
| f3_s0_com | 0.3355 | centre-of-mass (run 1) |
| f4_s0_com | 0.3205 | centre-of-mass (run 1) |

- Mean val log loss: hotspot arm **0.3240**, CoM control **0.3023** (hotspot gain = -0.022, i.e. the hotspot crop is slightly *worse* than the plain centre-of-mass crop; fold 0/1 hotspot seeds are the weak ones).
- 15-checkpoint average OOF log loss **0.2994**; LR arm 0.4661.
- Calibration fit: w = 1.00 (CNN only), T = 0.913, lam = 0.02, pi = 0.548 -> OOF **0.2999**.
- `submission.zip` = 50.7 MB, 31 files, 15 checkpoints. Local copy: `artifacts/run2_2026-09-05/` (zip + extracted checkpoints/JSON).

### Submissions

| id | type | zip | status | score |
|---|---|---|---|---|
| 321144 | smoke test | submission_prior.zip (prior only) | Completed | 0.6687 |
| 321152 | smoke test | submission.zip (run 2) | Completed (8 s, 15/15 checkpoints, exit 0) | 0.2567 |
| 321156 | **real** (slot 1 of 3) | submission.zip (run 2) | submitted 2026-09-05 15:38 UTC | pending |

Kaggle-side checks on the packed zip: end-to-end on the 20 smoke scans = 0.2567 (identical to DrivenData), 32 s wall on 4 CPUs; 150 training volumes in 204 s on 4 CPUs -> extrapolated ~12 min for 2,000 test volumes on the 24-vCPU/A100 box (limit 180 min).

### Takeaways for run 3

1. Hotspot anchoring did not beat CoM; the two crops are complementary in the ensemble (CoM checkpoints are among the best). Keep both, and train CoM checkpoints for seed 1 as well (5 more jobs, ~35 min).
2. Fold 0 and fold 1 hotspot checkpoints are the outliers (0.41, 0.37); a third seed would stabilise the average.
3. The alignment gate is still below target: the anchor is the known weak point. Candidates: larger crop (88x88x56), or a learned/registered anchor.

## Run 1 — original anchor (2026-09-05, superseded)

Hotspot folds 0.353/0.290/0.266/0.354/0.321 (mean 0.317); CoM control mean 0.302; calibration w = 1.0, T = 0.989, OOF 0.318. The five CoM checkpoints from this run are the `_com` members of the run-2 ensemble.

## Run 3 build log (2026-09-06)

**Diagnosis (docs/RUN3_BRAINSTORM.md):** the public gap is discrimination, not calibration (a calibrated classifier at AUROC 0.925 scores 0.346; the leader's 0.971 maps to 0.221). Weakest link = the spatial frame (hotspot anchor failed on 19.5 % of abnormals vs 2.4 % of normals; in CoM crops the striatal plane wandered between slice 7 and 30 of 48) plus site-fingerprint scalars in the heads and best-epoch selection inflating the OOF. Deep-research (5 verified claims before the agent quota ran out): the 97 %+ DaT-SPECT results (Wenzel 2019 PPMI 0.972; Buchert/Hamburg 2025, n=1,740, 97.0-97.5 % balanced accuracy) all use affine registration to a study template, reference-region (whole brain minus striata) 75th-percentile scaling and a 12 mm axial slab through the striatum fed to a 2D ResNet; blur augmentation makes the CNN robust to resolution mixing where SBR features degrade.

**Run-3 pipeline:** 2 mm CoM box -> SimpleITK similarity (7-DOF) + affine refinement registration to a label-blind, L/R-symmetrised study template (400 random scans, 2 iterations) -> NCC QC with CoM fallback -> reference-region normalisation (eroded brain minus dilated striatum, background-subtracted; units = SBR+1). Arms: GroupNorm/SiLU 3D ResNet (no scalar head, fixed 50 epochs, final-EMA weights, mixup 0.2) on the 3 mm-equivalent registered box; 2.5D torchvision EfficientNet-B0 / V2-S (ImageNet, BSD-3) on 3 x 8 mm axial slabs through the template striatal plane (z = 33.5 vox); leave-pseudo-site-out runs to fit the temperature (floored at 1.0); registered-ROI radiomics LR as the registration gate.

| template stat | value |
|---|---|
| registration time | 1.6-1.8 s / volume (8 + 4 mm pyramid, NCC metric) |
| synthetic recovery | 12 deg / scale 1.10 / 12 mm recovered exactly (NCC 0.72 -> 0.88) |
| NCC to template, unregistered p10/50/90 | 0.302 / 0.630 / 0.809 |
| NCC to template, registered p2/10/50/90 | 0.578 / 0.692 / 0.783 / 0.835 (QC threshold 0.578) |
| scale p10/50/90 | 0.965 / 1.037 / 1.162; rotation p50/90 = 5.8 / 14.7 deg |
| template masks | brain 3,317 mL; striatal 42.7 mL (2,671 voxels per side); reference 2,301 mL; striatal plane z = 33.5 |
| registered cache (cache_v4) | 1,360 / 1,362 registered (fallback 0.1 %: 0.2 % of normals, 0.1 % of abnormals); NCC p2/50/90 = 0.703 / 0.810 / 0.860 |
| gate G1: template-ROI features + L2 LR | AUROC **0.902** / log loss 0.402 on random folds (unregistered CoM frame: 0.852 / 0.482; run-2 hotspot radiomics: 0.862); site folds 0.893 / 0.418. Target 0.92 not met, fallback criterion met -> proceed to the CNN A/B, keep LR as a third arm |
| Kaggle T4 timing (run 3) | 3D GroupNorm ResNet 64x64x48 (2.5M params): 21 s/epoch -> 18 min per 50-epoch fold; EfficientNet-B0 on 3 slabs @192 px: 8 s/epoch -> 5 min per 30-epoch fold |
| queue (Sept 6 06:20 UTC, relaunched 06:45 after the EMA warm-start hotfix fc44f48) | 50 jobs: B0 5 folds x 3 seeds, 3D 5 x 3, B0 site-folds x 1, 3D site-folds x 1, V2-S 5 x 2 -> ~9.5 GPU-h, ~4.7 h wall on 2 GPUs |
| first run-3 folds (EMA 0.99, 2.5D B0, seed 0) | fold 2: 0.2787 / AUROC 0.9519; fold 3: 0.2942 / 0.9472 (final-EMA, no best-epoch selection). With the 0.998 EMA window the same folds gave 0.58 / 0.55 at AUROC 0.87 / 0.92 -> the EMA window was the whole story |
