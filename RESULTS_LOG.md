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
| run-3 2.5D B0 arm, 13 fresh folds (seeds 0-2) | 0.258-0.326 per fold, mean ~0.285, AUROC 0.935-0.963 |
| run-3 3D GroupNorm ResNet, first folds (seed 0) | f0 0.274 / 0.945, f1 0.334 / 0.931, f2 0.292 / 0.945; ~21 min per fold. Queue ETA ~13:45 UTC, then stale re-run + queue 2 (coronal, full-data) + TTA re-OOF + stack |
| Sept 6 ~09:00 UTC | Interactive Kaggle session lost (websocket drop, kernel unreachable, Quick Save refused); 15 B0 + 7 3D checkpoints gone. Relaunched as background commit "Version #2" at 10:30 UTC with trimmed queues (B0 x3 seeds, 3D x3 seeds, B0 site-folds, V2-S x1, coronal B0 x1, full-data B0 x3); ETA ~16:00 UTC |
| commit Version #2 (fresh machine, 09:50 UTC) B0 seed 0 + part of seed 1 | 0.258 / 0.268 / 0.283 / 0.294 / 0.292 (seed 0); 0.281 / 0.262 / 0.289 (seed 1 f0-f2); AUROC 0.946-0.962 — consistent with the lost interactive run |

## Run 3 — Kaggle commit Version #2 (Sept 6, 09:46–~14:45 UTC, 2×T4, scriptVersionId 347676112)

All 56 jobs finished, failures []. Per-arm OOF (final EMA weights, TTA z∈{-1,0,1} × flip for 2D):

| arm | ckpts | AUROC | log loss |
|---|---|---|---|
| a2d_b0 (EfficientNet-B0 axial, 3 seeds) | 15 (+3 full-data) | 0.9556 | 0.2695 |
| a2d_v2s (EfficientNet-V2-S axial) | 5 | 0.9549 | 0.2740 |
| a2d_cor (B0 coronal) | 5 | 0.9512 | 0.2773 |
| a3d (ResNet3D 64×64×48, 3 seeds) | 15 | 0.9476 | 0.2937 |
| lr (registered-ROI features) | 1 | 0.9024 | 0.4015 |
| **stack** (weights a3d .1 / b0 .3 / v2s .3 / cor .3 / lr 0; T=1.0 floor, random-fold T 0.81; λ=0.02) | 44 | **0.9614** | **0.2580** (equal-weight 0.2598) |

Site-fold B0 (seed 0): 0.2818 / AUROC 0.952 — close to random folds (0.2752), i.e. little site leakage in the registered frame.
EMA-lag guard: final vs best-epoch within ±0.01 everywhere (window 0.99/0.995 fixed the run-2 collapse).

**Bug found by the end-to-end cell:** every 2D checkpoint failed with `FileNotFoundError` inside the container — the manifest's
`model_kw` carried the training-time `weights_path=pretrained/efficientnet_b0_in1k.pt`, which `build_model` tries to load
before the checkpoint state dict is applied. Only a3d + LR ran (smoke log loss 0.2996). Fix (commit below): inference drops
`weights_path`, the manifest writer strips it, the e2e cell now asserts every arm used all its checkpoints. The Kaggle zip is
repaired locally with `scripts/repack_v3.py` (fresh code, same artifacts) and re-verified with `scripts/local_e2e_v3.py`.

**Submission (Sept 6):** local e2e on the repaired zip 0.2458 (CPU, 390 s / 20 vols); DrivenData smoke test **id-321261 = 0.2397**
(run 2 smoke: 0.2567). Real submission uploaded ~15:20 UTC from `data/submission_v3_fixed.zip` (slot 2 of 3 in the 7-day window) — public score below when scored.

**Public score (id-321263, Sept 6 15:50 UTC): log loss 0.2817, AUROC 0.9477, rank 66 / 450** (run 2: 0.3391 / 0.925 / 157; leader 0.2192 / 0.971).
Nested OOF of the shipped combiner 0.2603 → the 0.02 gap is test shift, not combiner optimism (OOF AUROC 0.961 → public 0.948). The
leave-pseudo-site-out B0 arm (0.2818) matches the public score almost exactly. Per-site nested (n, log loss): other 360 0.276 ·
2.46|256|high 265 0.247 · 2.30|128|low 207 0.274 · 2.40|128x87|ultralow 142 0.192 · 2.46|256|low 134 0.315 · 3.90|128|high 70 0.224 ·
2.46|256|ultralow 61 0.252 · 3.90 thin-FOV 0.30. Clip/hedge: no nested gain.

## Feature-engineered arm `feat` (Sept 6 15:57 UTC, local CPU)

`datsc/features_v4.py` + `datsc/featmodel.py` + `scripts/feat_arm_v4.py`: 17 run-3 ROI features + 44 engineered descriptors
(A→P uptake profile per striatum, threshold-volume ladder, hot-spot volume/shift, L/R asymmetries) + 40 PCA scores of the
48×48×32 striatal box (block-averaged to 24×24×16) → L2 logistic (C 0.3), fitted on both mirror orientations, predictions
averaged over both. Grid: n_comp 40 / C 0.3 best of {20,40,80}×{0.03,0.1,0.3}.

| | AUROC | log loss |
|---|---|---|
| random folds | 0.9398 | 0.3120 |
| site folds | 0.9277 | 0.3548 |
| (run-3 LR arm) | 0.9024 | 0.4015 |

Strongest single descriptors: anterior/posterior bins (AUROC 0.85–0.86) and the > 3.0 SBR volume fraction. Weak sites unchanged
(2.46 mm ultra-low AUROC 0.76). **Stack effect** (run-3 arms + feat, nested): 0.2603 → **0.2545**; plain OOF 0.2580 → **0.2532**;
weights a3d 0 / b0 .2 / v2s .3 / cor .3 / feat .2 / lr 0 (the whole-brain 3D arm is displaced). Artefacts: `artifacts/featarm_v4/`
(`feat_v4.npz` 1.4 MB); inference arm type `feat` in `datsc/infer_v3.py` (numpy only).

