# Effort 004: run 5 — site-invariant training, histogram standardisation, typicality-aware calibration

| | |
|---|---|
| **State** | delivered (assembled nested 0.2433; submission candidate for Sept 12 15:38 UTC) |
| **Spec** | [2026-09-06-run5-site-invariance-spec.md](../../../docs/superpowers/specs/2026-09-06-run5-site-invariance-spec.md) |
| **Plan** | [2026-09-06-run5-site-invariance-plan.md](../../../docs/superpowers/plans/2026-09-06-run5-site-invariance-plan.md) (Tasks 1–9, code included; Task 9 = PPMI, conditional on access) |
| **Baseline** | Run 4a gated stack: nested 0.2491 (crop-3D .375, V2-S .25, ConvNeXt .125, coronal B0 .125, feat .125); best single arm striatal-crop 3D 0.2568 / 0.960 |
| **Budget** | GPU ≈ 6.5 h this week (after stopping the stuck Version 3), 30 h next week; run 5 = one ≈ 3 h commit. Submissions: slot 1 Sept 12 15:38 UTC, slots 2–3 Sept 13 |
| **Depth** | standard |

## Requirements delta (vs baseline)

| Id | Requirement |
|---|---|
| R12 | Neural arms can be trained with a gradient-reversal pseudo-site head (DANN) and group-DRO site weighting; the head never enters the checkpoint; evaluated on random AND leave-site-out folds. |
| R13 | A histogram-standardised normalisation mode (template landmarks in `template.npz`) used identically by the trainer and the container. |
| R14 | Per-scan typicality (Mahalanobis distance in the feature arm's PCA space, training statistics only) can raise the combiner temperature; parameters fitted on the site pool, carried in the manifest, gated ≥ 0.003. |
| R15 | Affine test-time augmentation is defined once in `InputSpec.variants` and kept per arm only where the re-OOF shows a gain. |
| R16 | Human-runnable delivery: `kaggle/dat-parkinsons-train.ipynb` exported by the generator (import → attach datasets → Save & Run All). |

## Units

| Unit | Plan task | Cost | Gate |
|---|---|---|---|
| 5-1 | Tasks 1–2: features()/SiteHead, `--dann`, `--dro` | local 1.5 h | tests green; 2-epoch CPU smoke |
| 5-2 | Task 3: hist mode + landmarks | local 1 h | identity/monotonic tests; selftest |
| 5-3 | Task 4: affine TTA | local 45 min | measured on crop-3D checkpoints |
| 5-4 | Task 5: typicality temperature | local 1 h | nested site-pool gain |
| 5-5 | Task 6: run-5 commit (JOBS4) | GPU ≈ 3 h | per-arm gates on random + site folds |
| 5-6 | Task 7: assemble → smoke → submission 4 (Sept 12) | local + DD | nested < 0.2491 |
| 5-7 | Task 9 (conditional): PPMI pretraining | GPU next week | site-fold gate |

## Construction log

| # | When (UTC) | Unit | Result |
|---|---|---|---|
| 1 | Sept 6 23:05–23:20 | Tasks 1–2: `features()`/`feat_dim` on both model families, `SiteHead` + gradient reversal, `--dann` (ramped over the first third) and `--dro` (group-DRO over pseudo-sites, sites < 25 scans merged) in the trainer; site ids looked up by uid | 45 tests green; 1-epoch CPU smoke on the crop-3D arm with both switches finite (val 0.47 / 0.86 after one epoch) |
| 2 | 23:20–23:40 | Task 3: `hist` normalisation — landmarks (percentiles 1…99) of the **reference region** (not the brain: standardising the striatum would erase the label), piecewise-linear map, slope-1 above the top landmark so the striatal contrast keeps absolute units; landmarks stored in `template.npz`; train + inference + tests | identity on the template, monotonic; scan p95 striatal 5.30/4.38/4.56 → 4.73/3.37/3.78 with the reference region mapped exactly |
| 3 | 23:40–23:50 | Task 4: reproducible affine TTA in `InputSpec.variants` (`--tta-affine`), re-OOF override | measured on crop-3D fold 0: 0.2257 → 0.2269 (K=3) → **not used** |
| 4 | 23:50–00:10 | Task 5: typicality (Mahalanobis in the feat arm's PCA space, stats in `feat_v4.npz`), per-scan temperature in `combine`, `--typ` selection | run-4a stack nested 0.2499 → 0.2479 (κ 0.5, d0 = q0.9 = 1.19): below the 0.003 gate → **not selected** (kept available) |
| 5 | 00:10–00:20 | Task 6: `JOBS4`, trimmed `ARMS` (10 arms), `EXTRAS`, queue-4 cell, landmark template + feat artefacts preferred from the checkpoint dataset, `--typ` in the stack cell; `kaggle/dat-parkinsons-train.ipynb` regenerated (63 cells); checkpoint-dataset snapshot (342 files) upload started | ready to launch — needs a browser (Playwright down) or the human path: import the ipynb, attach the 3 datasets, GPU T4 x2, Internet on, secrets, Save & Run All |
| 6 | Sept 7 03:53–09:39 | Version 4 (run 5) ran on Kaggle (2xT4); 479 artefacts retrieved to `data/kaggle_output_v5/artifacts_v3` | completed |
| 7 | Sept 7 10:40 | Readout & assembly: `r3d_cropinv` fails gate (0.2884 vs 0.2589 on site folds) -> OUT. `r3d_crophist` improves single-seed (0.2539 vs 0.2568) and displaces crop-3D at 37.5% weight -> IN. Re-fitted nested combiner estimate: **0.2433** (vs run-4a 0.2491, gain -0.0058); stack OOF **0.2394** / AUROC **0.9672** (equal-weight 0.2370). Zip assembled. | gate passed (nested < 0.2491) |
| 8 | Sept 8 14:30 | DrivenData Smoke Test id-321496 (`submission_run5.zip`): **0.2388 log loss** (local container smoke loss 0.2241, acc@0.5 = 95.0%). Locked in as candidate for Slot 1 (Sept 12 15:38 UTC). | smoke test verified |

