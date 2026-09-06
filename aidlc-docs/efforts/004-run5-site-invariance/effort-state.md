# Effort 004: run 5 — site-invariant training, histogram standardisation, typicality-aware calibration

| | |
|---|---|
| **State** | planning → ready to build (spec + plan written Sept 6 22:45 UTC); construction starts after the slot-3 submission is in |
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
