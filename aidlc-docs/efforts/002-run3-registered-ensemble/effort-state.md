# Effort 002: run 3, registered frame + 3D / 2.5D / LR ensemble

| | |
|---|---|
| **State** | in progress: queue 1 training on Kaggle (relaunched 06:50 UTC Sept 6 after the EMA fix; ETA ~12:45 UTC) |
| **Started** | 2026-09-05 evening |
| **Resume runbook** | [HANDOFF.md](../../HANDOFF.md), section 3 |
| **Plan** | [2026-09-06-run3-finish-and-run4.md](../../../docs/superpowers/plans/2026-09-06-run3-finish-and-run4.md), Part A |
| **Depth** | standard |

## Requirements delta (vs baseline)

| Kind | Change | Baseline req |
|---|---|---|
| NEW | SimpleITK similarity + affine registration to a symmetric, label-blind study template; NCC quality check with CoM fallback | R4 |
| NEW | Reference-region intensity normalisation (eroded brain minus dilated striatum, background-subtracted) | R4 |
| CHANGED | No scalar side-input; GroupNorm / SiLU; fixed 50 (3D) / 30 (2D) epochs; final-EMA weights; mixup 0.2; AdamW weight decay 1e-2 | R5 |
| NEW | 2.5D torchvision EfficientNet-B0 / V2-S on 3 x 8 mm axial slabs at the template striatal plane (z = 33.5) | R3 (disclosed) |
| NEW | Registered-ROI features + logistic regression as a third arm | |
| NEW | Weighted-logit combiner; temperature fitted on leave-pseudo-site-out predictions, floored at 1.0 | R5 |

## Construction log

| # | When (UTC) | Unit | Result |
|---|---|---|---|
| 1 | Sept 5 late | `cache_v3`: 2 mm CoM boxes | 1362 / 1362 built |
| 2 | Sept 6 06:00 | Study template (400 scans, 2 iterations, L/R symmetrised) | NCC to template: unregistered 0.63 → registered 0.81 (median) |
| 3 | 06:10 | `cache_v4`: all boxes registered | fallback 0.1 % (label-independent) |
| 4 | 06:15 | Gate G1: ROI features + LR | AUROC 0.902 (unregistered frame 0.852; run-2 hotspot radiomics 0.862). Target 0.92 not met; accepted, LR kept as an arm |
| 5 | 06:20 | Tests | `tests/test_v3.py`, `tests/test_infer_v3.py` green; `main_v3.py --selftest` OK with the template |
| 6 | 06:30 | Kaggle dataset `dat-parkinsons-cache-v4` (2.08 GB); notebook re-injected (48 cells); queue 1 launched | 50 jobs |
| 7 | 06:45 | Hotfix 1: first B0 folds under-confident (AUROC 0.87 / 0.92 but log loss 0.58 / 0.55). Suspected EMA warm-start (0.999^1020 leaves 36 % random init) | step-adaptive decay (commit `fc44f48`); queue relaunched |
| 8 | 07:15 | Hotfix 1 changed nothing. Diagnosis on the OOF files: probabilities compressed to [0.3, 0.75], optimal temperature 0.18–0.41, EMA validation loss lags the raw model by ~15 epochs. Cause: EMA window too long (0.998 = 15 of 30 epochs) | Fix 2 (`fc136d8`): `--ema 0.99` (2D) / `0.995` (3D); raw-model validation metrics logged every epoch; queue relaunched from folds 2/3 |
| 9 | 07:40 | Fix confirmed | B0 folds 2 / 3: 0.279 / 0.294 log loss, AUROC 0.952 / 0.947 |
| 10 | 08:00–08:40 | B0 seeds 0–2 complete (0.258–0.326, AUROC 0.935–0.963). Added z-shift test-time augmentation (`InputSpec.variants`, shared by trainer, `reoof_v3.py`, `main_v3.py`), coronal-slab option, full-data mode (fold -1), stack `--extra`; queue 2 defined; notebook synced with `push_cells.py` without touching the live queue | |
| 11 | 08:50 | 3D seed 0 complete | 0.274 / 0.334 / 0.292 / 0.316 / 0.242 (AUROC up to 0.972); ~21 min per fold |

### Open item before stacking

The hotfix-2 cleanup used `pkill -f scripts/train_v3.py`, which killed its own shell before the `rm`, so `r2d_efficientnet_b0_f0_s0` and `_f1_s0` are still the stale 0.998-EMA checkpoints. Notebook cell 47 deletes them and cell 48 retrains only those two. Use `pkill -f "[s]cripts/train_v3.py"` in future.

## Next

1. Queue 1 ends (~12:45 UTC) → automatic report / stack / pack / end-to-end check (preliminary zip).
2. Cell 47 → run-all-below: cleanup → 2 stale jobs → queue 2 → re-OOF with TTA → stack / pack → e2e → split.
3. Download parts → join → local checks → DrivenData smoke test → real submission (slot 2 of 3).
4. Record OOF, per-site and public numbers in `RESULTS_LOG.md`; close this effort; open effort 003.
