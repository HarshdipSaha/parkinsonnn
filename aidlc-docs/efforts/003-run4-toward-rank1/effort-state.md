# Effort 003: run 4, measured additions toward rank 1

| | |
|---|---|
| **State** | planning (starts after effort 002's real submission) |
| **Plan** | [2026-09-06-run3-finish-and-run4.md](../../../docs/superpowers/plans/2026-09-06-run3-finish-and-run4.md), Part B |
| **Budget** | Kaggle: ~17 GPU-hours left this week, 30 hours next week. Real submissions: slot 3 available now; slot 1 returns Sept 12 15:38 UTC; slot 2 returns 7 days after tonight's submission |
| **Depth** | standard |

## Requirements delta (vs baseline)

| Id | Requirement |
|---|---|
| R7 | Every ensemble change is gated on a **nested** out-of-fold estimate of the stacked log loss (weights and temperature fitted on four folds, scored on the fifth, rotated), with per-pseudo-site reporting; keep only changes that gain ≥ 0.005. |
| R8 | Test-time augmentation is defined once (`InputSpec.variants`) and used identically for out-of-fold predictions (`scripts/reoof_v3.py`) and inference (`main_v3.py`). |
| R9 | Full-data models (fold -1) may be appended to an arm through `fit_stack_v3.py --extra`; they never contribute out-of-fold predictions. |

## Units (see the plan for steps and expected outputs)

| Unit | What | Cost |
|---|---|---|
| B1 | Nested calibration estimate; hard-fold and per-site analysis (fold 1 is the worst fold for both arms) | CPU, ~1 h |
| B2 | Diversity members: ConvNeXt-Tiny 2.5D, 5-slab B0, coronal V2-S; each gated | GPU ~4 h |
| B3 | 3D upgrades: full 2 mm resolution, longer schedule, multi-shift TTA | GPU ~4 h |
| B4 | More seeds and full-data models for the kept arms | GPU ~3 h |
| B5 | Registration refinement (template from all scans, 3 iterations), gated on gate G1 improving by ≥ 0.01 | CPU local |
| B6 | Submission cadence: Sept 9–10, Sept 13, Sept 15, Sept 16 (bracket) | |

## Honest expectation

Public 0.235–0.25 after B1–B4 (top 10 plausible). Rank 1 (< 0.219) needs a further step beyond ensembling plus favourable leaderboard noise: roughly 10–15 % by the deadline.
