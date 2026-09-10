# Run 8 — ensemble-corrected members and Master v3

## Why

Master v2 (five pillars, equal weights, temperature 0.9; nested out-of-fold log loss **0.2323**; DrivenData smoke **0.2381**) is the best zip on hand. Its loss is concentrated, not spread evenly: on the 1,362 training scans, the top 100 scans (7 %) carry **54 %** of the nested loss, 94 scans lose more than 1 nat, and 46 scans are confidently wrong (probability on the wrong side by more than 0.8) even though five architecturally different arms (3D hemisphere, 3D whole-crop, feature-based logistic, 2D coronal ConvNeXt, 2D axial EfficientNet-V2-S) agree. Every member trained so far learned from the hard label on these same scans. Run 7 tried one label-noise fix — self-bootstrapped targets, where a model softens its own hard labels during training — and it changed nothing (0.2565 vs 0.2568 nested). That result rules out "the model doesn't trust itself enough" and points at the alternative: the signal to soften toward has to come from the *ensemble*, which already disagrees with the hard label on exactly the scans that hurt.

## What

Two new per-sample mechanisms in the existing trainer, applied only to the loss (nothing else in the pipeline changes):

- **Soft targets** — `y* = (1-λ)·y + λ·p_nested`, where `p_nested` is Master v2's out-of-fold probability for that training scan, computed the same way the nested nested-CV nested estimate is: weights and temperature refit on the other four folds, never on the scan being predicted. λ = 0.4 (mid-point of the literature range 0.3–0.5 for this kind of target smoothing). A scan the ensemble already reads correctly barely moves; a scan the ensemble reads as 0.85 abnormal against a "normal" label gets pulled most of the way there.
- **Mistake weights** — `w = 1 + κ·loss_nested`, normalised to mean 1 over the training set, κ = 2. Scans the ensemble already loses on train with a proportionally larger gradient; easy scans are slightly down-weighted rather than ignored.

Both are computed once (on the laptop, from the master pool) into a single 1,362-row table (`uid, p, loss`) and embedded directly in the notebook — no new dataset, one file, one command. Validation is always scored against the hard label, so a soft-target arm's reported log loss is never inflated by its own training signal.

## Where the budget goes (≈ 20 GPU-h on `skyfall23`)

| job family | seeds/folds | purpose |
|---|---|---|
| `r3d_hemisoft` | 3 seeds × 5 random folds | primary test of soft targets on the strongest architecture (hemisphere, OOF 0.2379 alone) |
| `r3d_hemisoft_site` | 1 seed × 5 site folds | shift behaviour of the soft-target mechanism, temperature fit |
| `r3d_hemi48` (seeds 3–4) | 2 seeds × 5 folds | safe seed-averaging gain, independent of whether soft targets help |
| `r3d_hemimw` | 1 seed × 5 folds | isolates mistake-weighting from soft-targeting |
| `r3d_crophistsoft` | 1 seed × 5 folds | soft targets on a second, weaker architecture (mirrors run 7's DANN/DRO cross-check) |
| `r3d_hemisoft` (full-data) | 2 seeds, fold −1 | the arm actually shipped, trained on every scan |

42 training jobs, ≈ 8.7 GPU-hours, ≈ 4.5 h wall on two T4s (hemisphere fold ≈ 12.4 min per the run-7 metadata), leaving ≈ 10 GPU-hours for a bracket session (λ 0.3/0.5, or more seeds of whatever gates in) before the Sept 13 slots.

## Gate

The notebook prints the nested estimate of the Master v2 pillars alone and pillars + each new arm (equal weights, temperature floor 0.9) in its own cell, so the number is visible in the Kaggle log before anything is downloaded. On the laptop, `scripts/build_master_v3.py` recomputes the same table from the fetched checkpoints plus the local master pool, adds shrunk/fitted weights and a greedy best-subset search, and packs the zip with whatever set passes. **An arm ships only if it improves the nested estimate by at least 0.003**; the zip carries every seed of every member (the seed-0-only pruning mistake found in Master Final is not repeated) and equal weights (equal beat fitted by 0.0016–0.0017 in every set tried on Sept 10).

## Honest expectation

Capping each training scan's nested loss at 2.0 nats — an upper bound on "what if the ensemble stopped being badly wrong on its worst scans" — gives 0.2201, not lower. A realistic read is nested **0.226–0.229** (a 0.003–0.006 gain), public **≈ 0.245–0.250** at the measured 0.018 shift gap, rank **≈ 15–20**. That is a real improvement over Master v2's projected ≈ 0.253–0.258, and the best remaining use of the GPU hours — not a path to rank 1, which needs public ≈ 0.22, a further 0.025 this mechanism cannot supply alone.

## Rules

No test-set or pseudo-label use; the soft-target table is built entirely from training-fold out-of-fold predictions (never the hidden test set); only public torchvision ImageNet weights (unchanged from earlier runs); per-sample inference; no external data.
