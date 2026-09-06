# Effort 003: run 4 — from 0.2817 toward 0.21

| | |
|---|---|
| **State** | in-progress (opened Sept 6 15:55 UTC) |
| **Plan** | [2026-09-06-run4-toward-0.21.md](../../../docs/superpowers/plans/2026-09-06-run4-toward-0.21.md) (detailed, bite-sized tasks); background in [2026-09-06-run3-finish-and-run4.md](../../../docs/superpowers/plans/2026-09-06-run3-finish-and-run4.md) Part B and `docs/RESEARCH_SIMILAR_COMPETITIONS.md` |
| **Baseline** | Run 3: public 0.2817 / AUROC 0.9477 / rank 66; OOF stack 0.2580 (nested 0.2603); leader 0.2192 / 0.971 |
| **Budget** | Kaggle GPU: week reset Sat Sept 6 00:00 UTC; ≈ 8.5 h used today → ≈ 21 h left this week, 30 h next week. Real submissions: slot 3 free now; slot 1 back Sept 12 15:38 UTC; slot 2 back Sept 13 15:11 UTC. Deadline Sept 16 23:59 UTC |
| **Depth** | standard |

## Requirements delta (vs baseline)

| Id | Requirement |
|---|---|
| R7 | Every ensemble change is gated on the **nested** OOF log loss of the stack (weights + T fitted on 4 folds, scored on the 5th) with per-pseudo-site reporting; a member is kept if its stack weight ≥ 0.1 or the nested loss improves ≥ 0.003. |
| R8 | TTA is defined once (`InputSpec.variants`) and used identically for OOF (`scripts/reoof_v3.py`) and inference (`main_v3.py`). |
| R9 | Full-data models (fold −1) may be appended through `fit_stack_v3.py --extra`; they never produce OOF rows. |
| R10 | **Site robustness is the objective**: report the leave-pseudo-site-out score of every new arm variant; prefer variants that shrink the random-fold → site-fold gap. |
| R11 | Every zip is verified with `scripts/local_e2e_v3.py` (all checkpoints used, format OK) before any DrivenData upload. |

## Units

| Unit | What | Cost | Gate |
|---|---|---|---|
| 4a-1 | Runner `tag` override; `--aug strong` preset; `--z-jitter`; 3D striatal crop input (`--crop3d/--center3d`) + tests | local, 1 h | pytest green |
| 4a-2 | Commit "run 4a" (2×T4, ≈ 3.5 h): tiny slab-net ×5 seeds, V2-S seeds 1–2, B0 strong-aug, B0 crop64/224, ConvNeXt-T, coronal V2-S, 3D crop 48×48×32; resume the 43 run-3 checkpoints from the private dataset | GPU 3.5 h | each new arm gated (R7); TTA per arm from the re-OOF table |
| 4a-3 | Nested stack of all arms locally, repack, local e2e, smoke, **submission 3** (Sept 7–8) | local + DD | smoke sane; nested ≤ 0.250 target |
| 4b | Seeds 1–2 + full-data models for every kept new arm; 3D crop seeds; drop dead arms; **submission 4** Sept 12–13 | GPU ≈ 5 h | nested improves |
| 4c | Bracket submissions Sept 15–16: same weights, alternative T / λ (best private counts) | | |

## Honest expectation

Public 0.24–0.25 after 4a/4b (top 10–15). 0.21 requires the test AUROC to rise from 0.948 to ≈ 0.97: only the site-robustness items (strong augmentation, tiny slab net, striatal 3D crop) can deliver that, and whether they do is exactly what 4a measures. Probability of ≤ 0.22 by Sept 16: about 10–15 %.
