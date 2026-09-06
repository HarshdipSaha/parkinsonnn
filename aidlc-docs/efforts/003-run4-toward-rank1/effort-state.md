# Effort 003: run 4 — from 0.2817 toward 0.21

| | |
|---|---|
| **State** | in-progress (opened Sept 6 15:55 UTC) |
| **Plan** | [2026-09-06-run4-toward-0.21.md](../../../docs/superpowers/plans/2026-09-06-run4-toward-0.21.md) (detailed, bite-sized tasks); background in [2026-09-06-run3-finish-and-run4.md](../../../docs/superpowers/plans/2026-09-06-run3-finish-and-run4.md) Part B and `docs/RESEARCH_SIMILAR_COMPETITIONS.md` |
| **Baseline** | Run 3: public 0.2817 / AUROC 0.9477 / rank 66; OOF stack 0.2580 (nested 0.2603); leader 0.2192 / 0.971 |
| **Budget** | Kaggle GPU: editor shows quota 12:42 of 30 h on Sept 6 16:00 UTC (≈ 12.7 h left this week; the weekly reset is not Saturday 00:00 UTC), 30 h next week. Real submissions: slot 3 free now; slot 1 back Sept 12 15:38 UTC; slot 2 back Sept 13 15:11 UTC. Deadline Sept 16 23:59 UTC |
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

## Construction log

| # | When (UTC) | Unit | Result |
|---|---|---|---|
| 1 | Sept 6 15:15–15:40 | 4a-1: `--aug strong` preset, `--z-jitter`, 3D striatal crop (`--crop3d/--center3d`), runner `tag` override + `{XC}`, JOBS3, ARMS (11 arms, grid 8), checkpoint-dataset resume + API-download fallback; 3 new tests | 42 tests green; `main_v3.py --selftest` OK; commits `be94b69`, follow-up |
| 2 | 15:25–15:44 | Run-3 artefacts (48 ckpts + json + OOF, 475 MB) uploaded to the private dataset `harshdipsaha/dat-run3-checkpoints` (CLI must run from inside the folder on Windows); dataset attached to the notebook via Add Input (URL search; name search finds nothing for private datasets); cells synced (4 inserted, 13 updated); stale "stale-checkpoint cleanup" cell deleted | |
| 3 | 15:45 | **Version 3 = run 4a** started as "Save & Run All (Commit)" on GPU T4 x2 (editor quota reading 12:42 / 30 h before launch). Expected ≈ 3.5 h → ~19:15 UTC. Watcher re-armed with `-Out data/kaggle_output_v4` | running |
| 4 | 15:47–15:57 | Feature-engineered arm `feat` (user direction: a math/feature-engineering member; stack keeps it only if it earns weight): engineered A→P profiles, threshold ladder, hot-spot geometry, asymmetries + striatal-patch PCA(40) → logistic, mirror-averaged | OOF 0.9398 / 0.3120 (site folds 0.3548); **stack nested 0.2603 → 0.2545**, weight 0.2, 3D whole-brain arm → 0. Included in ARMS for future commits; local stack for submission 3 will use it |

