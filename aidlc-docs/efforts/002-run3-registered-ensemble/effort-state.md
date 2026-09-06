# Effort 002: run 3, registered frame + 3D / 2.5D / LR ensemble

| | |
|---|---|
| **State** | complete — public **0.2817** / AUROC 0.9477 / rank **66** (id-321263, Sept 6 15:50 UTC) |
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
| 12 | 09:00–09:50 | Interactive session died after the notebook websocket dropped (kernel status stuck at "connecting"; Kaggle then started a fresh session). 22 checkpoints lost. Switched to a background commit: queues trimmed (dropped 3D site-folds, V2-S seed 1, full-data 3D), tail cells made fail-safe, obsolete runner cell removed, `push_cells.py --no-run` sync, Version #2 "Save & Run All" started 09:50 UTC | commit running |
| 13 | 10:00–10:25 | Persistence: cache-mount cell resumes from any mounted `artifacts_v3/`/`out/`; runner uploads `artifacts_v3/` to the private dataset `harshdipsaha/dat-run3-checkpoints` every 4 jobs (background thread, upgrades the kaggle client first). User supplied a Kaggle API token in `.env`; local CLI works after `pip install -U kaggle` (2.2.4); checkpoint dataset created via CLI; `KAGGLE_USERNAME`/`KAGGLE_KEY` added as notebook Secrets by clipboard paste (token never written to a file or transcript). Draft synced; the running commit predates these and will not upload | commit progress: B0 folds 0.258 / 0.268 / 0.283 / 0.294 / 0.292 / 0.281 / 0.262 / 0.289 |

| 14 | 11:34 | Persistence on the laptop: `scripts/watch_kaggle_output.py` + Task-Scheduler job (every 5 min) downloads and verifies the version output when the run ends | at 14:34 found silently stopped: schtasks defaults to AC-power-only and the laptop was unplugged at 13:40 → task allowed on battery (`kaggle/register_watcher.ps1`) |
| 15 | 14:41 | Commit Version #2 COMPLETE after 4 h 55 min: 56 jobs, failures [] | per-arm OOF: b0 0.2695 (15 ckpts), v2s 0.2740 (5), cor 0.2773 (5), 3d 0.2937 (15), lr 0.4015; stack 0.2580 / AUROC 0.9614; weights 3d .1 / b0 .3 / v2s .3 / cor .3 / lr 0; T 1.0 (random-fold 0.81, floored) |
| 16 | 14:45 | e2e cell inside the commit: every 2D checkpoint `FileNotFoundError`; only 3D + LR ran (smoke 0.2996) | root cause: manifest `model_kw.weights_path=pretrained/…` (training-time ImageNet init) loaded by `build_model` inside the container. Fix `225d39d`: inference drops `weights_path`, manifest strips it, e2e asserts every arm used all checkpoints |
| 17 | 14:47–14:55 | Zip pulled with the CLI (`--file-pattern`), repaired locally (`scripts/repack_v3.py`: fresh code, same artifacts), verified with `scripts/local_e2e_v3.py` | 43/43 checkpoints + LR used; smoke-set log loss 0.2458; 390 s on CPU |
| 18 | 14:52–15:06 | DrivenData smoke test id-321261 (upload 419 MB ≈ 12 min; the old submit script's reload loop raised `beforeunload` dialogs — fixed in `f6b206d`) | **0.2397** (run 2 smoke: 0.2567) |
| 19 | 15:11–15:50 | Real submission id-321263 (slot 2 of 3), uploaded on the user's "now submit" before the smoke result was visible | **public 0.2817, AUROC 0.9477, rank 66** (from 157). Nested OOF of the same combiner 0.2603 → the gap is test-distribution shift (OOF AUROC 0.961 → public 0.948), not combiner optimism |

### Resolved item

The stale 0.998-EMA checkpoints from the interactive session no longer exist (that machine is gone); the commit trains every job fresh. Keep using `pkill -f "[s]cripts/train_v3.py"` for any future cleanup.

## Next

1. Commit Version #2 (started 09:50 UTC, 40 queue-1 jobs + 8 queue-2 jobs + re-OOF + stack/pack + e2e + split; ETA ~15:10 UTC): poll with `kaggle kernels status …` and `.playwright-mcp/poll_commit.js`.
2. When it succeeds: `kaggle kernels output harshdipsaha/dat-parkinsons-train -p data/kaggle_output` (whole output, no size cap) or the version's Output tab → `python scripts/join_parts.py <dir> <md5> --out submission_v3.zip` if only parts are present.
3. DrivenData smoke test → real submission (slot 2 of 3) → record public log loss / AUROC / rank in `RESULTS_LOG.md`.
4. Close this effort; open effort 003.

## Update — Sept 6, 15:20 UTC

- Kaggle Version #2 complete (56 jobs, 0 failures); stack OOF 0.2580 / AUROC 0.961. Container bug (training-time `weights_path` in the manifest) found by the e2e cell, fixed in `225d39d`, zip repaired locally (`scripts/repack_v3.py`) and verified (`scripts/local_e2e_v3.py`: all 43 checkpoints + LR used, smoke 0.2458).
- DrivenData smoke test id-321261 = **0.2397**. Real submission uploaded from `data/submission_v3_fixed.zip` (slot 2/3). Score to be recorded in RESULTS_LOG.md.
- Persistence: Task-Scheduler watcher (`scripts/watch_kaggle_output.py`, battery-safe) archives the version output to `data/kaggle_output/`.
- Next: run 4 per the plan (Part B) — tiny slab-net arm, nested clip/hedge selection, more seeds/TTA; sync the fixed cells to the notebook (`kaggle/gen_run3_inject.py` regenerated; push with `kaggle/push_cells.py --no-run`) before the next commit.

## Closing analysis (Sept 6 15:50 UTC)

- Leader: 0.2192 / AUROC 0.971. We are 0.2817 / 0.948. The public-test AUROC deficit (0.023) is the whole gap; calibration is already honest (nested 0.2603 ≈ plain 0.2580; clip/hedge give no nested gain).
- Per-site nested log loss (cache_v4/sites.csv): best on the 2.40 mm 87-slice ultra-low site (0.19) and 2.46 mm high (0.25); worst on 2.46 mm 256³ low (0.315, n 134), the 3.90 mm thin-FOV sites (0.30) and "other" (0.276).
- The site-fold B0 arm (0.2818) matches the public score (0.2817): treat the public test as a site shift and optimise for leave-site-out performance, not random folds.
- Members: V2-S with one seed (0.2740) already matches three B0 seeds (0.2695) → more V2-S seeds are the cheapest reliable gain. The 3D whole-brain arm is the weakest (0.2937, weight 0.1). Coronal B0 earned weight 0.3 with 5 checkpoints → orthogonal views help.
- Lessons: (1) run `scripts/local_e2e_v3.py` on every zip before uploading; (2) schtasks jobs need battery settings; (3) never reload the DrivenData page during an upload; (4) keep the Kaggle draft session off (GPU quota) and use commits only.

Effort closed; continues as effort 003 (run 4).
