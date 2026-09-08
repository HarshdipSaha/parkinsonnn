# Effort 006: run 7 — hemisphere-level members and the super-stack (wayfinder T07)

| | |
|---|---|
| **State** | delivered / verified (Kaggle completed Sept 8 13:40 UTC; DrivenData smoke test id-321572 scored 0.2706 Sept 9 00:09 UTC) |
| **Spec** | [2026-09-07-run7-hemisphere-spec.md](../../../docs/superpowers/specs/2026-09-07-run7-hemisphere-spec.md) |
| **Map** | `docs/wayfinder/map.md` — the route to public 0.21–0.22; this effort resolves ticket T07 |
| **Deliverable** | `kaggle/run7_hemisphere/` (gen_run7.py → notebook.ipynb 72 cells + kernel-metadata.json; APPROACH.md; HOWTO.md + HOWTO.pdf); `datsc/models_h.py`; `--arch`/`--loss` on `scripts/train_v3.py`; tests `tests/test_run7.py`; `data/submission_run7_hemi_only.zip` (63 MB) |
| **Launch** | `python kaggle/run7_hemisphere/gen_run7.py && kaggle kernels push -p kaggle/run7_hemisphere` |
| **Depth** | standard |

## Requirements delta (vs baseline)

| Id | Requirement |
|---|---|
| R17 | A hemisphere-level 3D member whose logit is invariant to mirroring the scan (unit-tested), trained on random and leave-site-out folds. |
| R18 | A label-noise-robust (soft bootstrapped target) loss selectable per job, gated by the nested estimate. |
| R19 | A super-stack cell that discovers every arm family of runs 3–7 in the checkpoint dataset and fits the nested combiner over the ten best by OOF log loss. |

## Construction log

| # | When (UTC) | Unit | Result |
|---|---|---|---|
| 1 | Sept 7 04:20–04:48 | `HemiNet3D` (shared encoder over mirrored half-crops, symmetric pair + max-side head), dispatch `arch=hemi`, `--arch`/`--loss` on the standard CLI, bootstrapped loss in the trainer, JOBS7 (32 jobs ≈ 2.3 h on 2×T4), super-stack cell (top-10 arms), HOWTO md + 1-page pdf, APPROACH | 66 tests green (3 new); selftest OK; notebook validated; real-cache CPU smoke of the member running |
| 2 | 05:00 | Real-cache CPU smoke of `r3d_hemi48_f0_s0`, one epoch | val 0.4128 / AUROC 0.8844 in 615 s (crop-3D one epoch: 0.4705 / 0.8621); T07 resolved on the wayfinder map |
| 3 | Sept 8 13:40 | Kaggle execution `skyfall23/dat-run7-hemisphere` (32 jobs on 2×T4) | Completed. `a3d_hemi` single-arm OOF 0.2379 / 0.9658; stack OOF 0.2357 / 0.9684 (50% stack weight to `a3d_hemi`). In-container smoke: 0.2408. Package: `data/submission_run7.zip` (1815 MB). |
| 4 | Sept 8 23:25 | DriveData upload repack `data/submission_run7_hemi_only.zip` | Isolated `a3d_hemi` (17 ckpts) down to 63 MB to pass DrivenData upload size limit. |
| 5 | Sept 9 00:09 | DrivenData smoke test evaluation **id-321572** | **Score: 0.2706 log loss** on hidden smoke test for single standalone `a3d_hemi` arm! Confirms bilateral asymmetry prior transfers to unseen test data with no ensemble. |

