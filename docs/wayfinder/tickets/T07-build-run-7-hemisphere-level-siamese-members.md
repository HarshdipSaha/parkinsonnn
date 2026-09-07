---
id: T07
title: Build run 7: hemisphere-level Siamese members and the super-stack notebook
label: wayfinder:task
type: task
mode: AFK
status: closed
assignee: claude
blocked_by: 
map: ../map.md
---

## Question

Execution carried into the map (see Notes). Build the hemisphere-level member (shared 3D encoder over the left and right striatal half-crops, mirrored to one orientation; head over [f_L, f_R, |f_L − f_R|] plus per-side logits with max pooling, so unilateral onset is an explicit inductive bias), a robust-loss variant, and the run-7 notebook that trains them (random + site folds, 3 seeds) and stacks EVERY arm present in the checkpoint dataset (runs 3–6) with `fit_stack_v3.py --nested --grid 8` over the ≤ 10 best arms by OOF. Deliver `kaggle/run7_hemisphere/` (generator, notebook, kernel-metadata, APPROACH, HOWTO md + pdf) runnable with `kaggle kernels push -p kaggle/run7_hemisphere`.

## Resolution

Built and committed Sept 7 04:48 UTC (commit `ec18566`, spec `docs/superpowers/specs/2026-09-07-run7-hemisphere-spec.md`, effort 006). `kaggle/run7_hemisphere/`: generator → 72-cell notebook + kernel-metadata (id harshdipsaha/dat-run7-hemisphere), APPROACH, HOWTO md + 1-page pdf; `datsc/models_h.py` (HemiNet3D: shared encoder over mirrored half-crops, symmetric pair + max-side head, mirror-invariant logit unit-tested), `--arch hemi` and `--loss bootstrap` on the standard CLI; JOBS7 = 32 jobs ≈ 2.3 h on 2×T4 (3 seeds, site folds, DANN/DRO variant, bootstrapped-loss crop-3D probe, 2 full-data models); super-stack cell keeps the 10 best arm families of runs 3–7 by OOF. Evidence: 66 tests green; selftest OK; real-cache CPU smoke of the member, one epoch: val 0.4128 / AUROC 0.884 (crop-3D at one epoch: 0.471 / 0.862). Launch: `python kaggle/run7_hemisphere/gen_run7.py && kaggle kernels push -p kaggle/run7_hemisphere` (after run 5 on our account, or on the teammate's).
