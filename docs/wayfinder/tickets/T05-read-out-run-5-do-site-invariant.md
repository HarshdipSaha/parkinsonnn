---
id: T05
title: Read out run 5: do site-invariant training or histogram standardisation shrink the shift gap?
label: wayfinder:task
type: task
mode: AFK
status: closed
assignee: claude
blocked_by: 
map: ../map.md
---

## Question

When Kaggle Version 4 (run 5) completes: assemble with `scripts/run4_assemble.py --src data/kaggle_output_v5 --extra-dir artifacts/featcarm_v5`, compare `r3d_cropinv` (DANN 0.1 + DRO 0.1) with `r3d_crop48x48x32` on random AND leave-site-out folds, `r3d_crophist` vs `r3d_crop48x48x32`, and record which mechanism (if any) reduces the random→site gap by ≥ 0.005 and the nested estimate by ≥ 0.003. The answer fixes the training objective and normalisation for run 7.

## Resolution

Resolved Sept 7 10:45 UTC.
- `r3d_cropinv` (DANN + DRO): random 0.2885 / AUROC 0.9493, site folds 0.2884 / AUROC 0.9501 vs baseline `r3d_crop` (random 0.2568 / 0.9603, site 0.2589 / 0.9598). While the gap closed, discrimination degraded by +0.030 across both sets. **OUT** (fails gate).
- `r3d_crophist` (reference histogram standardisation): random **0.2539** / AUROC **0.9614** (beats baseline 0.2568 / 0.9603). Enters stack as primary 3D arm with 37.5% weight, displacing plain crop-3D. **IN**.
- Stack nested estimate: **0.2433** (vs Run 4a 0.2491, gain -0.0058); stack OOF **0.2394** / AUROC **0.9672**.
- Decision for Run 7: Run 7 trains with `hist` normalisation enabled and without DANN/DRO.

