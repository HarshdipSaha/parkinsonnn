---
id: T05
title: Read out run 5: do site-invariant training or histogram standardisation shrink the shift gap?
label: wayfinder:task
type: task
mode: AFK
status: open
assignee:
blocked_by: 
map: ../map.md
---

## Question

When Kaggle Version 4 (run 5) completes: assemble with `scripts/run4_assemble.py --src data/kaggle_output_v5 --extra-dir artifacts/featcarm_v5`, compare `r3d_cropinv` (DANN 0.1 + DRO 0.1) with `r3d_crop48x48x32` on random AND leave-site-out folds, `r3d_crophist` vs `r3d_crop48x48x32`, and record which mechanism (if any) reduces the random→site gap by ≥ 0.005 and the nested estimate by ≥ 0.003. The answer fixes the training objective and normalisation for run 7.
