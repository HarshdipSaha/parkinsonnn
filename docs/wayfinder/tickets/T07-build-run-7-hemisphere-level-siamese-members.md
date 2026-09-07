---
id: T07
title: Build run 7: hemisphere-level Siamese members and the super-stack notebook
label: wayfinder:task
type: task
mode: AFK
status: open
assignee:
blocked_by: 
map: ../map.md
---

## Question

Execution carried into the map (see Notes). Build the hemisphere-level member (shared 3D encoder over the left and right striatal half-crops, mirrored to one orientation; head over [f_L, f_R, |f_L − f_R|] plus per-side logits with max pooling, so unilateral onset is an explicit inductive bias), a robust-loss variant, and the run-7 notebook that trains them (random + site folds, 3 seeds) and stacks EVERY arm present in the checkpoint dataset (runs 3–6) with `fit_stack_v3.py --nested --grid 8` over the ≤ 10 best arms by OOF. Deliver `kaggle/run7_hemisphere/` (generator, notebook, kernel-metadata, APPROACH, HOWTO md + pdf) runnable with `kaggle kernels push -p kaggle/run7_hemisphere`.
