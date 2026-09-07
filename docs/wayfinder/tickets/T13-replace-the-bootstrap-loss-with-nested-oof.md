---
id: T13
title: Replace the bootstrap loss with nested-OOF soft targets in run 7
label: wayfinder:task
type: task
mode: AFK
status: open
assignee:
blocked_by: 
map: ../map.md
---

## Question

T03 found that the only label-noise treatment with log-loss evidence is soft targets that carry uncertainty: y* = (1−λ)·y + λ·p_OOF (λ 0.3–0.5), where p_OOF is the current stack's out-of-fold probability for each training scan. Implement `--loss soft --soft-lam 0.4` reading `artifacts_v3/oof_stack.csv` (fallback: plain targets when a uid is missing), replace the `r3d_cropboot` job in `kaggle/run7_hemisphere/gen_run7.py` with it, unit-test the target construction, regenerate the notebook. Resolved when tests are green and the notebook is regenerated; the gate (≥ 0.003 nested) is read after the run.
