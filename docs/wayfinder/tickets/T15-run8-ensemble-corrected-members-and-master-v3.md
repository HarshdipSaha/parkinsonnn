---
id: T15
title: Run 8 — ensemble-corrected members (nested-OOF soft targets, mistake weights, extra seeds) and the final blend Master v3
label: ready-for-agent
type: task
mode: AFK
status: open
assignee:
blocked_by:
supersedes: T13
map: ../map.md
spec: ../../superpowers/specs/2026-09-10-run8-distill-spec.md
plan: ../../superpowers/plans/2026-09-10-run8-distill-plan.md
---

## Question

Can members trained on the ensemble's nested out-of-fold beliefs (soft targets λ 0.4; a mistake-weighted variant) plus two extra hemisphere seeds lift the nested estimate of the final blend by ≥ 0.003 over Master v2 (0.2323, equal weights, T 0.9) in one ≈ 5-h session on the teammate's account before the Sept 12 15:38 UTC slot?

## Evidence (Sept 10)

- Master v2 nested loss is concentrated: top-100 scans carry 54 %, 94 scans lose > 1 nat, 46 confident mistakes; capping per-scan loss at 2 would give 0.2201 (ceiling, not expectation).
- Self-bootstrap targets (run 7 `r3d_cropboot`) 0.2565 vs plain 0.2568: the soft target must come from the ensemble, not the model itself.
- Hemisphere fold ≈ 12.4 min on a T4; 42 jobs ≈ 8.7 GPU-h ≈ 4.5 h wall on two T4s.

## Resolved when

`tests/test_run8.py` green; `kaggle/run8_distill/notebook.ipynb` regenerated and pushed as `skyfall23/dat-run8-distill`; `scripts/build_master_v3.py` prints the gate table (Master v2 vs + each new arm vs best set, fitted/equal/shrunk) and the container check passes; the slot decision recorded in `RESULTS_LOG.md` by the written rule (v3 replaces v2 in slot 1 only if nested gain ≥ 0.003 and platform smoke within 0.01 of 0.2381).
