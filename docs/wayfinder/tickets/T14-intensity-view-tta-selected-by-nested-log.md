---
id: T14
title: Intensity-view TTA selected by nested log loss
label: wayfinder:task
type: task
mode: AFK
status: open
assignee:
blocked_by: T05
map: ../map.md
---

## Question

T03 ranks intensity-domain test-time views (e.g. small reference-scale jitter, blur/deblur, hist on/off) above geometric ones for the site gap. Add an `intensity` variant family to `InputSpec.variants` (reproducible, shared by re-OOF and inference), measure per arm with `scripts/reoof_v3.py` on the run-4a/5 checkpoints, keep only variants that improve the nested estimate by ≥ 0.002. Blocked by T05 because the hist decision changes the candidate set.
