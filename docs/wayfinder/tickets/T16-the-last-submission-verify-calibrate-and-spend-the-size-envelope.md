---
id: T16
title: The last submission — verify what shipped, price the prior mix honestly, and spend the remaining GPU inside the zip-size envelope
label: ready-for-agent
type: task
mode: solo
status: open
assignee:
blocked_by:
supersedes:
map: ../map.md
spec: ../../superpowers/specs/2026-09-13-run9-endgame-spec.md
plan: ../../superpowers/plans/2026-09-13-run9-endgame-plan.md
---

## Question

With one real submission left (slot opens Sept 14 03:38 UTC, deadline Sept 16 23:59 UTC), 30+ GPU-hours and
0.2266 nested banked, what is shippable that beats 0.2266 without tripping the 850 MB trim or the 150-minute
inference ladder — and is the recorded +0.0084 "λ gain" a model result or an artefact of arms being dropped?

## Evidence (Sept 13)

- **The slot opens, it does not expire.** Next expiry after Sept 14 03:38 is Sept 19, past the deadline, so
  this is the last slot; but it can be spent any time up to Sept 16 23:59. The standing plan spends it at
  03:38 on a zip (nested 0.2323) worse than what is banked (0.2266), discarding ~60 h of runway.
- **λ is exhausted as a lever.** With arms fixed, λ 0.02→0.01 is capped at 0.0055 nats by arithmetic and is
  worth **0.0020** on the real nested pool (1,362 scans, reproduces the shipped 0.2323). The recorded public
  gain of 0.0084 exceeds the hard ceiling, so it did not come from λ. λ 0.01→0 buys ≤0.0018 and raises the
  cost of a confident miss from 5.2 to 9.21 nats. See ADR 0006.
- **Size binds before GPU does.** 847 MB against `--max-mb 850` is ~3 MB of headroom. `DROP_ORDER`'s first
  action is deleting `a2d_v2s` seed 1, measured at ≈0.01 — five times any single new seed's gain. At
  checkpoint size is per-arm, not an average: fp16 3D checkpoints are ≈3.83 MB, `a2d_v2s` ≈37.1 MB,
  `a2d_cnx_cor` ≈51.2 MB. A 1,050 MB ceiling buys ~52 more 3D checkpoints but only ~5 v2s. Neither size nor
  the ladder binds in practice (1421 MB already smoke-scored as `id-321496`; a 71-ckpt run uses ~24 min of the
  150-min ladder). **Diminishing returns is the real cap**: `a3d_hemi` seeds 3→5 bought 0.0001 nats for 10
  checkpoints, so the spend goes only where seeds still pay — ~20 added checkpoints.
- **Two silent-degradation paths were open.** `build_master_v3.py` wrote an oversized zip with no assert once
  `DROP_ORDER` was exhausted; `local_e2e_v3.py`'s arm check could not see the non-torch arms (`featc`, LR),
  so a dead arm still passed the gate. Both are closed.
- **Neither real submission has a DrivenData id** in the log, though `HANDOFF.md`:126 requires one, and the
  entry recording the second was byte-corrupted (3 NUL bytes, collapsed lines). Repaired; the ids still need
  transcribing from the page.
- Rules: real run 3 h, smoke 6 min on 20 scans, no network, no test-set statistics in logs (disqualification).

## Resolved when

The submissions page is transcribed into `RESULTS_LOG.md` with ids and the two `View log` outputs are recorded
(seven `stage written` lines, no `ladder:`); `scripts/sweep_calibration.py` has been run on the real master
pool and its table recorded; the final zip is built at an explicit `--max-mb` with no drop line, its gate table
beats 0.2266 by ≥0.002 or the banked zip is kept; `pytest -q` is green (99); `local_e2e_v3.py` prints
`FORMAT OK` with every arm N/N; the zip is smoke-tested on DrivenData and uploaded as the real submission
before Sept 16 20:00 UTC, with its id, score, AUROC and G written into `RESULTS_LOG.md`.
