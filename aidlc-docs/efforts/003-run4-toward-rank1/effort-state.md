# Effort 003 — run 4: measured additions toward rank 1
state: planning (starts after effort 002's real submission)
plan: docs/superpowers/plans/2026-09-06-run3-finish-and-run4.md (Part B)
## Requirements delta (vs baseline)
- NEW R7: every ensemble change is gated on a NESTED out-of-fold estimate of the stacked log loss (>= 0.005 gain) with per-pseudo-site reporting.
- NEW R8: multi-shift test-time augmentation must be identical in OOF (reoof_v3.py) and inference (main_v3.py) — InputSpec.variants is the single source.
- NEW R9: full-data (fold -1) models may be added to arms via `--extra`; they never contribute OOF.
## Candidate units (see plan Part B): B1 nested estimate + hard-fold analysis; B2 diversity members (ConvNeXt-T, 5-slab B0, coronal V2-S); B3 3D upgrades (96^3, longer, shift TTA); B4 seeds + full-data; B5 registration refinement (gated on G1); B6 submission cadence.
## Expected outcome (honest): 0.235-0.25 public after B1-B4; rank 1 (< 0.219) ~10-15 % by deadline.
