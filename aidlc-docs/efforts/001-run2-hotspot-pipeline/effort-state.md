# Effort 001 — run 1/2: hotspot-anchored crop pipeline
state: complete (superseded)
started: 2026-09-05  finished: 2026-09-05
## Requirements delta
- NEW: template-free hotspot/CoM crop at 3 mm, 0.9M 3D CNN + radiomics LR, w/T/lambda calibration (council verdict).
## Outcome
- OOF log loss 0.2999 (15 checkpoints) but PUBLIC 0.3391, AUROC 0.925, rank 152 -> FAILED the top-rank goal.
- Root causes (docs/RUN3_BRAINSTORM.md): label-correlated hotspot anchor failures, unnormalised frame, site-fingerprint scalars in the head, best-epoch OOF optimism.
## Artifacts
- submission.zip (50.7 MB) -> DrivenData id-321156; artifacts/run2_2026-09-05/ (checkpoints, OOF csvs, folds); RESULTS_LOG.md "Run 2".
