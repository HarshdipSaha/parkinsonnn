# Effort 001: run 1/2, hotspot-anchored crop pipeline

| | |
|---|---|
| **State** | complete (superseded by effort 002) |
| **Started / finished** | 2026-09-05 / 2026-09-05 |
| **Depth** | minimal |

## Requirements delta

- NEW: template-free hotspot / centre-of-mass crop at 3 mm, 0.9M-parameter 3D CNN plus radiomics logistic regression, calibration with blend weight, temperature and prior mix (the council verdict).

## Outcome

| Metric | Value |
|---|---|
| Out-of-fold log loss (15 checkpoints, best-epoch selection) | 0.2999 |
| Public log loss / AUROC / rank | 0.3391 / 0.925 / 152 |
| Verdict | Failed the top-rank goal |

Root causes (see `docs/RUN3_BRAINSTORM.md`):

- label-correlated hotspot-anchor failures (19.5 % of abnormals vs 2.4 % of normals);
- unnormalised spatial frame (striatal plane wandered between slice 7 and 30 of 48);
- site-fingerprint scalars fed to the model heads;
- best-epoch selection on the validation fold made the OOF optimistic.

## Artefacts

- `submission.zip` (50.7 MB) → DrivenData id-321156.
- `artifacts/run2_2026-09-05/` (checkpoints, OOF csvs, canonical `folds.csv`).
- `RESULTS_LOG.md`, section "Run 2".
