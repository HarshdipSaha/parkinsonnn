# Results log

All numbers are 5-fold StratifiedGroupKFold out-of-fold (OOF) on the 1,362 training scans unless stated. Metric = log loss (lower is better). Prior-only baseline = 0.6884.

## Run 2 — fixed interior anchor, 2 seeds, dual-crop ensemble (Kaggle T4 x2, 2026-09-05)

Kaggle notebook `dat-parkinsons-train`, session of ~3 h. Cache: 1362/1362 volumes in 1281 s, 161 (11.8 %) fallback anchors, 0 failures. Alignment gate (14 hotspot radiomics + L2 logreg): OOF AUROC **0.8615**, log loss 0.4661 (gate target 0.90 not met; proceeded on CNN validation loss).

Per-checkpoint best validation log loss (40 epochs, batch 16, AMP, ~16 s/epoch):

| run | best val logloss | crop |
|---|---|---|
| f0_s0 | 0.4139 | hotspot |
| f0_s1 | 0.3104 | hotspot |
| f1_s0 | 0.3713 | hotspot |
| f1_s1 | 0.3552 | hotspot |
| f2_s0 | 0.2863 | hotspot |
| f2_s1 | 0.3267 | hotspot |
| f3_s0 | 0.2951 | hotspot |
| f3_s1 | 0.3189 | hotspot |
| f4_s0 | 0.2847 | hotspot |
| f4_s1 | 0.2778 | hotspot |
| f0_s0_com | 0.2661 | centre-of-mass (run 1) |
| f1_s0_com | 0.3082 | centre-of-mass (run 1) |
| f2_s0_com | 0.2811 | centre-of-mass (run 1) |
| f3_s0_com | 0.3355 | centre-of-mass (run 1) |
| f4_s0_com | 0.3205 | centre-of-mass (run 1) |

- Mean val log loss: hotspot arm **0.3240**, CoM control **0.3023** (hotspot gain = -0.022, i.e. the hotspot crop is slightly *worse* than the plain centre-of-mass crop; fold 0/1 hotspot seeds are the weak ones).
- 15-checkpoint average OOF log loss **0.2994**; LR arm 0.4661.
- Calibration fit: w = 1.00 (CNN only), T = 0.913, lam = 0.02, pi = 0.548 -> OOF **0.2999**.
- `submission.zip` = 50.7 MB, 31 files, 15 checkpoints. Local copy: `artifacts/run2_2026-09-05/` (zip + extracted checkpoints/JSON).

### Submissions

| id | type | zip | status | score |
|---|---|---|---|---|
| 321144 | smoke test | submission_prior.zip (prior only) | Completed | 0.6687 |
| 321152 | smoke test | submission.zip (run 2) | Completed (8 s, 15/15 checkpoints, exit 0) | 0.2567 |
| 321156 | **real** (slot 1 of 3) | submission.zip (run 2) | submitted 2026-09-05 15:38 UTC | pending |

Kaggle-side checks on the packed zip: end-to-end on the 20 smoke scans = 0.2567 (identical to DrivenData), 32 s wall on 4 CPUs; 150 training volumes in 204 s on 4 CPUs -> extrapolated ~12 min for 2,000 test volumes on the 24-vCPU/A100 box (limit 180 min).

### Takeaways for run 3

1. Hotspot anchoring did not beat CoM; the two crops are complementary in the ensemble (CoM checkpoints are among the best). Keep both, and train CoM checkpoints for seed 1 as well (5 more jobs, ~35 min).
2. Fold 0 and fold 1 hotspot checkpoints are the outliers (0.41, 0.37); a third seed would stabilise the average.
3. The alignment gate is still below target: the anchor is the known weak point. Candidates: larger crop (88x88x56), or a learned/registered anchor.

## Run 1 — original anchor (2026-09-05, superseded)

Hotspot folds 0.353/0.290/0.266/0.354/0.321 (mean 0.317); CoM control mean 0.302; calibration w = 1.0, T = 0.989, OOF 0.318. The five CoM checkpoints from this run are the `_com` members of the run-2 ensemble.
