# Effort 005: run 6 — three parallel approaches (one Kaggle notebook each, single-command launch)

| | |
|---|---|
| **State** | in-progress (opened Sept 7 03:50 UTC after submission 3 scored 0.2667 / rank 39) |
| **Ask** | User: three subagents, each a unique approach grounded in the EDA/research/results (+ web search), delivering an end-to-end Kaggle notebook runnable with one command from this repo, a one-page HOWTO PDF, and all code; use the smoke-vs-public ladder as a factor; runnable on a teammate's Kaggle account |
| **Delivery** | `kaggle/run6a_crop3d/`, `kaggle/run6b_triplanar/`, `kaggle/run6c_features_ssl/` (each: `gen_run6x.py` → `notebook.ipynb` + `kernel-metadata.json`, `APPROACH.md`, `HOWTO.md` + `HOWTO.pdf`, tests `tests/test_run6x.py`); launch = `kaggle kernels push -p kaggle/<folder>`; second account: `docs/how-to-run-on-a-second-kaggle-account.md` |
| **Depth** | standard |

## Units

| Unit | Approach | Status | Evidence |
|---|---|---|---|
| A | 3D striatal-crop maximiser: SE-ResNet3D family (GroupNorm/SiLU, SE gates, stochastic depth), 48 and 64-voxel crops, 60–80 epochs, new seeds, DANN/DRO variant, full-data models; 27 jobs ≈ 4.7 h on 2×T4 | merged `76252b1` | 53 tests; 1-epoch real-cache smoke (val 0.509 / 0.867); shift TTA measured flat (0.2257 → 0.2251); expected nested 0.240–0.244, public ≈ 0.258–0.262 |
| B | Tri-planar 2.5D fusion (`TriPlanarNet`: shared ConvNeXt-T / V2-S over axial, coronal and two sagittal striatal views, attention pooling, exact mirror TTA), with and without `hist`; 25 jobs ≈ 3.3 h + 40 min re-OOF | merged `5c9fdd1` | 52 tests; real-weights forward 28.1 M params; 1-epoch smoke 0.507 / 0.873 and re-OOF contract check; expected single tri 0.255–0.265, nested 0.240–0.245, public ≈ 0.255–0.26 |
| C | Richer radiomics/feature arm (numpy-serialisable) + self-supervised 3D encoder on training scans, linear probe + fine-tune | running | — |

## Construction log

| # | When (UTC) | What | Result |
|---|---|---|---|
| 1 | Sept 7 03:50 | Three fork agents launched in isolated worktrees with identical constraints (rules, resume + per-4-job uploads, ≤ 6 h, compositions stack ≤ 10 arms, fail-safe tail, smoke-set scores per arm, < 850 MB zip, no launches) | |
| 2 | 04:20–04:35 | B then A merged into master; notebooks regenerated after the merge so each carries the other's modules; 57 tests green; selftest OK | `fa30329` |
