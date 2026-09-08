# Effort 005: run 6 — three parallel approaches (one Kaggle notebook each, single-command launch)

| | |
|---|---|
| **State** | in-progress (B completed on Kaggle Sept 8 16:35 UTC; C running; A merged) |
| **Ask** | User: three subagents, each a unique approach grounded in the EDA/research/results (+ web search), delivering an end-to-end Kaggle notebook runnable with one command from this repo, a one-page HOWTO PDF, and all code; use the smoke-vs-public ladder as a factor; runnable on a teammate's Kaggle account |
| **Delivery** | `kaggle/run6a_crop3d/`, `kaggle/run6b_triplanar/`, `kaggle/run6c_features_ssl/` (each: `gen_run6x.py` → `notebook.ipynb` + `kernel-metadata.json`, `APPROACH.md`, `HOWTO.md` + `HOWTO.pdf`, tests `tests/test_run6x.py`); launch = `kaggle kernels push -p kaggle/<folder>`; second account: `docs/how-to-run-on-a-second-kaggle-account.md` |
| **Depth** | standard |

## Units

| Unit | Approach | Status | Evidence |
|---|---|---|---|
| A | 3D striatal-crop maximiser: SE-ResNet3D family (GroupNorm/SiLU, SE gates, stochastic depth), 48 and 64-voxel crops, 60–80 epochs, new seeds, DANN/DRO variant, full-data models; 27 jobs ≈ 4.7 h on 2×T4 | merged `76252b1` | 53 tests; 1-epoch real-cache smoke (val 0.509 / 0.867); shift TTA measured flat (0.2257 → 0.2251); expected nested 0.240–0.244, public ≈ 0.258–0.262 |
| B | Tri-planar 2.5D fusion (`TriPlanarNet`: shared ConvNeXt-T / V2-S over axial, coronal and two sagittal striatal views, attention pooling, exact mirror TTA), with and without `hist`; 25 jobs ≈ 3.3 h + 40 min re-OOF | completed `skyfall23/dat-run6b-triplanar` | 25 jobs finished on 2xT4 in 8925s; `tri_cnx` (10 ckpts) OOF 0.2560 / 0.9603; took **25.0% stack weight**; final stack OOF 0.2434 / 0.9664; in-container smoke test **0.2139 log loss** (acc@0.5 = 90.0%, 136s runtime); package: `data/run6b/submission_v3.zip` (2275 MB, 73 ckpts) |
| C | 116-descriptor feature arm `featc` (principal-axis A→P profiles, medial/lateral split, hot-region moments, radial shells, GLCM texture, asymmetries; linear won over RFF/MLP) + VICReg self-supervised pretraining of the crop-3D encoder on training scans with `--init` fine-tune; 17 jobs ≈ 2–2.5 h | running `skyfall23/dat-run6c-features-ssl` | 17 jobs launched Sept 8 16:45 UTC on 2xT4; SSL + 116 radiomics descriptors; kernel actively executing |

## Construction log

| # | When (UTC) | What | Result |
|---|---|---|---|
| 1 | Sept 7 03:50 | Three fork agents launched in isolated worktrees with identical constraints (rules, resume + per-4-job uploads, ≤ 6 h, compositions stack ≤ 10 arms, fail-safe tail, smoke-set scores per arm, < 850 MB zip, no launches) | |
| 2 | 04:10–04:25 | B then A merged into master; notebooks regenerated after the merge so each carries the other's modules; 57 tests green; selftest OK | `fa30329` |
| 3 | 04:25–04:30 | C merged; all three notebooks regenerated after the three-way merge (71 / 70 / 73 cells) and validated; 62 tests green; selftest OK | `HEAD` |
| 4 | Sept 8 14:00–16:35 | Kaggle execution of Unit B `skyfall23/dat-run6b-triplanar` (25 jobs on 2xT4) | Completed. `tri_cnx` OOF 0.2560; stack OOF 0.2434; container smoke 0.2139 log loss; artifacts retrieved to `data/run6b/`. |
| 5 | Sept 8 16:45 | Kaggle execution of Unit C `skyfall23/dat-run6c-features-ssl` (17 jobs on 2xT4) | Launched via `kaggle kernels push -p kaggle/run6c_features_ssl` after patching `--owner skyfall23`. Running. |

