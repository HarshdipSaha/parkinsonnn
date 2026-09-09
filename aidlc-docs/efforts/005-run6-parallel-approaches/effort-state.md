# Effort 005: run 6 — three parallel approaches (one Kaggle notebook each, single-command launch)

| | |
|---|---|
| **State** | complete / delivered (all units A, B, C trained and verified; ready for master assembly) |
| **Ask** | User: three subagents, each a unique approach grounded in the EDA/research/results (+ web search), delivering an end-to-end Kaggle notebook runnable with one command from this repo, a one-page HOWTO PDF, and all code; use the smoke-vs-public ladder as a factor; runnable on a teammate's Kaggle account |
| **Delivery** | `kaggle/run6a_crop3d/`, `kaggle/run6b_triplanar/`, `kaggle/run6c_features_ssl/` (each: `gen_run6x.py` → `notebook.ipynb` + `kernel-metadata.json`, `APPROACH.md`, `HOWTO.md` + `HOWTO.pdf`, tests `tests/test_run6x.py`); launch = `kaggle kernels push -p kaggle/<folder>`; second account: `docs/how-to-run-on-a-second-kaggle-account.md` |
| **Depth** | standard |

## Units

| Unit | Approach | Status | Evidence |
|---|---|---|---|
| A | 3D striatal-crop maximiser: SE-ResNet3D family (GroupNorm/SiLU, SE gates, stochastic depth), 48 and 64-voxel crops, 60–80 epochs, new seeds, DANN/DRO variant, full-data models; 27 jobs ≈ 4.7 h on 2×T4 | merged `76252b1` | 53 tests; 1-epoch real-cache smoke (val 0.509 / 0.867); shift TTA measured flat (0.2257 → 0.2251); expected nested 0.240–0.244, public ≈ 0.258–0.262 |
| B | Tri-planar 2.5D fusion (`TriPlanarNet`: shared ConvNeXt-T / V2-S over axial, coronal and two sagittal striatal views, attention pooling, exact mirror TTA), with and without `hist`; 25 jobs ≈ 3.3 h + 40 min re-OOF | completed `skyfall23/dat-run6b-triplanar` | 25 jobs finished on 2xT4 in 8925s; `tri_cnx` (10 ckpts) OOF 0.2560 / 0.9603; took **25.0% stack weight**; final stack OOF 0.2434 / 0.9664; in-container smoke test **0.2139 log loss** (acc@0.5 = 90.0%, 136s runtime); package: `data/run6b/submission_v3.zip` (2275 MB, 73 ckpts) |
| C | 116-descriptor feature arm `featc` (principal-axis A→P profiles, medial/lateral split, hot-region moments, radial shells, GLCM texture, asymmetries; linear won over RFF/MLP) + VICReg self-supervised pretraining of the crop-3D encoder on training scans with `--init` fine-tune; 17 jobs ≈ 2–2.5 h | completed `skyfall23/dat-run6c-features-ssl` | 17 jobs completed on 2xT4 in 12781s; VICReg 91 min pretraining; `featc` RFF OOF 0.2997 (site folds 0.3106 vs old 0.3548); `a3d_ssl` OOF 0.2519 (took **38.0% stack weight**); `featc` took **12.0% weight**; stack OOF 0.2412; in-container smoke test **0.2289 log loss**; package: `submission_v3.zip` (1367 MB) |

## Construction log

| # | When (UTC) | What | Result |
|---|---|---|---|
| 1 | Sept 7 03:50 | Three fork agents launched in isolated worktrees with identical constraints (rules, resume + per-4-job uploads, ≤ 6 h, compositions stack ≤ 10 arms, fail-safe tail, smoke-set scores per arm, < 850 MB zip, no launches) | |
| 2 | 04:10–04:25 | B then A merged into master; notebooks regenerated after the merge so each carries the other's modules; 57 tests green; selftest OK | `fa30329` |
| 3 | 04:25–04:30 | C merged; all three notebooks regenerated after the three-way merge (71 / 70 / 73 cells) and validated; 62 tests green; selftest OK | `HEAD` |
| 4 | Sept 8 14:00–16:35 | Kaggle execution of Unit B `skyfall23/dat-run6b-triplanar` (25 jobs on 2xT4) | Completed. `tri_cnx` OOF 0.2560; stack OOF 0.2434; container smoke 0.2139 log loss; artifacts retrieved to `data/run6b/`. |
| 5 | Sept 8 16:45–20:20 | Kaggle execution of Unit C `skyfall23/dat-run6c-features-ssl` (17 jobs on 2xT4) | Completed. VICReg SSL pretraining 91 min; `a3d_ssl` 38% stack weight; `featc` 12% weight; stack OOF 0.2412; container smoke 0.2289. |

