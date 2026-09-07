# Approach C — representation + feature-engineering hybrid (run 6c)

*Subagent C, Sept 7 2026. Branch `worktree-agent-a0d5ea53cdb45d43b`, folder `kaggle/run6c_features_ssl/`.*

## 1. Where we are and what this approach attacks

| Fact (from the repo) | Value |
|---|---|
| Public scores | run 2 0.3391 → run 3 0.2817 → run 4a **0.2667** (rank 39); leader 0.2192 / AUROC 0.971 |
| Nested OOF of the shipped run-4a stack | 0.2491 → public 0.2667: the gap (0.018) is test-site shift, not combiner optimism |
| Smoke → public ladder (20 labelled smoke scans) | 0.2567 → 0.3391, 0.2397 → 0.2817, 0.2176/0.2263 → 0.2667: the smoke set ranks candidates correctly, public ≈ smoke + 0.04–0.05 |
| Members | every kept arm is a CNN on the registered box (crop-3D 0.2568, V2-S 0.2708, ConvNeXt 0.2746, coronal B0 0.2773) plus one linear feature arm (0.3120, weight 0.125–0.25) |
| Diagnosed weakness | low-count / thin-FOV pseudo-sites (2.46 mm 256³ low 0.31, 3.90 mm 39-slice 0.36 nested) and generalisation to unseen sites |

The ensemble's members are highly correlated (same input, same ImageNet family). Stacking rewards members whose *errors* are
different. Approach C therefore adds the two most different learners we can build without external data or ImageNet:

1. **`featc` — a much stronger hand-crafted arm.** 116 descriptors of the registered striatum computed label-blind in template
   space (`datsc/features_c.py`): the run-3/4 blocks (17 + 44) plus 62 new ones — principal-axis uptake profiles per hemisphere
   (6 bins along each striatum's own geometric long axis, caudate head → posterior putamen, plus slope), medial/lateral halves
   (caudate vs putamen proxy) and their ratio, intensity-weighted second moments of the hot region (elongation, flatness,
   orientation vs the A-P axis, centroid offsets, hot volume), radial shell profiles (0–3–6–9–12–16 mm) around each hot-spot
   centroid, grey-level co-occurrence texture of the 48×48×32 striatal box (contrast, homogeneity, energy, correlation; numpy
   only), and the L/R asymmetry of all of them. Model (`datsc/featmodel_c.py`): descriptors + 40 PCA scores of the striatal patch
   → standardise → logistic on [linear features + random Fourier features] (RBF-kernel logistic with a linear pass-through) or a
   tanh MLP; everything is numpy at inference (npz, no pickle), fitted on both mirror orientations and averaged, with the
   Mahalanobis typicality statistics kept. Radiomics of the striatum (texture + shape) has matched or beaten SBR indices for
   PD vs. control in the DAT-SPECT literature (Rahmim-style radiomics signatures, EJNMMI Research 2022; radiomics + deep
   features, EJNMMI Physics 2024), which is why the block leans on shape/texture rather than more intensity statistics.
2. **`a3d_ssl` — self-supervised striatal-crop encoder.** VICReg pretraining (`datsc/ssl_c.py`) of the run-3 GroupNorm/SiLU
   ResNet3D on the *training* scans only, labels unused, test scans never touched: two views per scan from two independent
   draws of our physical augmentations (flip, small affine, blur, count noise, quantisation, reference-scale jitter) plus a
   ±2-voxel crop-centre jitter; invariance + variance + covariance terms (no negatives → stable at batch 32 on a T4). The
   encoder is then loaded into the supervised crop-3D arm (`scripts/train_v3.py --init`, backbone tensors only) and fine-tuned
   on random folds (2 seeds), leave-site-out folds (the R10 site-robustness report) and full data. Rationale: contrastive /
   non-contrastive pretraining is the best-established way to get more out of ~1k labelled medical volumes (Wolf et al.,
   *Sci. Rep.* 2023, contrastive and masked-autoencoder pretraining for small medical datasets; VoCo, CVPR 2024, volume
   contrast that exploits consistent anatomy — our registered frame gives that consistency for free), and it regularises
   toward augmentation-invariant, hence camera-invariant, features, which is the diagnosed failure mode.

Both are stacked with the run-4a survivors by `scripts/fit_stack_v3.py --nested` (compositions enumeration, ≤ 10 arms);
members that earn no weight are dropped by `scripts/run4_assemble.py`. The notebook's end-to-end cell prints the
**smoke-set log loss of every arm and of the stack** (`kaggle/run6c_features_ssl/e2e_c.py`) so the smoke→public ladder can be
read directly; the 20 smoke scans are never used for fitting.

## 2. Measured on CPU before any Kaggle run

Feature arm C (`python scripts/feat_arm_c.py --grid`, 1,362 scans × 2 orientations, 0.22 s per scan per orientation):

| Model | random-fold OOF log loss / AUROC |
|---|---|
| linear, PCA 40, C 0.1 (**chosen**) | **0.3011 / 0.9444** |
| RFF-kernel logistic (C 1.0, γ 0.25, 512 features) | 0.3075 / 0.9458 |
| linear C 0.3 | 0.3088 / 0.9441 |
| MLP (32) α 1.0 | 0.3404 / 0.9421 |
| run-4 feature arm (reference) | 0.3120 / 0.9398 |

Leave-pseudo-site-out: **0.3264 / 0.9362** (run-4 feature arm 0.3548 / 0.9277) — a −0.028 gain exactly on the axis that hurts on the
public test. Strongest new descriptors: principal-axis bins 1–2 (AUROC 0.86) and the 9–12 mm radial shell (0.85).

Stack effect with the run-4a survivors (nested, grid 10): 0.2499 → **0.2493**, plain OOF 0.2455 → 0.2448; `featc` takes the feature
weight (0.1) and the old feature arm drops to 0. Modest in the stack because the CNN arms already carry most of the signal;
the value of `featc` is its site robustness and its independence from the CNN family.

SSL entry point smoke (`scripts/ssl_pretrain_c.py --max-steps 3`, real cache, CPU): loss finite, encoder saved and loadable,
`--init` fine-tune loads 23 backbone tensors (tests). Its OOF value can only be measured on Kaggle (≈ 30 min pretraining + 60 min
fine-tuning on 2×T4).

Container-style check (`scripts/local_e2e_v3.py`) of a mini zip with the crop-3D arm + `featc`: see the line appended at the end.

## 3. Budget, risks, expectation

- Kaggle GPU T4 x2: features 15 min (CPU), SSL 200 epochs ≈ 30 min, 17 fine-tune jobs ≈ 60 min on two GPUs, tail ≈ 20 min → **≈ 2–2.5 h**.
  Resumes from `dat-run3-checkpoints`; every checkpoint is uploaded there every 4 jobs. Inference: featc adds ≈ 0.5 s per scan on
  CPU (two orientations) — negligible against the 3 h budget; zip stays < 850 MB (the SSL arm is 5 MB per checkpoint).
- Risks: VICReg on 1.3k scans may under-perform the from-scratch crop-3D (then `a3d_ssl` simply gets weight 0 — nothing lost but
  90 GPU-min); the first SSL run uses the base augmentation and 200 epochs (the strong preset hurt supervised training in run 4a;
  it could still be right for SSL and is a one-flag experiment). The RFF/MLP models are kept in the grid; on the current
  features the linear model wins.
- Expected gain: `featc` alone −0.001 nested (measured); `a3d_ssl` −0.003 to −0.01 nested if the pretraining transfers (it is the
  best single arm's backbone, warm-started), and a smaller random→site gap. Honest public expectation for this notebook's stack:
  0.255–0.262 (from 0.2667). The site-robustness mechanisms of run 5 and, above all, external multi-site data (PPMI) remain the
  levers for 0.24 and below; approach C composes with both (its members are independent of them).

**Container-style local check (mini zip: crop-3D 5 ckpts + featc, 21 MB) on the 20 smoke scans:** every arm N/N (`featc model (linear) on 20 volumes`), FORMAT OK, 64 s on CPU, smoke log loss 0.2889 for that 2-arm mini stack (the run-4a 5-arm stack scores 0.2176 on the same scans; this only verifies the featc inference path).
