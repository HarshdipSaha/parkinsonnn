# Approach A — the striatal-crop 3D maximiser (run 6a)

## 1. Where the evidence points

Run 4a (public 0.2667, rank 39; nested OOF 0.2491) was won by one member: the 48×48×32-voxel striatal-crop 3D net (2 mm, 2.0 M params, 5 folds, one seed) — OOF **0.2568 / AUROC 0.960**, better than any 2.5D ImageNet arm (V2-S 3 seeds 0.2651, ConvNeXt 0.2807, B0 0.2695) and it took the largest stack weight (0.375). Three facts from `analysis_a.py` on the run-4a artefacts say it is also the member with the most head-room:

| Fact | Number | Consequence |
|---|---|---|
| Under-trained: validation loss still falling in the last ten epochs (fold 0: 0.245 → 0.234, fold 1: 0.308 → 0.303, fold 4 flat) | 50-epoch cosine schedule | longer schedule (80 epochs) for the same net |
| One seed only; the B0 arm gained 0.0057 from single seed → 3-seed averaging | seed gain ≈ 0.005–0.008 | two more seeds (80 ep) + full-data models |
| Least correlated with the 2D arms: logit r = 0.93 vs V2-S and ConvNeXt, while V2-S–ConvNeXt is 0.97 | diversity | every improvement of this arm transfers to the stack |
| Shallow: 1 residual block per stage, no channel attention | 2.0 M params, ~15 s/epoch on a T4 | an SE / stochastic-depth family at 1.3–1.9× the cost |
| Per-site: worst on 3.90 mm 39-slice (0.469, n 33), 2.46 mm 256³ low (0.336) and "other" (0.287); best on the ultra-low 87-slice site (0.171) | site shift again | a site-invariant variant (DANN + group-DRO) of the same net |
| Flip TTA helps (0.2340 → 0.2257 on fold 0); integer x/y/z shift TTA adds nothing (0.2251/0.2252) | `tta_shift_probe_a.py` | keep flip-only TTA (run 5's affine TTA was also flat) |

## 2. The design

Five 3D arms, all sharing the trainer, the base augmentation (strong augmentation hurt on both fold types in run 4a), mixup 0.2, AdamW 5e-4, EMA 0.995 (short window — the run-3 lesson), fp16 NaN guards, and flip TTA:

1. **`r3d_crop48e80`** — the run-4a net trained for 80 epochs, seeds 1 and 2 (+ 2 full-data models). Expected: −0.005 to −0.01 per fold from the longer schedule plus the seed-averaging gain.
2. **`r3d_se48`** — `datsc/models_a.py::SEResNet3D`: widths 32–64–128–256, squeeze-and-excitation channel gates (Hu et al. 2018) and linearly increasing stochastic depth up to 0.1 (Huang et al. 2016) on every residual branch, GroupNorm/SiLU, 60 epochs. 3.6 M params, 1.35× the base cost. Channel attention is the one architectural change that repeatedly pays on small-volume medical classification (e.g. SE on fine-tuned CNNs for MRI tumour classification, [PMC11970440](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11970440/)); stochastic depth is the regulariser that lets a deeper net train on 1,089 scans per fold.
3. **`r3d_crop64e60`** — the base net on a 64×64×40 crop (128×128×80 mm): the crop-64 B0 beat plain B0 in 2D (0.2654 vs 0.2752), so more context around the striatum (occipital reference, ventricles, head shape) is worth one seed; 1.9× the cost, 60 epochs.
4. **`r3d_cropinv`** — DANN 0.1 + group-DRO 0.1 on the base net (the run-5 recipe, same tag: whichever notebook finishes first is reused). Targets the site shift that costs 0.018 between nested OOF and public.
5. **`r3d_crop`** — the existing run-4a checkpoints stay as their own arm.

Stacking: `scripts/fit_stack_v3.py` with the compositions enumeration (the 9^11 hang is fixed), ten arms (the five 3D arms + V2-S, ConvNeXt, coronal B0, feature arm, LR), grid 8, nested estimate, typicality option (`--typ`, gated), full-data extras appended to the 80-epoch arm. `scripts/run4_assemble.py` drops zero-weight arms after the run.

The smoke factor: the end-to-end cell (`e2e_check_a.py`) runs the packed zip exactly as the DrivenData container does and then scores **every arm** and the stack on the 20 labelled smoke scans (written to `out/smoke_per_arm.json`). The ladder so far is monotone (run 2 0.2567 → 0.3391 public, run 3 0.2397 → 0.2817, run 4a 0.2176 → 0.2667), so a member whose smoke loss is far worse than its OOF is a site-shift casualty; nothing is ever fitted on those scans.

Not done, and why: SWA (EMA already averages the trajectory; SWA on top of a short cosine schedule brought nothing in the run-3 diagnostics), affine/shift TTA (measured flat), strong augmentation (measured harmful), 3D-to-2D distillation ([Sci Rep 2024](https://www.nature.com/articles/s41598-024-80938-6), promising on PPMI DaTscan but needs a second training stage the budget does not have), transformer/diffusion 3D classifiers ([Diff3Dformer](https://arxiv.org/pdf/2406.17173): built for CT slice sequences, far too heavy for 1,362 scans on T4s). Recent evidence that test-time augmentation can hurt medical classifiers ([arXiv 2604.09697](https://arxiv.org/pdf/2604.09697)) is consistent with our flat affine-TTA measurement; EMA's calibration and robustness benefits ([Morales-Brotons et al. 2024](https://arxiv.org/abs/2411.18704)) are why the final EMA weights are kept.

## 3. Budget (two T4s, one Kaggle commit)

| Job | Folds × seeds | min / fold (T4) | GPU-min |
|---|---|---|---|
| crop64e60 | 5 × 1 | 27 | 138 |
| se48 (60 ep) | 5 × 1 | 20 | 99 |
| crop48e80 | 5 × 2 | 19.5 | 195 |
| cropinv (50 ep) | 5 × 1 | 17.5 | 88 |
| crop48e80 full-data | 2 | 19.5 | 39 |
| **total** | 27 jobs | | **≈ 560 → 4.7 h wall on 2 GPUs** (+ 3 min timing check, ≈ 35 min tail) |

Checkpoints are uploaded to the private checkpoint dataset every four finished jobs, so a crash or a 12-hour kill loses at most four jobs; the runner skips anything already present. Inference: the five 3D arms together are < 60 MB and add ≈ 2 min for 2,000 scans on an A100; the zip stays under the 850 MB that DrivenData accepted for run 4a.

## 4. Expected gain and risks

| | Nested OOF | Public (with the observed 0.018 shift gap) |
|---|---|---|
| run 4a (shipped) | 0.2491 | 0.2667 |
| 3D arm from 0.2568 → ≈ 0.245 (seeds + 80 ep + SE/crop-64 stacked) | ≈ 0.240–0.244 | ≈ 0.258–0.262 (rank ≈ 15–25) |
| if DANN/DRO also narrows the shift gap by 0.005 | | ≈ 0.253–0.257 (rank ≈ 12–20) |

Risks: (1) the SE net over-fits (stochastic depth + EMA are the guards; the stack simply gives it weight 0); (2) the 64-crop costs the most and may only tie the 48-crop (one seed only, gated); (3) the run-5 notebook and this one can both produce `r3d_cropinv` — the tag is identical on purpose; (4) any nested gain under 0.003 is noise — the assembler's gate applies.

## 5. Files

`datsc/models_a.py` (SE / stochastic-depth family; `build_model(arm="3d", arch="se", …)` dispatch added to `datsc/models_v3.py`, backward compatible), `scripts/train_a.py` (CLI + tag scheme), `kaggle/run6a_crop3d/{gen_run6a.py, e2e_check_a.py, notebook.ipynb, kernel-metadata.json, HOWTO.md, HOWTO.pdf, analysis_a.py, timing_a.py, tta_shift_probe_a.py}`, `tests/test_run6a.py`.
