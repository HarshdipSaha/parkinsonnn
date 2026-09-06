# Run 3 brainstorm — from public 0.339 to a rank-1 attempt

- **Date:** 2026-09-05 (evening). Deadline 2026-09-16 23:59 UTC. Written non-interactively under the `brainstorming` skill: context explored first, 2–3 approaches per decision, YAGNI, isolated units. Assumptions are stated where a question would normally have been asked.
- **Inputs read:** `EDA_REPORT.md`, `COUNCIL_VERDICT.md`, `RESULTS_LOG.md`, `LITERATURE_REVIEW.md`, `PIPELINE_REC_{A,B,C,D}.md`, `DAT_PARKINSONS_CHALLENGE.md`, `docs/superpowers/plans/2026-09-05-dat-pipeline.md`, all of `datsc/`, `main.py`, `scripts/*.py`, `kaggle/inference_cells.py`, `tests_synth.py`, the run-2 artefacts (`artifacts/run2_2026-09-05/`), the local cache (`cache/meta.csv`, `cache/folds.csv`, `cache/oof_lr.csv`, 1,362 `cache/*.npz`), `cache_anchor_diag.csv`, and the QC grids `qc/random.png`, `qc/short_z.png`, `qc/fallback.png`.
- **New measurements in this document** were produced by read-only analysis of the local cache and the LR arm (procedure in Appendix A). No code was modified. Numbers labelled *measured* come from that analysis or from `RESULTS_LOG.md`; numbers labelled *estimate* are judgement calls with a stated confidence.

---

## 0. TL;DR

1. **The public score is a discrimination problem, not a calibration problem.** Under a two-Gaussian (binormal) model, a *perfectly calibrated* classifier with AUROC 0.925 scores log loss 0.346; we scored 0.339 at AUROC 0.9254. The leader's 0.2192 at AUROC 0.9709 sits on the same curve (0.221). There is at most ~0.01 of calibration slack to harvest; the other ~0.11 to rank 1 is AUROC 0.925 → 0.971.
2. **The 0.04 OOF→public gap is ≈0.02 AUROC**, explained by three things we can fix: (a) per-fold best-epoch selection on the validation fold (`datsc/train.py:84-92`) makes the OOF optimistic by an estimated 0.01–0.03 nats; (b) four site-signature scalars (`native_spacing_mm`, `z_extent_mm`, `mask_ml`, `log_n_levels`; `datsc/config.py:20-23`) are fed to both the CNN head (`datsc/model.py:32-38`) and the LR (`datsc/lrmodel.py:13`) — *measured*: they identify the acquisition cluster with 97.3% accuracy and produce biased predictions under leave-one-cluster-out (LOCO); (c) public-LB sampling noise (test size undisclosed; if ~300 scans, SE ≈ 0.03).
3. **Why AUROC 0.925 and not 0.97:** the frame. The hotspot anchor is flagged on 19.5% of abnormal scans vs 2.4% of normals (*measured*), i.e. the crop fails on exactly the scans that carry the signal; the CoM control puts the striatum anywhere between slice 7 and 30 of 48 (*measured*, p10–p90); a median 20% of each hotspot crop is zero padding; the intensity reference is the p75 of a *head* mask that includes scalp (`datsc/preprocess.py:39-57`); the radiomics gate never passed (0.86). Every published ≥0.97 DaT-SPECT result registers to a template first (`LITERATURE_REVIEW.md` §1, `PIPELINE_REC_A_PAST_WINNERS.md` §1 Budd/Wenzel/Choi). The council cut registration on cost grounds, not evidence; the evidence now says the frame is the bottleneck.
4. **Recommendation: the "balanced" design (§3.2)** — own-code 7/9-DOF registration to a study template at 2 mm with a reference-region normalisation, a re-protocolled from-scratch 3D CNN (GroupNorm, EMA, fixed epochs, no scalar head, 3 seeds), plus a 2.5D ImageNet-pretrained torchvision ResNet18 family on registered striatal slabs as a second ensemble member, registered-ROI SBR radiomics + LR as a third, probability-averaged with two family weights, T fitted on LOCO-pooled predictions and floored at 1.0. Expected public log loss **0.26 (range 0.235–0.29)** after the 48-hour plan, **~0.24** after next week's seeds/TTA/second backbone; rank 1 (< 0.219) needs AUROC ≥ 0.971 and is the aggressive design's upside (§3.3), P ≈ 15–25%.
5. **48-hour plan (§5):** Sat morning registration + new cache on the 12-core box with a hard gate (registered-ROI LR AUROC ≥ 0.92, QC-fallback < 3%); Sat afternoon a three-way fold-0 A/B on Kaggle (registered 2 mm box vs registered whole-head vs CoM control); Sat night 5×2 seeds + two LOCO trainings unattended; Sun morning the 2.5D family + calibration; Sun afternoon pack → `--selftest` → Kaggle E2E → DrivenData smoke test → **real submission #2 Sunday evening**. ~9–11 GPU-hours of the 26 available.

---

## 1. Root-cause analysis

### 1.1 What the three numbers actually say

| Quantity | Value | Source |
|---|---|---|
| Prior-only log loss | 0.6884 (train) / 0.6687 (smoke, 20 train scans) | `RESULTS_LOG.md:3,38` |
| 15-checkpoint OOF (random-stratified group folds) | 0.2994 → 0.2999 after (w=1, T=0.913, λ=0.02) | `RESULTS_LOG.md:30-31`, `artifacts/run2_2026-09-05/artifacts/calib.json` |
| Smoke test | 0.2567 | 20 *training* scans → in-sample for the LR and for 4/5 of the CNN folds; not a generalisation signal |
| **Public LB** | **0.3391, AUROC 0.9254, rank #152** | task brief |
| Leader | 0.2192, AUROC 0.9709 | task brief |

**Decomposition via a binormal model** (*measured* in this session: equal-variance two-Gaussian scores, prior 0.548, Bayes-optimal posterior; Appendix A):

| AUROC | best achievable log loss | accuracy |
|---|---|---|
| 0.900 | 0.397 | 0.819 |
| **0.925** | **0.346** | 0.847 |
| 0.940 | 0.311 | 0.865 |
| 0.950 | 0.284 | 0.878 |
| 0.960 | 0.255 | 0.893 |
| **0.970** | **0.221** | 0.909 |
| 0.975 | 0.201 | 0.918 |
| 0.980 | 0.180 | 0.927 |

Our public pair (0.925, 0.339) and the leader's (0.971, 0.219) both lie on this curve within 0.01. Conclusions:

- **Calibration on the public set is already near-optimal for our discrimination.** The temperature question is a footnote: at true AUROC 0.925, shipping T=0.913 instead of T=1 costs 0.001 nats; T=1.2 costs 0.004; λ=0.02 costs 0.001 (*measured*, same model). The "0.04-nat calibration gap" in the brief is not a calibration gap.
- **The OOF 0.300 corresponds to AUROC ≈ 0.94–0.945** on the same curve, so the OOF→public gap is ≈ 0.02 AUROC of lost discrimination plus noise, not a shift in probability scale. (The OOF AUROC was never logged — `RESULTS_LOG.md` has no AUROC column and the OOF CSVs written by `datsc/train.py:95` were not copied into `artifacts/run2_2026-09-05/`. Fix: log OOF AUROC alongside log loss every run.)
- **Rank 1 requires AUROC ≈ 0.971 on the public set.** No post-processing gets there. Everything below is ordered by expected AUROC gain, with calibration hygiene as downside protection.

### 1.2 Why OOF 0.300 but public 0.339 (ranked by estimated contribution)

**(a) The OOF is optimistically biased by per-fold early stopping — est. 0.01–0.03 nats.** `datsc/train.py:84-92` evaluates the validation fold after every epoch and keeps the best epoch's weights; the OOF prediction (`train.py:93-95`) then comes from those weights. With n ≈ 272 per fold the per-epoch validation log loss has a standard error of roughly 0.03 (per-sample loss std ≈ 0.5–0.6), and the minimum over ~15–20 near-plateau epochs is biased low by about 1–1.5 SE. The 15-way ensemble averages some of that away, but every member was selected on the same labels, so the 0.2994 is not an honest out-of-fold number. The calibration (`scripts/fit_calibration.py:32`) was then fitted on it, which is why T came out < 1: the OOF looked more separable than the model is. *Predicted honest OOF: ~0.31–0.32*, i.e. half the gap disappears before any shift is invoked.

**(b) Site-signature scalars are a measured shortcut — est. 0.005–0.02 nats on a shifted test set.** `datsc/config.py:20-23` defines eight scalars; four of them (`mask_ml`, `z_extent_mm`, `native_spacing_mm`, `log_n_levels`) describe the scanner/protocol, not the patient. They are concatenated into the CNN head after a `BatchNorm1d` (`datsc/model.py:32-38`) and into the LR (`datsc/lrmodel.py:13`). *Measured on the local cache (Appendix A):*

| Feature set | site-ID accuracy (4 big clusters, chance 0.44) |
|---|---|
| all 8 scalars | 0.954 |
| the 4 site scalars alone | **0.973** |
| the 4 image-derived scalars | 0.559 |
| 14 hotspot radiomics | 0.558 |

and for the LR arm, leave-one-cluster-out vs random 5-fold on the same scans:

| held-out cluster (n) | with all 8 scalars: LOCO ll / rand ll (inflation), mean p LOCO vs rand | without the 4 site scalars: inflation | radiomics only: inflation |
|---|---|---|---|
| 2.46 mm (528) | 0.659 / 0.473 (**+39%**), 0.37 vs 0.50 | +37% | +12% |
| 3.90 mm (325) | 0.567 / 0.469 (**+21%**), 0.75 vs 0.61 | **+0.5%** | +1.6% |
| 2.30 mm (209) | 0.523 / 0.477 (+10%), 0.67 vs 0.61 | +3.6% | +0.4% |
| 2.40 mm (151) | 0.733 / 0.413 (**+77%**), 0.76 vs 0.56 | +16% | +3.7% |

AUROC barely moves under LOCO (0.87 vs 0.87); what moves is the *mean predicted probability* — a bias of 0.13–0.20 in probability. That is the signature of a model that learned "this spacing/head-volume combination has prevalence X" and extrapolates it wrongly to a cluster it has not seen. The CNN head receives the same four numbers and has far more capacity to exploit them. Random-stratified CV cannot see this (every site is in both train and validation); a differently mixed public set can. The EDA (`EDA_REPORT.md` §6) warned the long tail is 0%/100% abnormal — e.g. the 3.30 mm cluster (26 scans) is 73% abnormal and the 3.20 mm cluster (13) is 15% (*measured*). `PIPELINE_REC_D_BRAINSTORM.md` §2.2 explicitly excluded regime indicators as features for this reason; the shipped design ignored that.

**(c) Genuine acquisition shift in the public split — est. 0.00–0.02.** Unknown split; `DAT_PARKINSONS_CHALLENGE.md:37` says private "may differ" from public. The LR LOCO numbers above bound it: with site scalars removed, three of four clusters inflate < 20%, so an *image-only* model should transfer reasonably; the 2.46 mm cluster is the exception because holding it out also removes 39% of the training data and most of the high-count regime.

**(d) Public-LB noise.** Test size undisclosed. If the public split is ~300 scans, the standard error of log loss is ≈ std(per-sample loss)/√300 ≈ 0.6/17 ≈ 0.03, and of AUROC ≈ 0.013. Differences < 0.03 between leaderboard entries are noise; the top three (0.219/0.229/0.231) are within one SE of each other. Optimise the honest CV estimate, not the public number.

**(e) Prevalence.** Unknown on the public set; a constant-prediction real submission would invert it exactly, but costs one of two remaining real slots this week. The binormal table shows the effect is second-order (≤ 0.005–0.01 at our confidence levels); not worth a slot now.

### 1.3 Why AUROC 0.925 and not 0.97 (ranked by estimated AUROC loss)

**1. Spatial frame — est. −0.02 to −0.03 AUROC.** Evidence, all *measured*:
- Hotspot anchor flagged (`find_anchor`, `datsc/preprocess.py:60-92`, flag = component < 1 mL, > 30 mL or > 50 mm from the mask centroid) on 161/1,362 = 11.8% overall (`RESULTS_LOG.md:7`), **19.5% of abnormal vs 2.4% of normal scans**, 17% in the 2.46 mm cluster. The failure is label-correlated by construction: low striatal uptake is the disease, and it is exactly what makes the hotspot ambiguous. `qc/fallback.png` shows many flagged crops whose mid-slice contains lateral/occipital uptake rather than the striatum.
- Fold-level result: hotspot mean val log loss 0.324 vs CoM 0.302 (`RESULTS_LOG.md:29`); the two hotspot folds that blew up (f0 0.41/0.31, f1 0.37/0.36) are fine under CoM (0.27/0.31), so the instability is crop-driven, not seed-driven.
- CoM crops (`preprocess.py:150-153`) leave rotation, scale and z-position unnormalised: the z-slice of maximum intensity in the CoM crop spans slice 7 → 30 of 48 (p10 → p90), i.e. ±50 mm around the centre; a 144 mm crop puts the striatum near the edge in ~10% of scans.
- Zero padding: median 20% of the hotspot crop and 12.5% of the CoM crop is zero-filled (`crop_at`, `preprocess.py:95-106`); p90 is 50%. A zero edge is out-of-distribution for scans with small FOV and is a site signature in its own right.
- The alignment gate never passed: hotspot radiomics + LR = OOF AUROC 0.8615 (`RESULTS_LOG.md:7`), against a literature value of ~0.99 for registered-ROI radiomics on PPMI (`LITERATURE_REVIEW.md` §1, PMC9237203) and the council's own 0.90 gate (`COUNCIL_VERDICT.md` item 5). The fixed 30×30×18 mm boxes at the crop centre (`datsc/features.py:16-21`) are only as good as the anchor.
- Every high-performing DaT-SPECT pipeline registers: Choi 2017 (MNI 2 mm, AUC 0.988/0.997 external), Budd 2024 (affine to MNI 2 mm, 194k-param ResNets, 98%/94%/98%), Wenzel 2019 (`PIPELINE_REC_A_PAST_WINNERS.md` §1). The council's counter-argument was cost and failure visibility (`COUNCIL_VERDICT.md` "Where the Council Agrees" item 1), not evidence that alignment is unnecessary; the measured result says the template-free anchor is the weakest link.

**2. Intensity reference — est. −0.005 to −0.01 AUROC, plus site bias.** `brain_mask` (`preprocess.py:39-46`) is Otsu on a 6 mm-smoothed volume: a *head* mask. `normalize_p75` divides by the p75 of that mask, so scalp/skull uptake enters the reference. `qc/random.png` shows bright scalp rings on a subset of scans (e.g. `wujt4ftl`, `k0vm59m6`, `pk0qtl3q`, `9e7eobjl`) — a site-specific reconstruction/attenuation style. A scalp-inflated reference deflates the striatal ratio and pushes normals toward "abnormal" on those sites. Budd's DVR uses the 75th percentile of a reference region that *excludes* the striata and lies inside the brain (`PIPELINE_REC_A_PAST_WINNERS.md` §1). Separately, the ultra-low regime is one hospital: the 2.40 mm cluster has median raw p75 = 2.4 counts and 142/149 of its scans have p75 < 5 (*measured*); after division by p75 ≈ 2 their voxel values are quantised to multiples of 0.5. The quantisation augmentation (`datsc/augment.py:51-61`) is the right idea; a σ = 1-voxel blur at preprocessing would make those inputs continuous (`PIPELINE_REC_D_BRAINSTORM.md` §2.2).

**3. Training protocol and capacity — est. −0.005 to −0.01 AUROC.** `SmallResNet3D` uses `BatchNorm3d` with batch 16 (`datsc/model.py:12-13`, `train.py:52`); BN running statistics are computed over the training site mix and are a known cross-site failure mode (`PIPELINE_REC_B_ARXIV.md` 2508.20300: 3D brain CNNs 0.95 → 0.50 across domains). AdamW with `weight_decay=1e-4` (`train.py:53`) is effectively unregularised (decoupled decay at 1e-4 over ~2,700 steps is ≈ 0). 40 epochs, no warmup, no EMA/SWA, best-epoch selection (the optimism in §1.2a is also *variance*: the selected checkpoint is a noisy point on the trajectory). Only two seeds for the hotspot arm and one for CoM (`RESULTS_LOG.md:12-27`). The scalar head's `BatchNorm1d` over eight scalars with batch 16 adds noise of its own.

**4. Resolution — est. 0 to −0.005 AUROC.** 3 mm isotropic (`config.py:9`) with `order=1` zoom (`datsc/io.py:25-28`) downsamples 54% of the data (the 2.46 and 2.30 mm clusters) and represents the putamen (~10 mm wide) with 3 voxels. The physics argument for 3 mm (≈10 mm FWHM) is sound for detection of gross loss; the early sign — loss of the posterior putamen "tail" — is a shape change at the resolution limit. Literature standard is 2 mm. Cheap to test once the cache stores whole-head volumes.

**5. Missing family diversity — est. −0.005 to −0.01 AUROC.** All 15 members are the same 0.9M-param architecture on the same normalisation; the LR arm got zero weight (`RESULTS_LOG.md:31`). Ensembling two different *families* (3D from scratch + 2.5D pretrained + ROI features) is what both the RSNA winners and the DaT ensemble paper (PMC9139649, 93–95% single models → 98.45% ensemble on 645 PPMI images) did.

**6. Label-noise floor.** The leader's AUROC 0.971 is an empirical lower bound on what the test labels permit; inter-rater disagreement for expert-panel DaT reads is a few percent, so the floor is plausibly AUROC 0.98 / log loss ~0.18. Rank 1 is therefore within ~0.03–0.04 of the floor and will be decided by discrimination on the hard 5–8%, not by hedging.

Summing the estimates (−0.045 to −0.065 AUROC) reproduces the observed gap (0.925 vs a 0.97–0.99 ceiling). The order of attack follows the order above.

---

## 2. Candidate changes, ranked

Expected gains are on the **public log loss** (negative = better), given as point estimate (range) and P(gain ≥ half the point estimate). GPU cost is on Kaggle T4×2 (two processes). "Inference risk" is the risk of breaking the 3-hour / no-internet run. Gains are **not additive** — most address the same variance; §3 gives combined estimates.

| # | Change | Mechanism | Expected public Δ (confidence) | GPU-h | Impl h | Inference risk | Licence |
|---|---|---|---|---|---|---|---|
| 1 | **Affine (7–9 DOF) registration to a study template, own implementation** (scipy Powell on NCC, 6→4 mm pyramid, CoM init, coarse rotation grid, QC by NCC with CoM fallback) + registered striatal box at 2 mm | Removes rotation/scale/position variance; makes fixed ROIs valid (feature arm, reference region, slab positions); removes label-correlated crop failure; no zero padding | **−0.04 (−0.02 to −0.07), 60%** | 0 (CPU: ~1–2 s/vol; 1,362 vols ≈ 4 min on 10 cores) | 6–8 | **M** — mitigated by NCC QC + per-volume timeout (`infer.py:18`) + CoM fallback; pure numpy/scipy, no new deps | Template derived from competition data — disclose; no external asset |
| 2 | **Honest, stable training protocol**: GroupNorm(8) instead of BN, SiLU, EMA 0.999, fixed 60-epoch cosine with 3-epoch warmup, lr 5e-4, wd 1e-2, no best-epoch selection, 3 seeds | Removes OOF optimism (§1.2a), BN site-stat mismatch, checkpoint variance; honest T | **−0.02 (−0.01 to −0.03), 70%** (half is "the OOF becomes honest", half is real) | +2 (60 vs 40 epochs, 3rd seed) | 2 | L | — |
| 3 | **Remove site-signature scalars** from the CNN head (drop the scalar head) and the four site scalars from the LR | Kills the measured 97% site fingerprint and the LOCO bias; image-only model transfers | **−0.01 (0 to −0.03) public, 60%**; OOF may *worsen* ~0.005 | 0 | 0.5 | L | — |
| 4 | **2.5D ImageNet-pretrained family** (torchvision `resnet18` IMAGENET1K_V1, optionally `efficientnet_b0`) on three 6 mm axial slabs through the registered striatum, 80×80 @2 mm upsampled to 224, fine-tuned; TTA over slab offset ±2 mm and L/R flip | Second family with different inductive bias; DaT literature has direct precedent at n≈650 (Quan 2019 InceptionV3 3 slices 98.5%; PMC9139649 ensemble 98.45%); PDCADx winner MRN did ROI-slices + 2D backbone on 300 scans (`PIPELINE_REC_B_ARXIV.md` 2509.17566) | **−0.015 (0 to −0.03) as ensemble member, 55%**; standalone ≈ 3D CNN ± 0.02 | 1.5–2.5 (10 runs × ~6–8 min) | 4 | L–M — weights bundled fp16 (225 MB for 10), `weights=None` + `load_state_dict`; never call the download path; torchvision 0.27 in runtime | torchvision BSD-3; staff ruling thread 11475 ("permissively licensed weights are eligible"); DINOv3 excluded (11488). Disclose |
| 5 | **Reference-region normalisation** in template space: reference = brain mask eroded 8 mm minus a 20 mm-dilated striatal box; floor-subtract the p1 of non-brain; divide by reference mean (SBR+1 units); plus σ = 1 vox blur so ultra-low scans are continuous | Removes scalp from the reference; scale- and offset-invariant; the clinical quantity | −0.01 (0 to −0.02), 50% | 0 | 2 (needs #1) | L | — |
| 6 | **Registered-ROI SBR radiomics + LR** as a real member (per-side caudate/putamen mean & p90, asymmetry, put/caud ratio, posterior gradient, volume above thresholds) | Literature AUC 0.99 on registered data; diverse, calibrated by construction, and a live QC probe of #1 | −0.008 (0 to −0.015), 50% | 0 | 2 (needs #1) | L | — |
| 7 | **Calibration under shift**: fit T on pooled LOCO predictions of the 4 big clusters (89% of data), ship T = max(1.0, T_LOCO); nested 5-fold estimate of post-calibration loss; ε from the worst fold's most-confident decile; λ = 0.02 by the existing rule, λ_i = 0.15 for registration-QC failures | Downside protection: over-confidence is unbounded, under-confidence bounded; cost of T = 1.1 vs optimum is 0.001–0.004 | −0.005 (0 to −0.015), 65% | 0.7 (two LOCO CNN trainings) | 1.5 | L | — |
| 8 | **Mixup α = 0.2 and/or label smoothing 0.05** (ablate on fold 0, keep if grouped OOF improves after calibration) | Calibration regularisers; small-N 3D evidence mixed but cheap | −0.007 (−0.015 to +0.005), 50% | 1.5 (ablation) | 1 | L | — |
| 9 | **3 seeds + multi-view TTA** (flip × 3 shifts of ±2 mm × 2 rotations ±5°) | Variance reduction; inference budget allows 10–20× | −0.008 (−0.003 to −0.015), 70% | 3–4 | 1 | L (inference ×6 is still ~5 min) | — |
| 10 | **2 mm vs 3 mm** (given #1; box 80×80×48 @2 mm = 307k voxels vs 249k now) | Putamen tail at the resolution limit | −0.005 (0 to −0.015), 45% | ×1.3 per run | 0.5 | L | — |
| 11 | **Whole-brain input** (2.5 mm 77×90×64 registered head) vs registered striatal box | Context vs capacity; unknown sign — fold-0 A/B only | 0 (−0.01 to +0.01), 35% | 1 (one fold-0 run, ×2.5 slower/epoch) | 0.3 | L | — |
| 12 | **Bigger from-scratch 3D net** (widths 24/48/96/192 ≈ 2M params, GroupNorm) | Hamburg band is 0.2–2.9M; capacity is not the bottleneck | −0.003 (−0.01 to +0.005), 40% | ×1.5 | 0.3 | L | — |
| 13 | **Pretrained 3D backbone**: torchvision `r3d_18` (Kinetics-400, 33M) treating z as time, or MedicalNet ResNet-18 (MIT) | Regularised init; modality gap large (`PIPELINE_REC_B_ARXIV.md` 2605.06859: PET is an "island") | −0.005 (−0.02 to +0.01), 35% | 3–4 (10 runs × ~10 min at 20 epochs) | 3 (MedicalNet: +2 for the model source) | M — 33M × 10 fp16 = 660 MB in the zip; loading fine | torchvision Kinetics weights: `PIPELINE_REC_A` marks "uncertain, ask on forum"; MedicalNet: MIT repo, data terms unenumerated — disclose |
| 14 | **PSF/FWHM-matching blur** (blur coarser-spacing-adjusted to a common effective resolution, e.g. 12 mm FWHM) | Harmonises resolution across cameras without site labels (`PIPELINE_REC_B_ARXIV.md` §4.1 step 2) | −0.002 (−0.01 to +0.005), 35% | 0.7 (fold-0 A/B) | 1 | L | — |
| 15 | **Test-time robustness**: keep λ = 0.02, add ε-clip = worst-fold q, λ_i = 0.15 on QC failure; pad with masked p10 not 0 | Tail insurance; cost 0.001–0.002 when no shift | 0 to −0.005 expected; up to −0.02 in a bad-shift scenario, 60% | 0 | 1 | L | — |
| 16 | **Stacking (LR meta-learner) instead of averaging** | Sharpens boundaries → destroys calibration (`PIPELINE_REC_B_ARXIV.md` 2605.18696) | **+0.005 to +0.01 (worse)**, 70% | 0 | — | — | — |
| 17 | **Train-on-all-data final models** (fixed epochs from CV) added to the ensemble | +20% data per member | −0.005 (−0.008 to 0), 50% | 1.5 | 0.5 | L | — |
| 18 | **Whole-head registered cache** (one fp16 memmap `volumes_2mm.npy` + `index.csv`; crops/slabs/normalisation variants on the fly) | Enabler: no more 21-min rebuild per crop change (`scripts/cache_crops.py:28-31` stores crop-specific arrays) | 0 direct; saves ~1 h per ablation day | 0 | 1.5 | L | — |
| 19 | **Prior-only real submission** to read the public prevalence | π for λ-mix and bias | ≤ −0.005; costs a real slot | 0 | 0 | — | — |
| 20 | **Coronal-view 2.5D** model as an extra member | Diversity; putamen/caudate vertical extent | −0.003 to −0.01, 40% | 1.5 | 1 | L | as #4 |

Rows 1–7 are the run-3 core; 8–10 are fold-0 ablations that ride along; 11–14, 17, 20 are next-week options; 16 and 19 are rejected.

**On the two re-opened council decisions.** *Registration:* cut on the basis that a hotspot crop gives "containment, not correspondence" and fails visibly (`COUNCIL_VERDICT.md` item 1). Measured: it fails preferentially on abnormals (19.5% vs 2.4%), the gate stalled at 0.86, and the plain CoM crop beat it. Containment without correspondence also costs the feature arm, the reference region and every fixed-ROI trick. Reopen: yes, as own code with a QC gate. *2.5D:* cut on "a 10 mm blob has no texture for ImageNet priors" and the RSNA data-regime argument (item 2). The DaT literature itself contradicts the first claim at our n (`LITERATURE_REVIEW.md` §1: three separate ImageNet-transfer results at n = 645–659 with 98%+), and the Hamburg tiny-3D evidence argues for *also* keeping a small 3D net, not against a 2D member. Reopen: yes, as a gated second family, torchvision weights only.

---

## 3. Three run-3 designs

### 3.1 Conservative — "fix what is measurably broken, change no representation"

Keep the 3 mm CoM crop (drop the hotspot arm). Apply #2 (protocol), #3 (no site scalars), #7 (LOCO-based T ≥ 1, nested estimate), #9 (3 seeds, TTA), #15, and an eroded-mask reference (a cheap half of #5 that does not need registration).

- Expected public: **0.30–0.32** (AUROC ≈ 0.935–0.94). Δ ≈ −0.02 to −0.04.
- GPU: ~8 h. Human: ~8–10 h. Inference risk: none new.
- Trade-off: predictable, nearly guaranteed improvement, and the OOF finally tracks the LB; but it cannot reach the top 50 (0.2727) and leaves the frame — the dominant cause — untouched.

### 3.2 Balanced — "registration + protocol + one pretrained family" (**recommended**)

Everything in 3.1 plus #1 (own-code registration to a study template), #5 (reference-region normalisation), #10 (2 mm registered striatal box 80×80×48), #6 (registered-ROI radiomics LR), #4 (2.5D torchvision ResNet18 family), #18 (whole-head cache). Three families probability-averaged with two family weights fitted on grouped OOF; T on LOCO pool floored at 1; λ = 0.02; λ_i = 0.15 on QC failure; ε from worst fold.

- Expected public: **0.26 (0.235–0.29)** after the 48 h (AUROC ≈ 0.955–0.965); **~0.24** after next week's seed 3 / TTA / second 2.5D backbone. Top 10 (< 0.244) is the base case if the registration gate passes; rank 1 needs the aggressive add-ons and favourable noise.
- GPU: ~9–11 h in the 48 h (plan table sums to 8.5 h plus re-runs), ~20 h total by Sept 11. Human: ~24–30 h, front-loaded on Saturday.
- Inference risk: medium, fully mitigated by design — registration is numpy/scipy only, has an NCC QC gate with CoM fallback, runs inside the existing per-volume timeout, and adds ~2 s/volume (≈ 4 min for 2,000 volumes on 20 workers).
- Trade-off: the only design whose ceiling includes rank 1; it depends on one risky component (registration) that is gated on Saturday morning with a precommitted fallback (registration-lite: mirror-symmetry yaw + CoM + isotropic scale from head extent, ~2 h).

### 3.3 Aggressive — "everything, five seeds, three backbones"

3.2 plus #13 (r3d_18 or MedicalNet fine-tune), a second 2.5D backbone (`convnext_tiny` or `efficientnet_b0`), #20 (coronal 2.5D), #11/#12/#14 ablations, 5 seeds, #17 train-on-all, SWA.

- Expected public: **0.235 (0.215–0.26)**; P(< 0.219) ≈ 20%.
- GPU: 35–45 h (exceeds this week's 26 h; needs next week's 30 h). Human: 45+ h against a calendar with Rote (Sept 7) and Apart (Sept 11–13).
- Inference risk: medium-high (zip approaches 1 GB with r3d_18 checkpoints; more code paths). Trade-off: highest ceiling, lowest probability of a clean, tested submission by Sunday, and its gains are conditional on 3.2 working anyway.

### 3.4 Recommendation and reasoning

**Build 3.2, staged so each stage is submittable, and queue the 3.3 add-ons behind gates for next week's quota.** Reasons: (i) the root-cause analysis says the frame is 50–60% of the AUROC gap and only 3.2/3.3 address it; (ii) 3.2's risky component is gated on Saturday morning at zero GPU cost, with a fallback that still improves on run 2; (iii) 3.3's extras are strictly additive to 3.2 and cannot start before 3.2's cache exists, so choosing 3.2 now loses nothing; (iv) the public LB has ~0.03 noise, so "rank 1" is partly luck — the right objective is the best honest private-set estimate with a tested zip in hand by Sunday night, then compounding seeds/families through Sept 14.

**Assumptions made in lieu of questions:** Kaggle training remains the accepted compliance risk (`docs/superpowers/plans/2026-09-05-dat-pipeline.md` "Accepted risks" 1); the registered whole-head cache may be built locally and uploaded as a private Kaggle dataset version like the raw data already is; torchvision ImageNet weights are prize-safe per thread 11475 and will be disclosed; SimpleITK/antspyx are *not* relied on even though `PIPELINE_REC_A` reports them in the runtime `pyproject.toml`.

**Unit boundaries for the new code (design-for-isolation):**
- `datsc/register.py` — `build_template(volumes) -> template`, `register(vol_iso, spacing, template) -> (params, ncc)`, `resample_to_template(vol, params, out_mm=2.0) -> vol_t`, `qc_ok(ncc, params) -> bool`. numpy/scipy only; no knowledge of models.
- `datsc/normalize.py` — `reference_normalise(vol_t, template_masks) -> (vol_sbr, ref, floor)`; `extract_box(vol_t, box) -> crop`; `extract_slabs(vol_t, z_centre, offsets) -> [3,H,W]`. Shared verbatim by cache builder, trainers and `main.py`.
- `datsc/model.py` — `SmallResNet3D(norm="group", n_scalars=0, widths=...)`; `datsc/model2d.py` — `Slab2D(backbone="resnet18")` wrapping torchvision with `weights=None` and a bundled state dict.
- `datsc/train.py` — `train_fold(..., epochs_fixed=60, ema=0.999, select="final_ema")`, returns OOF + EMA weights; a `train_fold_2d` twin.
- `datsc/calibrate.py` — `Calibration(w_3d, w_2d, T, lam, pi, eps, lam_qc)`; `fit_family_weights`, `fit_T(pooled_loco)`.
- `datsc/infer.py` — stages: prior → register+normalise (pool) → LR → 3D ensemble → 2.5D ensemble, each overwriting `submission.csv` under the existing ladder.

---

## 4. Validation protocol that would have predicted the gap

**Pseudo-site definition (measured, 1,362 scans):** `(spacing rounded to 0.01 mm) × (raw-p75 regime: ultra-low < 5, low < 300, high ≥ 300)` gives 12 groups with ≥ 9 scans: 2.46|high 265, 2.46|low 263, 3.90|low 250, 2.30|low 207, 2.40|ultra-low 142, 3.90|high 75, 3.30|high 26, 1.48|high 25, 3.20|high 12, and a tail. Note `log_n_levels` cannot define regimes: it is computed *after* linear resampling (`datsc/preprocess.py:128`) and never falls below 2,131; use raw p75 (already in `cache/meta.csv`).

**Protocol (five rules):**

1. **Honest OOF.** Fixed epoch budget (chosen once on fold 0 by looking at the curve, then frozen) and EMA weights; no per-fold selection. Report OOF log loss *and* AUROC per fold, per pseudo-site, and worst-site. This alone would have reported ~0.31–0.32 for run 2 instead of 0.300.
2. **Model selection on random-stratified group folds** (keep `scripts/make_folds.py`; the near-duplicate audit found 0 groups at cosine 0.98 — leave the threshold, but re-run it on the registered 2 mm volumes where repeats are easier to see).
3. **Shift diagnostic = LOCO on the four big clusters (89% of data), run on the *shipped* model family**, not only on the LR (`scripts/shift_diagnostic.py:25-32` runs it on the LR only). Two extra CNN trainings (hold out 2.46 mm; hold out 3.90 mm) cost 40 min. Report LOCO log loss, AUROC and **mean predicted probability vs label prevalence** on the held-out cluster — the bias column is what exposed the site scalars.
4. **Feature-attribution check for shortcuts:** re-run LOCO with and without each scalar group; any feature whose LOCO inflation exceeds its random-CV gain is a shortcut and is removed. (Measured for the LR: site scalars +0.007 random-CV AUC, +20–60% LOCO inflation → removed.)
5. **Nested calibration.** Fit (w, T, λ) on four folds' OOF, evaluate on the fifth, rotate; report the mean as the honest post-calibration number; only ship changes that clear 0.02 on it.

**How it would have predicted 0.339:** honest random OOF ≈ 0.315 (rule 1), LOCO-pooled ≈ 0.35–0.40 with site scalars and ≈ 0.33–0.36 without (extrapolating the LR ratios to the CNN); the public 0.339 lies inside that bracket, i.e. "public ≈ between honest-random and LOCO" — which is the expectation for a test set drawn from the same hospitals with a different mix.

**Choosing T and λ under it:**
- T: fit once on honest random OOF (T_rand) and once on the pooled LOCO predictions (T_loco); ship **T = max(1.0, T_loco)**. The asymmetry justifies the floor: at our AUROC the cost of T being 10–20% too large is 0.001–0.004; the cost of being 10% too small under shift is the same order *plus* unbounded tail risk. Never ship T < 1.
- λ: keep the precommitted rule (`scripts/shift_diagnostic.py:31`: worst-cluster inflation < 50% → 0.02, else 0.05) but evaluate it on the CNN LOCO runs; per-volume λ_i = 0.15 when registration QC fails (frozen training-set NCC threshold — a per-sample rule, compliant with independent processing).
- ε: q = error rate in the most-confident decile of the worst fold (`PIPELINE_REC_C_ORACLE.md` §4, ε* = q); expect 0.005–0.01.
- π: 0.548 unless a spare real slot late in the week is used on a constant submission.

---

## 5. 48-hour build + ablation plan (Sat Sept 6 → Sun Sept 7, Kaggle T4×2)

Budget: ~9–11 GPU-h of the 26 h; real submissions: 2 left this week (slot 1 used 15:38 UTC Sept 5, returns Sept 12); smoke tests 3/day. Rote closes Sun 20:00 London — Sunday human work is capped at ~5 h; everything heavy runs unattended Saturday night.

| Step | When | What | Gate / output | GPU-h |
|---|---|---|---|---|
| 0 | Fri night, 1 h | Commit the uncommitted `RESULTS_LOG.md` edits. Small edits: `train.py` (GradScaler, `loss.detach()`, EMA, fixed epochs, `select="final_ema"`, log AUROC), `model.py` (GroupNorm option, `n_scalars=0`), `augment.py` unchanged. Unit tests green. | `pytest -q` passes | 0 |
| 1 | Sat 09:00–12:30, local 12-core box | `datsc/register.py` + `datsc/normalize.py` + `scripts/cache_volumes.py` (whole-head 2 mm registered memmap + `index.csv` with raw p75, spacing, NCC, params). Build template (label-blind: 200 random scans CoM-averaged → register all → re-average ×2; ≈ 3 × 4 min). Cache all 1,362 (~10 min). QC grids per pseudo-site (axial + coronal mid-slices, striatal box overlaid). Registered-ROI radiomics + LR, random 5-fold **and** LOCO. | **G1:** LR OOF AUROC ≥ 0.92 (run 2: 0.86), NCC-QC fallback < 3%, no pseudo-site with median NCC in the bottom decile. If G1 fails after ≤ 2 h of fixes (coarse rotation grid, mask erosion, metric mask), switch to registration-lite (mirror-symmetry yaw + CoM + isotropic head-extent scale) and continue. | 0 |
| 2 | Sat 12:30–13:30 | Upload cache as a private Kaggle dataset version (or rebuild on Kaggle: ~30 min on 4 CPUs). Ship `template_4mm.npz` + masks in `artifacts/`. | dataset visible in the notebook | 0 |
| 3 | Sat 13:30–15:30, Kaggle | Fold-0, seed-0 three-way A/B under the new protocol (no scalar head, GroupNorm, EMA, 60 epochs): **A** registered 2 mm box 80×80×48; **B** registered whole-head 2.5 mm; **C** CoM 3 mm 72×72×48 control. Two run in parallel. | **G2:** best of A/B beats C by ≥ 0.02 on fold-0 honest log loss (EMA final, no selection). Choose input. Also confirms epoch budget from the curve. | 1.5 |
| 4 | Sat 15:30 → Sun 01:00, unattended | Winner input: 5 folds × 2 seeds (10 × ~22 min). LOCO-2.46 and LOCO-3.90 trainings (2 × 22 min). Fold-0 ablations: mixup 0.2; label smoothing 0.05 (2 × 22 min). Notebook saves OOF CSVs + JSON histories + EMA state dicts. | OOF log loss + AUROC per fold/site; LOCO bias table; ablation deltas | 5 |
| 5 | Sun 09:00–11:00, Kaggle | 2.5D torchvision ResNet18 on registered slabs: fold 0 first (gate), then 5 folds × 2 seeds (10 × ~6 min). Bundled ImageNet state dict downloaded once in the notebook and saved fp16 into `artifacts/`. | **G3:** 2.5D fold-0 log loss ≤ 3D fold-0 + 0.03 and adding it lowers nested OOF ≥ 0.01. If it fails, ship without it. | 1.5 |
| 6 | Sun 11:00–12:30, local | Calibration: family weights (grid), T on LOCO pool floored at 1, λ rule, ε, λ_qc; nested 5-fold post-calibration estimate → `calib.json`. Pack (`scripts/pack_submission.py` extended for 2D weights + template). | nested OOF number written to `RESULTS_LOG.md` | 0 |
| 7 | Sun 12:30–15:00 | `python main.py --selftest` (add a rotated/scaled synthetic volume to exercise registration + QC fallback); Kaggle E2E on the 20 smoke scans from the packed zip (`kaggle/inference_cells.py`); 150-volume timing with the registration path → extrapolate to 2,000 (expect ~20–25 min); DrivenData smoke test. | exit 0, format OK, smoke wall < 3 min, extrapolated < 60 min | 0.5 |
| 8 | Sun evening | **Real submission #2** (slot 2 of 3). Record public log loss + AUROC vs honest OOF and LOCO in `RESULTS_LOG.md`. Keep slot 3 for Sept 9–10 after seed 3 + TTA. | public number inside the [honest-random, LOCO] bracket | 0 |
| 9 | Sept 8–11 (next quota) | Seed 3 (3D + 2.5D), multi-view TTA, second 2.5D backbone, r3d_18 ablation, train-on-all; final submissions Sept 14–15 with the λ bracket. | each gated at ≥ 0.01 nested OOF | 8–10 |

**Kill criteria:** if step 1 is not at G1 by 15:00 Saturday, ship the conservative design (3.1) on the CoM frame Sunday and move registration to Sept 8; if G2 shows registration ≤ CoM + 0.01, keep both frames as ensemble members (they were complementary in run 2, `RESULTS_LOG.md:45`) rather than arguing about it.

### 5.1 Every previous-run mistake / warning and its fix

| # | Where | Problem | Fix |
|---|---|---|---|
| 1 | `datsc/train.py:70` | `torch.cuda.amp.GradScaler(enabled=amp)` deprecated | `torch.amp.GradScaler("cuda", enabled=amp)` |
| 2 | `datsc/train.py:83` | `float(loss)` on a tensor with grad (warning; also forces a sync every step) | accumulate `loss.detach()` and `.item()` once per epoch |
| 3 | `datsc/train.py:84-92` | best-epoch selection on the validation fold → optimistic OOF → T < 1 | fixed epoch budget + EMA weights; OOF from final EMA; if selection is wanted, use an inner 10% split |
| 4 | `datsc/train.py:52-53` | 40 epochs, no warmup, `weight_decay=1e-4` under AdamW (≈ no regularisation) | 60 epochs, 3-epoch warmup, cosine, lr 5e-4, wd 1e-2 |
| 5 | `datsc/model.py:12-13,27` | `BatchNorm3d` with batch 16 and shifted test sites | `GroupNorm(8)`; SiLU |
| 6 | `datsc/model.py:32-38`, `datsc/config.py:20-23` | eight scalars incl. four site signatures into the head; `BatchNorm1d` over scalars | drop the scalar head (`n_scalars=0`); if any scalar is kept, image-derived only |
| 7 | `datsc/lrmodel.py:13` | LR uses all scalars → measured LOCO bias | radiomics + image-only scalars |
| 8 | `datsc/preprocess.py:128` | `n_levels` counted after resampling (min 2,131) — cannot flag the ultra-low regime | count on the native volume; keep as QC/CV grouping only, never as model input |
| 9 | `datsc/preprocess.py:60-92`, `config.py:12-15` | hotspot anchor fails on 19.5% of abnormals | replace with registration; CoM as fallback frame |
| 10 | `datsc/preprocess.py:39-57` | head mask (scalp) defines the p75 reference | eroded brain reference minus dilated striatal box, floor-subtracted |
| 11 | `datsc/preprocess.py:95-106` | zero padding (median 20%, p90 50% of the crop) | registered frame has no padding; pad with masked p10 where needed |
| 12 | `scripts/cache_crops.py:28-31` | cache stores crop-specific arrays → 21-min rebuild per crop change | whole-head registered 2 mm memmap; crops/slabs on the fly |
| 13 | `scripts/alignment_gate.py:16,36` | gate tied to hotspot radiomics, prints FAIL, blocks nothing | re-target to registered-ROI features; gate 0.92; make it an explicit G1 in the plan |
| 14 | `scripts/fit_calibration.py:32`, `datsc/calibrate.py:32-34` | T fitted on optimistic random OOF, unbounded below 1 | T on LOCO pool, `max(1.0, ·)`; nested estimate |
| 15 | `scripts/shift_diagnostic.py` | LOCO run on the LR only; sizes λ, never asks *why* the inflation | LOCO on the CNN family; report bias; feature-group ablation |
| 16 | `RESULTS_LOG.md` | no OOF AUROC, no per-site numbers, OOF CSVs not archived | log AUROC + per-site + worst-site; copy `oof_*.csv` into `artifacts/runN/` |
| 17 | `RESULTS_LOG.md:12-27` | 2 seeds hotspot, 1 seed CoM; folds 0/1 unstable | 3 seeds on a stable protocol; keep frames as members only if complementary |
| 18 | `RESULTS_LOG.md:39` | smoke-test score (0.2567) treated as informative | it is in-sample; use only as a format/timing check |
| 19 | `datsc/infer.py:99` | `torch.load(wpath, map_location=device)` without `weights_only=True` | add `weights_only=True` (runtime torch 2.12 defaults to it; be explicit) |
| 20 | `main.py:4-5` | parent sets `OMP_NUM_THREADS=4`; workers set 1 (`infer.py:22-24`) — fine, but torch inference threads on the A100 box are then 4 | leave; set `torch.set_num_threads(8)` before the GPU stage if CPU fallback is ever used |
| 21 | `datsc/augment.py:56-61` | quantisation levels (8,16,64,256) assume max-normalised input; after reference normalisation the ultra-low regime is ~5–10 levels | add levels 4 and 6; apply after normalisation, before blur |
| 22 | `scripts/make_folds.py:24-28` | near-duplicate audit found 0 groups — either none exist or the 16³ crop embedding is too noisy | re-run on registered 2 mm volumes; keep the threshold |
| 23 | git | `RESULTS_LOG.md` modified but uncommitted; `cache_anchor_diag.csv` untracked at repo root | commit the log; move/ignore the diag CSV |
| 24 | `docs/superpowers/plans/…` | plan says registration and 2.5D "do not appear in this plan" | supersede with this document once §5 step 1 passes G1 |

---

## 6. Alignment with the run-3 infrastructure that landed during this analysis (commit `74eaa21` + untracked `_v3` files)

While this document was being written, a parallel workstream committed "run 3 infrastructure: 2mm whole-brain memmap cache, 3D/2.5D model zoo, trainer with EMA/mixup/warmup, pseudo-site folds" and added `datsc/infer_v3.py`, `main_v3.py`, `scripts/fit_stack_v3.py`, `scripts/pack_v3.py`, `tests/test_infer_v3.py` (untracked). It was read, not modified. Mapping against §2:

**Already in place (keep):** whole-brain 2 mm memmap cache = #18 (`datsc/volume.py`, `scripts/cache_volumes.py`); GradScaler/`loss.detach()` fixes, EMA 0.998, warmup + cosine, optional mixup, grad clipping = most of #2 and #8 (`datsc/train_v3.py:115-142`); 2.5D torchvision family built with `weights=None` and bundled state dicts = #4 (`datsc/models_v3.py:33-49`, `scripts/fetch_pretrained.py`), plus `r3d_18` for #13; pseudo-site labels and leave-site-out folds = §4 (`scripts/pseudo_sites.py`); manifest-driven multi-arm inference with ladder and a T floor of 1.0 = #7 in part (`datsc/infer_v3.py`, `scripts/fit_stack_v3.py:44,68`); alternative normalisations via `renormalise` = part of #5 (`datsc/volume.py:47-59`).

**Gaps and conflicts with the evidence above, in order of importance (all small edits except #7):**

1. **Site-signature scalars still feed every head.** `ResNet3D` and `Net2D` concatenate `scalar_bn(scalars)` with all eight scalars (`datsc/models_v3.py:25-30, 64-65, 83-86`), `train_v3.py:101` and `infer_v3.py:86` pass `n_scalars=N_SCALARS`, `data_v3.py:49` serves them. Measured: 97% site-ID from four of them and LOCO probability bias of 0.13–0.20 (§1.2b). Fix before any training: `n_scalars=0` (drop `scalar_bn` and the concat) or an image-only subset.
2. **Best-epoch selection on the validation fold is retained** (`train_v3.py:149-152` keeps the best-val EMA state and predicts OOF from it) → the OOF stays optimistic (§1.2a) and the T floor only half-fixes the consequence. Fix: fixed epoch budget, ship the final EMA state; keep `best_val_logloss` for logging only.
3. **`Net2D` clamps its input at `in_scale=3.0` p75-units** (`models_v3.py:55,58,84`). Measured on the 1,362 CoM crops: the p99 of the central 72×60×36 mm box exceeds 3.0 in **93% of normals vs 50% of abnormals** (median 3.99 vs 3.00), and exceeds 6.0 in 0.5% — the clamp discards exactly the intensity range that separates the classes. Fix: `in_scale=6` (or 8).
4. **2.5D slabs are centred on the head CoM** (`data_v3.py:66`: `z0 = Z//2 - ...`). The striatum lies ~15 mm above the head CoM on average (`config.py:16`) and the CoM-crop z-of-maximum spans −51…+18 mm (p10…p90, §1.3); 3 × 12 mm = 36 mm centred on the CoM puts the p90 case at the edge. Until registration: 5 slabs × 8 mm centred 12 mm superior to the CoM (or 3 × 16 mm); with registration (#1): centre on the template striatal z.
5. **`BatchNorm3d` retained** in `ResNet3D` (`models_v3.py:20`, and `Block` imported from `model.py:12-13`) → GroupNorm(8) (§1.3 item 3).
6. **Defaults** `epochs=40, lr=1e-3, weight_decay=1e-4, warmup_epochs=1` (`train_v3.py:86-87`, `scripts/train_v3.py:24-28`) → 60 / 5e-4 / 1e-2 / 3 (§2 row 2).
7. **No registration**: `prepare_volume` is CoM-only (`volume.py:34,42`). The change is additive: register before `crop_at` (same `VOL_SHAPE`), store NCC and the 7–9 parameters in `stats`, keep CoM as the QC fallback. The §5 step-3 A/B/C then maps to `--size 96 96 72` (2 mm) vs `--size 64 64 48` (3 mm effective) on the registered vs the CoM cache.
8. **Reference normalisation is still head-mask p75** (`volume.py:30`; `renormalise("median")` rescales by the same head-mask statistics). The stored `masks.npy` allows an eroded-brain-minus-central-box reference to be computed without a cache rebuild for the CoM frame; the striatum-excluding version needs #7.
9. `scripts/fit_stack_v3.py` documents `--calib random|site` but defines no such flag (`:38-46`); fitting T on site-held-out predictions requires training each arm on `folds_site.csv` (5 extra runs per arm) or the two-cluster LOCO pool of §4 — decide which and wire it.
10. `scripts/fetch_pretrained.py` bundles `r3d_18` (Kinetics-400) and states the dataset licence as CC-BY; `PIPELINE_REC_A_PAST_WINNERS.md` §5 marks torchvision video weights "uncertain — ask on the forum". Disclose; do not make the final ensemble depend on it.
11. `infer_v3.py:87` and `models_v3.py:38` call `torch.load` without `weights_only=True` — add it for explicitness (runtime torch 2.12 defaults to it).
12. Arms are combined by a weighted **logit** average (`infer_v3.py:106-115`, `fit_stack_v3.py:63`), the more confident of the two averaging choices; acceptable with the T floor, but the §4 nested estimate should be computed on the same combiner.

## Appendix A — how the new measurements were produced (read-only)

All numbers labelled *measured* in §1 come from a scratch script run against the local cache on 2026-09-05; nothing in the repo was modified. Procedure:

1. Load `cache/meta.csv` (1,362 rows, all `ok`), `cache/folds.csv`, `cache/oof_lr.csv`, and `radiomics`/`scalars`/`crop`/`crop_com` from each `cache/{uid}.npz`.
2. Spacing clusters from `native_spacing_mm` rounded to 0.01 mm; regimes from raw `p75` (< 5 ultra-low, < 300 low, else high). Fallback rate by label from the `fallback` column.
3. LR arm = `StandardScaler + LogisticRegression(C=0.3)` (same as `datsc/lrmodel.py`) on (i) radiomics + 8 scalars, (ii) radiomics + 4 image scalars, (iii) radiomics only; random `StratifiedKFold(5)` OOF vs leave-one-cluster-out for the 2.46/3.90/2.30/2.40 mm clusters; log loss, AUROC and mean predicted probability on the held-out cluster, compared with the random-CV predictions restricted to the same scans.
4. Site fingerprint: 5-fold `LogisticRegression` predicting the 4-cluster label from each feature group.
5. Binormal table: equal-variance Gaussians separated by d′ = √2·Φ⁻¹(AUROC), prior 0.548, posterior integrated numerically for expected log loss and accuracy; temperature and λ costs applied to the calibrated logit of the same model.
6. Crop geometry on 300 random cached crops: fraction of exactly-zero voxels; index of the axial slice holding the maximum of the crop's per-slice maxima.

Key raw outputs: fallback 2.4% (normal) vs 19.5% (abnormal); 2.40 mm cluster median p75 2.36 counts, 142/149 ultra-low; site-ID accuracy 0.954/0.973/0.559/0.558 (8 scalars / 4 site / 4 image / radiomics); LR LOCO inflation with all scalars +39/+21/+10/+77% and without site scalars +37/+0.5/+3.6/+16%; hotspot crop zero-fraction p10/50/90 = 0.07/0.20/0.50, CoM 0.04/0.13/0.48; CoM z-of-max p10/50/90 = 7/23/30 of 48; run-2 checkpoint 907,665 params, 3.5 MB.
