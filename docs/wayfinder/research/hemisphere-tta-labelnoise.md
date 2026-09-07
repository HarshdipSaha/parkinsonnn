---
label: wayfinder:research
ticket: T03
title: Hemisphere-level and asymmetry-aware DaT classification; single-scan test-time adaptation; label-noise-robust losses
created: 2026-09-07 05:00 UTC
---

# T03 — hemisphere heads, single-scan test-time adaptation, label-noise losses

<!-- BOTTOM -->

## What the recommendations are measured against

- **Rule R2** (`aidlc-docs/inception/README.md`, DrivenData rules): every test scan is processed independently — no pseudo-labelling, no statistics across test samples, no state carried from one scan to the next. "Allowed" below means: uses only the one scan plus statistics stored from training, and resets afterwards.
- **Where we stand:** run 4a nested 0.2491 → public 0.2667 (gap 0.018; run 3: 0.021); leader 0.2192. Members: crop-3D SE-ResNet (GroupNorm) .375, EfficientNet-V2-S 2.5D .25 (110 BatchNorm layers), ConvNeXt-T .125 (LayerNorm), coronal B0 .125 (49 BatchNorm layers), feature arm .125; temperature floor 1.0, prior mix λ 0.02.
- **Already measured (RESULTS_LOG):** z-shift + mirror TTA flat (0.2257 → 0.2251), affine TTA negative (→ 0.2269), typicality temperature −0.002 (below the 0.003 gate), clip/hedge no nested gain. Label-noise floor: the 40 hardest scans carry 24 % of the stack loss, the 100 hardest 46 %; they are abnormal-labelled scans that every arm scores 0.03–0.15.
- **Run 7 already carries** `datsc/models_h.py` (`HemiNet3D`: one encoder over the two mirrored striatal half-crops, `z = pair([f_L + f_R, |f_L − f_R|]) + max(side(f_L), side(f_R))`) and `--loss bootstrap` (soft bootstrapping, `0.9·y + 0.1·σ(z)`). This note grades those designs against the literature rather than re-proposing them.

<!-- Q1 -->

## Q2 — Single-scan, non-transductive test-time adaptation under scanner shift

| Method | Evidence | Reported gain (setting) | Applicability to rule R2 and to our arms |
|---|---|---|---|
| **Source-anchored BatchNorm-statistics blend** from the one scan (pseudo-count prior N; TTN; SITA/AugBN; IABN) | Schneider 2020 NeurIPS; Lim 2023 ICLR; Khurana 2021; Gong 2022 NeurIPS | ImageNet-C mCE 76.7 → 71.4 at n = 1, N = 16; TTN at B = 1: CIFAR-10-C error 18.27 → 17.99, while *test-only* stats at B = 1 give 90.0 % error; AugBN (single image + augmented copies, λ 0.7–0.9 to source) CIFAR-10-C acc 61.4 → 71.8; ~0 gain on natural shifts (ImageNet-V2, CIFAR-10.1) | **Allowed** (stats from the scan's own slabs/TTA copies + stored source stats, nothing kept). Touches only the BN arms (B0, V2-S); the GroupNorm 3D and LayerNorm ConvNeXt arms are unaffected by construction. Expect a small effect; keep the un-adapted twin in the stack. |
| **Episodic entropy minimisation on one scan** (MEMO; TPT view selection) | Zhang 2022 NeurIPS; Niu 2023 ICLR (Table 4); Yoon 2024 ICLR (C-TPT); Joshi 2025 | R50 mCE 76.7 → 69.9 (BN prior alone: 71.4); at batch 1 MEMO keeps 23.9 % acc where Tent collapses to 0.1 %; but entropy minimisation raised ECE 5.04 → 12.0 (C-TPT); "MEMO hardly improves on the baseline" on 3D MRI; 55,980 s per 50 k images | Allowed only with a reset after every scan. It sharpens confidence, which is the wrong direction for a stack that is already calibrated; our z-shift/flip views are near-duplicates, so the marginal entropy it minimises is ≈ the conditional entropy (Kimura 2021). **Not recommended.** |
| **Online / continual TTA** (Tent, EATA, SAR, DUA, RoTTA, NOTE memory, RDumb, REALM, AdaBN) | Wang 2021 ICLR; Mirza 2022 CVPR; Press 2023 NeurIPS; Zhao 2023 ICML | Tent at batch 1 with BN: 0.1 % acc; accumulation collapses on long streams and periodic *reset* rescues it (RDumb 37.9 vs Tent 9.2); under label shift every method except MEMO is worse than no adaptation | **Not allowed** (state across scans = statistics across test samples). Also the literature's own verdict is that resetting is the safe regime. |
| **Input-space projection to the source domain** (DDA, GDA, Decorruptor) | Gao 2023 CVPR; Guo 2024 CVPR | ImageNet-C acc R50 18.7 → 29.7, ConvNeXt-T 39.3 → 44.2; no weight update | Allowed, but needs a source-domain diffusion model; out of budget before Sept 16. |
| **Test-time augmentation policy** (which views, how to aggregate) | Lyzhov 2020 UAI; Shanmugam 2021 ICCV; Kimura 2021; Medeiros 2026 (arXiv 2604.09697) | Policies selected by *calibrated log-likelihood* beat accuracy-selected ones (CIFAR-100 cLL −0.555 → −0.479); mean TTA on ImageNet +1.1 pt but ⅓ of the labels it flips are wrong; MedMNIST: TTA hurt in 11/12 model–dataset pairs (PathMNIST −31.6 pt), intensity-only 90.1 % vs geometric 88.1 % vs mixed 85.4 % — mechanism is BN-statistics mismatch, "a bias problem, not variance" | Allowed. Explains our flat z-shift and negative affine results. The one untried, theory-backed option is **intensity views** (the same scan under `ref`/`p75`/`hist` normalisation) selected by nested log loss — not more geometry. |
| **Per-scan intensity / histogram alignment to source statistics** | Llera 2019 Mol Imaging Biol; Nyúl & Udupa 2000 TMI; Zhao 2022 IJCAI (TAF-Cal) | GCDF per-scan intensity harmonisation on 1,342 PPMI scans / 7 camera models removed the camera-model differences in striatal intensity and SBR (AUC 0.94/0.98); Nyúl landmarks validated on MRI only; TAF-Cal (feature-amplitude calibration toward a stored source prototype, one sample) PACS 80.28 → 86.73 | Allowed; this is our `hist` mode (run 5 measures it). TAF-Cal-style feature calibration for the 2.5D arms is the cheap allowed extension if `hist` shows a gain. |
| **Per-image adaptable normaliser with a prior** (Karani 2021; He 2021; Valanarasu 2022; InTEnt 2024; MuVi 2025) | MedIA / MIDL / 2024–25 | Brain-MRI seg Dice 0.588 → 0.800; 1-shot Tent *hurt* (61.95 → 61.20); InTEnt: ensemble over BN-stat blends λ ∈ {0.2..0.8} 64.0 → 71.6 Dice, the best λ varies per shift; MuVi: keep source BN stats, adapt affine only | Priors live on label maps or reconstructions — segmentation only. Transferable lesson: never replace BN statistics wholesale; blend, or ensemble over blends. |
| **Training-time simulation of the camera shift** (context) | Buddenkotte & Buchert 2024 JNM; Budenkotte 2024 EJNMMI; Tossici-Bolt 2011 / 2017; Wenzel 2019 | "Unrealistic" blur + noise augmentation: 0.960/0.953 → 0.989/0.975 on two external sets; same site, different camera: 98.0 → 94.1 %; striatal-phantom recovery COV 46 % → 14 % after attenuation/scatter calibration; residual reconstruction differences 5.8–14.5 % | Allowed and the best-evidenced answer to *this* shift (resolution, scatter, reconstruction, count level); complements run 5's `hist`/DANN. No DaT-SPECT or PET/SPECT paper reports a single-scan TTA gain for classification. |

**Verdict.** (i) Consistent single-scan gains for classification exist only for source-anchored normalisation blending, episodic MEMO and input-space projection, and all three shrink toward zero on natural (non-corruption) shifts; none is reported as log loss. (ii) Everything that estimates statistics from the test stream breaks at batch 1 or is disallowed; pure single-scan BN statistics destroy contrast cues (they become instance norm), and single-sample entropy gradients drive class collapse. (iii) The BN-prior blend is worth one cheap variant on B0/V2-S, gated on nested log loss. (iv) MEMO-style adaptation is expected to over-sharpen and lose log loss; skip it. (v) Allowed: per-scan optimisation from frozen weights with reset, statistics from the scan's own slabs/augmentations blended with stored source statistics, per-scan histogram/Fourier alignment to a stored template. Not allowed: anything that persists across scans.

## Q3 — Label-noise-robust training when ~5 % of labels are ambiguous and the metric is log loss

| Method | Evidence | Reported effect on calibrated loss | Applicability (5 % boundary-concentrated ambiguity, log loss) |
|---|---|---|---|
| **Plain cross-entropy + temperature scaling** | Gneiting & Raftery 2007 JASA; Guo 2017 ICML; Penso 2024 TMI | The log score is strictly proper: its unique optimum is the label posterior. TS: NLL 1.498 → 1.044 (CIFAR-100), 0.329 → 0.210 (CIFAR-10). TS fitted on labels with 5 % noise: adaECE 4.34 vs 3.18 clean-fit (small bias); 11.8 vs 1.6 at 20 % noise | **Keep.** The test labels come from the same reader process as ours, so the quantity log loss rewards is the *reader* posterior P(ỹ\|x) — exactly what CE estimates. "De-noising" toward a sharper clean posterior is over-confident on borderline scans. |
| **GCE / SCE / MAE / APL / GJS / asymmetric / peer losses** | Zhang & Sabuncu 2018; Wang 2019 ICCV; Ma 2020 ICML; Englesson 2021; Zhou 2021; Wei 2022 ICML | Accuracy only in every paper (e.g. CIFAR-10 @ 20 % symmetric: CE 86.98 → GCE 89.83 %); none reports NLL/ECE. Non-proper: the population minimiser of L_q is p ∝ η^{1/(1−q)}, i.e. sharpened toward one-hot; Wei 2022 shows these losses act like *negative* label smoothing ("over-confident on correct predictions"); MAE/APL under-fit small data | **Likely to hurt.** Built for 20–80 % symmetric flips; on our data they sharpen exactly the borderline scans that carry 46 % of the loss. Do not use. |
| **Soft bootstrapping** (Reed 2015; our `--loss bootstrap`, β = 0.9) and self-prediction targets (SEAL) | Reed 2015 ICLR-W; Chen 2021 AAAI | Accuracy only (MNIST/TFD); SEAL (epoch-averaged own predictions as soft labels, retrain) CIFAR-10 instance-dependent noise 40 %: CE 75.68 → 82.98 | Allowed and mild at β = 0.9, but the target is the net's *own current* confidence, so near the boundary it reinforces whichever way the net leans. Gate on nested log loss; expect ≈ 0 ± 0.003. |
| **Uniform label smoothing** | Müller 2019 NeurIPS; Lukasik 2020 ICML; Thulasidasan 2019; Wei 2022 | ECE 0.150 → 0.024 at α 0.05, but "the model trained with hard targets achieves better NLL at all temperature-scaling settings"; α 0.2 ECE 0.156 vs 0.111; ε 0.1 on CIFAR-10 ECE 0.089 vs 0.023 (under-confidence); positive LS is optimal only at low noise | Not for a log-loss metric with TS on top (already the standing decision in `RESEARCH_SIMILAR_COMPETITIONS.md`); at most on flagged ambiguous cases, α ≤ 0.05. |
| **Soft targets that carry rater or ensemble uncertainty** (human soft labels; OOF-ensemble relabelling; self-distillation) | Peterson 2019 ICCV; Collins 2022 HCOMP; Zhang & Sabuncu 2020 NeurIPS; Uma 2021 JAIR; RSNA-2024 2nd place | Held-out CE 0.35 vs 0.57 (ResNet), 0.26 vs 0.51 (Shake-Shake) for soft vs hard training labels; CE 1.33 vs 2.03 with calibration RMSE 0.20 vs 0.28; self-distillation = instance-specific smoothing, improves ECE; Kaggle: 0.5·loss(labels) + 0.5·loss(OOF pseudo-labels), no ablation numbers | **The only family with primary evidence of lower log loss**, and it is training-time, so allowed. Nested OOF targets y* = (1 − λ)·y + λ·p_OOF with λ 0.3–0.5 put the smoothing where the ambiguity is (the 5 %) and leave clean cases hard. |
| **Sample selection / co-teaching / DivideMix / ELR / AUM / confident-learning removal** | Han 2018; Li 2020; Liu 2020; Pleiss 2020; Northcutt 2021; Chen 2021 AAAI; Liu 2021 ICML; DPC AAAI 2024 | Accuracy only; AUM removes 3–24 % of *clean* data; DivideMix confidence ≫ accuracy (over-confident); boundary-concentrated noise "misaligns the support" of selection; atypical instances do not benefit from denoising | **Hurts.** Our ambiguous cases are the hard clean tail, not random flips. Do not remove; down-weight at most (the audit route in `RESEARCH_SIMILAR_COMPETITIONS.md` §9 stays "flag, do not drop"). |
| **Mixup** | Zhang 2018 ICLR; Thulasidasan 2019 NeurIPS; Zhang 2022 ICML | ECE 0.023 → 0.019 (CIFAR-10), 0.049 → 0.035; stops memorisation of 20 % corrupted labels (test error 12.7 → 5.9 %); helps calibration when capacity/data is high — our regime; NLL never reported | Allowed, already in the trainer (`--mixup`); check per-member nested NLL (mixup can under-confidence; TS corrects that). |
| **Ambiguity-aware inference** (UDM: disagreement between high-sensitivity and high-specificity ensembles) | Budenkotte 2024 EJNMMI; Mathies 2022 | Flags 4.3 % of scans and catches 90 % of errors (99.8 % accuracy on "certain" scans); 7.8 % of test scans had discrepant expert reads; dichotomised inter-rater κ 0.90–1.00; borderline scans "5–10 %" in routine | Allowed (per scan, ensemble-internal). Our hedge-on-disagreement is the log-loss analogue and showed no nested gain, consistent with a well-calibrated stack; the number confirms the 5 % ambiguity estimate. |

**Verdict.** (i) Evidence of better NLL/Brier exists for soft targets that carry uncertainty (Peterson, Collins), for temperature scaling (Guo; robust to ≤ 5 % label noise per Penso) and, for ECE, for mixup. (ii) GCE/SCE/MAE-type losses, hard bootstrapping, forward correction and small-loss selection are expected to *raise* log loss here: they are non-proper or remove the hard clean tail, and no paper reports NLL for any of them. (iii) For 1,400 scans, 5 % ambiguity, log loss and a temperature-scaled stack: keep CE, add nested-OOF soft relabelling as a gated variant, keep mixup for the 3D members, no global label smoothing, no case removal.

<!-- RANKED -->

## Sources

<!-- SOURCES-Q1 -->

**Q2 — single-scan TTA**
- Schneider et al. 2020, NeurIPS, *Improving robustness against common corruptions by covariate shift adaptation* — https://arxiv.org/abs/2006.16971
- Nado et al. 2020, arXiv, *Evaluating prediction-time batch normalization* — https://arxiv.org/abs/2006.10963
- Lim et al. 2023, ICLR, *TTN: a domain-shift aware batch normalization in test-time adaptation* — https://arxiv.org/abs/2302.05155
- Khurana et al. 2021, arXiv, *SITA: single image test-time adaptation* — https://arxiv.org/abs/2112.02355
- Gong et al. 2022, NeurIPS, *NOTE: robust continual test-time adaptation against temporal correlation* — https://arxiv.org/abs/2208.05117
- Niu et al. 2023, ICLR, *Towards stable test-time adaptation in dynamic wild world* (SAR) — https://arxiv.org/abs/2302.12400
- Zhang, Levine, Finn 2022, NeurIPS, *MEMO: test time robustness via adaptation and augmentation* — https://arxiv.org/abs/2110.09506
- Wang et al. 2021, ICLR, *Tent: fully test-time adaptation by entropy minimization* — https://arxiv.org/abs/2006.10726
- Mirza et al. 2022, CVPR, *DUA: the norm must go on* — https://arxiv.org/abs/2112.00463
- Zhao et al. 2023, ICML, *On pitfalls of test-time adaptation* — https://arxiv.org/abs/2306.03536
- Press et al. 2023, NeurIPS, *RDumb: a simple approach that questions our progress in continual test-time adaptation* — https://arxiv.org/abs/2306.05401
- Yoon et al. 2024, ICLR, *C-TPT: calibrated test-time prompt tuning* — https://arxiv.org/abs/2403.14119
- Gao et al. 2023, CVPR, *Back to the source: diffusion-driven adaptation* (DDA) — https://arxiv.org/abs/2207.03442
- Sun et al. 2020, ICML, *Test-time training with self-supervision* — https://arxiv.org/abs/1909.13231
- Lyzhov et al. 2020, UAI, *Greedy policy search: a simple baseline for learnable test-time augmentation* — https://arxiv.org/abs/2002.09103
- Shanmugam et al. 2021, ICCV, *Better aggregation in test-time augmentation* — https://arxiv.org/abs/2011.11156
- Kimura 2021, ICONIP, *Understanding test-time augmentation* — https://arxiv.org/abs/2402.06892
- Medeiros 2026, arXiv, *Test-time augmentation on MedMNIST: when TTA hurts* (BN-statistics mismatch) — https://arxiv.org/abs/2604.09697
- Llera et al. 2019, Mol Imaging Biol, *Per-scan intensity harmonisation of DaT-SPECT across camera models (Gamma-CDF)* — https://pubmed.ncbi.nlm.nih.gov/29987621/
- Nyúl, Udupa, Zhang 2000, IEEE TMI, *New variants of a method of MRI scale standardization* — https://pubmed.ncbi.nlm.nih.gov/10784285/
- Zhao et al. 2022, IJCAI, *Test-time Fourier style calibration for domain generalization* (TAF-Cal) — https://arxiv.org/abs/2205.06427
- Yang & Soatto 2020, CVPR, *FDA: Fourier domain adaptation* — https://arxiv.org/abs/2004.05498
- Karani et al. 2021, MedIA, *Test-time adaptable neural networks for robust medical image segmentation* — https://arxiv.org/abs/2004.04668
- He et al. 2021, MedIA, *Autoencoder-based self-supervised test-time adaptation* — https://pmc.ncbi.nlm.nih.gov/articles/PMC8316425/
- Valanarasu et al. 2022, MIDL, *On-the-fly test-time adaptation for medical image segmentation* — https://arxiv.org/abs/2203.05574
- Dong et al. 2024, arXiv, *InTEnt: single-image test-time adaptation via integrated entropy weighting* — https://arxiv.org/abs/2402.09604
- Joshi et al. 2025, arXiv, *MuVi: single-volume test-time adaptation for 3D MRI* — https://arxiv.org/abs/2506.23705
- Buddenkotte & Buchert 2024, J Nucl Med, *Unrealistic augmentation for cross-camera DaT-SPECT robustness* — https://doi.org/10.2967/jnumed.124.267570
- Tossici-Bolt et al. 2011, EJNMMI, *Calibration of gamma camera systems for a multicentre European 123I-FP-CIT SPECT normal database* — https://pubmed.ncbi.nlm.nih.gov/21468761/
- Tossici-Bolt et al. 2017, EJNMMI Phys, *ENC-DAT: reconstruction-dependent quantification* — https://pmc.ncbi.nlm.nih.gov/articles/PMC5272851/
- Varrone et al. 2013, EJNMMI, *European multicentre database of healthy controls for 123I-FP-CIT SPECT (ENC-DAT)* — https://pubmed.ncbi.nlm.nih.gov/23160999/
- Wenzel et al. 2019, EJNMMI, *Automatic classification of dopamine transporter SPECT: CNNs robust to variable image characteristics* — https://pubmed.ncbi.nlm.nih.gov/31473800/

**Q3 — label noise and log loss**
- Gneiting & Raftery 2007, JASA 102(477):359–378, *Strictly proper scoring rules, prediction, and estimation* — https://doi.org/10.1198/016214506000001437
- Guo et al. 2017, ICML, *On calibration of modern neural networks* — https://arxiv.org/abs/1706.04599
- Penso, Frenkel, Goldberger 2024, IEEE TMI, *Confidence calibration of a medical imaging classification system that is robust to label noise* — https://www.eng.biu.ac.il/goldbej/files/2024/01/Coby_TMI_2023.pdf
- Zhang & Sabuncu 2018, NeurIPS, *Generalized cross entropy loss* — https://arxiv.org/abs/1805.07836
- Wang et al. 2019, ICCV, *Symmetric cross entropy* — https://arxiv.org/abs/1908.06112
- Ma et al. 2020, ICML, *Normalized loss functions for deep learning with noisy labels* — https://arxiv.org/abs/2006.13554
- Englesson & Azizpour 2021, NeurIPS, *Generalized Jensen-Shannon divergence loss* — https://arxiv.org/abs/2105.04522
- Zhou et al. 2021, ICML, *Asymmetric loss functions for learning with noisy labels* — https://arxiv.org/abs/2106.03110
- Wei et al. 2022, ICML, *To smooth or not? When label smoothing meets noisy labels* — https://arxiv.org/abs/2106.04149
- Reed et al. 2015, ICLR-W, *Training deep neural networks on noisy labels with bootstrapping* — https://arxiv.org/abs/1412.6596
- Chen et al. 2021, AAAI, *Beyond class-conditional assumption: instance-dependent label noise* (SEAL) — https://arxiv.org/abs/2012.05458
- Müller, Kornblith, Hinton 2019, NeurIPS, *When does label smoothing help?* — https://arxiv.org/abs/1906.02629
- Lukasik et al. 2020, ICML, *Does label smoothing mitigate label noise?* — https://arxiv.org/abs/2003.02819
- Thulasidasan et al. 2019, NeurIPS, *On mixup training: improved calibration and predictive uncertainty* — https://arxiv.org/abs/1905.11001
- Zhang et al. 2022, ICML, *When and how mixup improves calibration* — https://arxiv.org/abs/2102.06289
- Zhang et al. 2018, ICLR, *mixup: beyond empirical risk minimization* — https://arxiv.org/abs/1710.09412
- Peterson et al. 2019, ICCV, *Human uncertainty makes classification more robust* (CIFAR-10H) — https://arxiv.org/abs/1908.07086
- Collins, Bhatt, Weller 2022, HCOMP, *Eliciting and learning with soft labels from every annotator* — https://arxiv.org/abs/2207.00810
- Zhang & Sabuncu 2020, NeurIPS, *Self-distillation as instance-specific label smoothing* — https://arxiv.org/abs/2006.05065
- Uma et al. 2021, JAIR, *Learning from disagreement: a survey* — https://www.jair.org/index.php/jair/article/view/12752
- Irvin et al. 2019, AAAI, *CheXpert* (uncertain-label policies) — https://arxiv.org/abs/1901.07031
- Pham et al. 2021, Neurocomputing, *Label-smoothing regularisation of uncertain CheXpert labels* — https://arxiv.org/abs/1911.06475
- Han et al. 2018, NeurIPS, *Co-teaching* — https://arxiv.org/abs/1804.06872
- Li et al. 2020, ICLR, *DivideMix* — https://arxiv.org/abs/2002.07394
- Liu et al. 2020, NeurIPS, *Early-learning regularization* — https://arxiv.org/abs/2007.00151
- Pleiss et al. 2020, NeurIPS, *Identifying mislabeled data using the area under the margin ranking* — https://arxiv.org/abs/2001.10528
- Northcutt et al. 2021, JAIR, *Confident learning* — https://arxiv.org/abs/1911.00068
- Liu et al. 2021, ICML, *Understanding instance-level label noise: disparate impacts and treatments* — https://arxiv.org/abs/2102.05336
- Dirichlet-based prediction calibration for learning with noisy labels, AAAI 2024 — https://arxiv.org/abs/2401.07062
- Karimi et al. 2020, MedIA, *Deep learning with noisy labels: techniques and remedies in medical image analysis* — https://arxiv.org/abs/1912.02911
- Ju et al. 2022, IEEE TMI, *Improving medical images classification with label noise using dual-uncertainty estimation* — https://arxiv.org/abs/2103.00528
- Budenkotte et al. 2024, EJNMMI, *Automated identification of uncertain cases in deep learning-based classification of DaT-SPECT* — https://pmc.ncbi.nlm.nih.gov/articles/PMC10957699/
- Mathies et al. 2022 (Buchert group), inter-rater agreement of visual DaT-SPECT reads across reconstructions — https://pmc.ncbi.nlm.nih.gov/articles/PMC9385910/
- RSNA-2024 2nd-place write-up (Artley et al.), OOF pseudo-label loss and confident-learning removal — https://github.com/brendanartley/RSNA-2024-Competition
- Bulten et al. 2022, Nat Med, *PANDA challenge* (label-noise handling by top teams) — https://www.nature.com/articles/s41591-021-01620-2
