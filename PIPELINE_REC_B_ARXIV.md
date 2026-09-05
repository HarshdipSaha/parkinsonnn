# Pipeline Recommendation B — Fresh arXiv Sweep (2025-01-01 → 2026-09-05)

Author: Subagent B (arXiv skill). Date: 2026-09-05.
Scope: catch anything on arXiv **newer than `LITERATURE_REVIEW.md`** that changes how we should build the DaT Parkinson's Challenge pipeline. Everything cited below was retrieved from the arXiv Atom API (`https://export.arxiv.org/api/query`) during this session; abstracts were read, full texts were not. Nothing here is fabricated; where I could not verify a fact (e.g., a weight license) I say so.

Ground truth used: `DAT_PARKINSONS_CHALLENGE.md`, `LITERATURE_REVIEW.md`, `EDA_REPORT.md`.

---

## 0. Headline findings (read this first)

1. **arXiv has essentially no new DaT-SPECT classification work since 2025.** Across 110 queries, only one 2025+ paper is specifically about DaT-SPECT deep learning ([2506.20781](https://arxiv.org/abs/2506.20781), attenuation compensation, not classification). The DaT-classification literature lives in journals (EJNMMI, JNM, MDPI), which the existing lit review already covers. So this sweep's value is in **adjacent methodology**: small-data 3D classification, 2D-foundation-model-to-3D adaptation, multi-site robustness, and log-loss calibration.
2. **The strongest new evidence says: for small 3D datasets, 2D/2.5D approaches on frozen foundation models and *small* CNNs beat big 3D architectures** — MRN won the MICCAI 2025 PDCADxFoundation Parkinson's challenge with 300 scans using ROI crops + 2D VFM features ([2509.17566](https://arxiv.org/abs/2509.17566)); AnyMC3D (1st place VLM3D) found "2D-based methods surpass 3D architectures for 3D classification" ([2512.12887](https://arxiv.org/abs/2512.12887)); the SSL3D/FOMO25 brain-MRI challenge winners used a plain U-Net CNN that was 10× smaller and 1–2 orders of magnitude faster to train than transformer entries ([2601.13166](https://arxiv.org/abs/2601.13166)).
3. **But natural-image foundation models degrade on nuclear-medicine images.** The DINOv3 medical benchmark explicitly lists PET among the modalities where features "degrade" ([2509.06467](https://arxiv.org/abs/2509.06467)). DaT-SPECT is the closest modality to PET. Treat any FM branch as an *ablation*, not the backbone.
4. **Registered ROI features + a tabular head is now backed by fresh evidence, not just the 2022 radiomics paper.** In a 2026 benchmark of MRI foundation models, radiomics + TabPFN "matched or outperformed all visual encoders" ([2606.23172](https://arxiv.org/abs/2606.23172)); a lung-CT benchmark found classifier-head choice (TabPFN/CatBoost/LR) drives cross-cohort performance for several tasks ([2607.01001](https://arxiv.org/abs/2607.01001)).
5. **Calibration for log loss:** post-hoc calibrator choice matters more than most people assume. A 21-classifier × 5-calibrator study found **Venn-Abers gives the largest average log-loss reductions** ([2601.19944](https://arxiv.org/abs/2601.19944)), but the same group shows it *hurts* already-well-calibrated models (LR, CatBoost, TabPFN) and helps over-confident ones (XGBoost/LightGBM) ([2605.03816](https://arxiv.org/abs/2605.03816)). TabPFN as a head on frozen embeddings gives the best NLL/ECE among nine heads ([2607.11007](https://arxiv.org/abs/2607.11007)). Stacking with a logistic-regression meta-learner "destroys calibration" while improving accuracy ([2605.18696](https://arxiv.org/abs/2605.18696)) — do **not** stack; average probabilities and then calibrate once.
6. **Site/scanner effects dominate foundation-model embeddings** ([2604.14441](https://arxiv.org/abs/2604.14441)) and both DL and hand-crafted pipelines fail on unseen sites in leave-one-site-out tests ([2603.18050](https://arxiv.org/abs/2603.18050)). Our CV must be pseudo-site-grouped, and calibration temperature should be chosen on out-of-site folds.

---

## 1. Search log

**Method.** Four sequential sweeps against the arXiv Atom API, Python `urllib` + `xml.etree`, `sortBy=submittedDate desc`, page size 100, 3.5–4 s sleep between calls (well under the 3 req/s limit). Sweep 1 used the server-side `submittedDate:[202501010000 TO 202609062359]` filter combined with compound booleans; this visibly under-counted (e.g., 1 hit for `SPECT AND Parkinson`), so sweeps 2–4 used **simple booleans with client-side date filtering** (`published >= 2025-01-01`), paginating until results were older than the cutoff. Two batches hit HTTP 429 when sweeps 2 and 3 overlapped; those 13 queries were re-run sequentially in sweep 4 (`sweep_retry`) and all succeeded.

**Totals.** 110 query executions (97 distinct + 13 retries); **572 unique 2025+ papers retrieved**; ~95 abstracts read in full; **~45 judged relevant**, of which **41 are annotated in §2** (the rest were redundant with annotated ones). Date filter for everything: submissions dated 2025-01-01 or later.

**Query families and yield (relevant / kept 2025+):**

| Family | Example queries | Kept 2025+ | Relevant |
|---|---|---|---|
| DaT-SPECT / FP-CIT / DaTscan / SPECT+Parkinson | `all:DaTscan`, `all:"DaT SPECT"`, `all:ioflupane`, `all:"FP-CIT"`, `all:dopamine AND all:transporter AND all:SPECT`, `abs:SPECT AND abs:Parkinson`, `all:"specific binding ratio"` | 9 | 3 (only 1 is DL) |
| SPECT/PET brain DL, nuclear-medicine DL | `all:SPECT AND all:"deep learning"`, `all:"nuclear medicine" AND all:"deep learning"`, `all:PET AND all:Parkinson AND all:"deep learning"`, `abs:Parkinson AND abs:"deep learning" AND (abs:SPECT OR abs:PET OR abs:neuroimaging)` | 18 | 5 |
| Registration-free / spatial normalization / affine DL | `all:"registration-free" AND all:brain`, `all:"spatial normalization" AND all:brain`, `all:"affine registration" AND all:"deep learning" AND all:brain`, `all:atlas AND all:registration AND (PET OR SPECT) AND "deep learning"` | 5 | 3 |
| 3D foundation models / SSL pretraining (CT/MRI/PET/brain) | `all:"foundation model" AND (volumetric OR 3D) AND medical`, `all:"self-supervised" AND 3D AND medical AND pretraining`, `all:"brain foundation model"`, `all:"foundation model" AND all:PET`, `all:"foundation model" AND all:SPECT` (0 hits), `all:Med3D OR all:MedicalNet`, `all:SuPreM OR VoCo OR Merlin` | ~330 | 22 |
| 2D-FM → 3D, 2.5D, DINOv2/v3 in medical | `all:"2.5D" AND classification AND medical`, `all:DINOv3 AND all:medical`, `all:DINOv2 AND 3D AND medical AND classification`, `all:PDCADxFoundation`, `challenge AND "first place" AND 3D AND classification` | 75 | 9 |
| Multi-center DG / harmonization / shortcut / LOSO | `all:"domain generalization" AND all:"medical image classification"`, `all:harmonization AND PET AND "deep learning"`, `all:"scanner harmonization"`, `all:"leave-one-site-out"`, `all:shortcut AND "medical image" AND (site OR scanner)`, `all:"risk extrapolation" AND medical`, `all:"test-time adaptation" AND medical classification` | 40 | 10 |
| Calibration / log loss / proper scoring | `all:calibration AND all:"medical image classification"`, `all:"temperature scaling" AND medical`, `all:"log loss" AND calibration AND classification`, `all:"focal loss" AND calibration`, `all:"label smoothing" AND calibration AND medical`, `all:"test-time augmentation" AND calibration`, `all:"stochastic weight averaging" AND calibration`, `all:mixup AND calibration AND medical`, `all:ensemble AND "log loss" AND medical` | 70 | 12 |
| Tabular FMs on imaging features | `all:TabPFN AND (medical OR imaging OR radiomics OR clinical)`, `all:radiomics AND "foundation model" AND classification` | 35 | 5 |
| Intensity/count normalization, SPECT reconstruction, synthetic SPECT | `all:"intensity normalization" AND (PET OR SPECT)`, `SPECT AND reconstruction AND "deep learning" AND brain`, `SPECT AND (synthetic OR GAN OR diffusion) AND brain` | 2 | 1 |

<details>
<summary>Full per-query log (110 rows; "server total" is arXiv's all-time count for sweeps 2–4, or the date-filtered count for sweep 1; −1 = 429 during that sweep, retried in sweep_retry)</summary>

| # | sweep | query | server total / kept (2025-01-01+) |
|---|---|---|---|
| 1 | sweep1 | `(all:"dopamine transporter" AND all:SPECT AND (all:"deep learning" OR all:neural OR all:CNN OR all:classification))` | 1 / 1 |
| 2 | sweep1 | `(all:DaTscan OR all:"DaT-SPECT" OR all:"DAT SPECT" OR all:"DaT SPECT")` | 4 / 4 |
| 3 | sweep1 | `(all:"FP-CIT" OR all:ioflupane OR all:"123I-FP-CIT")` | 1 / 1 |
| 4 | sweep1 | `(all:SPECT AND all:Parkinson)` | 1 / 1 |
| 5 | sweep1 | `(all:"striatal binding" OR all:"specific binding ratio" OR all:"putamen" AND all:SPECT)` | 1 / 1 |
| 6 | sweep1 | `((all:SPECT OR all:PET) AND all:brain AND all:"3D" AND (all:convolutional OR all:CNN) AND all:classification)` | 1 / 1 |
| 7 | sweep1 | `((all:"nuclear medicine" OR all:SPECT) AND all:"deep learning" AND all:brain)` | 0 / 0 |
| 8 | sweep1 | `(all:PET AND all:Parkinson AND all:"deep learning")` | 1 / 1 |
| 9 | sweep1 | `(all:"registration-free" OR all:"registration free" OR all:"without registration") AND (all:brain OR all:medical) AND all:classification` | 0 / 0 |
| 10 | sweep1 | `(all:"spatial normalization" AND all:"deep learning" AND all:brain)` | 0 / 0 |
| 11 | sweep1 | `((all:"affine registration" OR all:"rigid registration") AND all:"deep learning" AND all:brain AND (all:PET OR all:SPECT OR all:template))` | 0 / 0 |
| 12 | sweep1 | `(all:"foundation model" AND (all:volumetric OR all:"3D") AND all:medical AND (all:CT OR all:MRI OR all:PET))` | 84 / 84 |
| 13 | sweep1 | `((all:"self-supervised" OR all:"masked autoencoder") AND all:"3D" AND all:medical AND (all:pretraining OR all:"pre-training"))` | 66 / 66 |
| 14 | sweep1 | `(all:"foundation model" AND (all:PET OR all:SPECT OR all:"nuclear medicine"))` | 36 / 36 |
| 15 | sweep1 | `(all:"foundation model" AND all:brain AND all:volumetric)` | 17 / 17 |
| 16 | sweep1 | `(all:"transfer learning" AND all:"3D" AND all:medical AND all:classification AND all:"small dataset")` | 0 / 0 |
| 17 | sweep1 | `((all:"domain generalization" OR all:"multi-center" OR all:"multi-site") AND all:medical AND all:classification AND (all:scanner OR all:harmonization))` | 6 / 6 |
| 18 | sweep1 | `((all:harmonization OR all:ComBat) AND all:"deep learning" AND (all:scanner OR all:site) AND (all:PET OR all:SPECT OR all:MRI))` | 6 / 6 |
| 19 | sweep1 | `(all:"test-time adaptation" AND all:medical AND all:classification)` | 5 / 5 |
| 20 | sweep1 | `(all:calibration AND all:"medical image" AND all:classification AND (all:"log loss" OR all:"cross-entropy" OR all:"temperature scaling" OR all:overconfidence))` | 7 / 7 |
| 21 | sweep1 | `((all:"probability calibration" OR all:"temperature scaling" OR all:"calibrated") AND all:"deep learning" AND (all:"small data" OR all:"limited data" OR all:"low-data"))` | 8 / 8 |
| 22 | sweep1 | `((all:"deep ensembles" OR all:"ensemble") AND all:calibration AND all:medical AND all:classification)` | 6 / 6 |
| 23 | sweep1 | `((all:"intensity normalization" OR all:"count normalization") AND (all:SPECT OR all:PET OR all:"nuclear medicine"))` | 1 / 1 |
| 24 | sweep1 | `(all:SPECT AND all:reconstruction AND all:"deep learning" AND all:brain)` | 0 / 0 |
| 25 | sweep1 | `(all:radiomics AND all:Parkinson AND (all:SPECT OR all:PET))` | 0 / 0 |
| 26 | sweep1 | `(all:"3D" AND all:medical AND all:classification AND (all:"few hundred" OR all:"small dataset" OR all:"limited data") AND all:CNN)` | 1 / 1 |
| 27 | sweep1 | `((all:"2.5D" OR all:"multi-view") AND all:medical AND all:"3D" AND all:classification)` | 8 / 8 |
| 28 | sweep1 | `((all:synthetic OR all:GAN OR all:diffusion) AND all:SPECT AND all:brain)` | 0 / 0 |
| 29 | sweep2 | `all:dopamine AND all:transporter AND all:SPECT` | 5 / 1 |
| 30 | sweep2 | `all:DaTscan` | 10 / 3 |
| 31 | sweep2 | `all:"DaT SPECT"` | 5 / 1 |
| 32 | sweep2 | `all:"FP-CIT"` | 2 / 0 |
| 33 | sweep2 | `all:ioflupane` | 3 / 1 |
| 34 | sweep2 | `all:SPECT AND all:Parkinson` | 14 / 1 |
| 35 | sweep2 | `all:SPECT AND all:"deep learning"` | 37 / 9 |
| 36 | sweep2 | `all:SPECT AND all:convolutional` | 21 / 2 |
| 37 | sweep2 | `all:PET AND all:Parkinson AND all:"deep learning"` | 2 / 1 |
| 38 | sweep2 | `abs:Parkinson AND abs:"deep learning" AND (abs:SPECT OR abs:PET OR abs:neuroimaging)` | 10 / 5 |
| 39 | sweep2 | `all:"nuclear medicine" AND all:"deep learning"` | 11 / 1 |
| 40 | sweep2 | `all:"registration-free" AND all:brain` | 5 / 1 |
| 41 | sweep2 | `all:"without registration" AND all:brain` | 1 / 0 |
| 42 | sweep2 | `all:"spatial normalization" AND all:brain` | 10 / 1 |
| 43 | sweep2 | `all:"affine registration" AND all:"deep learning" AND all:brain` | 5 / 2 |
| 44 | sweep2 | `all:"intensity normalization" AND (all:PET OR all:SPECT)` | 2 / 1 |
| 45 | sweep2 | `all:harmonization AND all:PET AND all:"deep learning"` | 2 / 2 |
| 46 | sweep2 | `all:"scanner harmonization"` | 2 / 1 |
| 47 | sweep2 | `all:"domain generalization" AND all:"medical image classification"` | 12 / 3 |
| 48 | sweep2 | `all:"multi-site" AND all:brain AND all:classification AND all:"deep learning"` | 13 / 5 |
| 49 | sweep2 | `all:calibration AND all:"medical image classification"` | 30 / 15 |
| 50 | sweep2 | `all:"temperature scaling" AND all:medical` | 21 / 12 |
| 51 | sweep2 | `all:"log loss" AND all:calibration AND all:classification` | 18 / 10 |
| 52 | sweep2 | `all:"focal loss" AND all:calibration` | 44 / 23 |
| 53 | sweep2 | `all:"label smoothing" AND all:calibration AND all:medical` | 5 / 1 |
| 54 | sweep2 | `all:radiomics AND all:Parkinson` | 0 / 0 |
| 55 | sweep2 | `all:"3D CNN" AND all:"small dataset" AND all:medical` | 3 / 0 |
| 56 | sweep2 | `all:"Med3D" OR all:"MedicalNet"` | 28 / 20 |
| 57 | sweep2 | `all:SuPreM OR all:"VoCo" OR all:"Merlin" AND all:CT` | 493 / 109 |
| 58 | sweep2 | `all:"brain foundation model"` | 28 / 27 |
| 59 | sweep2 | `all:"foundation model" AND all:brain AND all:MRI AND all:"3D"` | 28 / 23 |
| 60 | sweep2 | `all:"foundation model" AND all:PET` | 41 / 36 |
| 61 | sweep2 | `all:"foundation model" AND all:SPECT` | −1 (429) |
| 62 | sweep2 | `all:SPECT AND (all:synthetic OR all:GAN OR all:diffusion) AND all:brain` | −1 (429) |
| 63 | sweep2 | `all:"2.5D" AND all:classification AND all:medical` | −1 (429) |
| 64 | sweep2 | `all:"test-time adaptation" AND all:"medical image classification"` | −1 (429) |
| 65 | sweep2 | `all:"limited data" AND all:"3D" AND all:"medical image classification"` | −1 (429) |
| 66 | sweep2 | `all:striatum AND all:segmentation AND all:"deep learning"` | −1 (429) |
| 67 | sweep2 | `all:Parkinson AND all:diagnosis AND all:"deep learning" AND all:review` | −1 (429) |
| 68 | sweep3 | `all:PDCADxFoundation` | 1 / 1 |
| 69 | sweep3 | `abs:SPECT AND abs:Parkinson` | 13 / 1 |
| 70 | sweep3 | `all:dopaminergic AND all:"deep learning"` | 3 / 1 |
| 71 | sweep3 | `all:striatal AND (all:"deep learning" OR all:"neural network")` | 3 / 1 |
| 72 | sweep3 | `all:"quantitative SPECT"` | 9 / 3 |
| 73 | sweep3 | `all:"specific binding ratio"` | 0 / 0 |
| 74 | sweep3 | `all:TabPFN AND (all:medical OR all:imaging OR all:radiomics OR all:clinical)` | −1 (429) |
| 75 | sweep3 | `all:radiomics AND all:"foundation model" AND all:classification` | −1 (429) |
| 76 | sweep3 | `all:"test-time augmentation" AND all:calibration` | −1 (429) |
| 77 | sweep3 | `all:"stochastic weight averaging" AND all:calibration` | −1 (429) |
| 78 | sweep3 | `all:mixup AND all:calibration AND all:medical` | −1 (429) |
| 79 | sweep3 | `all:MedicalNet` | −1 (429) |
| 80 | sweep3 | `all:MONAI AND all:classification AND all:"3D"` | 2 / 1 |
| 81 | sweep3 | `all:"spatial transformer" AND all:brain AND all:classification` | 4 / 2 |
| 82 | sweep3 | `all:atlas AND all:registration AND (all:PET OR all:SPECT) AND all:"deep learning"` | 0 / 0 |
| 83 | sweep3 | `all:PET AND all:brain AND (all:"domain shift" OR all:"scanner variability" OR all:harmonization)` | 6 / 3 |
| 84 | sweep3 | `all:"leave-one-site-out"` | 15 / 12 |
| 85 | sweep3 | `all:shortcut AND all:"medical image" AND all:classification AND (all:site OR all:scanner)` | 2 / 1 |
| 86 | sweep3 | `all:"risk extrapolation" AND all:medical` | 3 / 3 |
| 87 | sweep3 | `all:"brain age" AND all:"foundation model"` | 15 / 12 |
| 88 | sweep3 | `all:PET AND all:brain AND all:"3D" AND all:classification AND (all:CNN OR all:transformer)` | 4 / 1 |
| 89 | sweep3 | `all:amyloid AND all:PET AND all:"deep learning" AND all:classification` | 4 / 4 |
| 90 | sweep3 | `all:FDG AND all:Parkinson AND all:"deep learning"` | 0 / 0 |
| 91 | sweep3 | `all:"self-supervised" AND (all:PET OR all:SPECT) AND all:classification` | 3 / 2 |
| 92 | sweep3 | `all:DINOv3 AND all:medical` | 53 / 53 |
| 93 | sweep3 | `all:DINOv2 AND all:"3D" AND all:medical AND all:classification` | 4 / 2 |
| 94 | sweep3 | `all:ensemble AND all:"log loss" AND all:medical` | 1 / 0 |
| 95 | sweep3 | `all:Kaggle AND all:"medical imaging" AND all:"3D" AND all:winning` | 0 / 0 |
| 96 | sweep3 | `all:challenge AND all:"first place" AND all:"3D" AND all:classification AND all:medical` | 1 / 1 |
| 97 | sweep3 | `all:"3D" AND all:medical AND all:classification AND (all:"parameter-efficient" OR all:LoRA) AND all:"foundation model"` | 3 / 2 |
| 98 | sweep_retry | `all:"foundation model" AND all:SPECT` | 0 / 0 |
| 99 | sweep_retry | `all:SPECT AND (all:synthetic OR all:GAN OR all:diffusion) AND all:brain` | 3 / 0 |
| 100 | sweep_retry | `all:"2.5D" AND all:classification AND all:medical` | 14 / 5 |
| 101 | sweep_retry | `all:"test-time adaptation" AND all:"medical image classification"` | 3 / 2 |
| 102 | sweep_retry | `all:"limited data" AND all:"3D" AND all:"medical image classification"` | 1 / 0 |
| 103 | sweep_retry | `all:striatum AND all:segmentation AND all:"deep learning"` | 1 / 0 |
| 104 | sweep_retry | `all:Parkinson AND all:diagnosis AND all:"deep learning" AND all:review` | 4 / 2 |
| 105 | sweep_retry | `all:TabPFN AND (all:medical OR all:imaging OR all:radiomics OR all:clinical)` | 31 / 31 |
| 106 | sweep_retry | `all:radiomics AND all:"foundation model" AND all:classification` | 5 / 4 |
| 107 | sweep_retry | `all:"test-time augmentation" AND all:calibration` | 15 / 8 |
| 108 | sweep_retry | `all:"stochastic weight averaging" AND all:calibration` | 10 / 4 |
| 109 | sweep_retry | `all:mixup AND all:calibration AND all:medical` | 4 / 1 |
| 110 | sweep_retry | `all:MedicalNet` | 7 / 5 |

</details>

---

## 2. Annotated new-paper list (not in `LITERATURE_REVIEW.md`)

Relevance verdict scale: **A** = directly shapes the pipeline; **B** = useful evidence/technique; **C** = context only / not applicable but worth knowing.

### 2.1 DaT-SPECT and nuclear-medicine specific

| arXiv | Title (date) | Summary | Verdict |
|---|---|---|---|
| [2506.20781](https://arxiv.org/abs/2506.20781) | *Development and in silico imaging trial evaluation of a deep-learning-based transmission-less attenuation compensation method for DaT SPECT* — Yu, Rahman, Li, Ying, An, Benzinger, Jha et al. (2025-06-25) | DaT-CTLESS: DL attenuation compensation (AC) for DaT-SPECT without a CT, evaluated in a virtual imaging trial on caudate/putamen/globus-pallidus quantification. Not a classifier. **Why it matters here:** it confirms that clinical DaT-SPECT is acquired both with and without CT-based AC depending on whether the site's camera has a CT — a hidden acquisition factor that changes striatal-to-background contrast. Our 10 hospitals almost certainly mix AC/non-AC; this is a domain axis our augmentation (contrast/gamma jitter, background-level jitter) should cover. | **B** |
| [2503.16706](https://arxiv.org/abs/2503.16706) | *ISIT-GEN: an in silico imaging trial to assess inter-scanner generalizability of CTLESS for myocardial perfusion SPECT* — Yu, Choi, Yang, Obuchowski, Siegel, Jha (2025-03-20) | Same group; tests a SPECT DL method across three scanner vendors and finds generalization holds when evaluated on the clinical task. Cardiac, not brain. Useful as a framing: evaluate cross-vendor generalization on the *task*, not on image metrics. | **C** |
| [2507.16065](https://arxiv.org/abs/2507.16065) | *Handcrafted vs. Deep Radiomics vs. Fusion vs. Deep Learning: review of ML-based outcome prediction in PET and SPECT* — Salmanpour et al. (2025-07-21) | 226-study systematic review (2020–2025). Deep-radiomics features had highest mean accuracy (0.862); **fusion (handcrafted + deep) had highest mean AUC (0.861)**; SPECT studies underperformed PET. Supports a hybrid CNN + handcrafted-ROI-feature ensemble. | **B** |
| [2511.04458](https://arxiv.org/abs/2511.04458) | *TRAECR: A tool for preprocessing PET imaging for statistical modeling* — Ambekar, Zielinski, Eloyan (2025-11-06) | A PET preprocessing pipeline: template registration, MRI co-registration, brain extraction, **ComBat/RAVEL harmonization**. A good template for the order of operations (register → extract → harmonize) even though we lack MRI. Note ComBat needs site labels at inference — not available on the test set, so only usable as a training-time diagnostic. | **B** |
| [2605.29932](https://arxiv.org/abs/2605.29932) | *Treatment-conditioned diffusion for forecasting neurodegenerative disease progression* (2026-05-28) | Diffusion model conditioned on screening DaTscan + levodopa dose to predict future brain states. Uses PPMI DaTscans. Not applicable (progression, PPMI). | **C** |
| [2603.15070](https://arxiv.org/abs/2603.15070) | *Hamiltonian dynamics for stochastic reconstruction in emission tomography* (2026-03-16) | HMC ensemble reconstruction; evaluated on one clinical DATSCAN acquisition. Reconstruction-level uncertainty, not classification. | **C** |
| [2502.06632](https://arxiv.org/abs/2502.06632) | *Few-shot classification and anatomical localization of tissues in SPECT imaging* (2025-02-10) | Prototypical networks with ResNet-18 on 2D cardiac SPECT slices with tiny labeled data. Cardiac; proof-of-concept only. | **C** |

### 2.2 Small-data 3D classification: 2D-FM-to-3D, 2.5D, compact CNNs

| arXiv | Title (date) | Summary | Verdict |
|---|---|---|---|
| [2509.17566](https://arxiv.org/abs/2509.17566) | **MRN: Harnessing 2D Vision Foundation Models for Diagnosing Parkinson's Disease with Limited 3D MR Data** — Ding, Liu, Zhou, Liu (2025-09-22). *1st place, MICCAI 2025 PDCADxFoundation classification track.* | Parkinson's diagnosis from QSM + neuromelanin MRI with **only 300 labeled scans**. Explicitly rejects fine-tuning pretrained 3D medical models because of "mismatches in voxel spacing and modality between pre-training and fine-tuning data." Instead: crop multiple **key ROIs** (brain nuclei), encode axial slices of each ROI with a frozen 2D VFM, fuse slices into an ROI token with an auxiliary segmentation head steering attention to the nucleus, combine ROI tokens, and train with **multi-ROI supervised contrastive loss**. 86.0% accuracy, +5.5% over 2nd place. This is the closest analogue to our problem (PD, tiny 3D dataset, deep nuclei) and its design choices — ROI cropping, 2D-FM slice encoding, SupCon — transfer directly to a striatum-cropped DaT-SPECT volume. | **A** |
| [2512.12887](https://arxiv.org/abs/2512.12887) | **Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification (AnyMC3D)** — Liu, Georgescu, Zhang, Yoo, Baumgartner, Gao et al., Siemens (2025-12-15). *1st place VLM3D.* | Frozen 2D FM + ~1M-parameter task plugins; 12-task benchmark. Findings: (1) adaptation is essential; (2) **general-purpose FMs match medical-specific FMs if adapted properly**; (3) **2D-based methods surpass 3D architectures for 3D classification.** Supports a frozen-2D-backbone + light head branch. | **A** |
| [2507.08254](https://arxiv.org/abs/2507.08254) | **Raptor: Scalable Train-Free Embeddings for 3D Medical Volumes Leveraging Pretrained 2D Foundation Models** — An, Jeong, Lee, Gorla, Yang, Sankararaman (ICML 2025 Spotlight) | Train-free: frozen natural-image 2D FM tokens from cross-sections, compressed by random projections. Beats SuPreM (+3%), MISFM (+6%), Merlin (+10%), VoCo (+13%), SLIViT (+14%) on 10 volume tasks. Cheap to try: embeddings for 1,362 volumes in minutes; then any tabular head. Prize-safe if the 2D backbone is (DINOv2, Apache-2.0). | **A** |
| [2509.06467](https://arxiv.org/abs/2509.06467) | **Does DINOv3 Set a New Medical Vision Standard?** — Liu et al. (2025-09-08) | DINOv3 is a strong baseline across 2D/3D medical classification/segmentation/registration and can beat BiomedCLIP/CT-Net, **but "features degrade in scenarios requiring deep domain specialisation, such as WSIs, EM, and PET"**, and it does not follow scaling laws in medical tasks (bigger ≠ better). Direct caution for DaT-SPECT: expect natural-image FM features to be weaker on our modality; use small model sizes; validate rather than assume. | **A** (as a caution) |
| [2509.12512](https://arxiv.org/abs/2509.12512) | *DinoAtten3D: Slice-level attention aggregation of DINOv2 for 3D brain MRI anomaly classification* (ICCV-W 2025). Code: github.com/Rafsani/DinoAtten3D | Frozen DINOv2 on axial slices, soft-attention slice weighting, SupCon + class-variance regularization; works with limited, imbalanced data. Ready-made recipe for a 2.5D branch. | **B** |
| [2603.14832](https://arxiv.org/abs/2603.14832) | *Halfway to 3D: Ensembling 2.5D and 3D models for robust COVID-19 CT diagnosis* (2026-03-16). Code: github.com/HySonLab/PHAROS-AIF-MIH | Multi-source CT: 2.5D DINOv3 tri-plane branch + 3D ResNet-18 pretrained with **VREx then supervised contrastive** for cross-source robustness; **logit-level ensemble** beats both. Almost exactly the two-branch ensemble I recommend below. | **A** |
| [2606.06950](https://arxiv.org/abs/2606.06950) | *When is 3D worth it? A resource–performance frontier for CNNs and transformers in lung CT* (2026-06-05) | n≈2k, fixed protocol: **2.5D CNN gave the best discrimination–stability trade-off; 3D CNNs showed threshold instability; transformers produced degenerate all-positive predictions.** Wide CIs, but a clear warning against ViTs at our data scale. | **B** |
| [2601.13166](https://arxiv.org/abs/2601.13166) | *From 100,000+ images to winning the first brain MRI foundation model challenges* — Gordaliza et al. (2026-01-19). *1st place SSL3D (ResEnc-L) and FOMO25 (methods).* Models: github.com/jbanusco/BrainFM4Challenges | Winners used a **U-Net CNN with anatomical priors**, "trained 1–2 orders of magnitude faster and 10× smaller than competing transformer approaches." Reinforces: compact CNN + domain priors over transformers. | **B** |
| [2606.02498](https://arxiv.org/abs/2606.02498) | *GloResNet: lightweight 3D CNN for preterm brain injury prediction* (2026-06-01). Code released. | A concrete small-data 3D recipe: **ResNet-10 initialized from MedicalNet, resample to 128³, subject-wise z-score, mixup + class weighting + test-time augmentation**, 5-fold CV. Not a strong result (75% acc) but a sane, reproducible baseline configuration. | **B** |
| [2511.06163](https://arxiv.org/abs/2511.06163) | *Cross-modal fine-tuning of 3D convolutional foundation models for ADHD classification with LoRA* (ISBI 2026) | 3D-LoRA factorizes 3D conv kernels into 2D low-rank updates; CT-pretrained 3D conv FM adapted to MRI with 1.64M trainable params (113× fewer). Parameter-efficient adaptation is a cheap way to use a 3D prior without over-fitting 1,362 scans. | **B** |
| [2604.11679](https://arxiv.org/abs/2604.11679) | *Towards brain MRI foundation models for the clinic: findings from the FOMO25 challenge* (2026-04-13) | 19 FMs from 16 teams, evaluated few-shot and out-of-domain on clinical data: **SSL pretraining improves generalization under domain shift.** Evidence that *some* pretraining is worth having even for small downstream sets, if a prize-safe source exists. | **B** |
| [2606.08164](https://arxiv.org/abs/2606.08164) | *How much MRI preprocessing is enough? A cost–utility study for brain MRI FMs* (2026-06-06) | Graded preprocessing P0–P7 for MAE/JEPA on 20k volumes: minimal-but-stable preprocessing (P2) gets most of the value; heavier preprocessing helps only in select regimes, and its benefit "can be recovered by applying stronger preprocessing downstream." Interpretation for us: do the heavy registration at the *supervised* stage; do not over-engineer it. | **C** |
| [2512.01310](https://arxiv.org/abs/2512.01310) | *Lost in Distortion: domain gap between CV and brain imaging — pretraining for age prediction* (2025-12-01) | Low-quality/distorted scans in pretraining measurably hurt downstream performance; argues for domain-aware curation. Relevant if we self-pretrain on our own 1,362 volumes: keep the ultra-low-count regime scans (they are valid, per EDA) but QC registration failures out of any SSL corpus. | **C** |
| [2605.06859](https://arxiv.org/abs/2605.06859) | *Knowledge Transfer Scaling Laws for 3D Medical Imaging* (2026-05-07) | Cross-modality transfer in 3D pretraining is strongly asymmetric; hub domains (CT/MRI) help others, isolated domains need direct investment. PET behaves closer to an "island" — implies CT/MRI-pretrained 3D weights will transfer only weakly to SPECT. Tempers expectations for MedicalNet/CT-FM initialization. | **B** |
| [2501.11755](https://arxiv.org/abs/2501.11755) | *A generalizable 3D framework and model for SSL in medical imaging (3DINO-ViT)* — Xu et al., npj Digital Medicine (2025-01-20) | 3D DINO ViT pretrained on ~100k multi-organ, multimodal scans; strong OOD transfer. Weights "will be made available" — license unverified. A candidate 3D prior if license permits. | **B** |
| [2502.02779](https://arxiv.org/abs/2502.02779) | *3D foundation model for generalizable disease detection in head CT (FM-CT)* — NYU, Nature BME 2026 | 361k head CTs, self-distillation + MIM, full 3D; big gains over from-scratch on scarce-label tasks. Head-anatomy prior, but CT not SPECT; weight availability/license unknown. | **C** |
| [2604.27277](https://arxiv.org/abs/2604.27277), [2510.23415](https://arxiv.org/abs/2510.23415), [2607.17782](https://arxiv.org/abs/2607.17782), [2509.10620](https://arxiv.org/abs/2509.10620), [2606.13315](https://arxiv.org/abs/2606.13315) | BrainDINO (6.6M axial slices, frozen encoder + light heads; code released), BrainFound (DINOv2 extended to 3D brain MRI), BrainNext (MAE + xLSTM-UNet, FOMO25 2nd), 3D-Neuro-SimCLR (44,958 scans, model released), MAE-vs-JEPA for MRI disease detection | The 2025–26 crop of brain-MRI foundation models. All MRI; none nuclear medicine. BrainDINO's finding — slice-wise SSL + frozen encoder + light heads generalizes "with particularly strong advantages under label scarcity" — again favors the frozen-2D-encoder pattern. Licenses unverified. | **C/B** |
| [2606.23172](https://arxiv.org/abs/2606.23172) | **A benchmark of (MRI-)foundation models to predict IDH status in glioma** (2026-06-22) | BrainIAC, MRI-CORE, BiomedCLIP, BrainDINO vs **radiomics + TabPFN / logistic regression**, with calibration assessed across 4 cohorts + external: "**TabPFN matched or outperformed all visual encoders**" (AUROC 0.92); MRI-specific encoders underperformed general ones. Strong support for a registered-ROI-feature tabular branch. | **A** |
| [2607.01001](https://arxiv.org/abs/2607.01001) | **Foundation models vs. radiomics for lung CT** — Neukirch, Maurer, Strodthoff (2026-07-01). Code released. | 5 extractors (Curia, Curia-2, DINOv3, Radiomics2D/3D) × 7 heads (TabPFN, TabICL, XGBoost, CatBoost, RF, LR, Ridge) × 3 segmentation regimes, **worst-case cross-cohort performance as primary metric**. Dominant factor is task-dependent; radiomics competitive; classifier head choice drives several tasks. Methodologically, "worst-case cross-cohort" is exactly the private-LB-proof metric we should optimize. | **B** |
| [2607.11007](https://arxiv.org/abs/2607.11007) | **TabPFN beyond tabular data: calibration and accuracy on multimodal embeddings** (2026-07-13). Code released. | 22,820 episodes, 14 datasets, 11 encoders: TabPFN as a zero-gradient head on frozen embeddings has the **best mean rank on NLL and ECE** among 9 heads (NLL −48–62%); accuracy advantage conditional on **k ≥ 50 shots and d ≤ 32 features**. For us: ≈1,100 train samples per fold is plenty, but we must reduce embedding dimensionality (PCA to ≤32) before TabPFN. | **A** |

### 2.3 Calibration and log loss

| arXiv | Title (date) | Summary | Verdict |
|---|---|---|---|
| [2601.19944](https://arxiv.org/abs/2601.19944) | **Classifier calibration at scale: model-agnostic post-hoc methods** — Manokhin, Grønhaug (2026-01-19) | 21 classifiers × 5 calibrators (isotonic, Platt, beta, **Venn-Abers**, Pearsonify) on TabArena binary tasks, scored by log-loss/Brier/ECE: "**Venn-Abers predictors achieve the largest average reductions in log-loss.**" Venn-Abers is distribution-free, works on small calibration sets, and is a drop-in replacement for Platt scaling on OOF predictions. | **A** |
| [2605.03816](https://arxiv.org/abs/2605.03816) | *The Manokhin Probability Matrix* (2026-05-05) | Same author; classifies models by calibration × discrimination. **CatBoost, TabICL, TabPFN, LR, RF are already well calibrated ("Eagles"); XGBoost/LightGBM/HGB are over-confident ("Bulls"). Venn-Abers cuts log loss 6.5–12.6% on Bulls but degrades Eagles by 2.1%.** Practical rule: calibrate the *neural nets*; do not re-calibrate TabPFN/LR outputs; calibrate the *ensemble* once. | **A** |
| [2605.18696](https://arxiv.org/abs/2605.18696) | *Ensembling tabular foundation models — a diversity ceiling and a calibration trap* (2026-05-18) | "**Stacking with a logistic-regression meta-learner … the worst log-loss rank among the ensembles. The meta-learner improves accuracy by sharpening class boundaries, which destroys calibration.** We recommend greedy selection." Implication: average probabilities (or use greedy forward selection on OOF log loss); never train a meta-classifier for a log-loss competition on 1,362 samples. | **A** |
| [2510.20011](https://arxiv.org/abs/2510.20011) | *Improving predictive confidence in medical imaging via Online Label Smoothing* (2025-10-22) | OLS (soft labels updated from the model's own predictions) beat hard labels, LS, and teacher-free KD on RadImageNet in accuracy and calibration. Cheap training-time regularizer; for binary tasks plain LS≈0.05 is the simpler stand-in. | **B** |
| [2509.24951](https://arxiv.org/abs/2509.24951) | *Evaluating temperature scaling for CNNs under varying noise levels in brain tumour detection* (2025-09-29) | TS "significantly reduces ECE and NLL under all noise conditions without degrading classification accuracy." Confirms TS remains the safe default for CNN outputs under input perturbation (our cross-site noise). | **B** |
| [2604.11798](https://arxiv.org/abs/2604.11798) | *Budget-aware uncertainty for radiotherapy segmentation QA using nnU-Net* (2026-04-13) | Compares TS, deep ensembles, **checkpoint ensembles**, and TTA individually and combined for calibration. Checkpoint ensembles + TTA are free at training time and reliably improve reliability — use them. | **B** |
| [2505.23463](https://arxiv.org/abs/2505.23463) | *Revisiting reweighted risk for calibration: AURC, focal, and inverse focal loss* — Zhou, Gruber, Popordanoska, Blaschko (2025-05-29) | Theory linking calibration error to selective classification; proposes a bin-based CDF loss that is competitive. Focal-style losses are *not* automatically better for NLL; treat them as an ablation, keep BCE(+LS) as default. | **C** |
| [2510.17915](https://arxiv.org/abs/2510.17915) | *Uncertainty-aware post-hoc calibration: mitigating confidently incorrect predictions* (2025-10-19) | Dual isotonic calibration stratified by proximity-based conformal reliability. Interesting for pushing "putatively wrong" predictions toward 0.5 — exactly what log loss rewards — but validated on CIFAR only. | **C** |
| [2602.02370](https://arxiv.org/abs/2602.02370) | *Uncertainty-aware classification in biomedical imaging using SNGP* (ISBI 2026) | Spectral-normalized GP last layer improves OOD calibration with a single model. Cheap architectural change worth an ablation. | **C** |
| [2506.18162](https://arxiv.org/abs/2506.18162), [2512.14727](https://arxiv.org/abs/2512.14727) | *Pitfalls of conformal predictions for medical image classification*; *A critical perspective on finite-sample conformal prediction in medical applications* (Schölkopf/Baumgartner group) | Conformal guarantees break under distribution shift and are of "limited practical value" with few classes; practical utility depends strongly on calibration-set size. Do not spend time on conformal sets for this competition. | **C** |
| [2606.20115](https://arxiv.org/abs/2606.20115) | *When average calibration fails: site-conditional federated conformal risk control* (2026-06-18) | On 20-institution brain tumour data, a single pooled calibration "protects the average hospital but violates coverage at 40% of individual institutions." Our single temperature will be miscalibrated per site; choosing it on out-of-site folds is the mitigation available without test-site labels. | **B** |
| [2603.15232](https://arxiv.org/abs/2603.15232) | *Decomposing probabilistic scores: reliability, information loss and uncertainty* — Charpentier, Fernandes Machado (2026-03-16) | Proper-loss decomposition (log-loss = miscalibration + grouping loss + irreducible), with analysis of post-hoc recalibration and aggregation of calibrated models. Theoretical justification for "calibrate after averaging." | **C** |

### 2.4 Multi-center generalization, harmonization, shortcut detection

| arXiv | Title (date) | Summary | Verdict |
|---|---|---|---|
| [2508.20300](https://arxiv.org/abs/2508.20300) | *Mitigating MRI domain shift in sex classification: 3D CNNs with ComBat harmonization* (2025-08-27) | 3D ResNet18/DenseNet/EfficientNet trained on IXI, tested on OASIS3: **within-domain 0.95 → cross-domain 0.50 (chance)**; ComBat on features recovers only to 0.61. Two lessons: (a) cross-site collapse in 3D brain CNNs is real and severe; (b) ComBat is a partial fix and needs site labels. Our EDA's chance-level unregistered feature is consistent with this. | **A** (warning) |
| [2604.14441](https://arxiv.org/abs/2604.14441) | **Batch effects in brain foundation model embeddings** — Tao, Baker, Wu, Sarwate, Panta, Plis (2026-04-15) | FM embeddings "encode substantial batch-related variability, **often dominating diagnosis-related information** across heterogeneous datasets." Any frozen-embedding branch must be checked for site leakage (train a site classifier on embeddings; if it is near-perfect, expect poor private-LB transfer). | **A** |
| [2512.13144](https://arxiv.org/abs/2512.13144) | *Weight Space Correlation Analysis: quantifying feature utilization* (2025-12-15) | Detects shortcut learning by correlating the clinical head's weights with auxiliary metadata (scanner) heads. Cheap diagnostic: train an auxiliary "spacing-cluster" head on our CNN's penultimate features and check correlation with the pathology head. | **B** |
| [2603.18050](https://arxiv.org/abs/2603.18050) | *Quality assessment of brain MRI: DL vs hand-crafted features generalization to new sites* (2026-03-17) | 1,098 volumes, 17 sites, **leave-one-site-out**: "both DL and traditional ML methods struggle to generalize to new scanners or sites." Expect a CV-to-LB gap; size it with LOSO. | **B** |
| [2506.23208](https://arxiv.org/abs/2506.23208) | *Multi-source COVID-19 detection via Variance Risk Extrapolation* (2025-06-29) | 4-hospital CT challenge solution: **VREx** (minimize variance of per-site risk) + **Mixup**. Simple to implement when site pseudo-labels exist (we have spacing/shape/intensity-regime clusters). | **B** |
| [2503.06759](https://arxiv.org/abs/2503.06759) | *Revisiting invariant learning for OOD generalization on multi-site mammogram datasets* (2025-03-09) | Counter-evidence: "**standard ERM consistently outperformed** specialized invariant mechanisms (IRM, VREx) on out-of-domain testing." So VREx is an ablation, not a default; a well-tuned ERM with strong augmentation is the baseline to beat. | **B** |
| [2603.00060](https://arxiv.org/abs/2603.00060) | *Learning under extreme data scarcity: subject-level evaluation of lightweight CNNs for fMRI-based prodromal Parkinson's detection* (2026-02-10) | 40-subject PPMI fMRI: image-level splits leak subject slices and give near-perfect accuracy; strict subject-level splits drop to 60–81%. Reminder for any 2.5D branch: **split by scan, never by slice**. | **B** |
| [2605.13813](https://arxiv.org/abs/2605.13813), [2606.05460](https://arxiv.org/abs/2606.05460), [2601.13385](https://arxiv.org/abs/2601.13385) | JANUS (anatomy-conditioned gating; weights on HF), ORACLE-CT (anatomy-aware support pooling), Organ-aware attention — Dahal, Bhandari, Rubin, Lo (2026) | CT triage: restricting attention pooling to anatomically relevant regions and fusing "macro-radiomic" scalar priors (volume, mean intensity) with visual embeddings improves external AUROC **and calibration**, and suppresses high-confidence false positives under domain shift. Direct analogue: pool CNN features inside striatal ROIs and fuse with SBR-like scalars. | **B** |
| [2602.21372](https://arxiv.org/abs/2602.21372) | *The Mean is the Mirage: entropy-adaptive model merging under heterogeneous domain shifts* (2026-02-24) | Online, forward-pass-only merging of site-specific models per test batch. Interesting but requires batch-level test adaptation, which conflicts with the rule that test samples be processed independently. **Not usable.** | **C** |
| [2509.06592](https://arxiv.org/abs/2509.06592), [2601.08193](https://arxiv.org/abs/2601.08193), [2512.07674](https://arxiv.org/abs/2512.07674), [2603.17219](https://arxiv.org/abs/2603.17219) | Diffusion/CycleGAN MRI harmonization methods (anatomy–contrast disentanglement, MMH, DIST-CLIP, SA-CycleGAN-2.5D) | All MRI, all need large multi-site corpora; none is feasible or prize-safe for us in 11 days. Listed for completeness. | **C** |
| [2607.03715](https://arxiv.org/abs/2607.03715), [2507.02437](https://arxiv.org/abs/2507.02437) | Test-time adaptation for medical classification (CoWA, F²TTA) | TTA updates the model on unlabeled *batches* of test data — conflicts with the independent-processing rule and the no-pseudo-labeling rule. **Not usable.** | **C** |

### 2.5 Registration

| arXiv | Title (date) | Summary | Verdict |
|---|---|---|---|
| [2501.13010](https://arxiv.org/abs/2501.13010) | *Learning accurate rigid registration for longitudinal brain MRI from synthetic data* — Fu, Dalca, Fischl, Moreno, Hoffmann (ISBI 2025) | Extends the "anatomy-aware, acquisition-agnostic affine registration" (SynthMorph) family with synthetic-pair training. Relevant as the state of the art for learned rigid/affine alignment that is robust to contrast; but the released weights live in the FreeSurfer ecosystem (license to verify) and are MRI-trained. Classical intensity-based affine (ANTs/SimpleITK, Mattes MI) remains the pragmatic choice for SPECT. | **B** |
| [2603.14086](https://arxiv.org/abs/2603.14086), [2508.14809](https://arxiv.org/abs/2508.14809) | DINO-pretrained features for 3D registration (ISBI 2026); DINOv3 with test-time training for registration | Feature-based registration is more robust to inter-scanner variability than intensity-based. Over-engineering for our need (rigid/affine alignment of a blob-like striatal image to a template). | **C** |
| [2608.10769](https://arxiv.org/abs/2608.10769) | *A framework for joint affine and diffeomorphic image registration* (2026-08-11) | Joint affine + LDDMM. We do not want deformable registration (it would normalize away the striatal shrinkage that *is* the signal). Affine only. | **C** |

### 2.6 Other context

| arXiv | Title (date) | Summary | Verdict |
|---|---|---|---|
| [2604.27195](https://arxiv.org/abs/2604.27195), [2508.17649](https://arxiv.org/abs/2508.17649) | TabPFN for MCI→AD conversion in data-limited settings (AUC 0.892 > LightGBM 0.860; robust at N=50); L2C-TabPFN on TADPOLE | More evidence that TabPFN is the strongest small-N tabular head on neuro-imaging-derived features. | **B** |
| [2609.01427](https://arxiv.org/abs/2609.01427) | Pix2Rep-v2 dense SSL (MICCAI 2026), code + weights released | Dense voxel-level SSL for few-shot; segmentation-oriented. If we needed a striatal segmenter on our own data, this is the current data-efficient option. | **C** |
| [2506.13306](https://arxiv.org/abs/2506.13306) | *Brain imaging foundation models, are we there yet?* — systematic review of 161 datasets, 86 FMs (2025-06-16) | Confirms no public SPECT/DaT foundation model exists as of mid-2025; PET is thinly covered. | **C** |
| [2608.23336](https://arxiv.org/abs/2608.23336) | *Can coding agents build robust baselines?* (MICCAI-W 2026) | An agentic literature→code→experiment loop produced competitive challenge baselines. Meta-relevant to how this team is working; no technical content for the pipeline. | **C** |

---

## 3. What's new since our lit review

**(a) The "how" of small-data 3D classification has shifted toward 2D-FM-on-ROI-crops and compact CNNs, away from big 3D nets and ViTs.** The PDCADxFoundation winner (MRN) is the single most transferable design: crop the nuclei, encode slices with a frozen 2D VFM, fuse with an attention/segmentation-guided pooling, train with supervised contrastive + CE on 300 scans. AnyMC3D, Raptor and DinoAtten3D independently show frozen 2D features + light heads matching or beating 3D-pretrained models. The lung-CT frontier paper and the FOMO25/SSL3D winners both say transformers are the wrong bet at our N. **Caveat unique to us:** DINOv3's PET degradation means we cannot assume the 2D-FM branch works on DaT-SPECT; it must earn its place in CV.

**(b) Registered ROI features + tabular foundation model is now a first-class contender, not a fallback.** 2026 benchmarks (IDH-glioma, lung CT) found radiomics + TabPFN matching or beating visual FMs, and TabPFN gives the best NLL/ECE as a head on embeddings. Combined with the 2022 manufacturer-stratified radiomics result already in our lit review, the case for a lean feature branch is stronger than it was.

**(c) Calibration specifics for a log-loss metric.** New, concrete guidance: (i) Venn-Abers is the best post-hoc calibrator on average for log loss, but only apply it to over-confident models (neural nets, XGBoost); LR/CatBoost/TabPFN are already calibrated and get *worse*; (ii) never stack with a learned meta-classifier — average probabilities, then calibrate once; (iii) checkpoint ensembles + TTA + temperature scaling compose well; (iv) conformal prediction is not worth the time here.

**(d) Site effects are worse than we assumed, and they contaminate FM embeddings.** 3D brain CNNs can collapse to chance across sites; FM embeddings are often dominated by batch; LOSO evaluations show both DL and hand-crafted features degrade on new sites. This upgrades "pseudo-site-grouped CV" from nice-to-have to mandatory, and adds a required leakage diagnostic (site classifier on embeddings; weight-space correlation).

**(e) Invariant-learning methods (IRM/VREx) are contested.** One multi-source CT solution used VREx + Mixup successfully; a larger multi-site mammography study found tuned ERM beats IRM/VREx OOD. Treat VREx as a one-day ablation.

**(f) Test-time adaptation and batch-level model merging are off the table** by competition rules (independent processing of test samples, no pseudo-labeling), regardless of their merits.

**(g) Registration: nothing changes.** No new registration-free classification method appeared for brain nuclear imaging; learned affine registration (SynthMorph family) advanced but is MRI-trained with a license to verify. Classical affine registration to a study-specific template remains the right call; deformable registration should be avoided because it would normalize away the pathology.

**(h) Nuclear-medicine-specific: acquisition with vs. without attenuation correction is a real hidden domain axis** (DaT-CTLESS), which our augmentation should simulate (global contrast/gamma and background-offset jitter).

---

## 4. Recommended end-to-end pipeline for this competition

Design goals: minimize private-LB log loss under unseen-site shift; fit in ~10–15 Kaggle GPU-hours total; run in well under 3 h on one A100 with no internet; use only prize-safe external assets.

### 4.1 Preprocessing (identical code path at train and test time)

1. **Load & canonicalize.** `nibabel` load → `as_closest_canonical` (RAS) → float32. All 1,362 have `qform/sform_code=1`, `scl_slope` unset (EDA §4), so use the affine as-is and ignore header scaling.
2. **Resample to 2.0 mm isotropic** (linear interpolation; MONAI `Spacing` or SimpleITK). Then **Gaussian-smooth to a common effective resolution**: apply `σ = sqrt(max(0, FWHM_target² − FWHM_native²))/2.355` with FWHM_target ≈ 12 mm so 2.3–3.9 mm-voxel scans converge to similar blur (the 3.9 mm cluster is intrinsically smoother). This is the cheapest, most defensible cross-scanner harmonization we can do without site labels at inference.
3. **Head/brain mask** via Otsu on the smoothed volume + largest connected component + hole fill. Compute center of mass and principal axes.
4. **Per-volume intensity normalization** (mandatory per EDA §5, three raw-count regimes): inside the mask, clip at the 99.8th percentile, divide by it → [0,1]. Keep a **second channel**: volume divided by the mean of a **reference (non-specific) region** — the mask minus a generous striatal box — which is an SBR-like normalization and is what clinicians look at. Two-channel input = `[p99.8-scaled, reference-scaled]`.
5. **Affine registration to a study-specific template.** Build the template from the training set only: start from the mean of ~30 registered *normal* scans in the largest cluster (256³ @ 2.46 mm), iterate 3× (register all normals → re-average). Register every scan (train and test) with **ANTsPy** `registration(type_of_transform='Affine', aff_metric='mattes')` at 2 mm, initialized by the CoM/principal-axes transform from step 3. Store the final Mattes MI value as a **QC scalar**; if it is below a threshold learned on training data, fall back to the CoM-aligned volume (never crash). **No deformable step.**
   - Why affine-to-own-template rather than MNI: there is no prize-safe public DaT-SPECT template (the common SPM DaT templates are PPMI-derived or academic-only), and MNI-space T1 templates have poor contrast correspondence to SPECT. Building our own from competition data keeps us clean.
   - Cost: ANTs affine at 2 mm ≈ 5–20 s/scan on one core. With `multiprocessing.Pool(20)` on the 24-vCPU box, 1,000 test scans ≈ 8–20 min. Bundle a **pure-PyTorch GPU rigid/affine fallback** (MI or NCC loss, Adam over 6–12 parameters, 200 iterations, batched) in case `antspyx` is not in the runtime image — **check `competition-sfmn-parkinsons-runtime` `pyproject.toml` on day 1 and file a package request immediately if needed** (SimpleITK is the more likely-to-be-present alternative).
6. **Crop a striatum-centered box** in template space: ≈ 96 × 112 × 80 mm → **48 × 56 × 40 voxels at 2 mm** (basal ganglia + generous margin, includes occipital background rows for the reference channel). Also keep a **whole-head 64³ low-res** view (4 mm) for an optional global branch.
7. **Metadata scalars for CV grouping only (never model inputs):** original spacing (rounded), original shape, intensity regime (ultra-low/low/high), registration MI. Define **pseudo-site = (spacing cluster, shape cluster, intensity regime)**.

### 4.2 Models (two required branches + one optional, ensembled by probability averaging)

**Branch 1 — compact 3D CNN on the striatal crop (primary).**
- MONAI `resnet18` (3D) or `resnet10`, 2 input channels, from scratch (default). Ablation: initialize from MedicalNet ResNet-18 if its weight license is confirmed (§5); expect only small gains because CT/MRI → SPECT transfer is weak (2605.06859).
- Augmentation (all applied in template space): random affine (rot ±10°, scale ±7%, shift ±3 voxels), **left–right flip** (label-preserving: laterality of loss is not the label), intensity gamma (0.8–1.25), global scale jitter (±15%), additive Gaussian + Poisson-like noise, **random extra Gaussian blur (σ 0–1 voxel) and background-offset jitter to simulate AC/non-AC and resolution differences**, Mixup (α = 0.3) on 50% of batches.
- Loss: BCE with label smoothing 0.05. AdamW, lr 3e-4 cosine, weight decay 1e-4, 60 epochs, batch 16, AMP. **SWA/checkpoint-average over the last 15 epochs.**
- Optional one-day ablation: VREx penalty across pseudo-site groups (2506.23208) — keep only if it lowers out-of-site OOF log loss (2503.06759 says it often will not).
- Compute: ~1,100 crops/fold at 48×56×40 → ≈ 20–40 s/epoch on a T4; **5 folds × 2 seeds × 60 epochs ≈ 4–6 GPU-hours.**

**Branch 2 — registered ROI features + calibrated tabular head (primary, and the interpretability story for the sponsor).**
- Define ROIs once on the template: left/right caudate, left/right putamen (anterior/posterior halves), occipital reference. Draw them by thresholding the *normal-scan* mean template at multiple levels + manual split (a few hours, no external atlas needed).
- Features per scan (~40): mean/max/p90 uptake per ROI in reference units (SBR = ROI/ref − 1), L–R asymmetry indices, putamen/caudate ratios, anterior/posterior putamen gradient, striatal volume above 50%/70% of peak, plus a small pyradiomics set (first-order + GLCM) on the striatal crop, plus the registration-QC scalar (kept *only* if it does not correlate with label in OOF checks).
- Heads: **logistic regression (L2, standardized)** and **TabPFN v2** (if license confirmed, §5); else CatBoost with `depth≤4`. Per 2605.03816 these are already well-calibrated — do not post-calibrate them individually.

**Branch 3 — frozen 2D-FM slice embeddings (optional; must earn its place).**
- DINOv2 ViT-S/14 (Apache-2.0), frozen. For each scan, take 9 axial + 5 coronal slices through the striatal crop (each channel rendered to RGB by replication), extract CLS + mean-patch tokens, concatenate, **PCA to 32 dims** (fit on train fold only), feed TabPFN/LR — this is the Raptor/AnyMC3D/2607.11007 recipe. Alternative: DinoAtten3D-style attention pooling with SupCon, trained ~10 min/fold.
- Gate: keep only if it lowers pseudo-site-grouped OOF log loss **and** a site classifier on its embeddings is not near-perfect (2604.14441). Expect it may fail on SPECT (2509.06467); that is fine.

### 4.3 Validation strategy (mirrors the multi-center test)

- **Primary CV:** `StratifiedGroupKFold(5)` grouped by pseudo-site, stratified by label — so each fold holds out whole acquisition configurations. Repeat with 2 seeds.
- **Stress test:** leave-one-major-cluster-out on the four dominant spacing clusters (2.46 / 3.90 / 2.30 / 2.40 mm; 89% of data). Report log loss and AUROC per held-out cluster. This is our estimate of the private-LB drop.
- **Also report** plain stratified 5-fold, to quantify the in-distribution vs out-of-site gap (expect the latter to be worse; 2603.18050, 2508.20300).
- **Leakage diagnostics (run once):** (i) train a pseudo-site classifier on each branch's features/embeddings; (ii) weight-space correlation between pathology head and an auxiliary site head (2512.13144); (iii) confirm no label–site shortcut beyond the EDA's 46–60% range.
- **Slice-level branch:** splits by scan, never by slice (2603.00060).

### 4.4 Ensembling and calibration for log loss

1. Collect OOF probabilities from every model (10 CNNs, 2–3 tabular heads, optional FM branch).
2. **Temperature-scale each CNN family** on its OOF logits (per-fold temperatures averaged); optionally compare Venn-Abers on the CNN OOF scores (2601.19944). Leave LR/TabPFN uncalibrated (2605.03816).
3. **Combine by probability averaging** with weights found by **greedy forward selection on grouped-OOF log loss** (2605.18696 — no meta-classifier). Typical expected weights: CNN 0.5–0.6, ROI-tabular 0.3–0.4, FM 0–0.2.
4. **One final calibrator on the averaged OOF probability**, chosen by nested grouped CV among {none, temperature/Platt, beta, Venn-Abers}; pick the lowest **out-of-site** OOF log loss, *not* in-distribution — shifted test sites make models over-confident, so the out-of-site choice will be slightly flatter and safer (2606.20115).
5. **Probability clipping** to `[ε, 1−ε]` with ε tuned on OOF (expect 0.01–0.03) as insurance against catastrophic single-scan penalties.
6. **Inference-time TTA:** average over {identity, L–R flip, ±1-voxel shifts} for the CNN branch (2604.11798).

### 4.5 Inference plan (≤3 h, 1× A100, no internet)

- `main.py` reads `/code_execution/data/niftis/*.nii.gz` and `submission_format.csv`; bundles: template, ROI masks, PCA/scaler/LR/TabPFN pickles, 10 CNN checkpoints (ResNet-18-3D ≈ 33M params ≈ 66 MB each in fp16 → < 1 GB total), DINOv2-S weights (~85 MB) if used.
- Stage A (CPU, 20 workers): load → resample → smooth → mask → normalize → ANTs affine → crop → save tensors. Budget ≈ 15–25 min for 1,000–2,000 scans; log per-scan wall time only in aggregate (300-line log cap; never print test-scan info).
- Stage B (GPU): batched CNN forward with TTA (seconds), FM embeddings if used (minutes).
- Stage C: features → tabular heads → ensemble → calibrator → clip → write `submission.csv`.
- **Hard time guard:** a wall-clock monitor; if elapsed > 2 h 15 min before Stage B, switch registration to the CoM-only fallback for remaining scans. Every stage wrapped so a single corrupt file yields the prior (0.548) rather than a crash.
- **Smoke test first** (6-min limit) with the Docker runtime locally; verify `antspyx`/SimpleITK availability there, not on the real submission.

### 4.6 Time plan (deadline Sept 16)
- Day 1–2: preprocessing + template + registration QC; runtime dependency check/request.
- Day 3–4: Branch 2 (fast, interpretable) + grouped CV harness → first smoke test + first real submission.
- Day 5–7: Branch 1 CNN (5×2), TS/Venn-Abers, greedy ensemble, clipping.
- Day 8–9: Branch 3 ablation, VREx ablation, MedicalNet-init ablation (only if licenses are confirmed).
- Day 10–11: freeze, full Docker dry run, final submissions; write MIT-licensed release notes and external-asset disclosure.

---

## 5. Prize-eligibility check (external resources)

Rule recap (`DAT_PARKINSONS_CHALLENGE.md`): external data/weights must be freely and publicly available to all participants **and** licensed to allow commercial use of the resulting model (no NC); everything external must be disclosed; PPMI is explicitly not eligible; the winning solution must be released under MIT. **License statements below are from my knowledge of the projects, not verified in this session — verify each against the repo/HF card before use and record it in the disclosure.**

| Resource | Role in pipeline | License (as best known) | Prize-safe? |
|---|---|---|---|
| Competition data (1,362 scans) | Everything | Competition rules | Yes (primary) |
| Study-specific template + ROI masks derived from competition data | Registration target, features | Derived from competition data | **Likely yes, but confirm with organizers** that a template derived from training data may be bundled and released under MIT with the winning code (data is slated for data.gouv.fr release, so this should be fine) |
| MONAI, PyTorch, nibabel, SimpleITK, scikit-learn, pyradiomics | Framework code | Apache-2.0 / BSD / Apache-2.0 / MIT / BSD-3 / BSD-3 | Yes |
| ANTs / ANTsPy (`antspyx`) | Affine registration | Apache-2.0 | Yes (if present in or addable to runtime) |
| DINOv2 ViT-S/B (Meta) | Optional frozen 2D encoder | Apache-2.0 | Yes |
| DINOv3 (Meta) | Alternative 2D encoder | Custom "DINOv3 License" (commercial use permitted with conditions/AUP) | **Uncertain** — prefer DINOv2; if used, read the license and disclose |
| MedicalNet / Med3D 3D ResNet weights (Tencent) | Optional CNN init | Repo MIT; weight files distributed separately — confirm they are covered | **Uncertain-leaning-yes**; verify |
| TabPFN v2 (Prior Labs) | Tabular head | "Prior Labs License" = Apache-2.0 + attribution clause for v2; **TabPFN-2.5 has different, more restrictive terms** | **Uncertain**: v2 probably OK with attribution, 2.5 probably not; also check compatibility with an MIT release of the winning solution. Fallback: LR / CatBoost (Apache-2.0) |
| 3DINO-ViT (2501.11755), 3D-Neuro-SimCLR (2509.10620), BrainDINO (2604.27277), BrainFM4Challenges (2601.13166), DinoAtten3D code, Raptor code, JANUS weights (HF) | Possible 3D priors / code | Unverified; several academic repos ship without a license file | **Uncertain** — do not rely on any of these for a prize submission unless a permissive license is confirmed |
| SuPreM, Merlin (Stanford), VoCo, Curia/Curia-2 (Raidium), FM-CT (NYU), NeuroVFM | 3D CT/MRI foundation models | Merlin is research/non-commercial; Curia and SuPreM are believed non-commercial; FM-CT/NeuroVFM availability unknown | **Assume NOT prize-safe** unless verified otherwise; also weakly relevant to SPECT |
| SynthMorph / FreeSurfer registration weights | Learned affine registration | FreeSurfer license (non-commercial restrictions) | **Not safe** — use ANTs/SimpleITK instead |
| MNI ICBM152 templates | Not needed with own template | Permissive (McConnell BIC copyright notice allows use/distribution) | Yes if ever used; but unnecessary |
| PPMI, NTUA dataset, any PPMI-derived DaT template/atlas | — | DUA / academic-only | **No** (already ruled out) |
| Kaggle for training | Compute | Kaggle ToS (retention license) | **Unresolved** compliance question (see `DAT_PARKINSONS_CHALLENGE.md` §"Open compliance question") — the pipeline above is small enough to train on a self-controlled machine or a wiped rented instance if that is chosen |

---

## 6. Confidence & caveats

- **Coverage.** High confidence that arXiv contains no 2025+ DaT-SPECT *classification* paper that we missed: seven different DaT/FP-CIT/SPECT+Parkinson queries (with and without server-side date filters) all returned the same three DaT items. The DaT-specific literature is journal-based; a PubMed/Semantic Scholar sweep (other subagents) is the right complement.
- **Abstract-level evidence only.** I read abstracts, not full texts. Numbers quoted are the authors' own; several results (e.g., 2606.06950's lung-CT frontier, 2508.20300's ComBat recovery) have wide CIs or single-dataset scope. Rankings among calibrators (2601.19944, 2605.03816) are on tabular benchmarks, not medical imaging.
- **Transfer risk.** The strongest "new" ideas (2D-FM-on-ROI, TabPFN heads) were demonstrated on MRI/CT; DINOv3's reported PET degradation is the only direct evidence about nuclear-medicine images, and it is negative. The recommendation therefore keeps the 3D-CNN and ROI-feature branches as primary and gates the FM branch on CV.
- **Licensing.** All license assertions in §5 are unverified recollections and must be checked before any weight is bundled; when in doubt, the pipeline is designed to work with zero external weights (from-scratch CNN + LR features), which is the only configuration that is unambiguously prize-safe.
- **Runtime dependencies.** ANTsPy availability in the DrivenData image is unknown; the pure-PyTorch affine fallback is a must-have, not a nice-to-have.
- **API caveats.** arXiv's compound-boolean + `submittedDate` filtering under-counted in sweep 1; sweeps 2–4 corrected this by client-side filtering. Some queries only paginated to 400 results (`q_suprem` truncated at 109 kept), so very generic families (CT foundation models) are not exhaustive — this does not affect the DaT-specific conclusions.
- **Expected performance.** Given the lit review's CV-to-blind-test drops and the LOSO evidence above, I would expect pseudo-site OOF log loss to be noticeably worse than random-split OOF, and the private LB to track the former. Optimize and select on the grouped/out-of-site number.

Scratch artifacts (raw JSON of all 572 retrieved records, per-sweep logs) are in the session scratchpad: `C:\Users\HARSHDIP\AppData\Local\Temp\claude\H--augsepthacks-parkinsonnn\8c4f9605-e416-47a5-8f35-a15f5cb8225a\scratchpad\merged.json`.
