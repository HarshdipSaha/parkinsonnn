# Pipeline Recommendation A — What Past Winners of Comparable Competitions Actually Did

- **Research date:** 2026-09-05 (deadline Sept 16, 2026)
- **Mode:** `deep-research` quick brief (RQ → source retrieval → verification against source text → synthesis). Every factual claim below is tied to a fetched source; where a source could not be fetched or only partially verified, that is stated explicitly.
- **Ground truth read first:** `DAT_PARKINSONS_CHALLENGE.md`, `LITERATURE_REVIEW.md`, `EDA_REPORT.md` (this repo). Nothing in `data/` was touched.

---

## 0. TL;DR

1. **Every recent 3D-classification competition on Kaggle's RSNA series was won by 2D/2.5D pretrained backbones + a slice-sequence head, not by 3D CNNs** ([Revisiting 2D FMs for 3D classification, arXiv:2512.12887](https://arxiv.org/html/2512.12887): "all the winning solutions for 3D classification challenges adopted the 2D/2.5D methods rather than 3D"). The only DrivenData 3D-volume classification winner found (Clog Loss) used a **3D DenseNet121 + 5-fold CV + TTA** ([drivendataorg/clog-loss-alzheimers-research](https://github.com/drivendataorg/clog-loss-alzheimers-research)). Both families are viable here; the domain-specific DaT-SPECT literature from the Hamburg (Buchert) group wins with **tiny 3D-slab CNNs (~0.2–2.9M params) on affine-MNI-normalized, reference-region-scaled 2 mm volumes** — and their multi-site robustness papers are the closest published analogue to this competition's actual problem.
2. **The recurring recipe is preprocessing-heavy, architecture-light:** standardize geometry (resample/register), standardize intensity, crop to the organ of interest, use segmentation/localization to focus the classifier, train with patient-level GroupKFold, ensemble across folds × backbones, apply cheap TTA, and tune probability scaling on out-of-fold predictions.
3. **For log loss, do the boring things that work:** average probabilities of a diverse ensemble, fit a single temperature on pooled OOF logits (Guo et al. 2017: temperature scaling "is surprisingly effective" — [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)), clip extremes, and weight ensemble members by OOF log loss. Do not chase raw accuracy.
4. **Prize-safety is now clearer than the ground-truth docs assumed:** DrivenData staff ruled ImageNet-pretrained weights **are** prize-eligible "as long as the model or weights themselves are permissively licensed (such as MIT or Apache 2.0)" ([forum thread 11475](https://community.drivendata.org/t/clarification-on-imagenet-pre-trained-weights-under-open-source-commercial-rules/11475)); DINOv3 is **not** ([thread 11488](https://community.drivendata.org/t/is-dinov3-allowed-to-use/11488)). The most-cited public FP-CIT MNI template is **CC BY-NC-SA → not prize-safe**; build your own template from training normals with `antspyx` (in the runtime) instead.
5. **MONAI, antspyx, SimpleITK, torchio, timm, nibabel, scikit-learn, lightgbm, xgboost are all already in the official runtime** ([runtime/pyproject.toml](https://raw.githubusercontent.com/drivendataorg/competition-sfmn-parkinsons-runtime/main/runtime/pyproject.toml)), so the whole recommended pipeline is importable offline.

---

## 1. Sources consulted (URL → what it contributed)

### Competition / runtime / rulings
| Source | Contribution |
|---|---|
| [runtime/pyproject.toml](https://raw.githubusercontent.com/drivendataorg/competition-sfmn-parkinsons-runtime/main/runtime/pyproject.toml) | Full dependency list: `monai`, `antspyx` (x86_64), `simpleitk`, `torchio==1.2.1`, `nibabel`, `timm`, `scikit-learn`, `lightgbm`, `xgboost`, `scikit-image`, `scipy`, `pytorch-lightning`, `transformers`, `peft`, `torch==2.12.1+cu129`, `torchvision==0.27.1+cu129`, Python `>=3.12,<3.13`. |
| [Runtime repo README](https://github.com/drivendataorg/competition-sfmn-parkinsons-runtime) | Package-request process (edit `pyproject.toml`, `just update-lockfile`, PR); test data mounted read-only at `/code_execution/data`; CUDA 12 GPU image. |
| [Forum 11475 — ImageNet weights ruling](https://community.drivendata.org/t/clarification-on-imagenet-pre-trained-weights-under-open-source-commercial-rules/11475) | Staff (`mdaley`): "External models are eligible for prizes as long as the model or weights themselves are permissively licensed (such as MIT or Apache 2.0)." |
| [Forum 11488 — DINOv3](https://community.drivendata.org/t/is-dinov3-allowed-to-use/11488) | Staff (`ishashah`): DINOv3 Agreement is incompatible with the required MIT release → "I don't think DINOv3 models would be eligible for prizes here." |
| [Forum 11480 — pretrained models/external data](https://community.drivendata.org/t/how-to-use-pretrained-models-and-external-publicly-available-datasets/11480) | Weights must be inside `submission.zip`; external *training* data need not be bundled. |
| [Forum 11493 — test set size](https://community.drivendata.org/t/test-set-size-and-final-submission-selection/11493) | Test size and public/private split are not disclosed; best private-set submission counts automatically. |
| [Forum 11470 — zip size](https://community.drivendata.org/t/limit-to-size-of-zip-files-for-uploading/11470) | A >1 GB zip uploaded successfully after staff extended the upload timeout; no explicit size cap stated. |
| [Forum 11489 — new packages](https://community.drivendata.org/t/how-to-install-new-python-package-on-this-competition/11489) | Confirms the PR-based request process. |
| [Forum category index](https://community.drivendata.org/c/dat-parkinsons-challenge/110) | 14 threads total as of research date; none report MONAI/antspyx import problems. |

### Past winners — DrivenData
| Source | Contribution |
|---|---|
| [drivendataorg/competition-winners](https://github.com/drivendataorg/competition-winners) | Index of all 52 DrivenData winner repos (as submodules). Medical-imaging ones: Clog Loss (3D vessel-video volumes), TissueNet (2D pathology), VisioMel (2D WSI). No prior 3D radiology classification competition on DrivenData. |
| [Clog Loss winners README](https://github.com/drivendataorg/clog-loss-alzheimers-research) | 1st (ZFTurbo): 3D DenseNet121, FC-512 + dropout 0.5, **5-fold CV, TTA**; 2nd: ResNet101 + BCE, flips/distortions/noise, 5 epoch-snapshot averaging; 3rd: Kinetics-400-pretrained R(2+1)D/MC3 ensemble. |
| [ZFTurbo 1st-place repo](https://github.com/ZFTurbo/DrivenData-Alzheimer-Research-1st-place-solution) | ROI extraction before 3D model; three-stage training; custom 3D augmentation library (`volumentations`); K-fold split script. |
| [VisioMel winners README](https://github.com/drivendataorg/visiomel-melanoma) | 1st: multi-stage (CNN → auxiliary targets → MLP with clinical metadata); 2nd: multi-task ResNet + focal loss; 3rd: SSL embeddings + logistic regression with **40-fold CV**. Shows DrivenData winners favor stacked, heavily cross-validated, small final classifiers. |

### Past winners — Kaggle RSNA 3D series
| Source | Contribution |
|---|---|
| [Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification (arXiv:2512.12887)](https://arxiv.org/html/2512.12887) | Survey statement: RSNA 2022 winner "finetunes 2D pretrained backbones with 2.5D input and uses bidirectional-LSTM for slice fusion"; same pattern for 2023; "all the winning solutions for 3D classification challenges adopted the 2D/2.5D methods rather than 3D." Benchmark: 3D ResNet 0.809 / 3D DenseNet 0.833 / 3D ConvNeXt 0.683 avg AUC vs 2D-FM-based 0.894 across 10 tasks. |
| [RSNA 2023 Abdominal Trauma 1st place (Nischaydnk)](https://github.com/Nischaydnk/RSNA-2023-1st-place-solution) | Stage 1: 3D segmentation → organ crops. Stage 2: 96 equidistant slices → (32, 3, 384, 384) 2.5D; CoaT-Lite / EfficientNetV2-S backbones + GRU; **4-fold GroupKFold at patient level**; AdamW + cosine warmup; BCE + auxiliary Dice (+0.01–0.03); light aug (perspective, flips, ±25° rotation); slice-level ensemble → max over patient; **post-processing = scaling factors tuned on CV**. |
| [RSNA 2019 IHD 1st place (SeuTao) README](https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection/blob/master/README.md) | 2D CNNs (SE-ResNeXt101, DenseNet169, DenseNet121) at 256/512 px → OOF features → two sequence models; **TTA in final submission**; private LB 0.04383 (log loss). Windowing details not in README. |
| [RSNA 2019 IHD 2nd place (darraghdog)](https://github.com/darraghdog/rsna) | ResNeXt101 at 480 px with DICOM windowing; GAP embeddings + delta features → LSTM over series; 5-fold split; TTA at embedding extraction; bagging. |
| [RSNA 2024 Lumbar 2nd place (brendanartley)](https://github.com/brendanartley/RSNA-2024-Competition) | Two-stage: keypoint localizer → crops; 2.5D (24 middle frames) → encoder → LSTM → attention pooling; heavy aug incl. sequence flip and manifold mixup; **9-rotation TTA**; confident-learning label denoising. (1st-place Kaggle writeup is JS-rendered and could not be fetched; the survey paper above confirms the same 2.5D pattern.) |
| [RSNA 2022 winners paper, Radiology: AI 2024 (PMID 38169426)](https://europepmc.org/article/MED/38169426) | Top-8 mean AUC 0.96 on 1,093-scan multi-site test set (12 institutions). Confirms multi-site generalization of the 2.5D+LSTM recipe. |
| [RSNA 2022 3rd place (darraghdog/RSNA22)](https://github.com/darraghdog/RSNA22) | Listed for completeness; not fetched in detail. |
| [BraTS-2021 MGMT external validation (PMC9562637)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9562637/) | Cautionary tale: 1st-place Kaggle MGMT model had public AUROC 0.62 and collapsed to 56.2% AUROC externally; 80% of 420 models were at chance. Lesson: when the signal is weak, no architecture saves you — but DaT-SPECT has strong signal (AUC ~0.98–0.99 in `LITERATURE_REVIEW.md`), so this is the opposite regime. |

### DaT-SPECT-specific multi-site pipelines (closest analogue to this competition)
| Source | Contribution |
|---|---|
| [Wenzel et al. 2019, EJNMMI (PMID 31473800)](https://pubmed.ncbi.nlm.nih.gov/31473800/) | 645 PPMI scans; CNN with 2,872,642 params kept accuracy 0.955–0.972 across unsmoothed/smoothed/mixed settings while SBR dropped to 0.900 in the mixed setting; only the CNN transferred acceptably to a 298-scan clinical sample. "A deep CNN can be trained to be robust with respect to variable site-, camera-, or scan-specific image characteristics." |
| [Budd et al. 2024, uncertainty detection (PMC10957699)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10957699/) | 3,025 scans (dev 1,740 / internal test 640 / PPMI 645). Preprocessing: **affine (no warping) normalization to MNI with SPM12 and custom DaT-SPECT templates, 2×2×2 mm; voxelwise scaling to the individual 75th percentile of a reference region (excluding striata etc.) → DVR image; 72×72 transversal slab input.** Model: **five identical tiny ResNets (194,390 params each)**, majority vote; extra high-sens/high-spec ensembles flag "uncertain" cases. Accuracy 98.0% (dev test), 94.1% (internal), 98.0% (PPMI). |
| [Unrealistic augmentation, JNM 2024 (PMID 39054285)](https://jnm.snmjournals.org/content/65/9/1463) | Trained on 1,100 homogeneous scans; **strong Gaussian blur + additive noise augmentation** lifted unseen-site/camera accuracy to 0.989 and 0.975, beating no-aug and standard nnU-Net aug. Directly addresses this competition's 10-hospital shift. |
| [Multi-template normalization, EJNMMI Physics 2023 (PMC10060476)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10060476/) | 1,702 scans, 4 cameras; same SPM12 affine-to-MNI 2 mm + 75th-percentile reference-region DVR pipeline; using normal + PD-like templates jointly improved effect size 3.83 → 3.96. Justifies a template that includes abnormal appearance. |
| [ThomasBudd/dat_spect_ud](https://github.com/ThomasBudd/dat_spect_ud) | MIT-licensed inference code for the above; expects "registration to the MNI space ... 2mm isotropic voxels ... 2d slabs". Confirms the exact preprocessing contract used by the strongest published multi-site DaT-SPECT CNNs. |
| [NITRC FP-CIT MNI template (Salas-Gonzalez)](https://www.nitrc.org/projects/fp_cit_atlas) | Public FP-CIT template in MNI space, built by posterization + affine registration of 40 scans — **license: CC Attribution Non-Commercial Share Alike → NOT prize-safe.** |
| [NITRC SPM templates (García-Gómez)](https://www.nitrc.org/projects/spmtemplates) | 123I-FP-CIT SPM normalization template from 30 controls — license stated only as "Freeware ... may have redistribution restrictions" → **ambiguous, avoid for prize path.** |
| [MNI ICBM152 2009 (McGill)](https://nist.mni.mcgill.ca/icbm-152-nonlinear-atlases-2009/) | "Permission to use, copy, modify, and distribute this software ... for any purpose and without fee is hereby granted" → permissive; usable if you want an external anatomical reference space (cross-modal). |

### Preprocessing frameworks / calibration / licenses
| Source | Contribution |
|---|---|
| [nnU-Net, Nature Methods 2021](https://www.nature.com/articles/s41592-020-01008-z) (+ [GitHub](https://github.com/MIC-DKFZ/nnUNet)) | Convention: crop to non-zero, resample to **median training spacing**, clip 0.5–99.5 percentiles then z-score; 3rd-order spline in-plane / nearest out-of-plane for anisotropic data. |
| [MONAI repo](https://github.com/Project-MONAI/MONAI) / [nets directory](https://github.com/Project-MONAI/MONAI/tree/dev/monai/networks/nets) | Apache 2.0; ships `densenet.py`, `resnet.py`, `efficientnet.py`, `senet.py`, `vit.py`, etc. with `spatial_dims=3`. |
| [ANTsPy](https://github.com/ANTsX/ANTsPy) / [build_template.py](https://raw.githubusercontent.com/ANTsX/ANTsPy/master/ants/registration/build_template.py) | Apache 2.0; `ants.registration(fixed, moving, type_of_transform=...)`; `ants.build_template(initial_template, image_list, iterations=3, gradient_step=0.2, blending_weight=0.75, ...)` "Estimate an optimal template from an input image_list" (default SyN, overridable via kwargs). |
| [Guo et al. 2017, On Calibration (arXiv:1706.04599)](https://arxiv.org/abs/1706.04599) | "temperature scaling — a single-parameter variant of Platt Scaling — is surprisingly effective at calibrating predictions"; depth, width, weight decay, BatchNorm drive miscalibration. |
| [Deep-ensembles survey results](https://arxiv.org/pdf/2107.03342) (via search) | 5-member ensembles already give most of the calibration gain; >10 members saturate; temperature scaling on the averaged prediction adds further calibration improvement. |
| [Kaggle log-loss tricks (Medium)](https://medium.com/@egor_vorobiev/how-to-improve-log-loss-score-kaggle-trick-3f95577839f1), [Platt/isotonic for log loss](https://www.analyticsvidhya.com/blog/2016/07/platt-scaling-isotonic-regression-minimize-logloss-error/), [NVIDIA KGM playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/) | Community practice: clip probabilities, hill-climb ensemble weights on OOF, stack with a log-loss-focused logistic regression. |
| [timm README — Licenses](https://github.com/huggingface/pytorch-image-models/blob/main/README.md) | Code Apache 2.0; weights: Facebook WSL/SSL/SWSL are CC-BY-NC → **not** prize-safe; Google (e.g., Noisy-Student EfficientNet) Apache 2.0 "with only ImageNet concerns applying". |
| [torchvision models docs](https://docs.pytorch.org/vision/stable/models.html) | Weights "may have their own licenses or terms and conditions derived from the dataset used for training"; 3D video nets (R3D_18, MC3_18, R2Plus1D_18, Swin3D) pretrained on Kinetics-400. |
| [Tencent MedicalNet](https://github.com/Tencent/MedicalNet/blob/master/README.md) | MIT-licensed 3D ResNet-10/18/34/50/101/152/200 weights pretrained on 23 medical datasets (CT/MRI segmentation). Dataset-level terms not enumerated in README. |
| [Kaggle GPU quota (secondary source)](https://www.geeksforgeeks.org/machine-learning/how-to-use-gpu-in-kaggle/) | "You get around 30 hours of GPU time per week"; P100 or T4 (x2). (Kaggle's own docs page is JS-rendered and could not be quoted.) |
| [ANTs affine runtime benchmark (C2FViT paper)](https://arxiv.org/pdf/2203.15216) | ANTs affine brain-MRI pair: mean 55 s, max >60 s single-threaded CPU — used as a conservative upper bound below. |

**Could not be verified this session (stated so, not assumed):** the RSNA 2024 1st-place Kaggle writeup body (JS-rendered), the RSNA 2019 1st-place Kaggle discussion body (same), SeuTao's exact windowing scheme, the MDPI 2026 "training protocols" DaT-SPECT paper full text (403), and the full list of `ants.registration` `type_of_transform` options (docs 404). None of the recommendations below depend on those.

---

## 2. Recurring winning recipe (distilled, with citations)

| Component | What winners did | Evidence |
|---|---|---|
| **Geometry standardization** | Resample to a fixed spacing; crop/locate the organ before classifying. RSNA 2023: 3D segmentation → organ crops; RSNA 2024: keypoint localizer → level crops; Clog Loss: ROI extraction → 3D model. DaT-SPECT SOTA: affine to MNI at 2 mm, then a striatal slab. | [Nischaydnk](https://github.com/Nischaydnk/RSNA-2023-1st-place-solution), [brendanartley](https://github.com/brendanartley/RSNA-2024-Competition), [ZFTurbo](https://github.com/ZFTurbo/DrivenData-Alzheimer-Research-1st-place-solution), [PMC10957699](https://pmc.ncbi.nlm.nih.gov/articles/PMC10957699/) |
| **Intensity standardization** | CT: fixed HU windowing → 3 channels. Generic: nnU-Net 0.5–99.5 percentile clip + z-score. DaT-SPECT: scale to 75th percentile of a non-striatal reference region (DVR). | [darraghdog](https://github.com/darraghdog/rsna), [nnU-Net](https://www.nature.com/articles/s41592-020-01008-z), [PMC10060476](https://pmc.ncbi.nlm.nih.gov/articles/PMC10060476/) |
| **Backbone** | Kaggle RSNA: 2D ImageNet-pretrained CNNs on 2.5D stacks + LSTM/GRU/attention over slices. DrivenData Clog Loss: 3D DenseNet121. DaT-SPECT: tiny 3D/slab ResNets (0.19–2.9M params). Survey: 3D CNNs trail 2D-FM approaches by ~6 AUC points on average across 10 tasks. | [arXiv:2512.12887](https://arxiv.org/html/2512.12887), [clog-loss](https://github.com/drivendataorg/clog-loss-alzheimers-research), [PMID 31473800](https://pubmed.ncbi.nlm.nih.gov/31473800/) |
| **CV scheme** | Patient/study-level GroupKFold, 4–5 folds; all folds ensembled at inference. | [Nischaydnk](https://github.com/Nischaydnk/RSNA-2023-1st-place-solution), [clog-loss](https://github.com/drivendataorg/clog-loss-alzheimers-research) |
| **Augmentation** | Modest geometric (flips, ±25° rotation, perspective) in RSNA 2023; heavy + sequence-flip + manifold mixup in RSNA 2024 2nd; **for DaT-SPECT specifically, strong Gaussian blur + additive noise is the proven multi-site robustness lever**. | [Nischaydnk](https://github.com/Nischaydnk/RSNA-2023-1st-place-solution), [brendanartley](https://github.com/brendanartley/RSNA-2024-Competition), [PMID 39054285](https://jnm.snmjournals.org/content/65/9/1463) |
| **Auxiliary supervision** | Auxiliary segmentation/Dice loss gave +0.01–0.03 in RSNA 2023; multi-task targets in VisioMel 2nd. | [Nischaydnk](https://github.com/Nischaydnk/RSNA-2023-1st-place-solution), [visiomel](https://github.com/drivendataorg/visiomel-melanoma) |
| **TTA** | Present in nearly every winner: flips (2019), 9 rotations (2024), TTA + 5-fold (Clog Loss). | [SeuTao](https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection/blob/master/README.md), [brendanartley](https://github.com/brendanartley/RSNA-2024-Competition), [clog-loss](https://github.com/drivendataorg/clog-loss-alzheimers-research) |
| **Ensembling** | Folds × backbones × preprocessing variants, probability-averaged; DaT-SPECT SOTA uses 5-CNN ensembles. | [Nischaydnk](https://github.com/Nischaydnk/RSNA-2023-1st-place-solution), [PMC10957699](https://pmc.ncbi.nlm.nih.gov/articles/PMC10957699/) |
| **Log-loss post-processing** | RSNA 2023: "scaling factor tuning based on CV scores"; community: clip, temperature/Platt on OOF, hill-climb weights. Theory: temperature scaling is the most reliable single-parameter fix; ensembles themselves improve calibration. | [Nischaydnk](https://github.com/Nischaydnk/RSNA-2023-1st-place-solution), [Guo 2017](https://arxiv.org/abs/1706.04599), [survey](https://arxiv.org/pdf/2107.03342) |
| **Inference-time discipline** | Precompute everything deterministic on CPU in parallel; keep GPU models small; RSNA 2023 used max-pool aggregation to avoid heavy per-slice post-processing. | [Nischaydnk](https://github.com/Nischaydnk/RSNA-2023-1st-place-solution) |

**Two things the winners did NOT do that matter here:** (a) none optimized for calibration beyond simple scaling — so a disciplined OOF temperature-scaling step is a cheap edge; (b) Kaggle RSNA winners had 3–10k studies; this competition has 1,362 volumes, which is the regime where the Hamburg group's tiny-network + aggressive augmentation approach was developed (1,100–1,740 training scans) and where architecture matters less than protocol (per `LITERATURE_REVIEW.md`'s reading of the MDPI 2026 paper).

---

## 3. Recommended end-to-end pipeline for THIS competition

### 3.1 Preprocessing (identical code path for train and inference; all libraries present in the runtime)

1. **Load & reorient.** `nibabel` → canonical RAS (`nib.as_closest_canonical`) or MONAI `Orientationd(axcodes="RAS")`. EDA shows `qform_code = sform_code = 1` for all 1,362 volumes, so this is safe.
2. **Resample to 2.0 mm isotropic** (MONAI `Spacingd(pixdim=(2,2,2), mode="bilinear")`, or `ants.resample_image`). Rationale: the strongest multi-site DaT-SPECT pipelines use 2 mm MNI ([PMC10957699](https://pmc.ncbi.nlm.nih.gov/articles/PMC10957699/)); nnU-Net's median-spacing rule would give ≈2.46 mm, which is also acceptable — pick 2 mm to keep the striatum at ≥10 voxels across.
3. **Provisional per-volume intensity scale** for registration only: clip to [p0.5, p99.5] and min-max to [0,1] (nnU-Net style). This collapses the three raw-count regimes (~5–10 / ~100–1,000 / ~10,000–65,000) found in the EDA.
4. **Build a study template from training data (no external template needed).** Use `ants.build_template(image_list=<~150 normal + ~100 abnormal training scans, downsampled>, type_of_transform="Affine", iterations=3)` ([build_template.py](https://raw.githubusercontent.com/ANTsX/ANTsPy/master/ants/registration/build_template.py)). Including abnormal scans mirrors the multi-template finding ([PMC10060476](https://pmc.ncbi.nlm.nih.gov/articles/PMC10060476/)). Ship the resulting `template.nii.gz` (~a few MB) in `submission.zip`. Optional: initialize with the permissively-licensed MNI152 T1 ([McGill](https://nist.mni.mcgill.ca/icbm-152-nonlinear-atlases-2009/)) so the template lives in an MNI-like frame — cross-modal, so only for orientation/box convenience, not required.
5. **Affine (no warping) registration of every volume to the template** with `ants.registration(fixed=template, moving=vol, type_of_transform="Affine")` (mutual-information metric). Affine-only is what the Hamburg pipelines use and is fast and stable for low-resolution SPECT. Robustness guard: first do center-of-mass alignment; if the affine result's MI is worse than the CoM-only result (or the transform's scale factors leave [0.7, 1.4]), fall back to CoM-only. Log the fallback count (within the 300-line log cap).
6. **Final intensity normalization on the registered volume** — two channels, both prize-safe and both cheap:
   - **Channel A (DVR-like):** divide by the 75th percentile of a fixed *reference-region mask in template space* (whole brain minus a generous striatal/thalamic/brainstem/ventricular box), exactly as in [PMC10060476](https://pmc.ncbi.nlm.nih.gov/articles/PMC10060476/). Define the exclusion box once on the template.
   - **Channel B (nnU-Net-like):** clip [p0.5, p99.5] → z-score within the brain mask.
   Feeding both as 2 input channels lets the network pick; ablate on CV.
7. **Crop a fixed template-space box around the striatum with margin**, e.g. 64×64×48 voxels at 2 mm (128×128×96 mm), which comfortably contains caudate + putamen + reference tissue. This is the 3D analogue of the winners' organ crops and the Hamburg 72×72 slab.
8. **Persist** preprocessed tensors as `float16 .npy` (train: ~1,362 × 2 × 64×64×48 × 2 bytes ≈ 0.53 GB). At inference, preprocessing runs once per test volume in a CPU `multiprocessing.Pool(20)`.

### 3.2 Models (train two DL families + one classical family for ensemble diversity)

- **M1 — small 3D CNN (primary).** MONAI `DenseNet121(spatial_dims=3, in_channels=2, out_channels=1)` *or* a custom 3-stage 3D ResNet (16/32/64 filters, 1–2 residual blocks per stage, ≈0.2–1M params) modeled on [PMC10957699](https://pmc.ncbi.nlm.nih.gov/articles/PMC10957699/). Train from scratch — the data regime (1,362 volumes, strong signal) matches the published from-scratch successes, and it removes every pretrained-weight license question for this member. Dropout 0.3–0.5 before the head (as ZFTurbo did).
- **M2 — 2.5D ImageNet-pretrained 2D backbone (the Kaggle winners' recipe).** Take the 12–16 axial slices through the striatal box, form them into 3-channel triplets of adjacent slices (RSNA 2023 style), encode each with a **permissively-licensed** `timm` backbone (e.g., `tf_efficientnetv2_s` / Google EfficientNet family, Apache 2.0 per [timm README](https://github.com/huggingface/pytorch-image-models/blob/main/README.md); **avoid** Facebook WSL/SSL/SWSL CC-BY-NC weights and **avoid DINOv3**), then a bi-GRU or attention pooling over slices → 1 logit. Input 128×128 after 2× upsampling of the 64×64 crop is enough (striatum is low-res anyway). Bundle the `.safetensors` weights in the zip; do not rely on `huggingface-hub` downloads (no internet).
- **M3 — registered-space SBR/radiomics + gradient boosting (cheap, calibrated, diverse).** On the template-registered DVR volume compute per-side caudate/putamen mean and "hottest-voxel" uptake, putamen/caudate ratio, L/R asymmetry indices, and ~20 first-order/GLCM texture features from small fixed template-space ROIs → `lightgbm` (both in runtime). The radiomics literature in `LITERATURE_REVIEW.md` matched CNNs on registered data and explicitly survived a manufacturer split; this member also acts as a sanity check that registration worked (its AUROC should jump from the EDA's 0.469 to >0.9).

### 3.3 Training strategy (fits Kaggle free tier — see §4)

- **Folds:** 5-fold **StratifiedGroupKFold** where the *group* is an acquisition-signature pseudo-site = (rounded voxel spacing, shape, intensity-regime bucket, dtype) — the EDA identified ≥4 dominant clusters + a long tail; this approximates the 10 unknown hospitals. Also run **one leave-one-major-cluster-out** sanity fit (train on 3 dominant clusters, test on the 4th) purely to estimate the CV→leaderboard gap; do not ensemble it.
- **Loss:** BCE-with-logits; try label smoothing ε=0.02–0.05 as an anti-overconfidence hedge (evaluate by OOF log loss, keep only if it helps *after* temperature scaling). Class weighting unnecessary (1.21:1).
- **Optimizer/schedule:** AdamW, lr 1e-3 (M1 from scratch) / 2e-4 (M2 pretrained), cosine with 3-epoch warmup, weight decay 1e-4 – 1e-2 (weight decay also affects calibration per Guo 2017), 40–60 epochs, AMP, batch 16–32. Snapshot/SWA of the last 5 epochs is a free ensemble (as Clog Loss 2nd place did).
- **Augmentation (the single most important protocol choice for multi-site shift):**
  - *Domain-shift aug:* random Gaussian blur σ ∈ [0, 2.5] voxels (5 mm) and additive Gaussian noise (σ up to ~10% of the DVR range) with p≈0.7, following the "unrealistic augmentation" result ([PMID 39054285](https://jnm.snmjournals.org/content/65/9/1463)); random intensity scale ×[0.8,1.25] and gamma [0.8,1.25] to simulate reconstruction/quantization differences seen in the EDA.
  - *Registration-error aug:* random affine — rotation ±10°, scale ±10%, translation ±3 voxels (`torchio.RandomAffine` or MONAI `RandAffined`).
  - *Left–right flip* (p=0.5): the label is symmetric under L/R mirroring (either side can be affected), so this doubles effective data safely.
  - Avoid elastic deformations and large crops that could move the striatum out of the box.
- **Early stopping / selection:** by OOF **log loss**, not accuracy/AUC.

### 3.4 Calibration & ensembling (the metric is log loss)

1. Collect OOF logits for every member (5 folds × {M1, M2} + M3 probabilities).
2. **Per-member temperature scaling** on its own OOF logits (single scalar, minimize NLL; Guo et al.).
3. **Blend** with non-negative weights found by **hill-climbing / coordinate descent on OOF log loss** (community standard; RSNA 2023 tuned scaling factors on CV). Start from equal weights; expect the 3D CNN and 2.5D model to carry most weight, M3 a minority weight that mostly helps hard cases.
4. **Second temperature on the blended logit** (ensembles are typically under-confident after averaging; a T<1 often helps — verify on OOF).
5. **Clip** final probabilities to [ε, 1−ε]; choose ε ∈ {0.005, 0.01, 0.02} on OOF. With ~95% accuracy, one confidently-wrong prediction at p=0.999 costs ~6.9 nats; clipping bounds that damage.
6. Freeze `T`s, weights and ε into a small JSON shipped in the zip. **Never** tune on the public leaderboard — the private set decides prizes.

### 3.5 Inference plan (must finish in 3 h, no internet, 1× A100, 24 vCPU)

1. Import guard: wrap `import antspyx`/`monai` in try/except with a pure-`scipy` CoM-alignment fallback so a runtime import surprise degrades gracefully instead of crashing.
2. `Pool(20)` over test uids: load → reorient → resample → provisional scale → affine register to bundled template → DVR + z-score → crop → `float16` array. Each volume is independent (rule: "test samples must be processed independently").
3. GPU pass: for each of the 10 DL checkpoints (5 folds × 2 families), batch-infer with **TTA = {identity, L-R flip} × {0, ±1-voxel shift}** → 6 views; average logits per checkpoint. M3 features → lightgbm.
4. Apply frozen calibration; write `submission.csv` with `uid, is_pathologic`.
5. **Time-budget watchdog:** record wall time after the first 50 volumes; if the projected total for preprocessing exceeds 100 min, switch remaining volumes to a faster transform type (rigid or CoM-only) and note it once in the log. Emit predictions for **every** uid regardless (fallback p = OOF prior 0.548 if a volume fails to load).

### 3.6 Validation strategy mirroring the held-out multi-center test set

- Report OOF log loss **per pseudo-site cluster** and the **worst-cluster** log loss, not just the mean — the private set may over-represent a hospital.
- Track the CV→smoke-test gap; the smoke test is drawn from *training* data so it only validates the code path, not generalization.
- Keep the leave-one-cluster-out run as the honest "unseen hospital" estimate; expect it to be noticeably worse than random-fold CV (Wenzel 2019: SBR fell 0.957 → 0.900 under mixed characteristics while the CNN held; Zhao 2022 in `LITERATURE_REVIEW.md` showed a CV→blind-test drop).

---

## 4. Compute budget check

**Training (Kaggle, ~30 GPU-h/week; two resets remain before Sept 16 → ~60 h ceiling; plan for ≤ 20 h):**

| Step | Estimate | Basis |
|---|---|---|
| Template build (CPU) | 0.5–1.5 h once | 3 iterations × ~250 affine registrations of 2 mm downsampled brains; ANTs affine ≤ ~55 s worst-case per pair on brain MRI ([C2FViT benchmark](https://arxiv.org/pdf/2203.15216)); SPECT at 2 mm is ~10× fewer voxels, so typically 5–20 s. |
| Register + normalize 1,362 train volumes (CPU, Kaggle 4 vCPU) | ~1–2 h once | 1,362 × ~15 s ÷ 4 workers ≈ 1.4 h. Cache to a private Kaggle dataset? **No** — the compliance thread in `DAT_PARKINSONS_CHALLENGE.md` flags Kaggle retention; keep preprocessed tensors in session storage only, or preprocess on a self-controlled machine. |
| M1 (3D CNN, 64×64×48×2, ~1M params) | ~5 folds × 50 epochs × ~1,090 vols ≈ 0.3–0.5 h/fold on T4 → **~2 h** | Each epoch is ~35 batches of 32; a small 3D CNN at this size runs well under 1 s/batch on a T4. |
| M1 with DenseNet121-3D instead | ~3–4 h | ~7M params; still small volumes. |
| M2 (2.5D EfficientNetV2-S + GRU, 16 slices at 128²) | ~0.5–0.8 h/fold → **~3–4 h** | Comparable to RSNA-style training but with ~15× fewer studies than RSNA 2023. |
| M3 features + LightGBM | minutes (CPU) | |
| Calibration/blending on OOF | minutes | |
| Ablations (2 extra M1 variants) | ~4 h | Optional. |
| **Total** | **~12–16 GPU-h + ~3 CPU-h** | Comfortably inside one weekly quota; leaves headroom for a second seed of M1 (another ~2 h) to widen the ensemble. |

**Inference (organizer A100 80GB, 24 vCPU, 3 h hard limit; test size undisclosed — plan for up to 3,000 volumes):**

| Step | Estimate |
|---|---|
| Preprocess + register 3,000 volumes with 20 workers | 3,000 × 15 s ÷ 20 ≈ **38 min** (worst case 55 s/vol → 2.3 h — hence the watchdog fallback to rigid/CoM after 100 min of projected time). |
| GPU inference: 10 checkpoints × 6 TTA views × 3,000 volumes at 64×64×48 | tens of thousands of tiny forward passes; **< 10 min** on an A100. |
| M3 features + LightGBM | **< 5 min** |
| Total (expected) | **~1 h**, with a hard-stop fallback keeping it under 3 h even in the slow-registration case. |

**Zip size:** 10 checkpoints (M1 ≈ 4–30 MB each, M2 ≈ 80–100 MB each) + template + lightgbm ≈ 0.5–1.2 GB; a >1 GB zip was accepted once the upload timeout was extended ([forum 11470](https://community.drivendata.org/t/limit-to-size-of-zip-files-for-uploading/11470)). Store weights as `safetensors` fp16 to halve it.

---

## 5. Prize-eligibility check

Rule (from `DAT_PARKINSONS_CHALLENGE.md`): external data/models must be freely & publicly available and licensed for commercial redistribution; winners release under MIT. Staff ruling on weights: "eligible for prizes as long as the model or weights themselves are permissively licensed (such as MIT or Apache 2.0)" ([forum 11475](https://community.drivendata.org/t/clarification-on-imagenet-pre-trained-weights-under-open-source-commercial-rules/11475)).

| Resource | License | Prize-safe? | Notes |
|---|---|---|---|
| Competition data only (no external training data) | competition license | **Yes** | Recommended default for all three model families. |
| Data-derived registration template (built from training scans) | derived from competition data | **Yes** | No external asset. Note: the template itself is a derivative of competition data; it ships inside the zip which is allowed ("include your model weights"), and an averaged affine template of ~250 scans is not an individual scan. Disclose it as part of the model artifacts. |
| `antspyx`, `monai`, `simpleitk`, `torchio`, `timm` (code) | Apache 2.0 | **Yes** | All present in runtime `pyproject.toml`. |
| `timm` **Google** EfficientNet / EfficientNetV2 ImageNet weights | Apache 2.0 (per timm README) | **Yes per staff ruling**; residual theoretical ImageNet-terms caveat acknowledged by timm itself | Bundle weights offline. Prefer `tf_efficientnet*`/`tf_efficientnetv2*` variants explicitly labeled Apache in timm. |
| `timm` Facebook WSL/SSL/SWSL ResNe(X)t weights | CC-BY-NC 4.0 | **No** | Explicitly non-commercial. |
| DINOv3 (any variant) | DINOv3 Agreement | **No** — staff ruling ([forum 11488](https://community.drivendata.org/t/is-dinov3-allowed-to-use/11488)) | Also excluded: any model requiring sign-up/acceptance of a custom license. |
| torchvision ImageNet / Kinetics-400 3D video weights (R3D_18, MC3_18…) | torchvision code BSD-3; docs say weights "may have their own licenses ... derived from the dataset" | **Uncertain** — ask on the forum before relying on them | Clog Loss 3rd place used Kinetics-pretrained R(2+1)D; benefit for SPECT is speculative anyway. |
| Tencent MedicalNet 3D ResNet weights | MIT (repo) | **Probably yes, but flag** | Weights are MIT, but trained on 23 CT/MRI datasets whose terms aren't enumerated in the README; the staff ruling keys on the *weights'* license, so MIT should pass — still disclose. Modality gap (CT/MRI → SPECT) makes the benefit doubtful; treat as an optional ablation. |
| Salas-Gonzalez FP-CIT MNI template (NITRC) | CC BY-NC-SA | **No** | Do not ship or use, even for template initialization. |
| García-Gómez SPM FP-CIT templates (NITRC) | "Freeware ... may have redistribution restrictions" | **Uncertain → avoid** | Not clearly commercial-redistributable. |
| MNI ICBM152 2009 T1 template | Permissive McGill notice ("for any purpose and without fee") | **Yes** | Only useful as an optional anatomical reference frame; cross-modal. |
| PPMI, NTUA datasets | DUA / academic-only | **No** (PPMI ruled out by staff; NTUA per `LITERATURE_REVIEW.md`) | Not used anywhere in this recommendation. |
| Hamburg `dat_spect_ud` code | MIT | Code yes; **it does not include preprocessing or templates**, and its trained weights' provenance/licensing should be checked before any use | Use as a reference implementation only. |

**Separate open compliance item (not a license question):** training on Kaggle with competition data remains an unanswered forum question (see `DAT_PARKINSONS_CHALLENGE.md`). This brief assumes Kaggle is used per the task statement, but a self-controlled, wipeable machine is the safer route for the prize path.

---

## 6. Confidence & caveats

**Well supported (multiple independent, fetched sources):**
- Affine-to-template at 2 mm + reference-region (75th-percentile DVR) intensity normalization + striatal crop is *the* published pipeline for multi-site DaT-SPECT CNNs and is fully implementable offline with `antspyx`/MONAI ([PMC10957699](https://pmc.ncbi.nlm.nih.gov/articles/PMC10957699/), [PMC10060476](https://pmc.ncbi.nlm.nih.gov/articles/PMC10060476/), [dat_spect_ud](https://github.com/ThomasBudd/dat_spect_ud), [runtime pyproject](https://raw.githubusercontent.com/drivendataorg/competition-sfmn-parkinsons-runtime/main/runtime/pyproject.toml)).
- Strong blur+noise augmentation is the demonstrated lever for unseen-site/camera generalization in DaT-SPECT ([PMID 39054285](https://jnm.snmjournals.org/content/65/9/1463), [PMID 31473800](https://pubmed.ncbi.nlm.nih.gov/31473800/)).
- Kaggle 3D-classification winners converge on 2.5D pretrained backbone + sequence head, patient-level GroupKFold, fold×backbone ensembles, TTA, CV-tuned scaling ([arXiv:2512.12887](https://arxiv.org/html/2512.12887), [Nischaydnk](https://github.com/Nischaydnk/RSNA-2023-1st-place-solution), [brendanartley](https://github.com/brendanartley/RSNA-2024-Competition), [darraghdog](https://github.com/darraghdog/rsna)).
- Temperature scaling on held-out logits + ensembling is the standard, well-evidenced calibration approach ([Guo 2017](https://arxiv.org/abs/1706.04599)).
- Licensing rulings: ImageNet permissive weights OK, DINOv3 not, PPMI not ([forum 11475](https://community.drivendata.org/t/clarification-on-imagenet-pre-trained-weights-under-open-source-commercial-rules/11475), [11488](https://community.drivendata.org/t/is-dinov3-allowed-to-use/11488)); NITRC FP-CIT template is CC-BY-NC-SA ([NITRC](https://www.nitrc.org/projects/fp_cit_atlas)).

**Moderately supported / partly inferred:**
- Whether 3D-from-scratch (M1) or 2.5D-pretrained (M2) will win *on this dataset* is genuinely uncertain: the Kaggle evidence favors 2.5D on large CT/MRI datasets; the DaT-SPECT evidence favors small 3D/slab CNNs on 1–3k scans. Training both and letting OOF log loss set blend weights is the hedge.
- Two-channel input (DVR + z-score) is my proposal, not a published recipe; ablate it.
- Registration timing (5–20 s/volume) is extrapolated from brain-MRI benchmarks to lower-resolution SPECT; measure it on Kaggle CPU early and keep the rigid/CoM fallback.
- Compute estimates assume ~1M–7M-param 3D nets on 64×64×48 crops; if you enlarge the crop or use DenseNet-3D at full 96³, expect 2–3× more GPU time.

**Speculative / not verified this session:**
- RSNA 2024 and 2019 *1st-place* Kaggle writeup bodies could not be fetched (JS-rendered); their recipes are represented via the survey paper, the GitHub READMEs, and the 2nd-place repos.
- The MDPI 2026 "training protocols" DaT-SPECT paper was not readable (403); its ablation magnitudes are not cited here.
- Exact `ants.registration` transform-type strings (`"Affine"`, `"AffineFast"`, `"Rigid"`, …) were not re-verified against the docs (404); check `help(ants.registration)` in the runtime image before relying on a specific string.
- MedicalNet weight usefulness for SPECT and the torchvision Kinetics weights' prize status are both unverified; neither is on the recommended critical path.
- The template built from competition scans is prize-safe on licensing grounds, but its status as a *derived artifact of competition data shipped in the zip* should be disclosed to organizers along with model weights; this is a judgment call, not a verified ruling.
