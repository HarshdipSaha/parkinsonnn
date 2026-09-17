# Official Competition Compliance & Methodological Disclosure

**Competition**: DrivenData Challenge #311 — DaT Parkinson's Challenge  
**Team / Participant**: `harshdipsaha`  
**Final Placement**: Rank #50 (Private Log Loss: 0.3003, AUROC: 0.9406)  
**Repository**: `HarshdipSaha/parkinsonnn`  

---

## 1. External Data & Pretrained Model Disclosure

In strict adherence to the competition official rules:
- **External Data**: **Zero external datasets** were used for training, validation, self-supervised learning, or calibration. All training was conducted exclusively on the 1,362 training scans provided by DrivenData.
- **PPMI Database**: Although the competition data originated from PPMI, no raw PPMI scans, clinical metadata, UPDRS scores, demographic fields, or external PPMI follow-up releases outside the official competition package were downloaded, accessed, or incorporated.
- **Pretrained Weights**:
  - The 3D backbones (`a3d_hemi`, `HemiNet3D`, `a3d_ssl`, `a3d_hemisoft`, `a3d_crophistsoft`) were trained **from scratch** with random initializations or initialized from self-supervised VICReg pretraining strictly on the competition scans.
  - The 2D backbones (`a2d_v2s`, `a2d_cnx_cor`) utilized public ImageNet-1K pretrained weights sourced strictly from official `torchvision.models` under the **BSD 3-Clause License**:
    - `torchvision.models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)`
    - `torchvision.models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)`
  - No medical foundation models (e.g. RadImageNet, MedicalNet, BiomedCLIP) or non-commercial research weights were used in the final submission packages.

---

## 2. Model Architecture & Pipeline Overview

Our final submission packages (`submission_master_v3_lam01.zip` and `submission_v9.zip`) implemented a multi-scale, multi-view bilateral ensemble consisting of 7 distinct model arms:

1. **`a3d_hemi` (Bilateral 3D HemiNet)**: Dual-stream 3D ResNet-18 sharing weights across left and right striatal volumes, computing bilateral feature differences to quantify asymmetric putaminal uptake loss.
2. **`a3d_ssl` (VICReg Pretrained 3D CNN)**: 3D ResNet encoder pretrained via Variance-Invariance-Covariance Regularization (VICReg) self-supervised learning on 1,362 unlabelled scans, fine-tuned for classification.
3. **`featc` (Radiomic Feature Arm)**: Classical geometric and photometric feature extractor calculating left/right striatal binding ratios (SBR), background non-specific binding, and putamen-to-caudate ratios, modeled via Ridge Logistic Regression.
4. **`a2d_cnx_cor` (Coronal ConvNeXt-Tiny)**: 2D ConvNeXt-Tiny processing 3-slice slab maximum intensity projections in the coronal plane through the anterior-posterior striatal center.
5. **`a2d_v2s` (Multi-Slice EfficientNetV2-S)**: 2D EfficientNetV2-S processing sequential axial slices centered at the striatal hotspot.
6. **`a3d_hemisoft` (Ensemble-Distilled Bilateral 3D CNN)**: 3D HemiNet trained on soft probability targets distilled from the Master v2 ensemble ($y^* = 0.6y + 0.4p$).
7. **`a3d_crophistsoft` (Histogram-Standardized Distilled 3D CNN)**: 3D ResNet trained on non-linear histogram-standardized striatal crops with soft distillation targets.

---

## 3. Inference Determinism & Rule Compliance

- **Per-Scan Isolation**: All preprocessing (hotspot localization, affine registration, intensity normalization) and model forward passes operate strictly on individual scans in isolation. No batch-level statistics across the test set are computed.
- **No Transductive Learning**: No test-set pseudo-labeling, transductive graph propagation, or unsupervised test-set domain adaptation was used.
- **Reproducibility & Execution**: Docker/container inference packages run headlessly via `main.py` and `infer_v3.py` within the prescribed DrivenData 2-hour execution limit, using GroupNorm and evaluation-mode layers to ensure 100% deterministic test predictions.

---

## 4. Compute Infrastructure & Data Hygiene

- **Training Venue**: Models were trained on Kaggle cloud instances utilizing dual NVIDIA Tesla T4 and single P100 GPUs.
- **Data Hygiene**: In compliance with DrivenData dataset confidentiality terms, all Kaggle datasets containing raw competition volumes (`dat-parkinsons-challenge-data`) are strictly private and marked for deletion upon competition audit conclusion.
- **Open-Source Release**: All model architectures, inference pipelines, registration routines, and AI-DLC engineering logs are open-sourced under the MIT License in the repository `HarshdipSaha/parkinsonnn`.

---

## 5. Verified Platform Submission Ledger

| Slot | Submission ID | Date (UTC) | Public Log Loss | Private Log Loss | Private AUROC | Model Configuration |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | `id-321156` | 2026-09-05 15:38 | 0.3391 | 0.3924 | 0.9082 | Run 2 dual-crop CNN + Ridge LR |
| 2 | `id-321263` | 2026-09-06 15:10 | 0.2817 | 0.3217 | 0.9477 | Run 3 registered-frame stack |
| 3 | `id-321298` | 2026-09-07 03:38 | 0.2667 | 0.3099 | 0.9536 | Run 4a 3D + V2-S + ConvNeXt-T |
| 4 | `id-321967` | 2026-09-12 01:40 | 0.2557 | 0.3029 | 0.9576 | Master v3 ($\lambda=0.02, T=0.90$) |
| 5 | `id-322091` | 2026-09-13 03:08 | **0.2473** | **0.3003** | **0.9608** | **Master v3 ($\lambda=0.01, T=0.90$) — Best Private** |
| 6 | `id-322254` | 2026-09-14 05:10 | 0.2522 | 0.3019 | 0.9582 | Master v9 (91 ckpts, $\lambda=0.01$) |
