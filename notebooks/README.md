# DaT-SPECT Parkinson's Challenge — Jupyter Notebooks Index

This directory contains the self-contained, educational, and reproducible Jupyter Notebooks for DrivenData Competition #311 (**DaT Parkinson's Challenge**).

---

## Notebook Guide

| # | Notebook | Focus & Contents |
|:---:|---|---|
| **01** | [01_eda_dat_spect_complete.ipynb](01_eda_dat_spect_complete.ipynb) | **Full Exploratory Data Analysis & Quality Audit**: 1,362 scans, geometry clusters, missingness audit, trimodal intensity quantization anomaly, orthogonal visualizations, SBR calculation. |
| **02** | [02_eda_scanner_heterogeneity_and_hard_scans.ipynb](02_eda_scanner_heterogeneity_and_hard_scans.ipynb) | **Scanner Domain Shift & Post-Mortem**: Multi-center camera fingerprints, why random 5-fold CV leaked, LOCO validation, the 88 hard scans carrying 46.9% loss, and 3D vs 2D transfer penalty. |
| **03** | [03_registration_and_spatial_preprocessing.ipynb](03_registration_and_spatial_preprocessing.ipynb) | **Spatial Normalization & Registration**: Hotspot detection, center-of-mass alignment, MNI affine registration, $48 \times 48 \times 32$ striatal cropping, and bilateral half-crops. |
| **04** | [04_radiomic_and_sbr_feature_engineering.ipynb](04_radiomic_and_sbr_feature_engineering.ipynb) | **Classical Radiomics & SBR Baselines**: 116 geometric & photometric descriptors, Caudate/Putamen SBR, asymmetry indices, and Ridge Logistic Regression baseline. |
| **05** | [05_deep_learning_models_zoo.ipynb](05_deep_learning_models_zoo.ipynb) | **Deep Learning Model Zoo**: PyTorch implementations of `HemiNet3D` (Bilateral Siamese), `TriPlanarNet` (2.5D Late-Fusion), `ConvNeXt-Tiny`, `EfficientNetV2-S`, and `Swin Transformer`. |
| **06** | [06_master_ensemble_calibration_and_submission.ipynb](06_master_ensemble_calibration_and_submission.ipynb) | **Master Ensemble Stacking & Calibration**: 7-arm multi-model blending, temperature scaling ($T=0.90$), prior mixture ($\lambda=0.01$), local container e2e testing, and zip packaging. |

---

## Cloud Training Kernels (`kaggle/`)
The production training scripts used on Kaggle are located in `kaggle/`:
- `kaggle/dat-parkinsons-train.ipynb`: Initial multi-backbone training kernel.
- `kaggle/run6a_crop3d/notebook.ipynb`: Run 6a 3D striatal crop models.
- `kaggle/run6b_triplanar/notebook.ipynb`: Run 6b Tri-Planar 2.5D multi-view models.
- `kaggle/run6c_features_ssl/notebook.ipynb`: Run 6c 116-descriptor radiomics & VICReg SSL pretraining.
- `kaggle/run7_hemisphere/notebook.ipynb`: Run 7 HemiNet3D bilateral Siamese training.
- `kaggle/run8_distill/notebook.ipynb`: Run 8 Distillation on soft ensemble targets ($y^* = 0.6y + 0.4p$).
- `kaggle/run9_seeds/notebook.ipynb`: Run 9 Master v9 91-checkpoint seed expansion.
- `kaggle/run10_swin/notebook.ipynb`: Run 10 2D Swin Transformer attention arm.
