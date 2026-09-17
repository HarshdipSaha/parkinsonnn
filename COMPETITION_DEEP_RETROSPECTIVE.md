# Deep Competition Retrospective: DaT Parkinson's Challenge

**A transferable field guide for ML competitions — written with blood, not ink.**

Written 2026-09-17, after 11 days, 9 runs, 6 scored submissions, 20 smoke tests, ~30 GPU-hours,
and a fall from Public Rank #16 to Private Rank #50 on DrivenData Challenge #311.

This document is not a summary. It is an exhaustive, deeply reasoned dissection of every
architectural choice, every decision heuristic, every failure mode, and every mental model
that shaped the outcome — structured so that any of it can be picked up and carried into a
completely different competition domain.

---

## Table of Contents

1. [What We Are Explaining](#1-what-we-are-explaining)
2. [The Textbook Statement: Final Numbers](#2-the-textbook-statement-final-numbers)
3. [Notation and Symbol Ledger](#3-notation-and-symbol-ledger)
4. [The Problem This Competition Posed](#4-the-problem-this-competition-posed)
5. [Prerequisite Ideas, Briefly Rebuilt](#5-prerequisite-ideas-briefly-rebuilt)
6. [The Complete Arc: Run by Run](#6-the-complete-arc-run-by-run)
7. [The Five Structural Failure Modes](#7-the-five-structural-failure-modes)
8. [The Key Insight](#8-the-key-insight)
9. [Mental Models for Competition ML](#9-mental-models-for-competition-ml)
10. [Worked Examples: Decisions That Mattered](#10-worked-examples-decisions-that-mattered)
11. [Non-Examples and Boundary Cases](#11-non-examples-and-boundary-cases)
12. [Confusion Clinic](#12-confusion-clinic)
13. [Reconstruction Checklist](#13-reconstruction-checklist-the-playbook)
14. [Transferable Laws](#14-transferable-laws-beyond-this-competition)
15. [What I Would Do Differently With a Time Machine](#15-what-i-would-do-differently-with-a-time-machine)

---

## 1. What We Are Explaining

**Target:** The complete decision process of a medical-imaging ML competition — from EDA through
9 training runs to final submission — and why a solution that reached Public Rank #16 collapsed to
Private Rank #50.

**Central question this document answers:** What are the *transferable* mental models, decision
heuristics, and failure patterns that determine whether a competition solution generalises to hidden
data — and how do you build a checklist that catches them *before* the private leaderboard reveals
them?

**Reader contract:** By the end of this document, you will have:
- A quantitative understanding of every decision that shaped this result
- 9 mental models you can apply to *any* ML competition
- A concrete reconstruction checklist you can print and tape to your monitor
- A confusion clinic answering the 12 questions you'll ask yourself at 3 AM on deadline night

---

## 2. The Textbook Statement: Final Numbers

### The Official Submission Ledger

| Slot | ID | Date | Public LL | Private LL | AUROC | What Changed |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | `id-321156` | Sept 5 | 0.3391 | 0.3924 | 0.9082 | First working pipeline |
| 2 | `id-321263` | Sept 6 | 0.2817 | 0.3217 | 0.9477 | Registration + multi-backbone |
| 3 | `id-321298` | Sept 7 | 0.2667 | 0.3099 | 0.9536 | Striatal crop + nested CV |
| 4 | `id-321967` | Sept 12 | 0.2557 | 0.3029 | 0.9576 | 7-arm bilateral ensemble |
| 5 | `id-322091` | Sept 13 | **0.2473** | **0.3003** | **0.9608** | λ=0.01 bracket variant |
| 6 | `id-322254` | Sept 14 | 0.2522 | 0.3019 | 0.9582 | 91 checkpoints, expanded seeds |

### The Leaderboard Context

| Rank | Private Log Loss | Key Architecture |
|:---:|:---:|:---|
| #1 | 0.2532 | 2D multi-plane projections |
| #6 (`paul-bd`) | 0.2732 | DenseNet-121 on MIP projections |
| #10 (`DataCouples`) | 0.2760 | 2D slice-sequence BiGRU |
| **#50 (us)** | **0.3003** | 3D ResNet bilateral ensemble |

### The Shape of the Gap

```
Our Private:    0.3003
Top-10 cutoff:  0.2760
Gap:            0.0243

Our AUROC:      0.9406   (ranking accuracy was competitive)
Top-10 AUROC:   ~0.95    (only slightly better)

Conclusion:     The gap was NOT discrimination.
                The gap was CALIBRATION under DOMAIN SHIFT.
```

**The entire 0.0243 gap can be decomposed** into five structural factors (§7), each of which
has a transferable lesson.

---

## 3. Notation and Symbol Ledger

| Symbol | Type | Meaning | Where It Appears | Common Confusion |
|:---:|:---:|:---|:---|:---|
| LL | metric | Log loss = $-\frac{1}{N}\sum_{i} [y_i \ln \hat{p}_i + (1-y_i)\ln(1-\hat{p}_i)]$ | Every score | Not the same as cross-entropy *training* loss (which may use label smoothing, mixup, etc.) |
| AUROC | metric | Area under ROC curve; measures ranking quality independent of calibration | Submission results | High AUROC ≠ low log loss. You can rank perfectly and still lose on LL |
| $T$ | parameter | Temperature scaling: $\hat{p} = \sigma(z/T)$ where $z$ is the logit | Calibration | $T < 1$ sharpens (more confident); $T > 1$ softens. We used $T=0.90$ — sharpening |
| $\lambda$ | parameter | Prior mixture: $\hat{p}_{final} = (1-\lambda)\hat{p}_{model} + \lambda\pi$ | Calibration | Insurance against confident mistakes. $\lambda=0$ means no floor on confident errors |
| $\pi$ | constant | Class prevalence in training data = 0.548 | Prior mixture | Not the test prevalence, which is unknown |
| OOF | abbreviation | Out-of-fold: prediction on the validation fold the model never trained on | CV estimates | OOF on random folds ≠ OOF on site-stratified folds |
| Nested | estimate | Leave-one-fold-out re-estimation of the stacking combiner | Decision gating | The "honest" estimate, but still random-fold if the base folds are random |
| LOCO | method | Leave-One-Cluster-Out: validation where each fold is a camera/site cluster | Top solutions | We never implemented this. It was the single biggest validation mistake |
| Smoke | test type | DrivenData's 20-scan hidden test set for format/runtime validation | Platform smoke tests | High variance (n=20). Spread of 0.0087 observed on identical zip |

---

## 4. The Problem This Competition Posed

### The Surface Problem
Classify 3D DaT-SPECT brain scans as normal ($y=0$) or pathologic ($y=1$, indicating
dopaminergic denervation consistent with Parkinson's disease). Metric: log loss. Format: code
submission (your inference code runs on their servers).

### The Real Problem (Discovered Too Late)
The training data came from a fixed set of hospital scanners (PPMI consortium). The hidden test
data included scanners the training set had never seen. The competition was not a classification
problem — it was a **domain generalisation** problem where the domain shift was
**scanner hardware**.

### Why the Naive Approach Fails
A naive approach (train a 3D CNN, minimise cross-entropy, submit) achieves AUROC ~0.91 and
LL ~0.39 (our Slot 1). This looks reasonable until you realise the leader is at 0.21. The 0.18
gap is not about model capacity. It's about:

1. **Spatial normalisation** (the scans come in wildly different shapes, spacings, orientations)
2. **Scanner invariance** (the model must not learn scanner-specific textures)
3. **Calibration under shift** (confident predictions on shifted data are catastrophically penalised)

> **Mental Model: The Iceberg**
>
> **Image:** The classification accuracy is the tip above water. The mass below is
> preprocessing (registration, normalisation, augmentation) and calibration. In medical
> imaging competitions, the underwater part is 3-5× larger than the modelling part.
>
> **What it clarifies here:** Why our first submission (AUROC 0.91) was Rank #152. The
> classification was fine; everything underneath it was wrong.
>
> **Formal anchor:** Slot 1 (LL 0.3391) vs Slot 2 (LL 0.2817) — a 0.0574 improvement from
> fixing registration alone, more than all subsequent modelling gains combined.
>
> **Where it breaks:** In tabular competitions or NLP, preprocessing is a smaller fraction.
> This model is specific to signal-processing-heavy domains (medical imaging, audio, remote sensing).
>
> **Shared by:** Kaggle RSNA competitions, satellite image segmentation challenges, speech
> recognition competitions.

---

## 5. Prerequisite Ideas, Briefly Rebuilt

### 5.1 Log Loss Is Asymmetrically Cruel

Log loss penalises confident wrong predictions exponentially:

$$\mathcal{L}(y=0, \hat{p}) = -\ln(1 - \hat{p})$$

| $\hat{p}$ (predicted probability of disease) | Loss when $y=0$ (patient is healthy) |
|:---:|:---:|
| 0.50 | 0.693 nats |
| 0.70 | 1.204 nats |
| 0.90 | 2.303 nats |
| 0.95 | 2.996 nats |
| 0.99 | 4.605 nats |

**One scan predicted at 0.99 when it's actually 0 costs as much as 6.6 scans predicted
correctly at 0.50.** This is the mechanism behind Factor 5 (§7.5).

**Where it's used later:** Explains why temperature sharpening ($T=0.90$) was catastrophic
on shifted private data, and why $\lambda > 0$ is insurance, not a lever.

### 5.2 Domain Shift in Medical Imaging

Different hospital scanners produce different image characteristics:
- **Voxel spacing:** 2.30 mm to 3.90 mm across 4 major clusters, plus a long tail
- **Matrix dimensions:** 128×128 to 256×256×256
- **Reconstruction filters:** FBP vs OSEM with varying iterations
- **Intensity scales:** Three distinct regimes (5-10, 100-1000, 10000-65000 raw counts)

A model trained on Scanner A's texture will confidently misclassify Scanner B's normals
because their noise floor looks different. This is not a bug in the model — it's a feature
the model learned that doesn't transfer.

### 5.3 Cross-Validation Designs

| CV Type | What It Measures | What It Misses |
|:---|:---|:---|
| Random k-fold | In-distribution generalisation | Cross-site shift |
| Stratified k-fold | Same, with balanced classes | Cross-site shift |
| Group k-fold (by site) | Cross-site transfer | Nothing, if sites are the right grouping |
| LOCO (leave-one-cluster-out) | Camera-invariant performance | Requires knowing the clusters |

**We used random 5-fold.** Top teams used LOCO. This single choice cascaded into every
downstream decision (temperature selection, model pruning, augmentation rejection).

### 5.4 Temperature Scaling

Given logits $z$, the calibrated probability is:

$$\hat{p} = \sigma(z / T)$$

- $T = 1.0$: no change (model's native calibration)
- $T < 1.0$: sharpens (more confident predictions)
- $T > 1.0$: softens (less confident, more conservative)

**Our choice:** $T = 0.90$ (sharpening), optimised on random-fold CV.
**Winners' choice:** $T \ge 1.05$ (softening), optimised on LOCO CV.

Random-fold CV sees no shift, so it always rewards sharpening. LOCO CV sees shift, so it
rewards the opposite. The validation design determined the calibration, and the calibration
determined the private score.

### 5.5 Ensemble Combination

We used a 7-arm ensemble with equal weights:
- `a3d_hemi` (bilateral 3D Siamese) — 25 checkpoints
- `a3d_ssl` (VICReg self-supervised 3D) — 10 checkpoints
- `featc` (116 radiomic descriptors + RFF kernel) — 1 checkpoint
- `a2d_cnx_cor` (coronal ConvNeXt-Tiny 2D) — 5 checkpoints
- `a2d_v2s` (axial EfficientNet-V2-S 2D) — 10 checkpoints
- `a3d_hemisoft` (soft-label distilled 3D bilateral) — 25 checkpoints
- `a3d_crophistsoft` (histogram-normalised soft-label 3D) — 15 checkpoints

**75 of 91 checkpoints were 3D ResNets.** The ensemble was wide but not diverse in
its fundamental representation type.

### 5.6 Code Submission Format

Unlike Kaggle notebook competitions, DrivenData code submissions:
- Run in an isolated container with no internet
- Must include all model weights in the zip
- Have wall-clock time limits (180 min for 2000+ scans)
- Evaluate on hardware you can't see (24 vCPU, A100 GPU)

This creates engineering constraints (zip size, inference time, pretrained weight packaging)
that interact with modelling choices in non-obvious ways.

---

## 6. The Complete Arc: Run by Run

### Run 1–2: Earning the Right to Compete (Sept 5)

**What happened:** Built the first working end-to-end pipeline. Centre-of-mass (CoM) crop
for spatial normalisation. 3D ResNet-18 with 2 seeds per fold. Ridge logistic regression
as an alignment gate.

**Key numbers:**
- OOF log loss: 0.2994 (15 checkpoints)
- Public: 0.3391 (Rank #152)
- The hotspot anchor (supposed improvement) was actually *worse* than the plain CoM crop
  (0.3240 vs 0.3023 per-fold mean)

**Lesson earned:** *Submit early. The pipeline is more valuable than the model.*
A working pipeline with a bad model is Rank #152. No pipeline is Rank ∞.

### Run 3: Registration Changes Everything (Sept 6)

**What happened:** Replaced CoM crop with proper affine registration to a study template
(SimpleITK, 7-DOF similarity + affine refinement). Added reference-region normalisation.
Multi-backbone stack: EfficientNet-B0, EfficientNet-V2-S, 3D ResNet, Ridge LR.

**Key numbers:**
- Stack OOF: 0.2580 (AUROC 0.9614)
- Public: 0.2817 (Rank #66) — jumped 86 places
- Registration alone improved the LR gate from AUROC 0.852 to 0.902

**The critical bug:** Every 2D checkpoint failed in the container because the manifest
carried `weights_path` from training time. Found by the end-to-end test cell. Without that
cell, the submission would have scored 0.2996 (3D + LR only) instead of 0.2817.

**Lesson earned:** *Preprocessing > modelling. End-to-end container tests find bugs that
unit tests miss.*

> **Mental Model: The Foundation**
>
> **Image:** You can't build a skyscraper on sand. Registration is the foundation.
> Every model improvement after Run 3 was less than 0.0574 — the gain from fixing
> the foundation.
>
> **What it clarifies here:** Why the first thing to do in any competition with
> heterogeneous inputs is to normalise the inputs, not to build better models.
>
> **Formal anchor:** Run 1 OOF 0.2994 → Run 3 OOF 0.2580 = -0.0414 from
> registration. All subsequent modelling: Run 3 → Run 9 = -0.0314 total.
>
> **Where it breaks:** In competitions where inputs are already normalised
> (e.g., pre-cropped ImageNet, tokenised text). Only applies when raw inputs
> have heterogeneous geometry/scale.
>
> **Shared by:** Any medical imaging competition, satellite/aerial image
> competitions, audio competitions with variable sample rates.

### Run 4a: Nested CV and the Feature Arm (Sept 6–7)

**What happened:** Added striatal-crop 3D (the winning inductive bias for 3D models),
3-seed V2-S, ConvNeXt-Tiny, and the 116-descriptor radiomics feature arm. Introduced
nested CV as the gating metric.

**Key numbers:**
- Nested estimate: 0.2491 (improvement over Run 3's 0.2603)
- Public: 0.2667 (Rank #39) — jumped 27 places
- Smoke-to-public gap: +0.049 (the largest in the project)

**The size limit scare:** We believed the zip size was capped at ~420 MB based on browser
upload failures. The actual limit was at least 1421 MB. Three plans were shaped by a
constraint that was never measured.

**Lesson earned:** *Measure your constraints before optimising against them.*

### Runs 5–7: Representations That Mattered (Sept 7–9)

**Run 5:** Site-invariance mechanisms (DANN, group-DRO) failed the gate. But histogram
standardisation worked (-0.0029). Scanner blur augmentation was tested and *dropped*
because it lost 0.003 on random CV.

**Run 6b (Tri-Planar):** Late-fusion 2.5D multi-view network. Smoke test: 0.2341 — the
strongest *standalone* score of the entire project. Yet the stacking optimiser assigned it
0% weight because its OOF was collinear with `a3d_hemi`.

**Run 7 (HemiNet3D):** Bilateral Siamese architecture (midline split, shared encoder,
symmetric pair head). Single-arm smoke: 0.2706. Took 50% weight in the final stack.

**Key insight missed:** The Tri-Planar 2.5D model was our strongest generaliser (0.2341
standalone smoke vs 0.2706 for HemiNet3D) but was excluded by the stacking algorithm.
The stacker optimised on random-fold OOF, which rewarded collinear 3D models over
orthogonal 2D representations. *The validation design infected the ensemble selection.*

> **Mental Model: Optimiser's Blindspot**
>
> **Image:** A flashlight that only illuminates the ground directly in front of you.
> You step confidently forward, missing the cliff to your left.
>
> **What it clarifies here:** The stacking optimiser could only see random-fold OOF.
> It optimised brilliantly for what it could see (in-distribution), but was blind to
> what mattered (cross-scanner generalisation). Tri-Planar's strength was invisible
> to the optimiser because Tri-Planar and HemiNet3D scored similarly on random folds.
>
> **Formal anchor:** Tri-Planar standalone smoke 0.2341 (strongest individual), but
> 0% stack weight. HemiNet3D standalone smoke 0.2706 (weaker individual), but 50%
> stack weight.
>
> **Where it breaks:** If your validation set accurately reflects the test distribution,
> the optimiser's view is correct and this model doesn't apply.
>
> **Shared by:** Feature selection with leaky cross-validation, hyperparameter tuning
> on non-representative validation sets, NAS on proxy tasks.

### Run 8: Distillation and Master v3 (Sept 11)

**What happened:** Soft-label distillation from the master pool. Added `a3d_hemisoft` and
`a3d_crophistsoft` arms. Greedy subset selection found the 7-arm combination.

**Key numbers:**
- Best nested estimate ever: 0.2289 (greedy 7-arm) vs 0.2323 (Master v2)
- Gate passed: +0.0034 ≥ 0.003 threshold
- DrivenData smoke: 0.2374
- Public: 0.2557 (Rank #23) at $\lambda=0.02$; **0.2473 (Rank #16)** at $\lambda=0.01$

**The $\lambda$ trap:** The +0.0084 public improvement from $\lambda=0.02 \to 0.01$
exceeded its own arithmetic ceiling of +0.0055 (and realistic effect of +0.0020). The gain
was not from $\lambda$ — something else differed (likely the wall-clock ladder dropping arms).
But the conclusion "push $\lambda$ lower" had already been recorded and acted upon.

**Lesson earned:** *Check that a claimed gain is arithmetically possible before believing it.*

### Run 9: The Endgame (Sept 13–14)

**What happened:** Expanded `a3d_hemisoft` to 25 checkpoints (5×5 fold-seed grid) and
`a3d_crophistsoft` to 15 checkpoints (3×5). Master v9: 91 checkpoints, 923 MB.

**Key numbers:**
- Nested: 0.2267 (gate passed: +0.0036)
- Smoke: 0.2400
- Public: 0.2522 (slightly worse than Slot 5's 0.2473)
- Private: 0.3019 (slightly worse than Slot 5's 0.3003)

**The diminishing returns wall:** 20 extra checkpoints bought 0.0001 on the nested estimate
and actually *hurt* on the public and private leaderboards. More capacity in the same
representation family does not help when the representation itself is the bottleneck.

**Lesson earned:** *When marginal models are worth 0.0001, switch to verification.*

---

## 7. The Five Structural Failure Modes

### 7.1 The 3D Convolution Spatial Memorisation Trap

**Estimated private cost: +0.035 to +0.043 log loss**

75 of our 91 checkpoints were 3D ResNets. In SPECT imaging, raw reconstructed volumes have
non-uniform axial slice thicknesses (2.0–4.5 mm across vendors) and varied reconstruction
filters (FBP vs OSEM). Even after 2 mm affine resampling, isotropic $3\times3\times3$ kernels
memorised:

- Interpolation textures from z-axis resampling
- Scanner-specific noise correlation patterns between adjacent slices
- Vendor-specific partial-volume effects at the striatal boundary

**What the winners did instead:**
- `paul-bd` (Rank #6): Collapsed 3D volumes into **2D maximum and mean intensity projections**
  (axial, coronal, sagittal), fed into ImageNet-pretrained DenseNet-121
- `DataCouples` (Rank #10): Processed sequential 2D slices with a 2D backbone + bidirectional
  GRU (BiGRU)

Both approaches *structurally cannot learn z-axis interpolation artifacts* because the z-axis
is either projected out (MIP) or sequenced as independent frames (BiGRU).

> **Mental Model: The Colander**
>
> **Image:** A colander lets water (noise) through while keeping the pasta (signal).
> 2D projections are a colander for z-axis scanner noise. 3D convolutions are a bowl
> that keeps everything.
>
> **What it clarifies here:** 2D projections don't just "reduce dimensions" — they
> structurally eliminate an entire class of confound (inter-slice artifacts) that 3D
> convolutions are free to memorise.
>
> **Formal anchor:** Our 2D arms (`a2d_v2s`, `a2d_cnx_cor`) had a smaller
> smoke-to-public gap than our 3D arms. Tri-Planar 2.5D scored 0.2341 standalone
> smoke — stronger than any 3D model.
>
> **Where it breaks:** When the z-axis carries genuine diagnostic signal (e.g., 3D
> shape of a tumour). In DaT-SPECT, the diagnostic signal is a 2D pattern
> (bilateral striatal uptake ratio) that is invariant to z-axis resolution.
>
> **Shared by:** CT competitions with variable slice thickness, MRI competitions with
> variable acquisition protocols, video competitions where frame rate varies.

**Transferable rule:** Before investing in 3D convolutions on reconstructed medical volumes,
test whether a 2D multi-plane projection preserves the clinical signal. If it does, use 2D.
The 2D model can't overfit what it can't see.

### 7.2 Validation Leakage: Random CV vs LOCO

**Estimated private cost: +0.015 to +0.020 log loss**

Our `folds.csv` split 1,362 scans randomly. Because multiple scans came from the same
acquisition site and camera model, identical scanner noise profiles appeared in both
training and validation folds. This meant:

1. **Temperature selection:** Random CV rewarded $T=0.90$ (sharpening). LOCO would have
   rewarded $T \ge 1.05$ (softening). The validation design *determined* the calibration.

2. **Augmentation rejection:** Scanner blur augmentation lost 0.003 on random CV and was
   dropped. On LOCO CV, it would likely have *gained* because it simulates cross-scanner blur.

3. **Model selection:** The stacker preferred collinear 3D models over orthogonal 2D models
   because random CV couldn't distinguish their generalisation properties.

4. **Nested estimate optimism:** Our best nested estimate was 0.2266. Our best private score
   was 0.3003. The 0.0737 gap is the *price* of validation leakage.

**We knew this was a risk.** The EDA report (Sept 5) said: "Still worth a spacing/hospital-aware
stratified k-fold for an honest cross-center generalization estimate." We did site-fold
*analysis* on several arms but never made site-fold CV the *gating* criterion.

> **Mental Model: The Map and the Territory**
>
> **Image:** You have a map that shows a flat plain. You plan your route across the
> plain. When you arrive, there's a mountain range. The map was of a different territory.
>
> **What it clarifies here:** Random-fold CV is a map of the in-distribution territory.
> The private test set is a different territory (different scanners). Every decision
> we made using random-fold CV was a decision about the wrong territory.
>
> **Formal anchor:** Random-fold nested: 0.2266. Private reality: 0.3003. Gap: 0.0737.
> Site-fold B0: 0.2818 vs random-fold B0: 0.2752 — the site fold was 0.0066 harder,
> signalling the shift we ignored.
>
> **Where it breaks:** When the test distribution genuinely matches the training
> distribution (e.g., i.i.d. tabular data with no group structure). Random CV is
> correct for i.i.d. problems.
>
> **Shared by:** Any competition with group structure (hospitals, users, devices,
> time periods). Kaggle competitions with temporal test splits. Fraud detection
> with customer-level grouping.

**Transferable rule:** On day one, identify the groups in your data. Build two validation
schemes: random (for capacity decisions) and grouped (for robustness decisions). When they
disagree about a hyperparameter, the grouped one is right about regularisation.

### 7.3 The Borderline Band and Counterfactual Lesion Synthesis

**Estimated private cost: +0.010 to +0.015 log loss**

88 scans (6.46% of the pool) carried 46.9% of all log loss. We characterised this as
"irreducible error matching the 7.8% clinical discordance rate" and moved on.

**What the winners did:** They recognised that log loss penalises uncertainty in the borderline
band more severely than anywhere else. Instead of accepting the band as a wall, they
*manufactured training data for it*:

- Take a high-confidence normal scan ($y=0$, clear bilateral uptake)
- Apply localised non-linear intensity erosion to the putamen
- Apply 9 mm Gaussian blur to simulate resolution degradation
- Label the result as pathologic ($y=1$)
- This creates a synthetic "early disease" scan that populates the decision boundary

**Why we missed this:** We tested whether the errors were *consistent* across models (they
were — all three representations failed on the same scans). We concluded "irreducible."
But all three representations shared the same *preprocessing pipeline*. Consensus among models
that share a pipeline is evidence about the pipeline, not about the data.

> **Mental Model: The Consensus Trap**
>
> **Image:** Three witnesses agree on what they saw. Convincing — unless all three
> were standing in the same spot with the same obstructed view.
>
> **What it clarifies here:** Our 3D CNN, 2.5D CNN, and radiomic feature model all
> agreed that 88 scans were hard. But they all consumed the same registered,
> reference-normalised input. Their consensus was about the preprocessing
> limitations, not about fundamental label noise.
>
> **Formal anchor:** "Errors do not cluster by fold (p=0.838)" — but folds are
> random, so this test could never find a site effect. We tested the wrong grouping.
>
> **Where it breaks:** When models genuinely do use independent features (e.g.,
> text + image + audio multimodal models with separate preprocessing). True
> multimodal consensus is stronger evidence.
>
> **Shared by:** Multi-model agreement in ensemble methods, committee-based active
> learning, unanimous wrong predictions in neural network verification.

**Transferable rule:** When multiple models agree on being wrong, list what they share.
Test the shared components before concluding the data is at fault.

### 7.4 Scanner Resolution Augmentation Discarded

**Estimated private cost: +0.010 to +0.015 log loss**

In Run 5, resolution degradation augmentation (heavy Gaussian smoothing $\sigma \in [1.5, 3.0]$
plus Poisson noise injection, following Buddenkotte & Buchert 2024) was evaluated and dropped
because it showed a -0.003 penalty on random 5-fold CV.

**Why it appeared to hurt:** Random CV held similar high-resolution cameras in both train and
validation folds. Adding blur to training images just made the training distribution noisier
without any corresponding benefit on the validation side (which had the same cameras).

**Why it was actually critical:** The private test set contained older, lower-resolution
collimator systems. Without blur augmentation, the models had never seen degraded-resolution
inputs during training and were brittle to them.

**The causal chain:** Random CV → augmentation appears harmful → drop augmentation →
models are brittle to resolution shift → private score degrades.

This is the same causal chain as §7.2 (validation leakage), applied to a specific decision.
The root cause is the same: *the validation design determined the augmentation strategy, and
the validation design was wrong about shift*.

**Transferable rule:** If an augmentation simulates a realistic deployment perturbation, don't
discard it based on random CV alone. Test it on your shift-aware validation.

### 7.5 Calibration Overconfidence Under Domain Shift

**Estimated private cost: +0.008 to +0.012 log loss**

We applied $T=0.90$ (sharpening) and $\lambda=0.01$ (minimal prior regularisation) to minimise
log loss on the public leaderboard. This worked: public score improved from 0.2557 to 0.2473.

**The arithmetic of the disaster:**

On a scan where the model predicts $\hat{p}=0.95$ but $y=0$:
$$\mathcal{L} = -\ln(1 - 0.95) = -\ln(0.05) \approx 2.996 \text{ nats}$$

On the same scan at $\hat{p}=0.70$:
$$\mathcal{L} = -\ln(1 - 0.70) = -\ln(0.30) \approx 1.204 \text{ nats}$$

Difference: **1.79 nats per scan.** With even 5 such scans in a 2000-scan test set,
that's $5 \times 1.79 / 2000 = 0.0045$ additional log loss.

At $\lambda=0.01$, the maximum loss on any single scan is capped at:
$$-\ln(\lambda \cdot \pi) = -\ln(0.01 \times 0.548) \approx 5.21 \text{ nats}$$

At $\lambda=0$ (which our nested CV kept arguing for):
$$-\ln(\epsilon) \approx 9.21 \text{ nats}$$

The difference is 4 nats per confident mistake. On the private test set with unseen cameras,
a handful of overconfident false positives paid this penalty.

> **Mental Model: The Insurance Premium**
>
> **Image:** $\lambda$ is an insurance premium. On a sunny day (in-distribution), it
> looks like a waste — you're paying 0.002 nats for nothing. In a storm
> (domain shift), it prevents catastrophic loss. Random CV only shows you sunny days.
>
> **What it clarifies here:** Our nested CV argued monotonically for $\lambda \to 0$
> because it contained no shift. Every reduction in $\lambda$ bought 0.001–0.002
> nats on the nested estimate while increasing tail-risk exposure by 1–4 nats per
> confident mistake on shifted data.
>
> **Formal anchor:** $\lambda=0.02$ nested: 0.2323. $\lambda=0.01$ nested: 0.2303.
> Gain: 0.002. But $\lambda=0.02$ private: 0.3029. $\lambda=0.01$ private: 0.3003.
> Gain: 0.0026. The nested estimate was *correct about the direction but wrong about
> the magnitude*.
>
> **Where it breaks:** In competitions with no domain shift (i.i.d. data), $\lambda=0$
> is genuinely optimal and the insurance premium is pure cost.
>
> **Shared by:** Bayesian priors in small-sample settings, dropout rates, weight
> decay in transfer learning, ensemble diversity penalties.

**Transferable rule:** Calibration decisions made on in-distribution validation will always
recommend sharpening. If you expect any domain shift, soften ($T \ge 1.0$) and keep your
insurance ($\lambda > 0$).

### Summary: The Five Factors

| Factor | Private Cost | Root Cause | Transferable Principle |
|:---|:---:|:---|:---|
| 3D CNN memorisation | +0.035–0.043 | Representation captures confounds | Match representation to invariance |
| Random CV leakage | +0.015–0.020 | Validation doesn't reflect test shift | Group-aware validation for shift |
| No lesion synthesis | +0.010–0.015 | Treated borderline as irreducible | Populate the decision boundary |
| Blur augmentation dropped | +0.010–0.015 | Random CV penalised useful aug | Test augmentation on shifted val |
| Calibration overconfidence | +0.008–0.012 | Sharpening optimised on no-shift CV | Soften under expected shift |
| **Total estimated** | **+0.078–0.105** | | |

Our actual gap: Private 0.3003 vs Rank #1 Private 0.2532 = **0.0471**. The factors sum to
more than the gap because they overlap (Factors 2, 4, 5 are all downstream of the validation
design choice).

---

## 8. The Key Insight

**The single most important sentence in this document:**

> *The validation design is not a modelling choice — it is the lens through which every
> modelling choice is evaluated. If the lens is wrong, every decision it informs is wrong
> in the same direction.*

### Expanding the Insight

**The intuition:** Your validation set is not just a score. It is the *decision function* for
every hyperparameter, every augmentation, every model selection, every calibration constant.
If it doesn't contain the shift your test set will have, then every decision it makes will
optimise for in-distribution performance at the expense of robustness.

**The formal content:** Let $\mathcal{V}$ be your validation distribution and $\mathcal{T}$ be
the test distribution. If $\mathcal{V} \neq \mathcal{T}$, then for any decision $d$ optimised
on $\mathcal{V}$:

$$d^* = \arg\min_d \mathbb{E}_{\mathcal{V}}[\mathcal{L}(d)] \neq \arg\min_d \mathbb{E}_{\mathcal{T}}[\mathcal{L}(d)]$$

And the gap grows with the divergence $D(\mathcal{V} \| \mathcal{T})$. In our case, $\mathcal{V}$
(random folds) had near-zero site shift, while $\mathcal{T}$ (private test) had substantial
camera shift. Every downstream decision was biased toward sharpness and against robustness.

**Why this insight is easy to miss:** Because random CV works *fine* for most decisions. Model
architecture, learning rate, batch size — these are relatively robust to the validation design.
The dangerous decisions are the ones at the *boundary* between in-distribution and out-of-distribution
performance: calibration temperature, augmentation intensity, regularisation strength, and
ensemble selection. These are exactly the decisions where random CV and LOCO CV disagree.

**A wrong but tempting mental model:** "My validation score is 0.2266 and my public score is
0.2473, so the gap is ~0.02 and I'm doing fine." The 0.02 gap on the public test set (which
has partial overlap with training cameras) dramatically understates the gap on the private test
set (which has unseen cameras). The private gap was 0.0737. The public gap was a *lower bound*,
not an estimate.

> **Mental Model: The Cascade**
>
> **Image:** A single wrong assumption at the top of a waterfall carries every
> downstream decision over the edge. The waterfall here is: Validation Design →
> Temperature → λ → Augmentation → Model Selection → Private Score.
>
> **What it clarifies here:** Factors 2, 4, and 5 are not independent failures.
> They are all consequences of Factor 2 (random CV). Fix the validation, and the
> downstream decisions self-correct.
>
> **Formal anchor:** If we had used LOCO CV: temperature would have been ≥1.0,
> blur augmentation would have been kept, and the stacker would have favoured
> 2D representations. Three of five factors collapse into one.
>
> **Where it breaks:** Some failures are genuinely independent (e.g., Factor 1,
> the 3D architecture choice, was made before validation results existed).
>
> **Shared by:** Bayesian decision theory (prior → posterior → decisions),
> compiler optimisation (wrong cost model → wrong optimisation choices),
> clinical trial design (wrong endpoint → wrong drug selection).

---

## 9. Mental Models for Competition ML

Here is a consolidated reference of the 9 mental models surfaced in this retrospective, each
with transfer examples outside this competition:

| # | Mental Model | One-Line Summary | Transfer Domains |
|:---:|:---|:---|:---|
| 1 | The Iceberg | Preprocessing is 3-5× larger than modelling in signal-heavy domains | Medical imaging, audio, satellite, sensor |
| 2 | The Foundation | Input normalisation gains > all model gains combined | Any heterogeneous-input competition |
| 3 | Optimiser's Blindspot | The optimiser can only see what the validation shows it | Feature selection, NAS, AutoML |
| 4 | The Colander | 2D projections structurally eliminate z-axis confounds | 3D medical, video with variable frame rates |
| 5 | The Map and the Territory | Random CV is a map of the wrong territory when shift exists | Any grouped/temporal data |
| 6 | The Consensus Trap | Models sharing a pipeline agree about the pipeline, not the data | Ensembles, active learning, multimodal |
| 7 | The Insurance Premium | Regularisation is insurance; its value is invisible in-distribution | Bayesian priors, dropout, weight decay |
| 8 | The Cascade | One wrong assumption cascades into every downstream decision | Any sequential decision process |
| 9 | The Diminishing Wall | When marginal models buy 0.0001, switch to verification | Late-stage competition strategy |

---

## 10. Worked Examples: Decisions That Mattered

### Example 1: The Temperature Decision

**Setup:** After Run 8, we had a 7-arm ensemble with logits. We needed to pick $T$.

**Goal:** Minimise expected private log loss.

**What we did:**
1. Computed random-fold nested log loss for $T \in \{0.80, 0.85, 0.90, 0.95, 1.00, 1.05\}$
2. Minimum at $T=0.90$ (nested 0.2289)
3. Submitted with $T=0.90$

**What we should have done:**
1. Computed both random-fold AND site-fold nested log loss
2. Random-fold minimum: $T=0.90$
3. Site-fold minimum: likely $T \ge 1.05$
4. Used site-fold for the robustness decision: $T = 1.05$

**Counterfactual:**
- At $T=1.05$, our public score would have been ~0.260 (worse by ~0.013)
- At $T=1.05$, our private score would have been ~0.290 (better by ~0.010)
- Net: worse public rank, better private rank (probably ~#40 instead of #50)

**Common mistake:** Optimising the public leaderboard. The public board has partial overlap
with training cameras. The private board is the one that counts.

### Example 2: The Augmentation Decision

**Setup:** Run 5 tested scanner blur augmentation ($\sigma \in [1.5, 3.0]$).

**Goal:** Decide whether to include blur augmentation in the final pipeline.

**What we did:**
1. Trained with and without blur on random 5-fold CV
2. Without blur: OOF 0.2568. With blur: OOF 0.2598
3. Blur lost by 0.003. Gate: failed. Decision: drop blur.

**What we should have done:**
1. Also evaluated on site-fold CV
2. Site-fold with blur would likely have shown improvement (blur simulates the exact shift
   that site folds measure)
3. When random-fold and site-fold disagree about an augmentation, site-fold wins on robustness

**Sanity check:** The blur augmentation was *designed* to simulate the exact degradation that
unseen scanners would produce. Dropping it because random CV (which has no such degradation)
showed a loss is like testing an umbrella indoors and concluding it's useless.

### Example 3: The λ Decision

**Setup:** Slot 1 used $\lambda=0.02$. We had one more slot.

**Goal:** Decide the bracket variant for Slot 2.

**What we did:**
1. Computed nested log loss for $\lambda \in \{0.00, 0.005, 0.01, 0.015, 0.02\}$
2. Monotone decreasing: lower $\lambda$ = better nested score
3. Submitted with $\lambda=0.01$. Public improved from 0.2557 to 0.2473.

**The arithmetic sanity check we did (correctly):**
- Maximum possible effect of $\lambda = 0.02 \to 0.01$: **0.0055** per scan
- Realistic effect on the pool: **0.0020**
- Observed public improvement: **0.0084**
- 0.0084 > 0.0055 (the hard ceiling). Therefore the gain was *not* from $\lambda$.

**What happened:** Despite correctly identifying the arithmetic impossibility, the 0.0084
improvement was real and was submitted. The actual cause was likely the wall-clock inference
ladder dropping arms under time pressure, with `combine()` renormalising over survivors.
We benefited from a different configuration but attributed it to the wrong cause.

**Lesson:** *Even when you do the sanity check, you can still act on the wrong conclusion
if time pressure prevents investigating the true cause.*

### Example 4: The Ensemble Composition Decision

**Setup:** We had both HemiNet3D (3D bilateral) and Tri-Planar (2.5D multi-view) architectures.

**Goal:** Select the optimal subset for the final ensemble.

**What we did:**
1. Ran greedy forward selection on random-fold nested CV
2. HemiNet3D: 50% weight. Tri-Planar: 0% weight.
3. Submitted 75/91 3D checkpoints, 16/91 2D checkpoints.

**What the smoke tests told us (which we ignored):**
- Tri-Planar standalone smoke: 0.2341 (strongest individual)
- HemiNet3D standalone smoke: 0.2706 (weaker individual)
- Tri-Planar generalised better to unseen data by 0.0365

**The lesson:** The smoke test (20 unseen scans) was a better predictor of generalisation
than random-fold nested CV (1,362 scans with leaked site signatures). **Small honest data >
large leaked data.**

**Common mistake:** Trusting sample size over data quality. 1,362 samples with leaked site
signatures is less informative than 20 clean samples for the specific question of
cross-scanner generalisation.

### Example 5: The Submission Timing Decision

**Setup:** Our last submission slot opened Sept 14 at 03:38 UTC. Deadline: Sept 16 23:59 UTC.

**Goal:** Maximise the value of the final slot.

**What we did:** Submitted Master v9 (91 checkpoints) at 03:38, the moment the slot opened.

**What we should have done:** Waited. The slot was live until the deadline (60 hours away).
Submitting immediately forfeited 60 hours of potential training/analysis time on a configuration
that was measurably worse (nested 0.2267 vs banked 0.2266). The marginal ensemble was
worth 0.0001.

**Lesson:** *Draw the submission timeline before the last week.* Free evaluation runs, slot
windows, rolling quotas — understand the economics before time pressure makes you spend slots
compulsively.

---

## 11. Non-Examples and Boundary Cases

### 11.1 When 3D Convolutions ARE the Right Choice

In competitions with genuine 3D diagnostic signal (e.g., tumour volume estimation, organ
segmentation where 3D shape matters), 2D projections lose critical information. The "colander"
model applies *only when the signal is 2D and the noise is 3D*. In DaT-SPECT, the diagnostic
pattern (bilateral striatal uptake asymmetry) is fundamentally a 2D spatial pattern.

### 11.2 When Random CV IS Sufficient

If there is no group structure in the data (e.g., i.i.d. tabular data, web-scraped images
with no source grouping, text classification with no author/source clustering), random CV
is the correct validation design. The "map and territory" model applies only when the test
distribution involves a *structured* shift along a grouping variable.

### 11.3 When Sharpening IS Correct

If the test distribution matches the training distribution and you have high discrimination
(AUROC > 0.95), then $T < 1.0$ is genuinely optimal. The "insurance premium" model applies
only under domain shift. On well-calibrated in-distribution problems, regularisation is
pure cost.

### 11.4 When "Irreducible Error" IS Real

Not all hard cases are preprocessing artifacts. If models with genuinely independent inputs
(different sensors, different features, no shared preprocessing) agree on errors, that is
stronger evidence of irreducible noise. Our mistake was treating consensus among *dependent*
models as evidence of *independent* agreement.

---

## 12. Confusion Clinic

**Q1: "We had AUROC 0.9406 — competitive with the top 10. How did we still finish #50?"**

AUROC measures ranking quality. Log loss measures calibration quality. You can rank every
patient correctly and still lose on log loss if your probabilities are too confident on the
ones you're wrong about. Our ranking was competitive; our calibration under shift was not.

**Q2: "The nested estimate was 0.2266 and the private score was 0.3003. Is nested CV just
unreliable?"**

No — nested CV is reliable *for the distribution it measures*. Our nested CV measured
in-distribution performance. The private test had distribution shift. The 0.0737 gap is
not nested CV being unreliable; it's nested CV on the wrong distribution being *accurately
irrelevant* to the actual question.

**Q3: "We tested site-fold CV on several arms. Why didn't it save us?"**

We measured site-fold performance but never *gated* on it. Site-fold B0 was 0.2818 vs
random-fold 0.2752 — the 0.0066 gap was a warning signal. But all gating decisions
(temperature, λ, augmentation, model selection) used random-fold nested CV. The site-fold
measurement was an observation, not a decision input.

**Q4: "Should we always use LOCO CV?"**

Use both. Random CV gives you the capacity ceiling (how good can this model be on the best
day?). LOCO CV gives you the robustness floor (how bad can it be on the worst day?). Make
capacity decisions on random CV. Make robustness decisions on LOCO. Where they disagree,
the competition winner is the one who listened to LOCO.

**Q5: "We dropped scanner blur augmentation because it hurt by 0.003. Was 0.003 significant?"**

Bootstrap noise on the nested estimate was ±0.013. The 0.003 loss was well within noise.
It was not significant, and it should not have cleared the decision gate. The correct decision
rule was: "Drop only if it loses by more than the bootstrap noise." We didn't have that rule.

**Q6: "More checkpoints (91 vs 71) actually hurt. Why?"**

The additional 20 checkpoints were all 3D ResNets (the architecture that memorises scanner
artifacts). Adding more capacity in the wrong representation family amplifies the bias
rather than diversifying it. Ensemble diversity comes from *different representations*, not
*more copies of the same representation*.

**Q7: "The Tri-Planar model scored 0.2341 standalone but got 0% stack weight. Is the stacker
broken?"**

No — the stacker is working correctly *on the data it sees*. On random-fold OOF, Tri-Planar
and HemiNet3D are collinear (they capture similar signal in-distribution). The stacker drops
the redundant one. But out-of-distribution, Tri-Planar generalises better because it doesn't
memorise z-axis artifacts. The stacker can't see out-of-distribution performance.

**Q8: "We spent the final submission slot immediately. Was there ever a reason to wait?"**

Yes. The slot was live until the deadline (60 hours away). Submitting immediately forfeits
any learning from those 60 hours. The option value of waiting is strictly positive when:
(a) you have training capacity left, (b) the banked score is competitive, and (c) the
new submission is not materially better. All three applied. We should have waited.

**Q9: "Is λ=0 ever correct?"**

Only if you have zero domain shift AND your model is perfectly calibrated AND you're
willing to accept catastrophic loss on any single confident error. In practice, the
expected cost of $\lambda > 0$ is ~0.002 nats, and the expected benefit on a single
confident mistake is 4 nats. The insurance is almost always worth it.

**Q10: "We did compute the arithmetic ceiling for the λ gain and found it impossible.
We submitted anyway. Why?"**

Time pressure. The calculation showed the +0.0084 public gain couldn't come from λ,
but the gain was real and we needed to submit before the slot expired. The correct response
was: "The gain is real but the cause is unknown. Submit (take the gain) but record that the
cause is unresolved." We did this, but we also *internalised* "lower λ is better" as a
mental model, which was the wrong conclusion.

**Q11: "Our smoke test scores had a spread of 0.0087 on identical zips. Are smoke tests
useful at all?"**

Yes, but only for *container validation* (does it run? does it use all checkpoints? does
it produce the right format?). They are not useful for *score estimation* with n=20. The
standard error of log loss on 20 samples is enormous. Use smoke tests for pass/fail, not
for numerical optimisation.

**Q12: "What's the single highest-value thing we could have done differently?"**

Implemented LOCO cross-validation on Day 1. This single change would have:
- Correct the temperature decision (T≥1.05 instead of T=0.90)
- Retain scanner blur augmentation
- Favour 2D representations in the stack
- Produce a realistic estimate of the private score (~0.27 instead of 0.23)
- Estimated cost: 2 hours of engineering. Estimated benefit: ~0.02 private log loss.

---

## 13. Reconstruction Checklist: The Playbook

Print this. Tape it to your monitor. Follow it in order.

### Day 1: The Foundation

- [ ] **Submit a working pipeline.** Any model, any score. You now have a baseline and a
  container. (Our Run 1: 0.3391, Rank #152 — but it worked.)
- [ ] **Read the rules.** Write down: submission limits, time windows, rolling quotas,
  free evaluation runs, file size limits, runtime limits, scoring metric. Put it in a file.
- [ ] **Identify the groups.** What varies between training and test that isn't the labels?
  Hospitals? Devices? Time periods? Users? Write it down.
- [ ] **Build two validation schemes.** Random k-fold for capacity. Group k-fold (LOCO) for
  robustness. Compare them immediately. If they disagree by >0.01, you have a shift problem.
- [ ] **Normalise the inputs.** Registration > architecture. Standardisation > hyperparameter
  tuning. Fix the foundation before decorating the building.

### Week 1: The Members

- [ ] **Build 3-5 genuinely different representations.** Different input types (2D projections,
  3D volumes, hand-crafted features), different backbones, different views. Diversity > depth.
- [ ] **Log everything.** Run number, configuration, date, OOF score, validation type, submission
  ID. If a number doesn't have a provenance, it's a rumour.
- [ ] **Test augmentations on LOCO.** If an augmentation simulates a realistic deployment
  perturbation (blur, noise, resolution degradation), don't reject it based on random CV alone.
- [ ] **Pre-register decision rules.** Before each run, write: "This ships only if it beats X
  by Y on validation type Z." This prevents post-hoc rationalisation.
- [ ] **Check arithmetic ceilings.** Before believing a result, compute the maximum effect the
  proposed cause could have. If the observed gain exceeds the ceiling, the cause is something else.

### The Endgame: Verification > Modelling

- [ ] **Freeze early.** Reserve the last 48 hours for packaging, verification, and submission.
- [ ] **Use equal weights.** Fitted ensemble weights score worse than equal weights in 9/10 cases
  when you have <10 members.
- [ ] **Calibrate conservatively.** Under expected shift: $T \ge 1.0$, $\lambda > 0$. The nested
  estimate will argue against this. The nested estimate is wrong about shift.
- [ ] **Verify the container.** Run the exact zip you'll submit through the exact container
  environment. Check that every model arm loads, every checkpoint is used, the output format is
  correct, and the score matches your local estimate.
- [ ] **Draw the timeline.** When does each slot open? When is the deadline? What's the option
  value of waiting? Don't spend a slot the moment it opens unless the new submission is
  materially better than what's banked.
- [ ] **Check for silent degradation.** Ask of every verification step: "What would this fail to
  notice?" A gate that can't see a dead component will ship one.

### Exercises

**Exercise 1:** You have LOCO CV giving log loss 0.28 and random CV giving 0.24. Your temperature
sweep shows $T=0.85$ is best on random, $T=1.10$ is best on LOCO. You have one submission left.
Which $T$ do you use?

*Answer:* $T=1.10$. The competition's private test set will have unseen groups (like LOCO),
not i.i.d. data (like random CV). The LOCO-optimal $T$ will produce a worse public score but
a better private score.

**Exercise 2:** Your ensemble has 5 members. Three are 3D CNNs, one is a 2D projection model,
one is a radiomics model. The stacker assigns 0% weight to both the 2D and radiomics models.
Standalone smoke tests show the 2D model generalises best. What do you do?

*Answer:* Force equal weights across representation types (e.g., 1/3 each to {3D pool, 2D model,
radiomics}). The stacker is optimising on in-distribution data where the 3D models dominate. The
smoke test is a better signal of out-of-distribution performance. When the stacker and the smoke
test disagree, the smoke test is right about generalisation.

**Exercise 3:** You find that 88/1362 scans carry 47% of your log loss, and all three of your
models are wrong on the same 88 scans. Is this irreducible?

*Answer:* Not necessarily. Check what the three models share. If they share preprocessing
(registration, normalisation, cropping), their consensus is about the preprocessing, not the
labels. Test with a model that uses a completely different input (e.g., raw unregistered scans
with a different normalisation). If it also fails on the same 88, the evidence for irreducibility
is stronger.

**Explain-it-back prompt:** Explain to a friend why a model with AUROC 0.94 can finish at
Rank #50 while a model with AUROC 0.95 finishes at Rank #6.

---

## 14. Transferable Laws Beyond This Competition

### Law 1: The Validation Cascade Law

*Every downstream decision inherits the assumptions of the validation design. If the validation
doesn't contain the shift, every decision it informs will optimise for in-distribution performance
at the expense of robustness.*

**Application:** Before tuning any hyperparameter, ask: "Does my validation set contain the type
of variation I expect in the test set?" If not, build one that does.

### Law 2: The Representation Invariance Law

*The representation should be structurally unable to capture the confound. Augmentation-based
robustness ("I trained with random blur") is weaker than architectural robustness ("I projected
out the axis where blur lives").*

**Application:** Before building a model, list the confounds (scanner noise, resolution, device
type, writing style, accent). For each confound, ask: "Can my architecture see this?" If yes,
either project it out or augment overwhelmingly.

### Law 3: The Insurance Law

*Regularisation that costs 0.002 in-distribution and saves 4.0 out-of-distribution is always
worth it unless you can prove there is no shift.*

**Application:** Default to conservative calibration ($T \ge 1.0$, $\lambda > 0$). The in-distribution
cost is tiny and measurable. The out-of-distribution benefit is large and invisible until the
private leaderboard.

### Law 4: The Consensus Dependency Law

*Consensus among models that share a pipeline stage is evidence about the pipeline, not
about the data. True consensus requires independent inputs.*

**Application:** When multiple models agree on errors, list their shared components.
Test the shared components before concluding "irreducible."

### Law 5: The Diminishing Returns Threshold

*When the marginal model is worth 0.0001, verification has a higher expected value. Spend the
last 20% of time verifying, not modelling.*

**Application:** Track the marginal improvement of each new model. When it drops below the
bootstrap noise of your estimate, freeze and verify. Our last 20 checkpoints bought 0.0001
and actually hurt on the private leaderboard.

### Law 6: The Arithmetic Ceiling Law

*Before believing any result, compute the maximum effect the proposed cause could produce.
A gain that exceeds its own ceiling is telling you the cause is something else.*

**Application:** Temperature, blending weights, prior mixing, label smoothing — all have
closed-form bounds on their maximum effect. Compute these before acting on surprising results.

### Law 7: The Timeline Law

*Submission economics (slots, windows, quotas, free runs) have option value. Don't spend
resources compulsively. Draw the timeline on Day 1.*

**Application:** On Day 1, write the submission timeline in a file. On the last day, re-read it.
A slot that is live for 60 hours should not be spent at hour 0 on a 0.0001 improvement.

### Law 8: The Provenance Law

*A number is data only if you can trace its origin. Every score needs a submission ID,
a timestamp, and a configuration. If you can't name where it came from, it's a rumour.*

**Application:** Log every run with its full provenance *at the time of the run*, not after.
Provenance discipline collapses under deadline pressure — precisely when its value peaks.

### Law 9: The Small Honest Data Law

*Small honest data (20 samples with no leakage) is more informative than large leaked data
(1362 samples with site leakage) for the specific question of generalisation.*

**Application:** When your smoke test and your cross-validation disagree about which model
generalises better, the smoke test is probably right — despite having 68× fewer samples.
What matters is whether the samples reflect the real test distribution.

---

## 15. What I Would Do Differently With a Time Machine

### Day 1 (Sept 5): +2 hours of engineering

1. **Build LOCO cross-validation immediately.** Cluster scans by `(matrix_size, voxel_spacing,
   background_noise_level)`. Create site-fold splits. Run the EDA feature test on both random
   and site folds. This takes 2 hours and changes every subsequent decision.

2. **Test 2D projections on Day 1.** Compute maximum intensity projections (axial, coronal,
   sagittal) and train DenseNet-121 on them. This takes 1 GPU-hour and was the #6 solution.
   We didn't try 2D projections until Run 6b (Day 5), and even then we let the stacker exclude
   them.

### Day 3 (Sept 7): Changed decisions

3. **Keep scanner blur augmentation.** When random CV and site-fold CV disagree about blur,
   listen to site-fold CV. The 0.003 random-CV loss is noise; the generalisation benefit is real.

4. **Force 2D representation into the stack.** Don't let the stacker optimise representation
   weights — force equal weight across representation *types* (3D, 2.5D, 2D, features).

### Day 7 (Sept 11): Calibration

5. **Use $T = 1.05$ and $\lambda = 0.02$.** Accept the ~0.01 public score penalty. The private
   score improvement would be larger.

### Day 9 (Sept 13): Endgame

6. **Don't spend the last slot immediately.** Wait until 12 hours before deadline. Use the
   time to verify the container, re-check provenance, and run one last LOCO evaluation.

### The Single Highest-Leverage Change

**LOCO cross-validation on Day 1.** This single 2-hour investment cascades into correct decisions
on temperature, augmentation, representation selection, and calibration. Estimated private score
improvement: ~0.02–0.03. Estimated rank improvement: #50 → #30–35.

The second highest-leverage change is **2D projections on Day 1**, which directly addresses
Factor 1 (the largest factor at +0.035–0.043). Combined with LOCO, the estimated private score
reaches ~0.27, putting us in the top 10.

---

## Closing

The meta-lesson underneath everything in this document is not about medical imaging, 3D
convolutions, or temperature scaling. It's about **decision quality under uncertainty.**

Every competition is a sequence of decisions made with imperfect information. The decisions
that matter most are not the ones about model architecture (those are recoverable). The
decisions that matter most are the ones about **how you evaluate your models** — because
those determine every subsequent decision.

Build two validation schemes. Listen to the one that contains the shift. Verify the container.
Log every number. Check every ceiling. And when the marginal model is worth 0.0001, stop
modelling and start verifying.

> **The failure mode of a competent team is not bad modelling. It is believing a number
> nobody checked.**

---

*Generated 2026-09-17 from the full project record: RESULTS_LOG.md, LESSONS.md, EDA_REPORT.md,
aidlc-docs/efforts/009-final-evaluation-and-postmortem/effort-state.md, and
aidlc-docs/HANDOFF.md. Quantitative evidence sourced from 6 scored submissions, 20 smoke tests,
and ~30 GPU-hours of training across 9 runs.*
