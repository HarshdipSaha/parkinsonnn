# Project Completion & Final Competition Handoff

| | |
|---|---|
| **Status** | **COMPLETED / CLOSED** (Sept 17, 2026) |
| **Competition** | DrivenData Challenge #311 — DaT Parkinson's Challenge |
| **Account** | `harshdipsaha` |
| **Final Platform Result** | **Rank #50** on Private Leaderboard |
| **Best Private Score** | **0.3003 Log Loss** (AUROC: **0.9406**) via Slot 5 (`id-322091`) |
| **Best Public Score** | **0.2473 Log Loss** (AUROC: **0.9608**, Rank ~#16) via Slot 5 (`id-322091`) |
| **Final Slot 6 Submission** | **0.2522 Public / 0.3019 Private** via `id-322254` (Master v9, 91 ckpts) |
| **Submissions Used** | **6 of 6 normal slots** (all evaluated and scored); **20 smoke tests** (14 evaluated, 6 canceled) |
| **AI-DLC Lifecycle** | 9 Efforts completed (`001` through `009`); fully audited and reproducible |

---

## 1. Executive Summary & Final Standing

The DaT Parkinson's Challenge concluded on **September 16, 2026, 23:59 UTC**. Upon evaluation of the complete hidden test set by DrivenData, the leaderboard experienced a significant upward domain shift across all participants (+0.04 to +0.06 log loss), driven by unseen hospital camera systems and collimator variations.

- **Leaderboard Anchors**:
  - Rank #1: **0.2532** Private
  - Rank #6 (`paul-bd`): **0.2732** Private
  - Rank #10 (`DataCouples`): **0.2760** Private
  - **Rank #50 (`harshdipsaha`)**: **0.3003** Private (AUROC: **0.9406**)
- **Key Validation Observation**:
  - Our ranking discrimination was strong (**AUROC 0.9406** vs Rank #6's 0.9489 and Rank #10's 0.9551).
  - The drop from public rank (~#16 at 0.2473) to private rank (#50 at 0.3003) was driven primarily by **calibrated log loss tail risk** under out-of-domain scanner shifts rather than ranking errors.

---

## 2. Official DrivenData Submission Ledger

### Scored Normal Submissions (6/6 Completed)

| Slot | ID | Date (UTC) | Public Log Loss | Private Log Loss | AUROC | Model / Package | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|
| 1 | `id-321156` | Sept 5 15:38 | **0.3391** | **0.3924** | 0.9082 | Run 2 (15 ckpts) | Fixed-anchor dual-crop CNN + Ridge LR |
| 2 | `id-321263` | Sept 6 15:10 | **0.2817** | **0.3217** | 0.9477 | Run 3 (44 ckpts) | Registered-frame multi-backbone stack |
| 3 | `id-321298` | Sept 7 03:38 | **0.2667** | **0.3099** | 0.9536 | Run 4a (30 ckpts) | Striatal-crop 3D + V2-S + ConvNeXt-T |
| 4 | `id-321967` | Sept 12 01:40 | **0.2557** | **0.3029** | 0.9576 | Master v3 (71 ckpts) | 7-arm bilateral ensemble ($T=0.90, \lambda=0.02$) |
| 5 | `id-322091` | Sept 13 03:08 | **0.2473** | **0.3003** | **0.9608** | Master v3 (71 ckpts) | **Team Best Private Score (Rank #50)** ($\lambda=0.01$) |
| 6 | `id-322254` | Sept 14 05:10 | **0.2522** | **0.3019** | 0.9582 | Master v9 (91 ckpts) | 923 MB expanded seed ensemble ($\lambda=0.01$) |

### Smoke Test Submissions (20 Total)

| ID | Status | Score | Description / Package |
|:---:|:---:|:---:|:---|
| `id-321144` | Completed | 0.6687 | Prior-only baseline container smoke test |
| `id-321152` | Completed | 0.2567 | Run 2 dual-crop CNN ensemble smoke |
| `id-321261` | Completed | 0.2397 | Run 3 registered-frame container smoke |
| `id-321293` | Completed | 0.2176 | Run 4a 30-ckpt ensemble smoke |
| `id-321297` | Completed | 0.2263 | Run 4a replicate smoke verification |
| `id-321496` | Completed | 0.2388 | Run 5 site-invariance container smoke |
| `id-321557` | Canceled | — | Canceled by submitter |
| `id-321558` | Canceled | — | Canceled by submitter |
| `id-321569` | Canceled | — | Canceled by submitter |
| `id-321570` | Canceled | — | Canceled by submitter |
| `id-321572` | Completed | 0.2706 | Standalone HemiNet3D arm smoke test |
| `id-321576` | Canceled | — | Canceled by submitter |
| `id-321610` | Completed | 0.2341 | Standalone Tri-Planar 2.5D arm smoke test |
| `id-321616` | Canceled | — | Canceled by submitter |
| `id-321617` | Completed | 0.2363 | Master Final 5-pillar unified stack smoke |
| `id-321786` | Completed | 0.2381 | Master v2 equal-weight 5-pillar smoke |
| `id-321850` | Completed | 0.2374 | Master v3 71-ckpt ensemble smoke |
| `id-321965` | Completed | 0.2558 | Master v3 parameter variant smoke |
| `id-322198` | Completed | 0.2400 | Master v9 91-ckpt package smoke test |
| `id-322199` | Completed | 0.2459 | Master v9 confirmation smoke test |

---

## 3. Structural Post-Mortem & Diagnostic Dissection

Why did our solution finish at Rank #50 (0.3003) while the top ten reached 0.2532–0.2760?
A deep analysis against public post-competition solutions (e.g., Rank #6 `paul-bd`, Rank #10 `DataCouples`) reveals five critical root causes:

```
                            [THE PERFORMANCE GAP]
                 Public 0.2473 (Rank #16) ──> Private 0.3003 (Rank #50)
                                      │
       ┌──────────────────────────────┼──────────────────────────────┐
       ▼                              ▼                              ▼
 [REPRESENTATION]               [VALIDATION]                   [CALIBRATION]
 • 3D Convolutions memorized    • Random 5-fold CV leaked      • Logit sharpening (T=0.9)
   scanner z-slice artifacts      site signatures                blew up on private tail
 • Top teams used 2D multi-     • Top teams validated with     • Insurance (λ) reduced
   plane projections / BiGRUs     LOCO (site clustering)         too aggressively
```

1. **The 3D Convolution Spatial Memorization Trap**:
   - Master v9 had 75 of 91 checkpoints based on 3D ResNets.
   - 3D convolutions with isotropic kernels operate across reconstructed voxels whose native z-slice spacing and interpolation filters vary across camera vendors (GE, Siemens, Philips).
   - In cross-scanner evaluation, 3D convolutions suffered a $+0.035$ to $+0.043$ penalty.
   - Top competitors (`paul-bd` Rank #6) used **2D multi-plane projections** (axial/coronal/sagittal MIPs into DenseNet-121) or **2D slice-sequence BiGRUs** (`DataCouples` Rank #10), naturally collapsing out scanner-specific z-axis artifacts.
2. **Validation Leakage (Random 5-fold CV vs LOCO)**:
   - Our CV splits (`folds.csv`) distributed scans randomly across folds.
   - Scans from the same hospital camera cluster were present in both train and validation folds, leaking scanner-specific features and producing overly optimistic nested scores ($0.2266$).
   - Top teams used **Leave-One-Cluster-Out (LOCO)** CV, which accurately simulated test-set camera transfer.
3. **The Borderline Band & Counterfactual Lesion Synthesis**:
   - 50% of the competition log loss lived in borderline early-disease cases.
   - While we characterized this as an irreducible error boundary (~7.8% clinical read ambiguity), top competitors actively populated this region using **counterfactual lesion synthesis** (denervating normal scans via 9 mm Gaussian blur to generate artificial borderline $y=1$ cases).
4. **Discarding Scanner Resolution Augmentation**:
   - In Run 5, strong scanner blur/noise augmentation was tested but dropped because it lowered random CV by $-0.003$.
   - On the private test set, which featured older, lower-resolution collimators, that blur was the vital invariant.
5. **Calibration Overconfidence Under Domain Shift**:
   - Sharpening temperature ($T=0.90$) and shrinking prior regularization ($\lambda=0.01$) improved public LB ($0.2473$), but under domain shift on private hospital cameras, confident misses were exponentially penalized by log loss.

---

## 4. AI-DLC Effort Registry Summary

All 9 efforts are complete and documented in `aidlc-docs/efforts/`:

| Effort | Focus | State | Outcome |
|:---:|:---|:---:|:---|
| `001` | [run2-hotspot-pipeline](efforts/001-run2-hotspot-pipeline/effort-state.md) | complete | Baseline CNN + Ridge; Public 0.3391, Private 0.3924 (`id-321156`) |
| `002` | [run3-registered-ensemble](efforts/002-run3-registered-ensemble/effort-state.md) | complete | Registered frame stack; Public 0.2817, Private 0.3217 (`id-321263`) |
| `003` | [run4-toward-rank1](efforts/003-run4-toward-rank1/effort-state.md) | complete | Multi-backbone 3D+2D; Public 0.2667, Private 0.3099 (`id-321298`) |
| `004` | [run5-site-invariance](efforts/004-run5-site-invariance/effort-state.md) | complete | Site invariance & crophist; smoke test `id-321496` (0.2388) |
| `005` | [run6-parallel-approaches](efforts/005-run6-parallel-approaches/effort-state.md) | complete | Tri-Planar 2.5D (smoke 0.2341) & VICReg SSL (smoke 0.2289) |
| `006` | [run7-hemisphere-superstack](efforts/006-run7-hemisphere-superstack/effort-state.md) | complete | HemiNet3D bilateral Siamese; smoke `id-321572` (0.2706) |
| `007` | [run8-distill-master-v3](efforts/007-run8-distill-master-v3/effort-state.md) | complete | Distillation & Master v3; Public 0.2473, Private 0.3003 (`id-322091`) |
| `008` | [run9-endgame](efforts/008-run9-endgame/effort-state.md) | complete | Master v9 (91 ckpts); Public 0.2522, Private 0.3019 (`id-322254`) |
| `009` | [final-evaluation-and-postmortem](efforts/009-final-evaluation-and-postmortem/effort-state.md) | complete | Official Private LB Rank #50; post-mortem & ledger closure |

---

## 5. Archival & Deliverables

- **Codebase**: Fully self-contained inference packages (`scripts/build_master_v3.py`, `scripts/local_e2e_v3.py`).
- **Tests**: All 94 unit/regression tests passing (`pytest tests/test_run9.py`).
- **Compliance**: `docs/DISCLOSURE.md` drafted and approved (fulfilling ticket T12). Zero external data, BSD-3 torchvision weights only, per-scan deterministic inference.
