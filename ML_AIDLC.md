# ML-AIDLC: AI-Driven Lifecycle for Machine Learning
## Operational Protocol for Autonomous Agents & Human-AI Pairing in Competitive and Scientific Data Science

> **The Core Problem Identified in DrivenData Challenge #311:**  
> On September 6, 2026, an autonomous research agent wrote `docs/RESEARCH_SIMILAR_COMPETITIONS.md`, retrieving the exact winning recipes from past literature and Kaggle/DrivenData champions:
> 1. *Use 2D multi-plane/slab CNNs rather than pure 3D CNNs* (Lesson 1 & 3)
> 2. *Use aggressive scanner blur and resolution noise augmentation to survive camera shift* (Lesson 2)
> 3. *Never trust random CV or public LB; validate with Leave-One-Cluster-Out (LOCO)* (Lesson 5 & 6)
> 
> Eleven days later, our final submission (Master v9, Rank #50) contained:
> - **75 of 91 checkpoints in 3D ResNets** (memorizing scanner z-axis interpolation artifacts)
> - **Scanner blur augmentation discarded** because random 5-fold CV showed a `-0.003` dip
> - **Validation gated strictly on random-fold nested CV**, leaking hospital signatures
> - **Tri-Planar 2.5D assigned 0% weight** by the stacking optimizer because of collinear random OOF
> - **Temperature sharpened to $T=0.90$**, blowing up exponentially on private test cameras
>
> **The research agent found the truth. The engineering agent ignored it and hill-climbed into a trap.**

This document establishes the **ML-AIDLC Protocol**: the operational doctrine that ensures AI models (and human practitioners) bridge the gap between high-level literature/winning insights and terminal-level code execution.

---

## 1. Anatomy of the Failure: The "Local Gradient Trap"

Why do advanced AI agents (like Claude, GPT-4, Gemini, Table 5.1, or deep-research teams) discover gold-standard winning solutions in Markdown files, yet completely abandon them when writing PyTorch code?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      STAGE 1: RESEARCH & BRAINSTORM                     │
│  Agent reads literature, winners' blogs, and domain papers.             │
│  Produces: `RESEARCH_SIMILAR_COMPETITIONS.md`, `RUN3_BRAINSTORM.md`      │
│  Insights: "2D projections beat 3D", "Blur is essential for cameras",   │
│            "Random CV leaks site signatures -> use LOCO"                │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ (Insight Lost Here)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     STAGE 2: LOCAL EXECUTION & CODE                     │
│  Agent enters rapid iteration loop: Edit -> Train -> Read OOF.          │
│  Metric in terminal: Random 5-Fold Stratified CV (leaking site data).   │
│                                                                         │
│  • Agent tests Blur Augmentation: Random CV says -0.003 -> DROPPED!    │
│  • Agent tests 3D ResNets: Random CV says 0.235 (memorizes site) -> 75 ckpts!│
│  • Stacker evaluates 2.5D Tri-Planar: Random CV says redundant -> 0% weight!│
│  • Agent sweeps T: Random CV says T=0.90 (sharper is better) -> APPLIED!│
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ (Collapse on Hidden Data)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      STAGE 3: PRIVATE LEADERBOARD                       │
│  Public LB: 0.2473 (Rank #16) ──> Private LB: 0.3003 (Rank #50)        │
│  Winners used EXACTLY what Stage 1 discovered.                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Three Cognitive Traps of Autonomous Coding Agents

1. **Context Amnesia Across Workflows**:
   When an AI switches from "Research Mode" to "Execution/Coding Mode", its active attention window shifts to compiler errors, tensor shapes, script arguments, and stdout numbers. The deep principles articulated in static `.md` files slip out of working memory unless explicitly enforced by an invariant validation gate.

2. **The Naive Proxy Metric Trap**:
   Language models acting as software engineers are instinctively reward-driven: if a script outputs `validation_logloss: 0.232` instead of `0.235`, the agent assumes progress. If the validation script suffers from data leakage (random folds instead of LOCO), the agent aggressively overfits to the leak, penalizing the very mechanisms (blur, invariance, regularization) designed to protect against it.

3. **Collinear Ensembling & Stacker Deception**:
   Automated stacking algorithms (L2 logistic, Ridge, SLSQP) optimize solely on the training/OOF matrix. On in-distribution data, multiple variants of the same memorizing architecture (e.g. 3D ResNets) will appear slightly better and displace truly orthogonal representations (e.g. 2D projections, physical radiomics). The AI coding agent accepts the optimizer's weights without asking: *"Is this ensemble diverse, or is it an echo chamber?"*

---

## 2. The ML-AIDLC Operational Protocol

Every AI agent executing machine learning tasks in this lifecycle MUST abide by the following non-negotiable rules.

---

### RULE 1: The Research Reconciliation Gate (No Insight Left Behind)

**Directive:** An AI agent is FORBIDDEN from finalizing a training run, ensemble, or submission candidate without performing an explicit **Research Audit** against all brainstorm and literature markdown files in the repository.

#### Mandatory Audit Step (Executed before every major Run / Effort transition):
Before building `submission_vX.zip`, the agent must populate this checklist:

```markdown
### Research-to-Implementation Reconciliation Table

| Insight / Recommendation from Research MDs | Implemented in Pipeline? | Status / Evidence | If Discarded, Why? (Must prove on shifted CV, not random CV) |
|---|:---:|---|---|
| E.g. 2D multi-plane projection (Buddenkotte 2024) | YES / NO | ... | Cannot discard based on random CV alone! |
| E.g. Heavy scanner blur/noise augmentation | YES / NO | ... | ... |
| E.g. Leave-One-Cluster-Out (LOCO) CV | YES / NO | ... | ... |
| E.g. Equal weights across representation types | YES / NO | ... | ... |
```

If a strategy recommended by high-trust research (e.g. past winners or domain literature) is omitted or set to 0% weight, the agent **must flag this as an active risk** in the effort state and request human concurrence or run a dedicated counter-experiment.

---

### RULE 2: Never Let In-Distribution CV Veto Domain Invariance Mechanisms

**Directive:** Augmentations and transformations derived from physical or clinical realities (e.g., scanner blur, resolution degradation, sensor noise, audio reverberation, atmospheric haze) MUST NOT be discarded due to small negative deltas ($\le 0.005$) on random-fold cross-validation.

#### The Invariance Principle:
- **Random CV measures Interpolation Capacity.**
- **Shifted / Grouped CV measures Generalization Robustness.**

When an augmentation simulates a known difference between training and test hardware:
1. It is expected to add entropy and slightly degrade pure in-distribution interpolation.
2. It MUST be evaluated on a **Grouped/LOCO validation split**.
3. If random CV says `-0.003` but the augmentation models a real-world physical perturbation, **the augmentation stays in the training mixture**.

---

### RULE 3: Representation Quotas (The Anti-Monoculture Rule)

**Directive:** Automated stacking algorithms (Ridge, Logistic Regression, Nelder-Mead) are structurally blind to out-of-domain shift. An AI agent must NEVER allow an automated combiner to allocate >60% of ensemble weight to a single architectural paradigm when diverse paradigms exist.

#### Enforcement: Fixed Representation Buckets
When assembling final submissions, the AI agent must enforce categorical representation quotas:

$$\text{Final Ensemble} = \sum_{k=1}^{K} w_k \cdot \text{Representation}_k$$

For DaT-SPECT / Medical Imaging, for instance:
- **Bucket A (3D Volumetric / Bilateral Asymmetry)**: max 35%
- **Bucket B (2.5D / Multi-Plane Orthogonal Projections)**: min 30%
- **Bucket C (Sequential / Slice Attention / BiGRU)**: min 20%
- **Bucket D (Domain Radiomics / Tabular Descriptors)**: min 15%

Within each bucket, multiple seeds can be averaged equally. But a stack of 91 checkpoints where 75 are 3D ResNets is a **structural monoculture**. When that monoculture hits an out-of-domain scanner, the entire ensemble sinks together.

---

### RULE 4: Split First, Model Second (The Validation Contract)

**Directive:** Before training a single neural network or fitting a single baseline, the AI agent must audit the dataset for **grouping variables** (hospital site, scanner vendor, camera matrix, acquisition date, patient ID, customer ID).

1. **If grouping metadata is explicit:** Create `GroupKFold` or `StratifiedGroupKFold`.
2. **If grouping metadata is hidden/unlabelled:** Unsupervised metadata clustering MUST be performed during EDA:
   - Cluster by image geometry: `(matrix_x, matrix_y, matrix_z, voxel_spacing_x, voxel_spacing_y, voxel_spacing_z)`.
   - Cluster by noise characteristics: background variance, intensity quantization regime.
   - Use these clusters to construct **Leave-One-Cluster-Out (LOCO)** validation folds.
3. **The Dual-Evaluation Standard:**
   Every experiment report must print two numbers:
   - `CV_random`: measures model capacity and learning stability.
   - `CV_loco` (or `CV_site`): measures camera/domain invariance.
   **Hyperparameters governing regularization ($\lambda$, weight decay, dropout) and calibration ($T$) MUST be tuned on `CV_loco`, NEVER on `CV_random`.**

---

### RULE 5: Arithmetic Plausibility & Ceiling Checks

**Directive:** An AI agent must never attribute an empirical gain to a cause without proving that the cause is arithmetically capable of generating that gain.

- **Post-hoc Transforms:** If modifying $\lambda$ from $0.02$ to $0.01$ has a mathematical ceiling of $\Delta_{\max} = 0.0055$ on log loss, and the observed leaderboard delta is $+0.0084$, the AI must NOT write *"Tuning $\lambda$ yielded $+0.0084$"*.
- **The True Cause Must Be Investigated:** In competitive systems, unaccounted-for gains usually mean:
  1. A silent fallback or timeout dropped an arm, changing the effective ensemble.
  2. A race condition or container-level difference occurred.
  3. Leaderboard sample variance (small test slices).
- **Rule:** When $\Delta_{\text{observed}} > \Delta_{\text{ceiling}}$, mark the result `[UNEXPLAINED CONFOUND]` and inspect the execution trace before pulling that lever again.

---

### RULE 6: The Option Value of Time & Submission Economics

**Directive:** Submissions are non-renewable options. An AI agent must never spend a submission slot impulsively simply because the slot has unlocked.

- **Timeline Mapping on Day 1:** The AI must calculate:
  - Total submission slots available over the competition window.
  - Rolling reset interval (e.g., 3 submissions per 7 days).
  - Free evaluation / smoke-test capabilities.
- **The Option Value Rule:**
  If a submission slot opens 60 hours before the final deadline, and the proposed package has only marginal local gains ($\le 0.001$ NLL) over an already-banked submission:
  - **DO NOT SUBMIT IMMEDIATELY.**
  - Hold the slot. Use the remaining GPU time to explore a truly orthogonal representation (e.g. 2D projections or synthetic boundary cases) or run comprehensive verification.
  - Spend the final slot only when a materially distinct, validated package is ready or when approaching the hard drop-dead cutoff (e.g. $T - 8$ hours).

---

## 3. The Multi-Agent Operational Handshake

In complex projects, different agents (or different turns of the same agent) play different roles:

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          RESEARCH AGENT (Table 5.1)                       │
│  Role: Literature review, past winner mining, domain physics, EDA.        │
│  Output Artifact: `docs/RESEARCH_SPEC.md`                                 │
│  Constraint: Must output EXPLICIT TESTABLE HYPOTHESES, not generic advice.│
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                        ARCHITECT AGENT (AI-DLC Lead)                      │
│  Role: Converts research into Effort Specs and Validation Architecture.   │
│  Output Artifact: `aidlc-docs/efforts/XXX/effort-state.md`                │
│  Mandate: Creates LOCO validation split and representation budget.         │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                       ENGINEERING AGENT (Coder/Trainer)                   │
│  Role: Trains models, manages GPUs, builds packages.                      │
│  Restriction: Cannot drop research hypotheses without passing Rule 1 & 2. │
│  Output Artifact: `RESULTS_LOG.md` with explicit provenance & ceilings.   │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                         AUDIT AGENT (Verification)                        │
│  Role: Container smoke tests, arithmetic checks, representation audits.   │
│  Mandate: Enforces the Anti-Monoculture Rule before any upload.           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Transferable Playbook for Any Future ML Competition

When an AI agent is dropped into a new competition or production modeling task, follow this exact chronological sequence:

### Phase 1: Inception & Domain Discovery (Hours 0–12)
1. **Identify the Core Physics / Generation Process:**
   - What sensor generated this data? What are the physical failure modes?
   - What changes between training environments and deployment environments?
2. **Cluster Unsupervised Metadata:**
   - Group by resolution, aspect ratio, audio sample rate, text source, time period.
   - Build both **Random CV** and **Grouped / Shifted CV**.
3. **Execute Deep Literature & Winner Mining:**
   - Mine top solutions from similar historical competitions (e.g. RSNA, ISIC, OSIC, Kaggle Grandmaster writeups).
   - Write `docs/RESEARCH_SIMILAR_COMPETITIONS.md`.
4. **Build the "Dumbest" End-to-End Container Pipeline:**
   - Generate submission format on Day 1. Submit it. Verify platform mechanics.

### Phase 2: Representation Construction (Days 2–7)
1. **Never Invest Entire Capacity into One Architecture:**
   - Implement at least 3 distinct inductive biases (e.g. 2D multi-plane projections, 3D volumetric, and tabular/radiomic descriptors).
2. **Invariance Augmentation First, Hyperparameters Second:**
   - Implement heavy domain augmentations (blur, noise, mixup, elastic) that target the sensor differences.
   - Retain them if they preserve or improve Grouped/LOCO CV, regardless of minor fluctuations on random CV.
3. **Address the Decision Boundary (Populate the Borderline):**
   - Inspect the loss distribution. If 10% of samples account for 50% of the loss, don't label it "irreducible" and walk away.
   - Synthesize boundary cases (counterfactual lesions, boundary interpolation, hard-negative mining).

### Phase 3: Assembly, Calibration & Endgame (Final 48 Hours)
1. **Freeze Modeling Early:**
   - Stop launching marginal seeds that offer $+0.0001$ theoretical gains.
2. **Audit the Ensemble Composition:**
   - Enforce the Anti-Monoculture Rule (Rule 3). Verify representation diversity.
3. **Calibrate for the Shifted World:**
   - Under domain shift, soften predictions ($T \ge 1.0$) and preserve prior insurance ($\lambda > 0$).
   - Never sharpen logits ($T < 1.0$) to overfit the public leaderboard.
4. **End-to-End Container Verification:**
   - Verify that every arm runs, every checkpoint loads, and no silent fallbacks are triggered.

---

## 5. Summary Law for Autonomous Systems

> **The First Law of ML-AIDLC:**  
> *A machine learning agent must never allow the convenience of an in-distribution evaluation metric to override the physical realities of domain shift documented in its own research.*
>
> If literature and physics indicate that cameras blur, that slice spacings vary, or that sensors drift, the agent's primary duty is to build invariance to those realities—even when naive validation suggests that memorization is momentarily cheaper.
