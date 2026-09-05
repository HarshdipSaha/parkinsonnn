# Pipeline Recommendation C — Independent Second-Model Design & Critique (no new retrieval)

- **Date:** 2026-09-05 (deadline Sept 16, 2026 23:59 UTC — 11 days)
- **Role:** Subagent C. Pure reasoning over the documents already in this repo; **no web search, no new sources**. This is a critique of `PIPELINE_REC_A_PAST_WINNERS.md`, not a replacement for its literature work.
- **Nothing in `data/` was touched. No input file was modified.**

---

## 1. Method note

**What was attempted.** The `oracle` skill (`@steipete/oracle` CLI v0.9.0) was used to bundle a task prompt plus all four documents (`DAT_PARKINSONS_CHALLENGE.md`, `LITERATURE_REVIEW.md`, `EDA_REPORT.md`, `PIPELINE_REC_A_PAST_WINNERS.md`; ~24k tokens total, verified with `--dry-run --files-report`) and send it to a second model via the browser engine (ChatGPT, GPT-5.x Pro). **Both browser attempts failed**: Chrome disconnected within seconds of launch, before the prompt was submitted (session logs empty; error "Chrome window closed before oracle finished"). A reattach hung. The API engine was not usable: no `OPENAI_API_KEY` is set, and API runs require explicit user consent for cost, which a subagent cannot obtain. **So the oracle skill could not reach GPT in this environment.**

**Fallback actually used (this is still a two-model result).** An independent reviewer agent was spawned inside the harness on a **different model (self-reported "Claude Opus 5")** with a fresh context, web/network tools forbidden, file writes forbidden, and a forced read order: it had to read the challenge, EDA, and literature docs and **write its first-principles design (Section 2 below) before it was allowed to open Rec A**. Its full answer was then synthesized by this subagent (Claude Fable 5.1), which had written its own independent draft *before* seeing the Opus answer, so convergence between the two is meaningful rather than anchored.

**Provenance labeling used below:** "Oracle" = the Opus reviewer. "Synth" = this subagent's own view. Where both independently reached the same conclusion it is marked **[converged]** — treat those as the highest-confidence items in this document.

**Caveat:** this is Claude-family × Claude-family, not Claude × GPT. Both models share training biases; a third opinion from a non-Anthropic model would still add value if the user can log in to ChatGPT and rerun `oracle session dat-spect-review-retry --render` or `npx -y @steipete/oracle --engine browser ...` with the prompt at the scratchpad path `00_TASK_oracle_prompt.md`.

---

## 2. Independent first-principles pipeline (the Oracle's own design, written before reading Rec A)

**Framing (Oracle).** Three facts dominate: (a) signal is strong (literature AUC 0.98–0.99) but n=1,362 is small, so *overfitting, not underfitting, is the binding constraint*; (b) the metric is log loss, so the score is dominated by the *rate of confident errors*, not mean accuracy; (c) the pipeline runs once, black-box, on unseen volumes with a 3-hour cap, so *robustness beats peak accuracy*. Build the least clever thing that captures the signal; spend saved time on guards and calibration.

### 2.1 Preprocessing / alignment — no external template, no ANTs on the critical path

1. **Reorient RAS** via `nib.as_closest_canonical` (safe: qform=sform=1 on all 1,362).
2. **Resample to 3.0 mm isotropic** (Oracle) — native spacings are 2.3–3.9 mm and DaT-SPECT reconstruction FWHM is ~8–10 mm; 2 mm oversamples into noise and triples voxel count against a tiny dataset. (Synth had independently picked 2.5 mm; either is defensible — see §7.)
3. **Brain mask:** Gaussian smooth σ≈6 mm → threshold at 0.15 × p99.9 of the smoothed volume → largest connected component → binary close (6 mm ball) → fill holes.
4. **Mid-sagittal (yaw/roll/x-shift) correction** (Oracle): 3-parameter Powell search on a 6 mm downsample maximizing normalized cross-correlation between the volume and its L–R mirror. Brain SPECT is strongly bilaterally symmetric; well-conditioned, <1 s.
5. **Striatal anchor — the key insight [converged]:** the striatal complex is the *brightest structure in every DaT scan*, normal or abnormal (severe PD retains caudate-head uptake). Smooth σ≈6 mm, threshold at 0.6 × smoothed max, take the centroid of the largest supra-threshold component. That centroid is the translation origin. **No template needed.** This is exactly why the EDA's naive feature scored 0.469: it used a fixed *relative-position* ROI rather than a hotspot-anchored one.
   - *Do not* use whole-volume PCA/moment alignment for rotation: z-extent ranges 33–256 slices, so principal axes reflect FOV, not anatomy.
   - Synth adds a guard: if the anchor lies >40 mm from the brain-mask centroid, fall back to brain-mask centroid + a fixed anterior/inferior offset and flag the volume as anomalous.
6. **Crop** a fixed box around the anchor: cache **72×72×48 @ 3 mm** (216×216×144 mm), model input **64×64×40** (192×192×120 mm) so random crops give translation augmentation. **Pad with the volume's own background percentile (p10 of masked region), never 0** — a zero edge is out-of-distribution and produces confident garbage.
7. **Optional Tier-2** (gated, not MVP): 9-DOF rigid+scale registration to an in-house mean of ~30 Tier-1-aligned scans, shipped only if OOF log loss improves by more than the noise floor (~0.02), with a per-volume QC gate (NCC to template ≥ threshold) falling back to Tier-1.

### 2.2 Intensity normalization — template-free DVR [converged on "no template ROI"; Oracle's formula adopted]

Nothing in the header can rescue the ~5–10 / ~100–1,000 / ~10,000–65,000 regimes, so normalization is per-volume and image-derived. **Do not divide by a whole-volume high percentile — that percentile *is* the striatum, i.e. the signal you want to preserve.** (Synth's draft had proposed p99.5-of-brain scaling; the Oracle's argument is correct and Synth concedes: dividing by the striatal peak throws away the absolute specific-to-nonspecific ratio, which is the single most diagnostic quantity.)

```python
ref = np.percentile(vol[brain_mask], 75)        # striatum is ~2–3% of brain volume,
if not np.isfinite(ref) or ref <= 0:            # so p75 is dominated by non-striatal tissue
    ref = max(vol[brain_mask].mean(), 1e-6)
x = np.clip(vol / ref, 0, 12)
x = (x - 1.0) / 2.0                             # ~zero-mean background, unit scale
```

This is the published 75th-percentile reference-region (DVR) idea without a template-space mask, using ~50% of brain voxels, immune to a 5–10 mm misalignment. Single input channel. Instead of a second image channel, append ~6 per-volume scalars (log p99/p75, mask volume in mL, masked-histogram skewness, mirror-symmetry NCC, z-extent in mm, native spacing, number of distinct grey levels) to the pooled feature vector before the head.

**Synth addition — quantization:** the ultra-low regime (max ~5–10) has only ~10 grey levels. Add a *quantization augmentation* (round the normalized volume to k ∈ [6, 256] levels, p≈0.3) or the model will be miscalibrated on exactly those scans.

### 2.3 Model

- **Primary: small 3D CNN from scratch.** Four stages, 16/32/64/128 channels, two (3×3×3 conv → GroupNorm(8) → SiLU) blocks per stage, strides (2,2,2)/(2,2,2)/(2,2,1)/(2,2,1) to respect the shallow z, GeM pool, dropout 0.3, one logit. **≈0.9M params**, inside the published 0.2–2.9M band for this data regime.
- **Second family: SBR/radiomics + L2 logistic regression (or a heavily regularized LightGBM: num_leaves 8, 300 trees, lr 0.03, feature_fraction 0.6)** on ~40 hotspot-anchored features: per-side caudate/putamen uptake, putamen:caudate ratio, asymmetry index, supra-threshold blob volume and eccentricity, first-order histogram stats. Cheap, decorrelated, calibrated by construction, and a **live QC probe**: if its OOF AUROC does not jump from 0.469 to >0.90, alignment is broken and everything else is premature.
- **No ImageNet 2.5D backbone** (see §3).

### 2.4 Training protocol

AdamW, lr 3e-4, wd 3e-2, 5-epoch warmup + cosine to zero, ~120 epochs, batch 16 (~8k steps), AMP, EMA 0.999, label smoothing 0.05 (Oracle; Synth would ablate it — low stakes), selection by fold **log loss**. ~15 s/epoch on a T4 → ~30 min/fold, ~2.5 GPU-h for 5 folds. Augmentation, ranked by expected value:

1. Gaussian blur σ ~ U[0, 4] mm, p=0.7 — the demonstrated cross-site lever.
2. Additive Gaussian noise σ ~ U[0, 0.15] normalized units + √-intensity (Poisson-like) noise, p=0.7.
3. Random affine: rotation ±12°, isotropic scale 0.9–1.1 (+ ±5% anisotropic), translation ±6 mm.
4. **L–R flip p=0.5** — label is mirror-invariant; free doubling.
5. **Reference-scale jitter: multiply `ref` by U[0.9, 1.1]** — regularizes directly against reference misestimation at test time (Oracle's proposal; not seen in the literature; cheap and targeted).
6. Mild gamma [0.9, 1.1]; coarse dropout 1–3 cubes of 8–12 mm, p=0.25; quantization aug (Synth); mixup α=0.2 (ablate).

### 2.5 Cross-validation [converged]

- **Primary: repeated StratifiedKFold (5 folds × 3 seeds).** Nothing in the competition text says the test set comes from *held-out hospitals*; the stated design is one corpus from 10 centres split into train/test. Absent evidence otherwise, test is IID with train across the same centres, and random stratified CV is the right estimator of test loss and the right basis for calibration.
- **Near-duplicate / repeat-patient audit (Oracle; absent from Rec A):** 1,362 "examinations" with no patient ID almost certainly include follow-ups of the same patient. After Tier-1 alignment + normalization, compute a 16³ embedding per scan, full 1,362² cosine similarity (seconds), flag pairs >0.98, force clusters into the same fold. Otherwise CV is optimistic and the shipped temperature too small.
- **Leave-one-major-spacing-cluster-out, run once, as a diagnostic only** — used to *size the safety margin* (λ and temperature inflation), never to fit anything.

### 2.6 Calibration for log loss

1. Average members in **probability space** (mildly under-confident = safe direction).
2. One temperature `T` fit on pooled OOF logits; ship **`T* = max(T_pooled, P75 of per-fold T)`** — deliberately biased toward under-confidence.
3. **Prior mixing:** `p = (1−λ)·p_cal + λ·0.548`, λ ∈ [0.02, 0.05] tuned on OOF but floored at 0.02.
4. Hard clip to [1e-4, 1−1e-4] as a NaN guard only; the real floor is set by λ (λ=0.03 ⇒ effective floor ≈ 0.016).
5. **Three free calibration parameters total** (one blend weight CNN-vs-classical, one T, one λ). No isotonic, no per-member temperatures, no hill-climb.

### 2.7 Inference plan (3 h, no internet, 1× A100, 24 vCPU)

Budget for 3,000 volumes. Tier-1 preprocessing ≈2.5 s/volume single-core → ~6 min with 20 workers. 15 checkpoints × 2 TTA views × 3,000 volumes of a 0.9M-param net ≈ 5 min on the A100. **Total < 30 min, a ~6× margin — that margin is the point.**

Non-negotiable guards [converged on 1, 3, 4, 7, 8; Oracle adds 2, 5, 6]:

1. **Write `submission.csv` with the prior (0.548) for every uid before any inference starts**, then overwrite rows as they complete, flushing every ~200. A hard kill at 2:59 still leaves a scored file. *Highest-value guard in the design.*
2. **Set `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` in every worker *before* importing ITK/ANTs/numpy.** `Pool(20)` × ITK's default multithreading on 24 vCPU = 480 threads; this alone can turn 15 s/volume into 200 s/volume and is the single most likely real cause of a timeout.
3. Per-volume try/except → prior; NaN/Inf check on inputs and outputs → prior; **per-volume timeout** via `apply_async(...).get(timeout=120)` / `as_completed(timeout=...)`.
4. Degenerate guards: max==0, constant volume, empty mask, <8 slices → prior.
5. Pad value = masked p10, never 0.
6. **Deadline ladder** on `t0` (continuous, not a one-shot projection after 50 volumes): >100 min → Tier-1 only; >140 min → 5 checkpoints, no TTA; >165 min → prior for all remaining, exit 0.
7. Lazy `try: import ants` (only in workers); an import failure degrades, never crashes. Measure import + CUDA-init time explicitly — the 6-minute smoke test can lose 60–90 s to imports.
8. Catch `BrokenProcessPool` (ANTs can segfault and hang `Pool.imap`), rebuild once, then fall through to serial Tier-1.
9. Strictly per-volume; **no statistic computed across the test set** (rule and robustness). Never log per-volume (300-line cap).

---

## 3. Critique of Rec A — keep / cut / change

**Overall (Oracle, Synth agrees):** Rec A is well-sourced and largely correct; its literature synthesis and license table are genuinely valuable and better than either reviewer would produce unaided. Its engineering judgement, however, is **calibrated for a six-week project with 10× the data**. It specifies three model families, a template build, a two-channel input, per-member temperatures, hill-climbed blend weights, and a second temperature — ~24 free calibration parameters — on 1,362 samples. That is more machinery than the data supports and more code than one or two part-time engineers can harden in 11 days.

### Keep
RAS reorient · isotropic resample · per-volume intensity standardization · fixed striatal crop · small 3D CNN from scratch as primary · strong blur+noise augmentation · L–R flip · probability-averaged fold ensemble · single OOF temperature · selection on log loss · per-cluster and worst-cluster OOF reporting · leave-one-cluster-out as a *non-ensembled diagnostic* · the radiomics/LightGBM member (framed as a sanity check — exactly right) · fallback prior for failed loads · import guards · float16 caching · offline-bundled weights · the entire license table · "never tune on the public leaderboard".

### Cut
| Item | Why |
|---|---|
| **`ants.build_template` on the critical path** [converged] | *Leakage:* essentially a non-issue — unsupervised mean over ~250 scans, each contributes O(1/250) to a coordinate frame, applied identically at test. *Real problems:* (1) selecting template scans **by label** (150 normal + 100 abnormal) is unnecessary and awkward for the "disclosed derived artifact" story — select label-blind; (2) an average template has a hot striatum, and MI-affine of a severely abnormal scan to it can be pulled by the template's bright putamen — a **label-correlated registration bias** that lands on the hardest cases; (3) the cost is not CPU-hours but the **debug loop** (drifting frames, degenerate scans poisoning the mean, unverified `type_of_transform` strings that Rec A itself flags). If registration is wanted later, a single hand-picked, Tier-1-aligned high-count normal scan as fixed reference gives ~90% of the benefit for ~5% of the machinery. |
| **DVR via a template-space reference-region mask** [converged; Oracle's strongest technical disagreement] | A fixed box in template space is only as good as the affine fit *on that volume*, and the fit will be visibly off on some fraction of test scans (33-slice, ultra-low-count, unfamiliar FOV). When the mask clips into striatum, `ref` inflates and DVR deflates → a normal scan is pushed toward "abnormal"; when it clips into skull/air, `ref` deflates → an abnormal scan is pushed toward "normal". **Both directions produce confident errors — the log-loss disaster case — on exactly the scans you can least inspect.** Volume-derived brain-mask p75 captures the same physics with no registration in the loop. |
| **M2: 2.5D ImageNet EfficientNetV2-S + bi-GRU** [converged] | Cargo-culting. RSNA winners had 3–10k studies × 400+ slices of high-frequency CT texture at 384 px (≈10⁶ weakly-correlated slice examples). Here: 1,362 × 16 ≈ 22k highly-correlated slices of a blurry single-channel blob, natively 64×64 and upsampled to 128 (zero added information), into a ~20M-param net. ImageNet's early filters encode natural-image edge/texture statistics at scales this data does not contain. **Rec A's own closest-analogue evidence (Hamburg, 1,100–1,740 scans) uses 194k-param 3D nets; Rec A cites both and picks the wrong one for our n.** Also: ~80–100 MB × 5 folds in the zip, license bookkeeping, and a second training loop to debug. If 2D diversity is wanted, train a small 2D CNN *from scratch* on three fixed slabs (hours, no license question). |
| **Two-channel DVR + z-score input** [converged] | The channels differ by a *per-volume affine map*; the marginal information is one scalar (≈ masked std / p75). Append it as a scalar to the head instead of doubling the input tensor and every downstream shape. |
| **Per-member temperatures + hill-climbed weights + second temperature** [converged] | OOF log-loss standard error at n=1,362 with heavy-tailed per-sample losses ≈ 0.5/√1362 ≈ **0.014 nats**. Differences below ~0.02 are noise. Fitting ~24 parameters to a target with that noise floor yields 0.01–0.03 optimism bias *and* mis-estimates the temperature you ship. Reduce to 3 parameters. |

### Change
| Item | Change to | Why |
|---|---|---|
| 2 mm isotropic | 3 mm (Oracle) / 2.5 mm (Synth) | FWHM 8–10 mm; 2 mm oversamples, inflates voxel count 3.4× against a tiny dataset; 2 mm is a day-9 ablation, not a day-1 default. |
| StratifiedGroupKFold by pseudo-site as *primary* | Repeated StratifiedKFold primary; grouped/leave-cluster-out as diagnostic to size hedges [converged] | (1) Test is almost certainly IID across the same 10 centres; grouped OOF is deliberately pessimistic, so **T fitted on it is too large → systematically under-confident shipped probabilities** (~0.01–0.03 nats on a well-separated model — the gap between 3rd and 8th on a tight board). (2) Mechanically, four clusters hold 89% of the data: a "5-fold" grouped split can put all 528 scans of the 2.46 mm cluster in one validation fold — that is four wildly unbalanced folds plus noise, not a 5-fold. |
| Watchdog: one-shot projection after 50 volumes | Continuous deadline ladder + pre-written CSV | Sites are not uniformly shuffled; 256³ scans cost ~8× the 128³ ones; a single hung worker blocks a one-shot projection forever. |
| Write `submission.csv` at the end (Rec A §3.5 step 4) | Pre-write priors, overwrite incrementally | A crash/kill at 179 min scores nothing. |
| Template-selected by label | Label-blind (largest FOV, highest count) if a template is ever built | Cleaner disclosure story; avoids the label-correlated bias above. |
| Zero padding (implicit) | Pad with masked p10 | Zero edges are OOD → confident garbage on short-z scans. |

### Single points of failure at inference Rec A misses or under-specifies (ranked by likelihood of actually killing a run)
1. Thread oversubscription (`Pool(20)` × ITK/BLAS threads) — unmentioned.
2. Submission written only at the end.
3. One-shot watchdog with no per-volume timeout; pool worker segfault hangs the parent.
4. Zero padding on 33-slice volumes.
5. Smoke-test import budget (6 min total).
6. No explicit prohibition on dataset-level test statistics (e.g., estimating the test prior from predictions) — should be a codebase invariant, both for the rules and for robustness on a small public split.

---

## 4. Log-loss-specific failure modes & mitigations

**Cost arithmetic.** `−ln(0.001) = 6.91` nats. Against a typical mean of ~0.12–0.20, one such error in 1,000 adds 0.007 to the mean — survivable. **The rate is the story:** at 95% accuracy, if 2% of samples (20 in 1,000) land at p=0.999 on the wrong side, that is 20 × 6.91 / 1,000 = **0.138 nats — it roughly doubles your score.** Optimizing the tail is worth more than optimizing the mean.

**Choosing ε — derived, not guessed (Oracle).** Let `f` be the fraction predicted more confidently than ε and `q` the true error rate within that confident bucket. Expected contribution ≈ `f[(1−q)(−ln(1−ε)) + q(−ln ε)]`; setting the derivative to zero gives **ε* = q** (Oracle wrote q/(1−q) using the −ln(1−ε)≈ε approximation; same thing for small q). The optimal clip is *the error rate among your most confident predictions*. If the top OOF decile has a 1% error rate, ε ≈ 0.01; at 0.3%, ε ≈ 0.003. **Ship ε ∈ [0.005, 0.02], likely 0.01**, estimated from the most-confident OOF decile of the **worst fold**, not the mean. Rec A's {0.005, 0.01, 0.02} grid brackets the answer but by guesswork. Synth's independent break-even estimate (clipping at 0.02 costs ~0.012 on correct confident predictions and saves ~3 nats per catastrophic error → worth it above ~1 catastrophe per 250 confident predictions) lands in the same range.

**Calibration drift across unseen centres.** T fitted on in-distribution OOF is optimistic for shifted data, and the asymmetry is stark: over-confidence is unbounded in cost, under-confidence is bounded and cheap. Bias deliberately toward under-confidence: (1) `T* = max(T_pooled, P75 of per-fold T)` (Oracle) or `T* = 1.15–1.25 × T` (Synth) — equivalent in spirit; (2) prior mixing λ ≥ 0.02; (3) ε from the worst fold; (4) if the leave-cluster-out diagnostic inflates loss by >50%, raise λ to 0.05.

**Which method at n=1,362 (ranked).** Probability-averaged ensembling ≫ temperature scaling > prior mixing/clipping > label smoothing ≫ isotonic. **Isotonic is a hard no**: O(n) knots on 1,362 points, exact-0/1 bins that need clipping anyway, catastrophic off-distribution extrapolation. Two-parameter Platt is the ceiling of what the data supports; if used, shrink its intercept ~50% because it encodes a prior you cannot verify transfers. Label smoothing is a training regularizer, not a substitute for T; if kept, evaluate it *after* calibration or you read the wrong signal. (Synth would drop it — it fights T and adds noise at this n; low-stakes either way.)

**Anomalous-volume hedging — compliant and targeted [converged].** Compute a per-volume anomaly score from: Tier-2 QC below threshold / fallback fired; mirror-symmetry NCC below the training 1st percentile; brain-mask volume outside training [p1, p99] mL; z-extent below training p1; fraction of crop padded; grey-level count in the ultra-low regime; anchor >40 mm from mask centroid. Map to per-volume shrinkage `λ_i = λ_base + 0.15·1[anomalous]` (Synth: up to 0.3–0.5 for hard failures). Thresholds are frozen from *training* distributions and shipped in JSON — each test volume is judged against fixed constants, never against other test volumes, so the "processed independently" rule is respected. Validate on training scans with induced anomalies (zeroed, truncated to 33 slices, NaN-injected, quantized to 8 levels).

**Class-prior mismatch — a third-order effect.** If the private set were 65% abnormal instead of 54.8%, the correct log-odds shift is ln(0.65/0.35) − ln(0.548/0.452) = 0.619 − 0.193 = 0.43. On a typical confident logit of ±4 that moves p from 0.982 to 0.988 or 0.973 — per-sample loss 0.018 → 0.012 or 0.027. Do nothing beyond the λ mixing. **Do not estimate the test prior from your own predictions** — it is a dataset-level statistic across test samples (rule risk) and it blows up on a small public split.

**Statistical-power discipline.** With an OOF noise floor of ~0.014 nats you cannot reliably rank a 5- vs 3-member ensemble, tune ε to three decimals, or trust any "improvement" under ~0.02. Gate every added component on ≥0.02 OOF gain; run one nested loop to get an honest post-calibration estimate.

---

## 5. Minimum-viable pipeline (days 1–3) → full pipeline (days 4–10)

Organizing principle [converged]: **have a scored real submission by Sept 8**, then improve monotonically against a known-good baseline. Every day without a leaderboard number is a day of unvalidated assumptions.

### MVP — Fri Sept 5 (evening) → Mon Sept 8
| Day | Deliverable |
|---|---|
| **Sept 5 (eve)** | Repo skeleton with **one shared `preprocess(path) -> array`** used identically by training and `main.py` (this single decision prevents the most common code-execution failure). Implement Tier-1: RAS → 3 mm (or 2.5) → brain mask → p75 normalize → mirror-symmetry → striatal anchor → 72×72×48 crop → fp16 cache. Cache all 1,362 (~1.5 GB, session storage only per the Kaggle-retention concern). Visual QC grid of ~30 crops. |
| **Sept 6** | Near-duplicate / repeat-patient audit → fold groups. Hotspot-anchored radiomics + L2 logreg: **OOF AUROC must jump from 0.469 to >0.90 — if not, alignment is broken; fix before anything else.** Train the 0.9M 3D CNN on one fold for a first honest number. |
| **Sept 7** | Full 5-fold CNN. Fit T, λ, ε on OOF; freeze to JSON. Write `main.py` with all nine guards. End-to-end run in the official Docker image on a 100-volume subset; measure import time and per-volume wall clock. |
| **Sept 8** | Platform smoke test → **first real submission.** You now have a validated code path, a wall-clock measurement, and a leaderboard anchor. |

### Full pipeline — Sept 9 → Sept 14, ranked by expected log-loss gain per engineering hour
1. **Sept 9–10 — augmentation sweep** (blur σ ceiling 2/4/6 mm; noise level; reference-scale jitter on/off; quantization aug on/off; mixup on/off). Pure config, runs unattended, and the literature says it is *the* multi-site lever. Expect the largest single gain here.
2. **Sept 10 — seeds × crop-size diversity.** 3 seeds × 5 folds × 2 crop sizes = 30 tiny checkpoints of code you already have. Ensembling gain for near-zero engineering.
3. **Sept 11 — classical member blended with a single weight.** Cheap, decorrelated, and a live QC probe.
4. **Sept 11–12 — Tier-2 registration, gated and time-boxed (1.5 days).** Ship only if OOF improves >0.02; if not clearly winning by end of Sept 12, cut it. This is the item both reviewers expect to be tempted by and would be disciplined about.
5. **Sept 12 — calibration hardening:** nested estimate, per-fold T, worst-fold ε, grouped + leave-cluster-out runs to size λ; anomaly-hedge thresholds from training distributions.
6. **Sept 13 — one architecture variant** (2 mm input, or a small 2D-from-scratch slab model) *only if* 1–5 are done. Otherwise skip.
7. **Sept 14 — freeze and scale-test.** Full Docker run over all 1,362 volumes as a throughput proxy for 3,000; **fault-inject every guard** (zeroed volume, 33-slice volume, NaN volume, simulated `import ants` failure, simulated worker hang); set deadline-ladder thresholds from measured numbers.
8. **Sept 15 — smoke test → full real submission.**
9. **Sept 16 — buffer.** One further submission only if a ≥0.02 OOF gain is already in hand. Do not touch the pipeline on the last day.

**Explicitly off the plan:** `ants.build_template`, M2 (2.5D EfficientNet + GRU), per-member temperatures, hill-climbed weights, isotonic regression, any pretrained-weight license adjudication (the recommended pipeline uses **no external weights at all**, which sidesteps the entire question and simplifies the MIT release).

---

## 6. Explicit disagreements with Rec A (+ confidence)

| # | Rec A says | Oracle / Synth position | Confidence | Converged? |
|---|---|---|---|---|
| 1 | Build affine template from ~250 training scans with `ants.build_template`, register everything to it | Cut from MVP. Hotspot-anchored crop is template-free and robust; template build is a debug-loop sink and a label-biased selection. If registration is added later, use a single hand-picked reference scan | Medium-high | Yes |
| 2 | DVR via a template-space reference-region box | Volume-derived brain-mask p75. Registration-dependent ROI yields confident errors in both directions on exactly the scans where registration fails | **High** | Yes |
| 3 | Three model families incl. 2.5D ImageNet EfficientNetV2-S + GRU | Drop M2. Wrong data regime by ~50× vs the RSNA precedents; Rec A's own Hamburg evidence points to ~0.2M-param 3D nets | Medium-high | Yes |
| 4 | Two-channel DVR + z-score input | One channel + per-volume scalars in the head | Medium | Yes |
| 5 | StratifiedGroupKFold by pseudo-site as primary CV and calibration basis | Repeated StratifiedKFold primary; grouped/leave-cluster-out as diagnostic to size hedges. Test is almost certainly IID across the same 10 centres; grouped OOF mis-calibrates toward under-confidence and produces structurally lopsided folds | Medium-high | Yes |
| 6 | Per-member T + hill-climbed weights + second T (~24 params) | 3 free parameters. OOF noise floor ≈ 0.014 nats | **High** | Yes |
| 7 | Write `submission.csv` at the end | Pre-write priors for every uid, overwrite incrementally | **High** | Yes |
| 8 | `Pool(20)` with no thread settings | Set ITK/OMP/MKL threads = 1 in workers before import; most likely real timeout cause | **High** | Oracle-only (Synth agrees on reading it) |
| 9 | One-shot watchdog after 50 volumes | Continuous deadline ladder + per-volume timeout + BrokenProcessPool handling | **High** | Yes |
| 10 | 2 mm isotropic | 3 mm (Oracle) / 2.5 mm (Synth) | Medium | Partly (both say "coarser than 2 mm") |
| 11 | (implicit) zero padding | Pad with masked background percentile | Medium-high | Oracle-only (Synth agrees) |
| 12 | No repeat-patient audit | Embedding-similarity audit, force clusters into same fold | Medium that repeats exist; **High** that the 2-hour check is worth it | Oracle-only (Synth agrees) |
| 13 | ε chosen from a {0.005, 0.01, 0.02} grid | Derive ε ≈ q (error rate in most-confident OOF decile, worst fold); ~0.01 | Medium | Yes (same range, Oracle's derivation is cleaner) |
| 14 | No explicit ban on dataset-level test statistics | Make it a codebase invariant | Medium-high | Oracle-only (Synth agrees) |
| 15 | Label smoothing 0.02–0.05 "keep if it helps" | Oracle: keep 0.05, evaluate after calibration. Synth: drop; it fights T at this n | Low (low stakes; ablate) | **No — the one internal disagreement** |
| 16 | Ultra-low-intensity regime handled by percentile normalization alone | Add explicit quantization augmentation and an anomaly flag for <16 grey levels | Medium | Synth-only |

**Agreements with Rec A (both reviewers):** preprocessing-heavy / architecture-light framing (the most important sentence in Rec A); small 3D CNN from scratch as primary (0.2–3M params); strong blur + noise as the key augmentation (both would push σ higher and add reference-scale jitter); L–R flip; fixed striatal crop (also removes FOV as a site shortcut); probability-averaged fold ensemble + single OOF temperature; keep the radiomics/LightGBM member as sanity check; per-cluster and worst-cluster OOF reporting; leave-one-cluster-out as a non-ensembled diagnostic; select on log loss; never tune on the public LB; the license table (accurate and time-saving); no PPMI/NTUA; degraded-fallback watchdog *instinct*; fallback prior for failed loads; offline-bundled weights.

---

## 7. Confidence & caveats

**High confidence (independent convergence of two models + first-principles argument):**
- The striatum is self-localizing (brightest structure in every DaT scan) → a template-free, hotspot-anchored crop is a valid and more robust MVP than template registration.
- A registration-dependent reference-region mask is the single riskiest element in Rec A for log loss.
- Pre-written submission file, thread limits, per-volume timeouts, and a continuous deadline ladder are the guards that actually decide whether a run scores.
- ~24 calibration parameters on 1,362 samples is over-fitting the OOF; three is the right number.
- Random stratified CV is the correct basis for calibration unless organizers state that test hospitals are held out.

**Medium confidence:**
- 2.5–3 mm vs 2 mm spacing (the two reviewers differ by 0.5 mm; either beats 2 mm as a default; cheap to ablate late).
- Dropping M2 entirely vs deferring it. Both reviewers would never ship it in 11 days; if the deadline were 6 weeks the answer might differ.
- The magnitude of the grouped-CV calibration error (0.01–0.03 nats is an estimate, not a measurement).
- Whether repeat-patient scans exist in the data (the audit is cheap regardless).

**Low confidence / open:**
- Label smoothing on/off (the one point on which the two reviewers disagree; low stakes).
- Exact augmentation ranges; the sweep on Sept 9–10 is what settles them.
- Whether Tier-2 rigid+scale registration adds anything over Tier-1 + affine augmentation for the CNN (both reviewers expect "small or nothing" but would test it, time-boxed).

**Caveats on this document itself:**
- **Not a Claude × GPT review.** The oracle skill could not reach ChatGPT/OpenAI in this environment (browser disconnected twice before submission; no API key; API runs need user consent). The second model was Claude Opus in a fresh, web-less context. Shared training lineage means shared blind spots are possible; the strongest hedge against that is that the two reviews were written independently and still converged on all major calls, and that every quantitative claim above (cost arithmetic, ε derivation, prior-shift arithmetic, noise floor) was re-checked by hand during synthesis. One correction was made: the Oracle's ε* = q/(1−q) is the small-q approximation of the exact ε* = q.
- **Key assumption:** test volumes come from the same 10 hospitals as training. If organizers later state otherwise, promote grouped CV to primary and raise λ to ~0.05 — but do not change anything else.
- No new sources were consulted; every factual claim about the competition, data, or literature is inherited from the four input documents and carries their caveats.
- **If the user can sign in to ChatGPT**, rerunning the oracle prompt (scratchpad `00_TASK_oracle_prompt.md`, session slug `dat-spect-review-retry`) would add a genuinely non-Anthropic third opinion; that is the one cheap thing left on the table.
