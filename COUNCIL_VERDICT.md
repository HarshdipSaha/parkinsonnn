# LLM Council Verdict — DaT Parkinson's Challenge ML Pipeline

Date: 2026-09-05
Method: Karpathy-style LLM Council — 5 advisors with distinct thinking lenses (Contrarian, First Principles, Expansionist, Outsider, Executor) answered independently; 5 anonymized peer reviews; one Chairman synthesis. Inputs: `PIPELINE_REC_A_PAST_WINNERS.md`, `PIPELINE_REC_B_ARXIV.md`, `PIPELINE_REC_C_ORACLE.md`, `PIPELINE_REC_D_BRAINSTORM.md`, plus `EDA_REPORT.md`, `LITERATURE_REVIEW.md`, `DAT_PARKINSONS_CHALLENGE.md`.

The Chairman verified two reviewer claims against the challenge doc before ruling: (i) "best private-set submission counts automatically" **is** recorded (`DAT_PARKINSONS_CHALLENGE.md` line 37); (ii) the Kaggle-training prize-eligibility risk **is** recorded (lines 95–97) and unresolved.

---

## Where the Council Agrees

Five advisors, working independently, converged on the same spine. Treat these as settled:

1. **Ship Rec C's MVP, template-free.** Hotspot-anchored crop, not affine registration to a template. Reasons converge from three directions: cost (50 lines of scipy vs. an `ants.build_template` debug loop), failure visibility (a bad hotspot is a bad picture; a bad affine is a confident wrong answer), and the First-Principles framing that the CNN needs *containment*, not voxel correspondence. Registration is at most a gated later ablation.
2. **Cut the 2.5D ImageNet branch.** Everyone, including the Expansionist. A 10 mm-FWHM blob has no texture for ImageNet priors to help with; the RSNA evidence is 50× the data on a different modality; Rec A's own Hamburg citation (194k-param 3D nets on ~1,100–1,740 scans) argues against Rec A. It adds a second training loop, a licence audit and ~500 MB of zip — three new ways to produce no valid file.
3. **Three calibration parameters, not 24.** Probability-average the ensemble, one blend weight, one temperature, one prior-mix λ. The OOF noise floor is ~0.014 nats; anything with more degrees of freedom (A's hill-climb, D's L2-logistic blender, which B's own citation says destroys calibration) is fitting noise.
4. **Random-stratified CV fits the calibrator; grouped/leave-cluster-out runs once as a diagnostic to size λ.** Nobody wants T fitted on pessimistic grouped folds.
5. **The alignment gate is non-negotiable.** Hotspot-anchored radiomics + L2 logreg must lift OOF AUROC from 0.469 to >0.90 before any CNN is trusted. Until then there are zero validated model families, not two.
6. **Two arms sharing one `preprocess()`**: 0.9M-param 3D CNN + feature logreg. Not three families.
7. **The score is a tail problem plus a P(no valid file) problem.** Blur+noise (and quantization) augmentation is the uncontested multi-site lever and is config, not code.
8. **Adopt from Rec D**: quantization augmentation, stage-wise CSV writes, export LR/calibration to JSON (no pickles), `--selftest` on synthetic shapes.

## Where the Council Clashes

**Is the test set IID with training?** First Principles says yes flatly ("one corpus, split — IID"). Contrarian and Outsider say that is a guess dressed as a finding. Verified: `DAT_PARKINSONS_CHALLENGE.md` line 37 says the private test set "may differ from the public-leaderboard test set," and the split is undisclosed. So the IID premise is unproven. Resolution: not to pick a side but to keep the cost asymmetry in view — over-confidence is unbounded, under-confidence is bounded — and to buy exactly one hedge (λ) sized by the shift diagnostic.

**How much hedging?** Rec C ships three stacked hedges (T inflated to the P75 of per-fold T, λ floored at 0.02, ε from the worst fold). The Contrarian's arithmetic is right that each is a ~0.01-nat tax and three of them is "3rd vs 8th." The Expansionist goes the other way and adds per-volume anomaly shrinkage λ_i on top. All five reviewers named the Expansionist's version as the biggest blind spot: ~10 fitted thresholds on 1,362 samples against a 0.014-nat floor — exactly the overfitting A was scolded for, relocated. Both extremes are wrong. **One hedge.**

**Bracketing λ across submissions.** The Expansionist proposed it on an "unverified" claim; the Chairman checked and it *is* verified — line 37 says the best private-set submission counts automatically. So bracketing is free upside in principle. What is *not* recorded is the submission-slot limit, and the execution machines are a shared resource (line 76). So: bracket with two submissions, not a grid.

**Where the 40 spare GPU-hours go.** The Expansionist wants 30+ checkpoints and an augmentation sweep. Executor and Contrarian point out the builder has three other hackathons in this window (Rote closes Sept 7, Apart is Sept 11–13) and that the binding constraint is solo wall-clock, not GPU. And the GPU-hours may not be usable at all — see next section.

## Blind Spots the Council Caught

1. **The training venue may void the prize.** No advisor mentioned it; a reviewer caught it. Verified at `DAT_PARKINSONS_CHALLENGE.md` lines 95–97: staff's compliance test is "every copy must be fully deletable" and the provider's terms must not grant retention rights; a participant pointed out Kaggle's ToS grants Kaggle a perpetual, irrevocable licence over uploaded content and deletion takes up to 6 months; staff had not answered as of today. The doc itself names the safe route: a self-controlled, wipeable machine or a rented instance. **Every one of the five advisors optimised GPU-hours on Kaggle. This changes the plan.**
2. **Repeat patients.** Only the Contrarian raised it, and it is real: EDA §8 ran content hashing on *one* group (460 of 1,362 volumes) and only for exact near-copies; follow-up scans of the same patient across folds will inflate every OOF number and shrink the shipped T. Rec C's 16³-embedding cosine audit is a two-hour job and must precede fold construction.
3. **Nobody set a baseline number.** Prior-only scores ~0.69; a center-of-mass-crop CNN with no hotspot logic scores an unknown. Without those, the "≥0.02 OOF gain" gate for any later addition has no denominator, and the 0.469→0.90 jump is uninterpretable.
4. **The prior-only submission is a free measurement of test prevalence.** The public LB log loss of a constant p=0.548 file inverts to the public test's class balance. It also proves the container path exists. One submission, two payoffs.
5. **The 3-hour limit was never made measurable.** Registration timing was argued about; the end-to-end Docker wall-clock (image cold start, import time, thread pinning, per-volume preprocess) was never scheduled. Term-3 risk is asserted everywhere and instrumented nowhere.
6. **Hotspot's failure mode is the ultra-low regime.** EDA: 25/500 sampled scans (5%) have `intensity_max < 10` — roughly 3-bit images. Rec C's "threshold at 0.6 × smoothed max" can tie or blob on those. Mitigated by smoothing in mm before thresholding, but must be *looked at* on day 1.
7. **The antspyx contradiction is resolved.** Rec A cites the runtime `pyproject.toml` listing `antspyx`, `monai`, `simpleitk`; Rec B's "unknown" is stale. Irrelevant to the MVP now that registration is cut.
8. **Label noise bounds achievable log loss.** DaT visual reads disagree ~5–10% between raters; this is a floor under everyone's score and a reason T > 1 is right regardless of fold scheme. It also means top pipelines will be indistinguishable on OOF — a further argument for shipping early rather than tuning.
9. **Laterality.** A reviewer's worry that a hotspot crop centering on the brighter putamen flips laterality is real for crop *centering* but not for mirror-flip augmentation: the target is binary "pathologic," and flipping L↔R preserves it. Fixed in the crop design below (x-center from the brain-mask midline).

## The Recommendation

Build the Rec C MVP, with one hedge, trained on a wiped rented GPU, with a measured container. Concretely:

### Preprocessing — one `preprocess(path) -> (crop, scalars)` used identically in training and `main.py`
- `nibabel` load → `as_closest_canonical` (RAS) → resample to **3 mm isotropic** (linear). 10 mm FWHM physics makes 2 mm a 3× cost for no information.
- Brain mask: Otsu on a σ=6 mm-smoothed volume, largest component, fill holes.
- Intensity: per-volume divide by the p75 of masked voxels (Rec C's template-free DVR); NaN/zero guard falls back to p99.
- Anchor: smooth σ=6 mm (in mm, so the 3-bit scans become continuous fields), threshold at 0.6× smoothed max, largest component centroid. **x-center = brain-mask midline** (kills the laterality-centering problem); y and z from the hotspot. Sanity rule: if the supra-threshold component exceeds 30 mL or is under 1 mL, or the centroid is >40 mm from the brain-mask centroid, fall back to brain-mask center-of-mass with a fixed superior offset and set a flag.
- Crop **72×72×48 at 3 mm** (nearly whole-brain, so containment is guaranteed even with a 15 mm anchor error), zero-pad, fp16 cache.
- Scalars appended to the CNN head and used as LR features: log(p99/p75), mask volume mL, masked skewness, mirror-symmetry NCC about the midline, z-extent mm, native spacing, number of distinct grey levels, anchor-fallback flag.
- Hotspot radiomics for the LR arm: striatal-ROI mean/max/asymmetry ratios in the anchored frame (Rec C §5).

### Models — two arms
- **Arm 1:** 0.9M-param 3D CNN (small ResNet-style, MONAI or hand-written, single channel + scalar head). AdamW, cosine schedule, ~40 epochs, BCE. Augmentation: Gaussian blur σ∈[0, 4 mm], additive noise, **quantization to k∈{8,16,64,256} levels** (Rec D), reference-scale jitter ±20%, random affine ±10°/±10%/±3 vox, mirror flip. **5 folds × 3 seeds = 15 checkpoints.**
- **Arm 2:** L2 logistic regression on hotspot radiomics + scalars. Serves as the alignment gate, the QC probe, and the inference fallback if the CNN path dies.
- **Control:** the same CNN on a plain brain-mask CoM crop, 5 folds × 1 seed. Run once so the hotspot gain is a measured number.
- **Registration: cut.** Not on any day's schedule. Reopen only if Sept 11 arrives with a scored submission and a free day, and only with a ≥0.02 OOF gate. **2.5D: cut.**

### Training venue
Do not put competition data on Kaggle unless DrivenData staff clears it in writing on thread 11479. Rent a wiped instance (RunPod/Lambda/vast, an A10/4090-class card at ~$0.5/hr); 15 checkpoints × ~15 min plus the control is ~5 GPU-hours, well under $10. Pin seeds, write a lockfile, snapshot the exact training command so the Sept 8 artefact is rebuildable Sept 15. Wipe the instance at the end and record it in the disclosure.

### CV and calibration
- Run the 16³-embedding cosine audit (>0.98 → same group). Stratified 5-fold with those groups forced together. That is the only grouping.
- OOF: p_cnn = mean of the 3 seeds' fold-OOF probabilities; p_lr = OOF LR. p_blend = w·p_cnn + (1−w)·p_lr, w on a coarse grid. T on logit(p_blend). Then p = (1−λ)·p_cal + λ·π. Clip to [1e-4, 1−1e-4] as a NaN guard only. **Three numbers. No T inflation, no worst-fold ε.**
- λ is the single hedge and is set by a precommitted rule: run leave-out-2.46 mm-cluster and leave-out-3.90 mm-cluster once as diagnostics; if the worse of the two inflates log loss by <50%, λ=0.02; otherwise λ=0.05. π = the class balance inverted from the prior-only public LB score, falling back to 0.548.
- One nested loop for an honest post-calibration number; anything under 0.02 OOF gain is noise and does not ship.

### Inference engineering (`main.py`)
- First action: write `submission.csv` with p=π for every uid. Every later stage overwrites it in place (Rec D), so a crash at any point leaves a valid file.
- Stage A: multiprocessing pool over preprocess with `OMP/ITK/MKL threads=1`, per-volume try/except → prior and per-volume timeout. Stage B: LR predictions → overwrite CSV. Stage C: CNN ensemble on GPU, batched, TTA = mirror flip only → overwrite CSV.
- Deadline ladder against a wall clock started at import: past 2h10m skip remaining CNN checkpoints; past 2h35m stop and keep what is written.
- Weights as `state_dict` `.pt`; LR, scaler and (w, T, λ, π) as JSON. `--selftest` on three synthetic volumes including a 3-bit 25-slice one. Log aggregates only; never print anything about a test scan.
- Measured, not assumed: full Docker run on the CPU-only path over 100 volumes, per-stage wall-clock, extrapolated to 2,000 test volumes with a 2× margin before any real submission.

### Submissions
(1) Prior-only, as early as possible. (2) LR + 5-fold CNN, Sept 8. (3) Full 15-checkpoint ensemble with rule-set λ, Sept 14. (4) Same weights at the other λ end, if slots allow — line 37 makes the best private score count automatically.

### Dated build order
- **Sept 5 (tonight):** post the Kaggle question on thread 11479 (five minutes). Repo skeleton, `preprocess()`, prior-only `main.py`, pull the runtime image, smoke test.
- **Sept 6 (Sat):** cache all 1,362 crops on the 12-core box. QC grid: 30 random + **all** ultra-low-regime scans + all short-z scans. Embedding audit → fold groups. Hotspot radiomics + LR: **gate at OOF AUROC > 0.90**. Submit prior-only for real; invert π. Provision the rented GPU.
- **Sept 7 (Sun, Rote closes 8pm London):** light day. One CNN fold on the rented GPU for a first honest number and an epoch-time measurement; kick off 5 folds × 1 seed plus the CoM control overnight.
- **Sept 8 (Mon):** fit w, T, λ; full `main.py` with ladder; Docker CPU dry run with wall-clock; **scored submission #2**.
- **Sept 9–10:** seeds ×3; one augmentation ablation (with vs. without quantization + scale jitter), gated at 0.02; the two leave-cluster-out diagnostics; nested calibration estimate; lockfile and rebuild test.
- **Sept 11–13 (Apart sprint):** frozen. Only unattended runs already queued.
- **Sept 14:** final ensemble, final Docker dry run, submissions #3 and #4.
- **Sept 15:** buffer, MIT-release README and disclosure, wipe the rented instance.
- **Sept 16:** deadline. Nothing new.

**Resolutions, explicitly:** (1) hotspot crop, registration cut; (2) 2.5D cut; (3) random-stratified CV with repeat-patient groups for fitting, grouped runs only to set λ via a precommitted rule; (4) probability average → w, T, λ, and nothing else; (5) two arms sharing one `preprocess()`, 15 CNN checkpoints, trained off-Kaggle.

## The One Thing to Do First

Tonight, write `preprocess()` and run the alignment gate on the 12-core box: hotspot-anchored radiomics + L2 logreg on the cached 3 mm crops, with the QC grid that includes every ultra-low-regime scan. If OOF AUROC does not clear 0.90, every other line of this verdict is on hold and the anchor is the only thing you work on; if it does, dispute (1) is closed by measurement and both model arms are unblocked. (Post the Kaggle question on thread 11479 before you start — it is a five-minute action whose answer has days of latency, and it decides where the GPU work runs.)

---

# Appendix A — Peer reviews (reviewers saw anonymized letters; A=Outsider, B=Executor, C=Contrarian, D=Expansionist, E=First Principles)

**Review 1.** Strongest: C (Contrarian) — only one attacking assumptions: hotspot's failure mode is the 25 ultra-low ~10-grey-level scans; repeat-patient audit must precede fold debate; C's own hedge stack = three stacked under-confidence taxes; names competing hackathons. Biggest blind spot: D (Expansionist) — per-volume anomaly hedge thresholds are fit on 1,362 scans against a 0.014-nat noise floor; "bracket λ" leans on an unverified forum claim. All missed: the IID-test claim is itself unverified; left/right asymmetry is label-relevant (hotspot crops centered on the brighter putamen can flip laterality); nobody proposes the cheapest control baseline (3D CNN on a CoM crop with no alignment).

**Review 2.** Strongest: C (Contrarian). B and E close seconds; E's base/tail/crash decomposition is the cleanest framing. Biggest blind spot: D — per-scan λ_i = ~10 tuned hyperparameters on 1,362 samples; burns 40 GPU-hr before a scored file exists. All missed: nobody makes the 3h inference budget MEASURABLE (instrumented Docker dry run at CPU speed); label noise / prevalence shift as a calibration failure mode; a fixed seed/lockfile so the Sept 8 artefact is rebuildable Sept 15.

**Review 3.** Strongest: B (Executor) — dated build order with a hard go/no-go gate. Biggest blind spot: D — hedge stack + 30 checkpoints before a validated model; bracketing is grounded (best-private-counts IS in the challenge doc line 37) but assumes unlimited slots. **All missed: the training venue is a prize-eligibility risk** — challenge doc lines 95–97 record that Kaggle's ToS grants a perpetual licence and staff have NOT cleared Kaggle notebooks. Also: private test "may differ" from public (line 37) — keep grouped-CV λ-sizing as a genuine hedge.

**Review 4.** Strongest: E (First Principles) — score decomposition derives every sub-answer; spots the label-selected template as a disclosable bias. Take E's frame, B's calendar. Biggest blind spot: D. All missed: the antspyx contradiction is resolved (Rec A cites the runtime pyproject.toml); nobody proposed using the public-LB delta on a prior-only submission as a free estimate of test class balance; label noise / inter-rater disagreement (~5–10%) bounds achievable log loss.

**Review 5.** Strongest: B (Executor). E close on reasoning. Biggest blind spot: D. All missed: nobody set a baseline number (prior-only ~0.69; CoM-crop CNN unknown) — the "≥0.02 gain" gate has no denominator; repeat-patient audit should precede fold construction; wall-clock accounting on the actual runtime (Docker image, thread pinning, cold-start).

---

# Appendix B — Round-1 advisor responses (full text)

## Response A — The Outsider

**The plan's foundation is an untested sentence.** Every one of the four documents is downstream of one EDA fact: "a fixed relative-position ROI scored AUROC 0.469." From that, all four conclude alignment is mandatory. But nobody has run the five-minute experiment that decides the biggest disagreement: *does a hotspot-anchored ROI (C's idea) score >0.9?* If yes, question (1) dissolves and the whole template/ANTs/DVR-mask apparatus in A, B, and D becomes optional. If no, C's "self-localizing striatum" claim — asserted with "high confidence" by two Claude models with zero data — is dead. Run that before writing any other line of code. It's an afternoon on the 12-CPU box.

**The documents contradict each other on facts, not opinions:**

- **Is antspyx in the runtime?** A quotes `pyproject.toml` and says yes. B says "availability unknown, build a pure-PyTorch fallback." One of these was written without reading the other's source. Check once; stop hedging.
- **What does the test set look like?** A, B, D assume hospitals may be held out; C says "one corpus from 10 centres split into train/test." The prompt says "same 10 hospitals." This isn't a modeling philosophy — it's a fact about the competition, and it determines question (3). If the organizers said same-hospital, grouped CV for calibration is simply wrong (C is right). If they didn't, C's "almost certainly IID" is a guess dressed as a finding.
- **Registration timing:** A says 5–20 s/vol (extrapolated from MRI), D says 5 s at 4 mm ("from experience, not measured"), B says 5–20 s. Three estimates, zero measurements, and the entire inference-timeout design depends on it.
- **Voxel size:** 2 mm (A, B, D) vs 3 mm (C). Everyone cites the same "~10 mm FWHM" physics. Same fact, opposite conclusions — so the fact isn't driving the choice.

**Things a non-specialist finds bizarre:**

- A proposes ~24 fitted calibration parameters on 1,362 samples for a metric whose noise floor C computes at ~0.014 nats. Nobody in A checked whether their tuning signal exceeds their noise. C's arithmetic is the only place anyone did.
- D's blender is an L2-logistic meta-learner. B says, citing a paper, that exactly this "destroys calibration." They were written the same day. Pick one.
- B's headline evidence for the 2.5D branch is RSNA winners with 3–10k studies of CT. C points out that's ~50× more data of a different modality. A cites *both* the RSNA and the Hamburg (tiny 3D nets, similar n) results and picks the one that doesn't match its data size.
- "Pseudo-site" appears everywhere as if it were a known quantity. It's spacing×shape×intensity-regime, which the EDA says has 30+ clusters with four holding 89%. C notes a "5-fold grouped" split under that grouping yields lopsided folds. So the phrase hides an unresolved decision.

**What I'd actually do:** C's MVP, with D's stage-wise CSV writes and quantization augmentation, one model family plus the linear feature probe, three calibration parameters, random-stratified CV. Ship by Sept 8. Registration becomes a gated Sept 11 experiment — it only earns its place with a measured ≥0.02 OOF gain, not a citation.

---

## Response B — The Executor

**Monday morning is today, Friday. The only question that matters: what produces a scored `submission.csv` by Sept 8?** Answer: Rec C's MVP, almost verbatim. Everything else is a Sept 9+ decision gated on a leaderboard number.

**Build order, no negotiation:**

1. **Tonight:** one `preprocess(path) -> array` function (C §5). RAS → 3mm → brain mask → p75 normalize → hotspot-anchored 72×72×48 crop → fp16 cache. Runs on the 12-core box; zero GPU. Visual QC grid of 30 crops.
2. **Sat:** hotspot radiomics + L2 logreg. This is the go/no-go gate: OOF AUROC must jump from 0.469 to >0.90. If it doesn't, alignment is broken and nothing downstream matters.
3. **Sun:** `main.py` with the pre-written prior CSV, `ITK/OMP threads=1`, per-volume timeout, deadline ladder. Docker CPU run on 100 volumes.
4. **Mon Sept 8:** smoke test → scored submission (features + whatever CNN folds finished overnight on Kaggle).

**Resolving the five disagreements by "what's the first step":**

**(1) Registration.** A/B/D all need `ants.build_template` on day 1. D's own plan says "measure ANTs affine time on day 1 — this number drives everything." That is a dependency sitting on the critical path before anyone has a number. C's hotspot crop is ~50 lines of scipy with no debug loop. Ship hotspot. Registration is a time-boxed Sept 11–12 ablation with a >0.02 OOF gate, exactly as C §5 item 4. If it isn't winning by Sept 12 evening, it's dead.

**(2) 2.5D branch.** Cut. D lists it as "first to drop under time pressure" — we are under time pressure now. It costs a second training loop, license verification, 5×45MB in the zip, and A's own Hamburg evidence uses 194k-param 3D nets. Zero Monday-morning value.

**(3) CV.** Random stratified (C). Grouped 5-fold with 528 scans in one cluster isn't a 5-fold; it's a leave-cluster-out with extra steps. Run grouped once, as C says, to size λ. Don't fit T on it.

**(4) Blending.** Probability average → one T → one λ → ε≈0.01. Three parameters. D's L2-logistic blender needs nested CV to be honest — that's a day. A's hill-climb fits 24 params to a 0.014-nat noise floor. Neither ships anything.

**(5) Families.** Two: 0.9M-param 3D CNN (5 folds × 3 seeds, ~2.5 GPU-hr — one Kaggle session) + logreg on features. A's 12–16 GPU-hr and D's 18 GPU-hr both assume the solo builder isn't also doing other hackathons.

**Adopt from D:** quantization augmentation (free), export LR to JSON/LightGBM to text instead of pickle (kills a real runtime failure mode), `--selftest` on three synthetic shapes.

**Where time actually goes if you ignore this:** debugging `ants.build_template` drift on Sept 7, writing a GRU dataloader on Sept 10, and submitting for the first time on Sept 14 with no wall-clock measurement. The pipeline that wins is the one that has four scored submissions by Sept 13, not the one with three model families on paper.

---

## Response C — The Contrarian

**The fatal flaw is shared by all four: 100+ pages of design, zero measurements.** ANTs "5 s/scan at 4 mm" (D), "Tier-1 2.5 s/volume" (C), "15 s/epoch on T4" (C), "striatum is the brightest structure in every scan" (C) — every load-bearing number is asserted by a model that never opened a voxel. Meanwhile the one real measurement you have (EDA §7, AUROC 0.469) says your intuitions about this data are wrong. Anything you build on unverified numbers is the thing that "never produces a valid scoring submission."

**Attacking the disagreements:**

**(1) Registration vs hotspot.** C's self-localizing-striatum claim is plausible and untested. Its failure case is precisely the EDA's ultra-low regime: 5% of scans have ~10 grey levels, so "threshold at 0.6×smoothed-max" on a 3-bit image can produce a blob covering half the brain, or a tie. D's answer is no better — Mattes MI with default bins on a 10-level image at 4 mm is degenerate. Neither camp has evidence. The tiebreaker is cost: hotspot is 50 lines and fails visibly; `build_template` is a debug loop (C is right). Ship hotspot, but **Day 1 = run it on the 25 ultra-low scans and the 33-slice scans and look at them.**

**(2) 2.5D.** Cut it. D's own sacrifice order ("drop C first") is a confession. A's cited evidence for M2 is 3–10k CT studies; its cited DaT evidence is 194k-param 3D nets. A argues against itself.

**(3) Grouped vs random CV.** This argument is theater. C computes the OOF noise floor at 0.014 nats, then estimates the grouped-CV calibration error at 0.01–0.03 — inside the noise. And "pseudo-site" is fiction either way: 2.46 mm is 528 scans, probably three hospitals sharing one scanner model. The CV risk nobody except C mentions is **repeat patients**: the EDA duplicate check hashed only exact near-copies in one cluster; follow-up scans of the same patient across folds will inflate every OOF number and shrink your shipped T. Do that audit before arguing about fold schemes.

**(4) Blending.** D's L2-logistic blender directly contradicts B's strongest finding ("LR stacking destroys calibration"). Average, one T, done. But notice C's hedge stack — T inflated to P75, λ≥0.02, ε from the worst fold — is three separate under-confidence taxes, each ~0.01. C says 0.02 is "3rd vs 8th." Hedges aren't free; pick one.

**(5) Families.** All four say "CNN + ROI features are decorrelated." They aren't — both die if alignment is off. Until the ROI probe jumps 0.469→0.90, you have zero validated families. Realistically: one CNN, seeds × folds, plus the probe as QC.

**What's missing everywhere:** you have three other hackathons in this window (Rote closes Sept 7, Apart Sept 11–13). "Scored submission by Sept 8" is fantasy. Submit a prior-only `main.py` **today** to prove the container path exists, then a CoM-crop CNN. Everything else is optional.

---

## Response D — The Expansionist

**The upside everyone is leaving on the table: this competition is scored on the tail, and nobody on the leaderboard is engineering the tail.**

Rec C's arithmetic is the most important number in all four documents: at 95% accuracy, 2% of scans confidently wrong at p=0.999 *doubles* your log loss. A and B chase the mean via architectures and calibrator taxonomies. The leaderboard edge is C §4's **per-volume anomaly hedge** — mirror-symmetry NCC, mask volume, padded fraction, grey-level count, anchor distance, each thresholded against frozen training percentiles, mapped to per-scan shrinkage λ_i. It's rules-compliant, costs a day, and no competitor will have it. Ship it as a first-class component, not a Sept 12 afterthought.

**Resolutions, from the "what if it works" angle:**

1. **Hotspot crop, and it's bigger than an MVP.** C frames it as safer; I frame it as *liberating*. Template-free means zero registration debug loop, zero external assets, cleanest possible MIT release, and — the hidden bonus — if the Tier-2 registered crop later works, you now have **two crops as decorrelated ensemble members** rather than one pipeline. Diversity from geometry, free.

2. **Cut 2.5D, but keep what it stood for: diversity.** C's 50× data-regime argument is right. Get the diversity elsewhere at zero engineering: C's 3 seeds × 5 folds × 2 crop sizes = 30 checkpoints, plus a 2 mm vs 3 mm spacing variant. B's evidence (2604.11798) says checkpoint ensembles + TTA are the cheapest calibration gain that exists. Compute isn't the constraint — D budgets 18 of ~60 GPU-hr. **Forty GPU-hours are sitting unused.** Spend them on seeds and the augmentation sweep, which run unattended while you sleep.

3. **Random stratified CV for the calibrator (C)**, and here's the undervalued fact: Rec A's forum table says **"best private-set submission counts automatically."** If that holds (verify on thread 11493), you don't have to *guess* the under-confidence safety margin — **bracket it**: submit the same models at λ ∈ {0.02, 0.05}, T inflation ∈ {1.0, 1.2}. The grouped-vs-random debate becomes moot; let the private set pick.

4. **Average probabilities, calibrate once** (B/C). Three parameters, not 24. Every degree of freedom you don't spend on blending you can spend on hedging.

5. **Two families, many seeds.** Not three families, one seed.

**Undervalued cheap edges:** D's quantization augmentation — the ultra-low regime is 5% of training; if one private hospital is regime-heavy, everyone else is miscalibrated there and you aren't. C's reference-scale jitter (unpublished, targeted at exactly the test-time failure mode). A's blur+noise at C's higher σ ceiling.

**The bigger prize:** the sponsor wants a deployable, interpretable, releasable model. Zero-external-weights + hotspot-anchored SBR features (B's "interpretability story") is simultaneously the fastest path *and* the strongest post-competition story. The conservative pipeline is also the ambitious one.

---

## Response E — The First Principles Thinker

**You're asking "which pipeline." The real question is "what is the score made of, and what kills it?"**

Log loss on a ~0.98-AUC problem decomposes into three terms: (1) a base term any competent registered-or-cropped small CNN reaches, (2) a tail term from confident errors (C's arithmetic: 2% of samples wrong at p=0.999 doubles your score), and (3) P(no valid file) × ~0.69. Term 1 is nearly fixed across A–D. Terms 2 and 3 are the entire competition. Every "disagreement" collapses once you optimize for those.

**1. Registration.** Ask what registration buys. The CNN needs the striatum *inside the box*, not voxel correspondence. The striatum is the brightest blob in every scan; a hotspot centroid puts it in the box with zero moving parts (C). Correspondence matters only for atlas-mask features, and C shows hotspot-anchored features work too. A's template-space reference mask is worse than useless: when the affine slips, ref inflates or deflates and produces confident errors in *both* directions — it feeds term 2 directly. A's label-selected template (150 normal + 100 abnormal) is a bias you have to disclose. Registration is optional infrastructure, not spine. Cut it from the critical path.

**2. 2.5D ImageNet branch.** What problem does ImageNet pretraining solve? Missing edge/texture priors. A 10mm-FWHM blob has no texture. A cites the Hamburg 194k-param 3D nets on 1,100–1,740 scans as the closest analogue, then picks the RSNA recipe built for 10k studies × 400 CT slices. Follow A's own evidence: cut it. It's also a second training loop, a licence audit, and 500MB of zip — three new ways to fail term 3.

**3. Grouped vs random CV for calibration.** Ask what distribution the test set is drawn from. One corpus, 10 hospitals, split. IID. Grouped CV answers "what if a new hospital?" — a question the organizers didn't pose. Fitting T on pessimistic OOF ships under-confident probabilities forever (C). Use grouped/leave-cluster-out *only* to size λ. B and D conflate a diagnostic with an estimator.

**4. Blending.** OOF noise floor ≈0.014 nats. A's ~24 parameters and D's L2 blender (which B's own citation says destroys calibration) are fitting noise. Average probabilities, one T, one ε≈q from the worst fold's top decile. Three numbers.

**5. How many families?** Count code paths that can crash, not models. Two arms sharing one `preprocess()`: 0.9M 3D CNN + hotspot radiomics/LR (the LR is also your alignment sanity probe: AUROC must jump from 0.469 to >0.9 or stop everything).

**Where the remaining gain actually is:** all four recs agree blur+noise augmentation is *the* multi-site lever, and nobody disputes it. It's config, not code. After the Sept 8 scored MVP, spend GPU-hours there and on seeds, not on templates.

You don't have a modelling problem. You have a scored-file-by-Sept-8 problem with a calibration tail attached.
