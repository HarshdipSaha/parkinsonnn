# Pipeline Recommendation D — Unanchored Design-Space Brainstorm

**DaT Parkinson's Challenge (DrivenData #311) — 3D DaT-SPECT normal/abnormal, scored by log loss**
Written 2026-09-05 (11 days to the Sept 16 deadline). Subagent D of 4.

---

## 1. Method note

This document is a **from-scratch, engineering-tradeoff-driven design-space exploration**, produced under the `brainstorming` skill's discipline (explore context → enumerate 2–3+ real alternatives per decision with tradeoffs → recommend → write the design doc). Because this ran as a subagent with a fully specified brief and no human in the loop, the skill's interactive one-question-at-a-time / approval-gate steps were collapsed into the explicit options tables below; each "Pick" line is the decision that would otherwise have been put to the user.

It is **deliberately unanchored**: I did not read `PIPELINE_REC_A_PAST_WINNERS.md` or any other subagent's proposal, and I used `LITERATURE_REVIEW.md` only for constraints (PPMI is prize-ineligible; all published methods register to a template first; nobody in the literature optimizes for log loss). Nothing below is copied from a paper; where a choice happens to coincide with published practice that is convergence, not anchoring.

Ground truth used: `DAT_PARKINSONS_CHALLENGE.md`, `EDA_REPORT.md`, plus three cheap facts checked in this session:
- the entire training set is **629 MB compressed (~0.46 MB/scan)** — data volume is a non-issue everywhere;
- the smoke-test archive is **10 MB** (roughly 15–25 scans) and must finish in **6 minutes** including cold imports;
- the local dev box has **12 logical CPUs, no NVIDIA GPU** — registration/preprocessing can run locally, training must go to Kaggle (T4x2/P100, ~30 GPU-hr/week).

Constraint summary that every decision below is measured against:

| Constraint | Value | Consequence |
|---|---|---|
| Inference box | 1× A100 80GB, 24 vCPU, 220 GB RAM, **3 h hard cap**, **no internet**, no root | GPU is nowhere near the bottleneck; CPU registration and cold-start are. Everything (weights, template, masks) must be in the zip. |
| Smoke test | 6 min cap | Imports + template load + registering ~20 scans + inference must fit; forces a fast registration path. |
| Metric | log loss (private LB decides) | Calibration and worst-case-per-sample control matter more than the last 1% of AUROC. |
| Training data | 1,362 scans, 747/615, 10 hospitals, 3 intensity regimes, 30+ spacing clusters | Small-N: parameter-frugal models, heavy but *physically meaningful* augmentation, grouped CV. |
| Training compute | Kaggle ~30 GPU-hr/week (T4x2 / P100), 11 days | ~60 GPU-hr total across two quota resets; plenty if models are small. |
| Prize safety | External weights must be public + commercially licensed; ImageNet weights OK if MIT/Apache/BSD; DINOv3 and PPMI not OK | Only ImageNet 2D backbones are usable as pretrained components; all 3D models are from scratch. |
| Rules | No pseudo-labeling; process test samples independently; log ≤300 lines; **never log info about test data** | No test-time adaptation / batch statistics across test scans; watchdog logging must be aggregate-only. |

---

## 2. Design-space exploration

### 2.1 Spatial alignment

The EDA's chance-level fixed-ROI result (AUROC 0.469) says the striatum is not in a predictable voxel location across centers. Something must put it there. Key physics fact driving cost: DaT-SPECT has an intrinsic resolution of roughly 8–12 mm FWHM, so **registration at 4 mm voxels loses nothing** relative to the native 2.3–3.9 mm grids. That single observation drops registration cost by ~5–10× and is the main lever in this axis.

| Option | Pros | Cons | CPU cost / volume (est.) | Chosen? |
|---|---|---|---|---|
| **A. None** — isotropic resample + center crop/pad | Trivial, zero failure modes, <0.5 s | EDA proves position varies; CNN must learn translation invariance from 1,362 scans; fixed atlas masks impossible | ~0.3 s | No (but it is the *last-resort* fallback) |
| **B. Center-of-mass + isotropic resample** (+ optional PCA principal-axis rotation) | Never "fails", ~1 s, fixes translation which is the biggest nuisance; robust to partial-FOV scans | Leaves rotation/scale; PCA axes have sign ambiguities on a near-spherical head; fixed atlas masks only approximately valid | ~1 s | **Yes — as initialization and as the guaranteed fallback** |
| **C. Rigid (6-DoF) to a data-derived template** | Handles rotation; cheap-ish; no scale distortion of pathology | Head size varies ±10%, so atlas masks still slop; not much cheaper than affine in ANTs | 3–8 s at 4 mm | No (dominated by D) |
| **D. Affine (12-DoF) to a data-derived template, run at 4 mm, transform applied at 2 mm** | Normalizes size → fixed striatal/reference masks become valid → enables hand-crafted SBR features and a tight striatal crop; MI metric is intensity-scale invariant; ANTs is already in the runtime | Can mis-converge on odd scans (25-slice partial-FOV volumes, ultra-low-count 3-bit images); needs a template we must build ourselves; ~5 s/scan | **~3–8 s at 4 mm** (15–40 s at full res) | **Yes — primary** |
| **E. Deformable (SyN)** | Best anatomical overlap | 1–5 min/scan; *the pathology is a shape/intensity change of the striatum* — a deformable warp toward a normal template can literally warp the disease signal away; useless for a blob image at 10 mm resolution | 60–300 s | No |

**Pick: B → D with B as fallback**, i.e. COM-initialized affine registration to a **template built from our own training normals**, computed on 4 mm downsampled copies, with the resulting transform applied to produce a 2 mm isotropic volume in template space.

Why: it is the cheapest option that makes fixed masks valid, which unlocks the feature arm (§2.4) and the striatal crop (§2.3). The template must be data-derived because no permissively licensed DaT-SPECT template exists and a template from *these* scanners is better matched anyway.

**Template construction (offline, once):** take ~50 normal scans from the largest cluster (256³ @ 2.46 mm, the most homogeneous group) → normalize intensity (§2.2) → COM-align → mean → affine-register all ~600 normals to that mean → re-average → repeat once more. Output: `template.nii.gz` (2 mm, cropped to a ~96×112×96 brain box), `brain_mask`, `striatum_L/R`, `reference_mask` (brain minus 10-mm-dilated striatum). Masks come from thresholding the template itself (striatum = top ~2% voxels inside the central box, split L/R at the midline, ant/post halves ≈ caudate/putamen).

**Failure handling at inference (this is the part that actually matters for the 3 h box):**
1. Run registration in a subprocess-pool worker with a **per-scan wall-clock timeout (60 s)**.
2. After registration, **validate the transform**: scale factors (singular values of the 3×3 block) in [0.7, 1.4], shear small, translation < 60 mm, and normalized cross-correlation with the template above a threshold learned from the training distribution (e.g. below the 1st percentile of training NCC ⇒ suspect).
3. On timeout / exception / failed validation ⇒ fall back to **B** (COM + isotropic resample, cropped to the same box). The models are trained with strong affine augmentation (§2.5) precisely so they degrade gracefully on B-aligned inputs rather than collapse.
4. Count fallbacks and log *only the count* (not uids, not shapes) — rules forbid logging test-set information.

Partial-FOV scans (z-depth 25 × 3.9 mm ≈ 97 mm of coverage) need the registration metric restricted to a mask around the central brain (template brain mask eroded), otherwise the missing cerebellum/vertex drives the fit. Set `mask=` on the fixed image in `ants.registration`.

### 2.2 Intensity normalization

Three raw-count regimes (~5–10, ~100–1,000, ~10,000–65,000) with **no header field** to fix them. The ultra-low regime is the nasty one: a max of 5–10 means the image has 3–4 bits of dynamic range — heavy quantization, not just scaling. Whatever we choose must be (a) invariant to a global multiplicative constant, (b) tolerant of an *additive* background shift (scatter correction differs by center), (c) not destroy the clinically meaningful quantity, which is the striatal-to-non-specific ratio.

| Option | Pros | Cons | Robust to unseen center? | Chosen? |
|---|---|---|---|---|
| **A. Per-volume percentile clip [p1, p99.5] → [0,1]** | One line; scale-invariant; standard | The p99.5 reference *is the striatal peak*, i.e. the diseased structure; a linear rescale preserves ratios mathematically, but it puts the network's "1.0" on the thing that shrinks with disease, so the absolute uptake signal must be re-derived from background level — harder with small N | Medium | As a **second input channel** only |
| **B. z-score inside a brain mask** | Scale + offset invariant | Mean/std are a mixture of specific + non-specific uptake; std depends on striatal size → mild signal leakage into normalization | Medium-high | No (B ≈ C but less interpretable) |
| **C. Reference-region ratio** — divide by mean of non-specific brain (brain mask minus dilated striatum), after subtracting an air/background floor | Makes voxel value ≈ (SBR + 1); exactly the clinical quantity; scale-invariant; offset handled by floor subtraction; masks are valid after affine registration (§2.1) | Requires registration for the mask (fallback: use the COM-aligned approximate mask, or "brain voxels below the 90th percentile" as a registration-free proxy); sensitive if the reference region is partially outside a partial-FOV scan | **High** | **Yes — primary channel** |
| **D. Histogram matching to the template** | Removes all acquisition differences | Non-linear; matching an abnormal histogram to a normal template *pushes the abnormal striatum up* — actively destroys signal | High robustness, wrong target | No (rejected outright) |
| **E. Learned (InstanceNorm / GroupNorm at the stem)** | Free; adapts per-sample | Cannot fix quantization; on its own re-introduces the "normalize by the diseased peak" problem | Medium | Yes, as the network's first normalization layer, on top of C |

**Pick: C as channel 1, A as channel 2, E inside the network**, preceded by a **light Gaussian smoothing (σ = 1 voxel at 2 mm)** applied identically to every scan. The smoothing is not cosmetic: it turns the 3-bit ultra-low-regime images into something continuous and brings all centers slightly closer to a common effective resolution. The two-channel input costs nothing and gives the network both the physical ratio and a contrast-stretched view.

Registration-free proxy for the reference region (used in the B-fallback path and as a sanity feature): brain mask = largest connected component of (smoothed volume > Otsu threshold); reference = brain voxels below the 90th percentile of the brain distribution. Verified on paper, must be checked visually on ~20 scans from each regime on day 1.

**Explicitly excluded:** the intensity regime itself (log10 of raw max) as a model feature. The EDA shows a modest regime-label association (61% vs 53% abnormal). It is a center signature; exploiting it might give +0.01 on the public LB and lose on an unseen private center. Not worth it.

### 2.3 Input representation

| Option | Pros | Cons | Cost | Chosen? |
|---|---|---|---|---|
| **A. Full brain 3D** (~96×112×96 @ 2 mm ≈ 1M voxels) | Nothing discarded; robust to residual registration error | 1M voxels × 1,362 samples = extreme overfitting surface; most of the volume is irrelevant to the label; slower | ~4 MB/scan float32 | No (as primary) |
| **B. Striatal-centered 3D crop** (64×64×48 @ 2 mm = 128×128×96 mm around the striatal centroid in template space) | 7× fewer voxels, focuses capacity on the diagnostic region while keeping cortical context; fast; less overfitting | If registration lands >~30 mm off, the striatum leaves the box → covered by the 60-mm-ish margin + fallback validation | ~0.8 MB/scan | **Yes — primary for the 3D CNN** |
| **C. 2.5D slab stack** (3 channels = 3 adjacent ~6 mm axial slabs through the striatum, evaluated at several slab offsets) | Unlocks ImageNet-pretrained 2D backbones (prize-eligible); very fast; a natural "TTA over depth" | Loses some 3D shape info; pretrained filters are for photos, so gains over scratch are uncertain but non-zero for small N | ~0.2 MB/scan | **Yes — secondary arm (diversity)** |
| **D. Max-intensity projections** (axial/coronal/sagittal → 3-channel 2D) | Fastest baseline; visually how radiologists often skim | MIPs of the striatum alone hide depth; the EDA's MIP grid showed no obvious separation *unregistered* — may be fine registered | trivial | Day-1 sanity baseline only |
| **E. Hand-crafted SBR/asymmetry features** (L/R × caudate/putamen mean & p90 ratios, asymmetry indices, putamen/caudate ratio, volume above ratio thresholds, background stats) | ~20 numbers; directly the clinical read-out; well-calibrated with a linear model; interpretable; near-zero compute; independent error profile from CNNs | Depends entirely on registration + masks being right; no texture/shape nuance | trivial | **Yes — third arm, and the fallback predictor** |

**Pick: B (3D CNN) + C (2.5D) + E (features).** Resolution 2 mm isotropic: finer than the coarsest native grid (3.9 mm) but not beyond the physics (10 mm FWHM), and it keeps the crop at 64×64×48. 2.5 mm would also be defensible and 30% cheaper; decide on day 3 by OOF, default 2 mm.

### 2.4 Model family

Overfitting budget: 1,362 scans (≈1,090 per training fold). Rule of thumb for 3D medical CNNs at this N: 0.5–3M parameters with global average pooling, dropout, weight decay, and augmentation is comfortable; 30M+ parameters from scratch is not.

| Option | Pros | Cons | Params | Chosen? |
|---|---|---|---|---|
| **A. Small 3D CNN from scratch** (5 stages, conv3×3×3 + GroupNorm + SiLU, channels 16→32→64→128→192, GAP, dropout 0.3, linear) | Right-sized for N; fast (batch 16 of 64×64×48 fits in <4 GB); fully controllable; trivial to bundle | No pretraining; needs augmentation discipline | ~1.5–2.5M | **Yes — primary** |
| **B. 3D ResNet-18/34 (MONAI)** | Standard; more capacity | 33M+ params on 1.1k samples; no prize-safe 3D pretrained weights (MedicalNet is trained on DUA-gated datasets; skip); slower with little expected gain | 33M | No |
| **C. 2.5D ImageNet-pretrained 2D backbone** (torchvision `resnet18` BSD-3 or timm `resnet18.a1_in1k`/`efficientnet_b0.ra_in1k` Apache-2.0) over slabs + mean over slab offsets | Pretrained low-level filters help with small N; prize-eligible per staff ruling if license is permissive; 11M params but mostly frozen-ish via low LR | Must bundle ~45 MB weights; ImageNet stats are meaningless for SPECT so the stem is effectively retrained; must **verify the exact weight file's license tag** before bundling (some timm `*in22k*` Meta weights are CC-BY-NC) | 11M (fine-tuned at low LR) | **Yes — secondary arm** |
| **D. Radiomics/SBR features + logistic regression (L2) or shallow LightGBM** | Best-calibrated per parameter; robust; interpretable; a fallback that runs in seconds | Ceiling limited by mask/registration quality | 20–200 | **Yes — third arm** |
| **E. Hybrid: CNN embeddings + features → GBM** | Could capture interactions | Stacking embeddings from OOF models is leakage-prone and fiddly in 11 days; a level-2 *logit* blender captures most of the benefit safely | — | No |
| **F. Ensemble of families** | Log loss loves averaging: different error profiles, reduced overconfidence | Inference cost (negligible here); more moving parts to bundle | — | **Yes — the assembled choice** |

**Pick: F = A (5 folds × 2 seeds = 10 models) + C (5 folds) + D (5 folds, LR and LGBM, keep the better)**, blended at the logit level by a tiny L2-logistic regression fit on OOF logits (§2.6). If time pressure bites, the order of sacrifice is: drop C first, then reduce A to 5 models. D is never dropped — it is also the fallback predictor.

### 2.5 Training strategy

**CV scheme — must mirror a multi-center held-out test.** Hospital IDs are not given, so define **pseudo-site groups from the acquisition signature**: `(shape, spacing rounded to 0.01 mm, dtype, intensity-regime bucket, NIfTI `descrip`/`db_name`/`aux_file` strings if populated, qoffset/pixdim quirks)`. The EDA found qform/sform codes uniform, but the free-text header strings often carry the reconstruction software name and are the strongest site proxy — inspect them on day 1. Then:

- **Primary: `StratifiedGroupKFold(5)` with pseudo-site groups.** The 460-scan 256³/2.46 mm cluster (34%) will dominate one or two folds — accept that; it is the honest estimate. OOF predictions from this scheme are what the blender and calibration are fit on.
- **Diagnostic only: leave-one-big-cluster-out** (train without the 2.46 mm cluster, test on it; same for 3.90 mm) to measure the cross-center penalty and to choose the calibration safety margin (§2.6). Not used for model selection.
- Nothing about the test split is disclosed (same 10 hospitals or not). Grouped CV is the right hedge either way; if the private set is same-site, grouped CV is merely pessimistic.

**Augmentation — which ones simulate real scanner/center shift for SPECT:**

| Augmentation | Simulates | Verdict |
|---|---|---|
| Random affine: rotation ±10°, scale ±10%, translation ±6 mm (done on-GPU with `affine_grid`/`grid_sample` on the batch — avoids a CPU dataloader bottleneck on Kaggle's 4 cores) | Residual registration error, the COM-only fallback path | **Essential** |
| Left–right flip | PD asymmetry can be on either side; label-preserving | **Essential** (also a TTA view) |
| Random Gaussian blur σ ∈ [0, 2] voxels | Different collimators / FBP vs OSEM / post-filters ⇒ different effective resolution — *the* dominant physical inter-center difference | **Essential** |
| Random intensity quantization to k ∈ {6, 12, 32, 128, ∞} levels before normalization | The ultra-low-count regime observed in the data | **Essential, cheap, and probably unique to this dataset** |
| Poisson/Gaussian noise at random count level; random gamma 0.8–1.25; additive background offset ±10% of reference mean before ratio normalization | Count statistics, scatter/attenuation-correction differences | Useful |
| Random anisotropic downsample→upsample (z factor 1–2) | Coarse 3.9 mm / thick-slice centers | Useful |
| Mixup (α = 0.2, on 30% of batches) | Nothing physical, but improves calibration and reduces overconfidence — directly relevant to log loss | Try; keep if OOF log loss improves |
| Elastic deformation | Nothing here — striatal *shape* is the signal | Reject |
| Superior–inferior or anterior–posterior flips, 90° rotations | Anatomically invalid | Reject |
| Cutout over the striatum | Destroys the label | Reject (small cutout elsewhere is harmless but pointless) |

**Loss:** plain **BCE-with-logits**, no class weighting (1.21:1 is mild and re-weighting distorts calibration). Focal loss is rejected: it systematically under-confidences the easy majority and is known to hurt calibration. Label smoothing (0.02–0.05) is a soft yes: it caps the network's confidence, which post-hoc temperature scaling then re-expands only as far as the OOF data justify — a reasonable safety property for log loss. Try both; select by OOF log loss *after* calibration, not before.

**Optimizer/budget:** AdamW, lr 1e-3 (3D CNN) / 1e-4 backbone + 1e-3 head (2.5D), cosine schedule, weight decay 1e-2, **60–80 epochs**, early-stopping checkpoint on validation log loss (not AUROC), EMA of weights. **fp16 autocast + GradScaler on T4** (tensor cores; bf16 is not supported on T4; P100 gains nothing from fp16 so prefer T4x2 sessions). With T4x2, run **two folds in parallel as two processes**, one per GPU — a free 2× on wall-clock.

**Data cache:** register + normalize once, store 2 mm template-space crops as `float16` `.npy` (0.4 MB each, ~0.55 GB total). The whole training set then fits in RAM and each epoch is dataloader-free. Compliance note carried over from `DAT_PARKINSONS_CHALLENGE.md`: whether Kaggle-persisted notebook outputs are acceptable is an unresolved forum question; the conservative pattern is to run preprocessing inside the training session (30 min on 4 cores, see §5) or locally on the 12-core dev box and never persist competition data as a Kaggle dataset.

### 2.6 Calibration for log loss

Expected picture after training: AUROC ≈ 0.97–0.99, accuracy ≈ 93–96%. Log loss will then be dominated by the ~5–8% of hard/ambiguous scans. The whole game is being *honestly unsure* on those.

| Option | Pros | Cons | Chosen? |
|---|---|---|---|
| **A. Temperature scaling** (1 param, on OOF logits) | Minimal variance; standard; fixes over/under-confidence | Cannot fix a bias shift | Subsumed by C |
| **B. Platt** (2 params) | Also fixes bias | Slightly more variance | Subsumed by C |
| **C. L2-logistic blender over the K arms' OOF logits** (K+1 params) | Does temperature + bias + ensemble weighting in one step; fit on OOF only; validated by nested CV | Must be nested-validated or it looks optimistic | **Yes** |
| **D. Isotonic regression** | Non-parametric | With 1,362 OOF points it produces step functions and 0/1 bins — catastrophic under log loss | No |
| **E. Ensemble averaging in probability space** (rather than logit space) | Averaging probabilities is more conservative than averaging logits when members disagree — desirable under shift | Slightly worse when members agree and are right | Yes, *within* the 3D-CNN arm across seeds/folds/TTA; the blender then works on the arm-level logit |
| **F. Probability clipping to [ε, 1−ε]** | Caps the worst-case per-sample loss at −ln ε; the single most important insurance against a handful of confidently wrong test scans | Costs a tiny amount on easy scans | **Yes** |
| **G. Shrinkage toward the prior**: p' = (1−λ)p + λ·0.55 | Insurance against systematic shift | Costs more than F on easy scans | Yes, small λ chosen from the leave-big-cluster-out diagnostic |

**Pick: C → E → F → G, all fit on OOF predictions only.** Concretely: for each arm, OOF logits from the grouped 5-fold; blender = `LogisticRegression(C=1.0)` on the K OOF logits, evaluated by nested 5-fold to get an honest log loss; ε chosen by grid on OOF (expect ε ≈ 0.01–0.03) then **inflated ×1.5** because the private set is more shifted than OOF; λ ≈ 0.02–0.05 chosen so that the leave-big-cluster-out log loss does not blow up. Use at most 2–3 public-LB submissions to check that the calibration transfers (compare public log loss to nested-OOF log loss); do **not** tune to the public LB beyond that — the private set decides.

### 2.7 Inference engineering for the 3 h / no-internet box

The budget is generous on paper; the risk is not throughput, it is a single unhandled exception at hour 2 or a pathological scan hanging one worker forever. Design principles: **write a valid file first, refine progressively, never let one scan kill the run, never print test information.**

**`main.py` skeleton (priority-ordered stages, each overwriting `submission.csv`):**
1. `t0`; set `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2` and `OMP_NUM_THREADS=2` *before* importing ants/torch; read `submission_format.csv`.
2. **Stage 0 (seconds):** write `submission.csv` with p = 0.548 (the training prior) for every uid. A crash from here on yields log loss ≈ 0.69, not a failed submission.
3. **Stage 1 — preprocessing pool:** `multiprocessing` with `spawn` context, **10 workers × 2 ITK threads** (ANTs scales sub-linearly with threads; 10×2 ≈ saturates 24 vCPU with headroom), `imap_unordered`, per-task `try/except` and a per-scan 60 s timeout (`signal.alarm` in the worker or a wrapper subprocess). Output per scan: the 2 mm crop (float16), the registration-validity flag, and the hand-crafted features. Progress printed every 100 scans (aggregate only). A **preprocessing watchdog**: if elapsed × (remaining/done) projects past `t0 + 100 min`, switch the remaining scans to the fast path (COM + resample only).
4. **Stage 2 — feature arm (seconds):** predict with the LR/LGBM models, apply the calibrated blender *for the feature arm alone* (a pre-fitted single-arm blender bundled as JSON) → overwrite `submission.csv`.
5. **Stage 3 — 3D CNN arm:** load 10 state_dicts (`torch.load(..., weights_only=True, map_location='cuda')`), batch 64 on the A100, fp16, TTA = identity + L/R flip (+ 2 small translations if time), average probabilities → arm logit. Overwrite `submission.csv` using the two-arm blender.
6. **Stage 4 — 2.5D arm:** same pattern → full 3-arm blender → overwrite. Apply ε-clip and prior shrinkage on every write.
7. Every stage checks `time.time() − t0` against a **150-minute soft deadline** (leaving 30 min of slack for container overhead) and skips itself if it cannot finish; skipping never invalidates the file because each stage's blender is self-contained.

**Bundle contents:** `main.py`, `pipeline/` (preprocess, models, features, blend), `assets/template.nii.gz`, `assets/masks.npz`, `assets/cnn3d_f{0..4}_s{0,1}.pt` (10 × ~8 MB), `assets/cnn25d_f{0..4}.pt` (5 × ~45 MB), `assets/feature_models.pkl` (scikit-learn/lightgbm objects — pin the same library versions as the runtime's `uv.lock`, otherwise pickle-loading can fail; safer: export LR coefficients to JSON and LightGBM via `booster.save_model()` text), `assets/blenders.json`, `assets/preproc_config.json`. Total ≈ 350 MB. Nothing is downloaded; nothing is pip-installed.

**Memory:** 2,000 test scans × 0.8 MB crops ≈ 1.6 GB; ANTs workers ≈ 1–2 GB each; total < 30 GB of 220 GB.

**Smoke-test-first discipline:** the 6-minute smoke run is the *only* pre-scoring check of the real container. Cold imports of torch + antspyx + lightgbm ≈ 15–30 s; ~20 scans × 5 s / 10 workers ≈ 10 s; model loading ≈ 10 s; comfortably < 2 min. Before every real submission: (1) run the zip locally in the official Docker image (`competition-sfmn-parkinsons-runtime`, CPU mode is enough to catch import/version errors; Docker Desktop + WSL2 on this Windows box), (2) submit the smoke test, (3) only then the scored run. Also keep a `--selftest` flag in `main.py` that runs the full pipeline on 3 synthetic volumes of shapes 256³/2.46 mm, 128×128×25/3.9 mm and 128×128×128/2.3 mm, so the three shape regimes are exercised before upload.

### 2.8 Risk register (summary — full version in §6)

Top five: registration failure on odd test scans; an intensity/quantization regime unseen at training; overconfident errors on the hard 5–8%; a hang or crash inside the 3 h run; Kaggle quota/compliance friction. Each has a concrete mitigation baked into the pipeline above (fallback path + validation; quantization/blur augmentation + ratio normalization; ε-clip + shrinkage + probability averaging; stage-wise writes + timeouts + watchdog; small models + local preprocessing).

---

## 3. Recommended end-to-end pipeline (the assembled choice)

```
OFFLINE (Kaggle + local 12-core box)                    ONLINE (A100 box, ≤3 h, no internet)
──────────────────────────────────────                  ──────────────────────────────────────
1. Header scan → pseudo-site groups                     0. write prior-only submission.csv
2. Build template from ~600 normals (2 iters affine)    1. Pool(10): per scan
3. For all 1,362: smooth σ=1vox → brain mask →               load → smooth → mask → floor-subtract
   floor-subtract → COM → 4mm affine to template →           → COM → 4mm affine (60 s timeout)
   apply @2mm → ratio-normalize (ch1) + p1/p99.5 (ch2)      → validate transform, else COM fallback
   → 64×64×48 striatal crop (fp16 .npy) + 20 features       → 2-ch crop + features + validity flag
4. StratifiedGroupKFold(5) OOF for 3 arms:              2. feature arm → blend → write csv
   A. small 3D CNN ×(5 folds × 2 seeds), GPU aug         3. 3D CNN ×10, TTA flip → blend → write csv
   B. 2.5D resnet18 (BSD/Apache weights) ×5             4. 2.5D ×5 → 3-arm blend → write csv
   C. LR / LightGBM on features ×5                       every write: ε-clip, prior shrinkage
5. Blender (L2-LR on OOF logits, nested-CV'd),            watchdog at t0+150 min skips late stages
   ε and λ from OOF + leave-big-cluster-out
6. Bundle zip → Docker CPU test → smoke test → submit
```

**Concrete specs:**

- **Template space:** 2 mm isotropic, RAS, brain box 96×112×96; striatal crop 64×64×48 centered on the template striatal centroid (covers ±64 mm L/R, ±64 mm A/P, ±48 mm S/I — generous against residual mis-registration).
- **Registration call:** `ants.registration(fixed=template_4mm, moving=scan_4mm, type_of_transform='Affine', aff_metric='mattes', mask=brain_mask_4mm_eroded, initial_transform=com_translation, aff_iterations=(200,100,50), aff_shrink_factors=(4,2,1), aff_smoothing_sigmas=(2,1,0))`, then `ants.apply_transforms` from the native scan to the 2 mm template grid with linear interpolation. Validate scale/shear/translation/NCC as in §2.1.
- **Normalization:** `floor = 1st percentile of non-brain voxels`; `v = max(v − floor, 0)`; `ref = mean(v[reference_mask])`; `ch1 = clip(v/ref, 0, 6)`; `ch2 = clip((v − p1)/(p99.5 − p1), 0, 1)`. In the fallback path, `reference_mask` is replaced by "brain voxels below the 90th percentile".
- **Features (20):** for L/R × {whole striatum, anterior half, posterior half}: mean and p90 of `ch1` (12); asymmetry index `|L−R|/(L+R)` for the three regions (3); posterior/anterior ratio L and R (2); voxel count with `ch1 > 2.0` (1); registration NCC (1); fallback flag (1).
- **3D CNN:** stem conv 3×3×3 (2→16) → 4 blocks of [conv-GN-SiLU ×2, stride-2 conv] with channels 32/64/128/192 → GAP → dropout 0.3 → linear(1). ≈ 2M params. Batch 16, AdamW 1e-3, wd 1e-2, cosine, 70 epochs, EMA 0.999, fp16. GPU-side augmentation as in §2.5.
- **2.5D:** three 6 mm slabs (mean over 3 slices) centered at striatal z − 6 / 0 / +6 mm as the 3 channels, resized to 128×128, `torchvision.models.resnet18(weights=IMAGENET1K_V1)` (BSD-3), backbone lr 1e-4, head lr 1e-3, 40 epochs; predict at 5 slab-center offsets (−8…+8 mm) and average.
- **Feature arm:** `LogisticRegression(C=0.3)` on standardized features; and `LGBMClassifier(num_leaves=7, n_estimators=300, learning_rate=0.03, min_child_samples=30, subsample=0.8)`. Keep whichever has lower nested-OOF log loss (expect LR).
- **Blend/calibrate:** `LogisticRegression(C=1.0)` on `[logit_A, logit_B, logit_C]`; separate blenders fitted for the subsets {C}, {A,C}, {A,B,C} so each inference stage is self-contained. Then `p = clip(p, ε, 1−ε)`, `p = (1−λ)p + λ·0.548`.

---

## 4. Day-by-day plan (Sept 5 → Sept 16)

| Day | Date | Deliverable |
|---|---|---|
| 1 | Fri Sept 5 | Header/`descrip` inspection → pseudo-site groups; brain-mask + floor + reference-region normalization checked visually on ~20 scans per intensity regime; **measure ANTs affine time at 4 mm vs full res on the 12-core box** (this number drives everything); build template v0 from 50 normals. |
| 2 | Sat Sept 6 | Full preprocessing of 1,362 scans (local, 12 cores); features extracted; feature-arm LR with grouped 5-fold → first honest OOF log loss. Write `main.py` skeleton with stage-0 prior write, pool, timeouts, watchdog, feature arm only. Docker CPU test of the zip. |
| **3** | **Sun Sept 7** | **Milestone: smoke test passes + first scored submission (feature arm only, calibrated + clipped).** This establishes the container works, the log budget, and the public-LB-vs-OOF gap. Start Kaggle 3D CNN fold 0 the same evening. |
| 4 | Mon Sept 8 | 3D CNN 5-fold on Kaggle (T4x2, two folds in parallel); GPU augmentation implemented; ablate 2 mm vs 2.5 mm crop and label smoothing on fold 0 only. |
| 5 | Tue Sept 9 | Second seed of 3D CNN; blender + nested-CV calibration; leave-big-cluster-out diagnostic → choose ε, λ. **Submission #2 (features + 3D CNN).** |
| 6 | Wed Sept 10 | 2.5D arm (verify weight license tag, bundle weights); 5-fold; add to blender. Template v1 (re-average with affine-registered normals) only if registration NCC distribution shows a tail — otherwise freeze the template. |
| 7 | Thu Sept 11 | Robustness pass: run the whole inference zip on synthetic odd shapes (`--selftest`), inject registration failures, confirm fallback + watchdog + stage-wise writes; time the full pipeline on 1,362 training scans as a proxy for the test run. **Submission #3 (3 arms).** |
| 8 | Fri Sept 12 | Quota reset. Spend it on what OOF says matters most: mixup / label smoothing / blur-augmentation strength sweep, or a third seed. No new architectures after today. |
| 9 | Sat Sept 13 | Freeze models. Final blender/calibration fit. Re-run Docker + smoke test on the frozen zip. **Submission #4 (final candidate).** |
| 10 | Sun Sept 14 | Buffer: fix whatever the public LB revealed (only calibration-level changes allowed — ε, λ, blender weights). |
| 11 | Mon Sept 15 | Buffer / documentation for the MIT-licensed release (required for winners); ensure external-data disclosure (ImageNet weights) is filed with organizers. |
| 12 | Tue Sept 16 | Deadline 23:59 UTC. Nothing new is submitted on this day unless a previous run failed. |

Rule for the plan: a scored submission goes in on **days 3, 5, 7, 9** at the latest, so there is never more than two days without a known-good file on the leaderboard.

---

## 5. Compute budget check

**Training (Kaggle, T4x2 unless noted):**

| Item | Arithmetic | Cost |
|---|---|---|
| Preprocessing (CPU, local 12-core box) | 1,362 scans × ~5 s (4 mm affine) / 10 workers ≈ 11 min; worst case full-res 30 s/scan → 68 min | 0 GPU-hr |
| 3D CNN, one fold | ~1,090 samples/epoch, batch 16 → 68 steps; 64×64×48×2-ch fp16 through a 2M-param net ≈ 0.15 s/step on T4 incl. GPU aug → ~10 s/epoch; 70 epochs + val ≈ **15 min** | 0.25 GPU-hr |
| 3D CNN, 5 folds × 2 seeds | 10 × 15 min | 2.5 GPU-hr |
| 3D CNN ablations (crop res, smoothing, mixup, blur strength: 4 variants × 5 folds) | 20 × 15 min | 5 GPU-hr |
| 2.5D resnet18, 5 folds | 1,090 × 3 slab offsets @128², resnet18 ≈ 25 s/epoch × 40 epochs ≈ 17 min/fold | 1.5 GPU-hr |
| Feature arm, blender, calibration | CPU seconds | 0 |
| Re-runs / mistakes / a third seed | ×2 safety factor on everything above | 9 GPU-hr |
| **Total** | | **≈ 18 GPU-hr of the ~60 available across two weekly resets** |

Two-GPU trick: running two folds as separate processes on T4x2 halves wall-clock; the quota counts session hours, not GPU-count, so this is also a real 2× on the quota.

**Inference (A100 box, 3 h cap):** test-set size N is undisclosed; budget for **N = 2,000** (≈1.5× the training set — almost certainly an over-estimate).

| Stage | Arithmetic | Time |
|---|---|---|
| Cold start: imports + template + 15 model loads | measured in smoke test; estimate | ~1 min |
| Preprocessing, 4 mm affine | 2,000 × 5 s / 10 workers | **17 min** (worst case full-res 30 s: 100 min — still fits, and the watchdog degrades before that) |
| Feature arm | 2,000 rows × LR/LGBM | seconds |
| 3D CNN ×10 models × 2 TTA | 40,000 forwards of a 2M-param net on 64×64×48, batch 64 on A100 ≈ 2,000 fwd/s | ~1 min |
| 2.5D ×5 models × 5 slab offsets | 50,000 resnet18 forwards @128² ≈ 3,000 fwd/s | ~1 min |
| CSV writes, checks | | seconds |
| **Total** | | **≈ 20 min typical, ≈ 105 min worst case; watchdog soft deadline at 150 min** |

Smoke test (6 min): ~20 scans → cold start 1 min + preprocessing 10 s + inference seconds ≈ **< 2 min**.

---

## 6. Risk register

| # | Risk | Likelihood | Impact | Mitigation (already in the design) | Residual |
|---|---|---|---|---|---|
| 1 | **Affine registration fails / hangs on odd test scans** (25-slice partial FOV, 3-bit ultra-low-count, unusual orientation) | Medium | High per-scan; catastrophic if it kills the pool | COM initialization; masked metric; 60 s per-scan timeout; transform validation (scale/shear/translation/NCC); COM-only fallback; models trained with strong affine aug so fallback inputs still score; per-task try/except; `spawn` context; aggregate-only logging | A few scans get fallback-quality predictions; the ε-clip bounds their loss |
| 2 | **Intensity/quantization regime unseen at training** (a private-set center with yet another scale, or a heavier scatter correction) | Medium | Medium | Ratio normalization is scale-invariant; floor subtraction handles offsets; quantization + blur + gamma + offset augmentation; two-channel input; the regime indicator is deliberately *not* a feature | Residual shift on background level; shrinkage λ absorbs some |
| 3 | **Overconfidence on the hard 5–8%** (the log-loss killer) | High without mitigation | High | Probability averaging across 10+ models and TTA; nested-CV blender; ε-clip inflated ×1.5 over OOF-optimal; small prior shrinkage; checkpoint on val log loss not AUROC; mixup/label smoothing evaluated by post-calibration OOF | Private-set shift larger than the leave-big-cluster-out diagnostic suggests |
| 4 | **Running out of the 3 h / a crash mid-run** | Low-medium | Catastrophic (no file = no score) | Stage-0 prior file written in seconds; every stage overwrites a valid CSV; watchdog at 150 min; preprocessing projection switches to fast path; `--selftest` on synthetic odd shapes; Docker CPU test + smoke test before every scored run; log ≤300 lines by printing per-100 progress only | Container-level failures outside our control |
| 5 | **Kaggle quota exhaustion or compliance friction** (persisted data on Kaggle is an unresolved gray area) | Medium | Medium (delays) | Models are small (≈18 of 60 GPU-hr); preprocessing runs locally on the 12-core box and is never uploaded as a Kaggle dataset; per-session re-preprocessing costs only ~11–30 min on 4 cores if needed; two folds per T4x2 session | If the organizers rule Kaggle non-compliant for prizes, training must move to a self-controlled instance — the pipeline is portable |
| 6 | Template built from training normals encodes a center bias (the 2.46 mm cluster) | Medium | Low-medium | Templates are for *alignment*, not prediction; mattes MI is robust to intensity style; check registration NCC per pseudo-site on day 2, re-average with a balanced site mix (template v1) if one site's NCC tail is heavier | — |
| 7 | Pretrained 2D weights turn out to be non-permissively licensed | Low | Prize-eligibility | Use torchvision `IMAGENET1K_V1` (BSD-3) or timm weights whose model card says `apache-2.0`; record the license in the disclosure; the 2.5D arm is optional and can be dropped without touching the other arms | — |
| 8 | scikit-learn / LightGBM pickle incompatibility with the runtime's pinned versions | Medium | Feature arm crashes (caught, but lost) | Export LR as JSON coefficients and LightGBM via `save_model()` text; test in the official Docker image | — |
| 9 | Public-LB over-tuning | Medium | Private-LB drop | Hard rule: ≤ 4 scored submissions, calibration-level changes only after day 9, model selection by nested OOF | — |

---

## 7. Confidence & caveats

**Well-established (I would bet on these):**
- Spatial alignment + per-volume scale-invariant normalization are mandatory (the EDA's chance-level unregistered ROI is direct evidence).
- Registering at 4 mm is lossless for a ~10 mm FWHM modality; the cost savings are real.
- Small 3D CNNs with GAP and heavy augmentation are the right capacity for ~1k volumes; 30M-param 3D backbones from scratch are not.
- Probability averaging over many members, temperature/Platt-style post-hoc calibration on OOF, and ε-clipping each reduce log loss reliably; isotonic at this N does not.
- Focal loss and re-weighting harm calibration; BCE + post-hoc calibration is the safe default for a log-loss metric.
- Stage-wise writing of a valid submission file and a wall-clock watchdog are cheap and remove the single worst failure mode.

**Engineering judgment calls (reasonable, but should be checked by data on days 1–3):**
- The 5 s/scan estimate for a 4 mm ANTs affine is from experience, not measured here; **measure on day 1** — the whole inference budget is insensitive to it up to ~30 s/scan, but the smoke-test comfort margin and the local preprocessing plan depend on it.
- Reference-region ratio normalization vs. plain percentile scaling: ratio normalization is more physically faithful, but with a learned InstanceNorm stem the practical gap may be small. Keeping both as channels hedges this.
- Quantization augmentation is an inference from the EDA's ultra-low-count regime; its value is untested but it is essentially free.
- The 2.5D pretrained arm may add little over the 3D CNN on this modality; it is included for ensemble diversity and can be dropped without affecting the rest.
- ε inflation ×1.5 and λ ≈ 0.02–0.05 are heuristics for unknown private-set shift; the leave-big-cluster-out diagnostic gives them some grounding, the public LB gives one more sanity check.
- Pseudo-site groups derived from header signatures will not perfectly recover the 10 hospitals; they are a proxy that makes CV *more* honest than random folds, not perfectly honest.

**Unknowns that could change the picture:**
- Test-set size and whether its centers overlap the training centers (undisclosed). The design tolerates both; it cannot be optimized for either.
- The Kaggle data-retention compliance ruling (open forum question). The pipeline is portable to a self-controlled machine; the local 12-core box handles everything except GPU training.
- Whether the NIfTI `descrip` strings actually carry a per-center signature — checked on day 1; if not, groups fall back to shape × spacing × regime, which is coarser but still useful.
