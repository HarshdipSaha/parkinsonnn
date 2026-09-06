# Spec: Run 5 — site-invariant training and typicality-aware calibration (effort 004)

*Draft written Sept 6 2026 17:15 UTC while run 4a trains; to be finalised with the run-4a numbers after submission 3. Produced with the `to-spec` synthesis format (no issue tracker is configured for this repo; the spec lives here and the plan is `docs/superpowers/plans/2026-09-06-run5-site-invariance-plan.md`).*

## Problem Statement

Our DaT-SPECT classifier is honest inside the training distribution (nested out-of-fold log loss 0.2603, plain 0.2580) but scores 0.2817 on the public test. The whole gap is discrimination under **site shift**: AUROC 0.961 out-of-fold versus 0.948 on the test, and the leave-pseudo-site-out score of our main arm (0.2818) matches the public score almost exactly. The leader is at 0.2192 / AUROC 0.971. Adding ensemble members has been giving about −0.005 per member and cannot close a 0.06 gap. We need methods that make each scan's prediction less dependent on the camera and protocol it came from, and that admit uncertainty on scans that look unlike anything in training — without touching the test set collectively (the rules forbid pseudo-labelling and require per-sample processing).

## Solution

Run 5 adds four mechanisms to the registered-frame pipeline, each gated on the nested / leave-site-out estimate exactly like run 4:

1. **Site-invariant training** for every neural arm: a gradient-reversal pseudo-site head and group-DRO loss weighting during training (no inference change), so the features that distinguish normal from abnormal are the ones that do not distinguish sites.
2. **Histogram standardisation** in the registered frame: each scan's brain-mask intensity landmarks are mapped piecewise-linearly onto the study template's landmarks after reference normalisation (a new normalisation mode used identically in training and inference), removing site-specific intensity nonlinearity (quantisation, scatter, partial volume).
3. **Typicality-aware temperature**: the combiner's temperature grows with the scan's Mahalanobis distance to the training distribution in the feature arm's PCA space, with the growth fitted on leave-site-out predictions, so unfamiliar scans are shrunk toward the prior. Inference-only and per-sample.
4. **Tri-planar fusion member** and **affine test-time augmentation**: one backbone reading axial, coronal and sagittal slabs with late fusion (learns cross-view interactions the separate arms cannot), and a small set of random affine perturbations averaged at test time for every arm (robustness to residual registration error), defined once in the shared variant generator.

Plus one delivery mechanism: the notebook generator exports a ready `.ipynb` so a person can import and run it on Kaggle without Playwright.

## User Stories

1. As the competitor, I want every neural arm to be trainable with a site-adversarial head, so that its features stop encoding the camera.
2. As the competitor, I want group-DRO weighting to up-weight the worst pseudo-sites during training, so that the thin-FOV and ultra-low-count sites stop being the tail of the loss.
3. As the competitor, I want the pseudo-site head to be discarded at inference, so that the submission container needs no site information.
4. As the competitor, I want a histogram-standardised normalisation mode, so that a 2.46 mm ultra-low scan and a 3.90 mm high-count scan present the same intensity distribution to the nets.
5. As the competitor, I want the standardisation landmarks stored in the template file, so that inference reproduces training exactly.
6. As the competitor, I want the standardisation to be a no-op on the template itself and monotonic on every scan, so that it can never invert contrast.
7. As the competitor, I want a per-scan typicality score computed from training statistics only, so that the calibration adapts per sample without looking at other test scans.
8. As the competitor, I want the temperature-versus-typicality curve fitted on leave-site-out predictions, so that it reflects how the model behaves on unseen sites.
9. As the competitor, I want the typicality parameters in the manifest, so that the container applies them without code changes.
10. As the competitor, I want a tri-planar member that shares one backbone across three views, so that the ensemble gains a member whose errors differ from the single-view arms.
11. As the competitor, I want affine test-time augmentation defined in the shared variant generator, so that out-of-fold predictions and inference use identical variants.
12. As the competitor, I want every mechanism gated on the nested / leave-site-out estimate, so that nothing that only helps random folds ships.
13. As the competitor, I want the strong-augmentation site-fold result from run 4a to decide the augmentation default for run 5, so that the budget follows the evidence.
14. As the competitor, I want the combiner weights fitted on the leave-site-out pool where it exists, so that the weights themselves generalise across sites.
15. As the competitor, I want a Kaggle notebook file exported from the generator, so that I can import it and press Run All without any automation.
16. As the competitor, I want each Kaggle commit to resume from the checkpoint dataset and to upload its new checkpoints, so that no training is lost or repeated.
17. As the competitor, I want a one-command assembly (nested stack, gate, pack, local container check), so that a submission never ships unverified again.
18. As the competitor, I want the plan's steps small enough to hand to any agent, so that the work continues if this session ends.
19. As the competitor, I want the inference budget re-measured after adding members, so that the 3-hour container limit is never at risk.
20. As the competitor, I want the site-fold score of every new arm reported next to the random-fold score, so that the random→site gap is tracked per member.
21. As the competitor, I want the final two submissions to be a bracket (best stack, then the same stack with the alternative prior mix), so that the best private score is captured automatically.
22. As the competitor, I want the data-hygiene rule kept (no raw scans in notebook outputs, private datasets, delete after the competition), so that the prize-eligibility risk stays where it was accepted.

## Implementation Decisions

- **Site-invariant training** lives in the training loop of the run-3 trainer as two optional switches: a gradient-reversal layer feeding a small pseudo-site classifier on the pooled features (loss weight λ_dann, warm-started at 0 and ramped over the first third of training), and group-DRO weighting where each pseudo-site's loss is weighted by an exponentially updated worst-group weight. Pseudo-site labels come from the existing pseudo-site table in the cache; sites with fewer than 25 training scans are merged into "other" for the head. The head is not saved into the checkpoint's inference state, so the submission container is unchanged. Default off; enabled per job from the job list.
- **Histogram standardisation** is a new normalisation mode ("hist") next to the existing reference mode: after reference-region normalisation, the scan's brain-mask intensities at fixed percentiles (1, 5, 10, 25, 50, 75, 90, 95, 99) are mapped piecewise-linearly onto the template's corresponding percentiles; values beyond the last landmark are extrapolated with the last segment's slope. The template's landmarks are computed once from the study template and stored in the template file; the mode name is recorded in each arm's metadata and manifest so inference applies the same mapping. Training arms compare "ref" against "hist" on identical folds.
- **Typicality-aware temperature** is an extension of the combiner: a per-scan distance d = Mahalanobis distance of the feature arm's PCA scores to the training mean/covariance (both stored with the feature model), and T(d) = T0 · (1 + κ · max(0, d − d0)), with (κ, d0) chosen from a small grid on the pooled leave-site-out predictions of the stack (falling back to random folds if unavailable). The manifest carries T0, κ, d0 and the distance statistics; the container computes d from the same feature code path it already runs for the feature arm. Gate: nested site-fold log loss must improve by ≥ 0.003, otherwise κ = 0.
- **Tri-planar member** is a new 2D model variant: three slab stacks (axial at the striatal z, coronal at the striatal y, sagittal at each hemisphere's x averaged) pass through one ImageNet backbone; the pooled features are concatenated and fed to the head. It reuses the existing slab derivation for each plane, adds a "sagittal" plane option, and is exposed as a backbone option so the job list, tags and manifest need no new arm type.
- **Affine test-time augmentation** extends the shared input-variant generator with K random affine perturbations (rotation ≤ 3°, scale ≤ 2 %, shift ≤ 1 voxel; fixed seed so variants are reproducible), applied to the registered box before the arm-specific input derivation, for both 2D and 3D arms. The re-OOF step decides per arm whether it stays on.
- **Combiner on the site pool**: when every kept arm has leave-site-out predictions, weights and temperature are fitted on that pool and the random-fold pool is used only for reporting; otherwise the current behaviour (weights on random folds, temperature on the site pool when available) stays.
- **Notebook export**: the generator gains an option that writes the generated cell set as a Kaggle-importable notebook file, with the same cell contents the Playwright sync would push, so the human path is "import notebook → attach the three datasets → Save & Run All".
- **Job list for run 5** (one Kaggle commit of ≈ 4 h on 2×T4, chained from the run-4 checkpoints): the two strongest run-4 backbones retrained with site-invariant training on random and site folds (the direct test of the mechanism), the same two with "hist" normalisation, the tri-planar member, and seeds for whichever wins. Members that fail their gate are dropped before packing.
- **Schedule**: run 5 must be assembled and smoke-tested before Sept 12 15:38 UTC (slot 1 returns); Sept 13 carries two slots (run 5 best and, if run 4b/5 disagree, the alternative); Sept 15–16 bracket with the last slot.

## Testing Decisions

- Tests exercise external behaviour at the existing seams: the input-variant generator (shapes, determinism of affine variants under a fixed seed, mirror invariance), the normalisation function (hist mode is a no-op on the template, monotonic on random scans, identical between the cache path and the inference path), the training entry point (one epoch with the site head and group-DRO on the tiny synthetic cache; metadata records the switches; checkpoint contains no head parameters), the combiner (larger typicality distance moves the probability monotonically toward the prior; κ = 0 reproduces today's output bit-for-bit), the model builder (tri-planar forward shape), and the submission zip (the container-style local check reports every arm at N/N checkpoints).
- Prior art: `tests/test_v3.py` (synthetic tiny cache, one-epoch training, InputSpec round trips), `tests/test_infer_v3.py` (combine and stage writes), `scripts/local_e2e_v3.py` (container-style run on the 20 smoke volumes).
- A good test here checks a property the competition cares about (a mapping cannot invert contrast; a variant set is identical in OOF and inference; a head cannot leak into the container), not internal tensors.

## Out of Scope

- Pseudo-labelling or any use of test scans collectively (test-time adaptation across scans, transductive normalisation, prevalence estimation from test predictions).
- New pretrained weights that are not public and commercial-use (DINOv3, non-commercial medical foundation models).
- Rebuilding the registration template or the cache spacing.
- Self-supervised pretraining on our 1,362 scans (cost/benefit too uncertain for the remaining budget).
- Per-site priors at inference.

## Further Notes

- Evidence so far: the feature-engineered arm moved the nested estimate 0.2603 → 0.2545 and displaced the whole-brain 3D arm; gradient boosting on the same features was worse than the linear PCA model, so the linear feature arm stays. The label-noise bound shows the 100 hardest scans carry 46 % of the loss; those are not a modelling target.
- Budget: Kaggle GPU ≈ 12.7 h left this week (editor reading 12:42 / 30 h at 16:00 UTC Sept 6, 3.5 h of which run 4a uses), 30 h next week; real submissions: slot 3 tonight, then Sept 12 15:38 UTC, Sept 13 15:11 and ~20:30 UTC.
- To be filled after run 4a: per-arm results, the site-fold strong-augmentation comparison (the augmentation default for run 5), and which backbones run 5 retrains.
