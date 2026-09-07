# Run 7 — hemisphere-level members and the super-stack (wayfinder ticket T07)

## Why this route

- Public 0.2667 / AUROC 0.9536 (run 4a) against a leader at 0.2192 / 0.971. Our nested out-of-fold estimate is 0.2491; the shift gap (nested → public) has been 0.021 and 0.018. Reaching 0.21–0.22 needs both a large discrimination gain and a smaller gap.
- In every run the left/right **asymmetry** descriptors were among the strongest single features (AUROC 0.85–0.86 alone), and Parkinsonian loss of striatal uptake typically starts on one side. The whole-crop 3D nets must learn asymmetry through convolutions spanning the midline; the hemisphere member gets it for free: one shared encoder over the two mirrored half-crops, a symmetric head over `[f_L + f_R, |f_L − f_R|]` plus `max(side_L, side_R)`. Mirroring a scan swaps the hemispheres and leaves the logit unchanged (unit-tested), so the label-preserving flip augmentation and the flip TTA are exact.
- Each scan contributes two hemispheres in one orientation → the encoder sees twice the samples per epoch at the same cost as one 48×48×32 crop (the halves are batched through the encoder in a single pass).
- Label noise: the 100 hardest training scans carry 46 % of the stack loss and are mostly abnormal-labelled scans every arm scores 0.03–0.15. A soft **bootstrapped-target** variant of the crop-3D arm (target = 0.9·y + 0.1·p) is trained as a gated member to see whether pulling less on those labels improves calibrated log loss.
- Site invariance: the hemisphere member is also trained with the run-5 DANN + group-DRO switches (`r3d_hemiinv`) and on the leave-site-out folds (`r3d_hemi48_*_site`) so its shift behaviour is measured, not assumed.
- **Super-stack**: the stack cell scores every arm family of runs 3–7 that has out-of-fold files in the checkpoint dataset (whole-brain 3D, crop-3D and its SE / 64 / 80-epoch / SSL / invariant / hist variants, B0, V2-S, ConvNeXt and its coronal and tri-planar versions, the feature arms), keeps the ten best by OOF log loss and fits the nested combiner (compositions enumeration, temperature floor 1.0, prior mix 0.02, typicality gate) over them. Arms from notebooks that have not run yet are simply absent and skipped.

## Jobs (2×T4, ≈ 2.3 h + 35 min tail)

| Job | Folds × seeds | Purpose |
|---|---|---|
| `r3d_hemi48` | 5 × 3 | the member itself (50 epochs, base augmentation, mixup, EMA-final) |
| `r3d_hemi48_*_site` | 5 × 1 | leave-site-out read-out of the member |
| `r3d_hemiinv` | 5 × 1 | DANN 0.1 + group-DRO 0.1 variant |
| `r3d_cropboot` | 5 × 1 | crop-3D with bootstrapped targets (label-noise probe) |
| `r3d_hemi48_fall` | 2 | full-data models appended through `--extra` |

## Expected effect (honest)

- Hemisphere member alone: 0.245–0.255 OOF (crop-3D is 0.2568); if it is at least as good and less correlated with the 2D arms, the super-stack moves from 0.2491 (run 4a) to ≈ 0.235–0.242 nested with the run-5/6 members, i.e. public ≈ 0.25–0.255 at today's gap. That is rank ≈ 10–15, not 0.21: the remaining distance is the shift gap and the PPMI branch (wayfinder tickets T05, T11).
- The bootstrapped loss is a probe: keep only if the nested estimate improves by ≥ 0.003.

## Rules and safety

No test-set use beyond per-sample inference; only torchvision ImageNet weights in the 2D arms; every job resumes from and uploads to the private checkpoint dataset; fail-safe tail (re-OOF, stack, pack, e2e with per-arm smoke-set scores, split); the zip is packed from the manifest, so an oversized ensemble is trimmed at assembly (`scripts/run4_assemble.py`) rather than in the notebook.
