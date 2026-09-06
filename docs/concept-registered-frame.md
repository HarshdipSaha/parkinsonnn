# Why run 3 is built the way it is

This explains the reasoning behind the run-3 design. It does not tell you how to run anything (see the how-to) or what the exact parameters are (see the reference).

## The problem was the frame, not the model

Run 2 reached out-of-fold log loss 0.300 but public 0.339 at AUROC 0.925, while the leader sits at 0.219 / 0.971. Under a simple binormal model a perfectly calibrated classifier with AUROC 0.925 scores about 0.346, so run 2 was already close to optimally calibrated for its discrimination; the missing 0.12 is almost entirely AUROC. The post-mortem (`RUN3_BRAINSTORM.md`) traced the discrimination loss to the spatial frame: the hotspot anchor failed on 19.5 % of abnormal scans against 2.4 % of normals (the disease is exactly what makes the hotspot ambiguous), and in the plain centre-of-mass crop the striatal plane wandered between slice 7 and 30 of 48. Every published DaT-SPECT result above 97 % registers to a template first.

## Registration to a study template

Each 2 mm head box is registered to a template built from the training scans themselves (label-blind random subset, two mean-and-re-register iterations, then symmetrised about its mid-sagittal plane). A 7-parameter similarity transform followed by an affine refinement, computed on smoothed images at 8 and 4 mm, is enough because the modality's resolution is about 10 mm FWHM. The normalised cross-correlation with the template is the quality check: if a registration is implausible or does not beat the unregistered alignment, the box falls back to the centre-of-mass frame and the fact is recorded, so a failure is visible rather than a confident wrong answer. In practice 99.9 % of scans register, independent of label, and the ROI-feature gate rises from 0.85 to 0.90 AUROC.

Symmetrising the template matters for two reasons: left/right flipping becomes an exact, label-preserving augmentation, and the left and right striatal masks used by the feature arm are mirror images.

## Reference-region intensity

Dividing by the 75th percentile of a head mask lets scalp uptake and site-specific reconstruction styles leak into the reference. In the registered frame a proper reference region exists: the brain eroded by 8 mm minus the striatum dilated by 20 mm. Dividing by its mean (after background subtraction) expresses every voxel in specific-binding-ratio units, the clinical quantity, and removes the per-site scale.

## No scalar side-input

Four of the eight run-2 scalars (native spacing, z extent, mask volume, grey-level count) identify the acquisition site with 97 % accuracy. Fed to the heads, they let the model learn "this scanner has prevalence X", which is right on random folds and wrong on a differently mixed test set (leave-one-site-out bias of 0.13–0.20 in probability). Run 3 models see only the image.

## Honest validation and the EMA lesson

Run 2 picked the best epoch on the validation fold and then used that fold as its out-of-fold estimate, which flatters the number and produced a temperature below 1. Run 3 trains for a fixed number of epochs and ships the final exponential moving average of the weights. The first attempt used a long EMA window (decay 0.998, about 15 of 30 epochs): validation loss crawled while the raw model was already good, and the shipped predictions were compressed into [0.3, 0.75]. The averaged weights blended solutions from when the model was still poor. A short window (2–3 epochs) fixed it immediately (0.58 → 0.28 log loss on the same fold). The trainer now logs the raw model's validation metrics next to the EMA's so a lag is visible within an epoch.

## Temperature from site folds, floored at 1

Over-confidence under distribution shift is unbounded in log loss; under-confidence is bounded. The temperature is therefore fitted on predictions from models trained with whole pseudo-sites held out, and never allowed below 1.0. The cost of a temperature 10–20 % too large is a few thousandths; the cost of one too small on a shifted test set is open-ended.

## Test-time augmentation is defined once

Predictions used to fit the stack must be the predictions inference will produce. `InputSpec.variants` is the single definition of the test-time variants (slab-centre shifts and the mirror flip); the trainer, the re-OOF script and `main_v3.py` all iterate over it, so the combiner is never fitted on one thing and applied to another.

## Diversity over depth

Two families with different inductive biases, a small 3D network trained from scratch on the whole registered box and an ImageNet-pretrained 2D network on a 24 mm striatal slab, plus a linear model on registered-ROI features, are combined with a handful of fitted numbers (simplex weights, one temperature, one prior mix). The ensemble buys more than any single larger model would on 1,362 scans, and each member fails differently.
