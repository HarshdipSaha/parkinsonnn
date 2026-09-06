---
status: accepted (2026-09-06); supersedes the council verdict of 2026-09-05 that cut registration
---

# Register every scan to a study template instead of cropping around an anchor

Run 2 cropped a fixed box around a hotspot or the head centre of mass and never registered. The hotspot anchor failed on 19.5 % of abnormal scans against 2.4 % of normals (low striatal uptake is the disease), the striatal plane in centre-of-mass crops wandered across half the box, and the public score (0.339, AUROC 0.925) was limited by discrimination, not calibration. We now register each head box to a label-blind, left/right-symmetrised template built from the training scans, with a normalised-cross-correlation check and a centre-of-mass fallback per scan.

**Considered options**: keep the CoM crop and rely on augmentation (cheap, but the frame stays the dominant error source); a learned anchor (label-dependent by construction); registration (chosen: every published DaT-SPECT result above 97 % does it, and the ROI-feature gate rose from 0.85 to 0.90 AUROC with a 0.1 % fallback rate).

**Consequences**: the template and the registration code ship inside the submission zip and must stay bit-compatible with the caches; changing the template invalidates every cache, mask and checkpoint; inference gains ~2 s per scan (well inside the 3-hour limit).
