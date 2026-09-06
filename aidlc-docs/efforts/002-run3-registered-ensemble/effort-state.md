# Effort 002 — run 3: registered frame + 3D/2.5D/LR ensemble
state: in-progress (Kaggle training queue launched 2026-09-06 06:20 UTC, 50 jobs, ~4.7 h)
started: 2026-09-05 evening
## Requirements delta (vs baseline)
- NEW: SimpleITK registration to a symmetric label-blind study template with NCC QC + CoM fallback (R4).
- NEW: reference-region intensity normalisation (eroded brain minus dilated striatum).
- CHANGED: no scalar side-input; GroupNorm/SiLU; fixed 50/30 epochs, final EMA; mixup 0.2; AdamW wd 1e-2 (R5).
- NEW: 2.5D torchvision EfficientNet-B0 / V2-S on 3x8 mm slabs at template z=33.5 (R3 disclosed).
- NEW: registered-ROI LR arm; weighted-logit combiner; T fitted on leave-pseudo-site-out OOF, floored at 1.0.
## Construction log
1. cache_v3 (2 mm CoM boxes, 1362/1362) — done
2. template (400 scans, 2 iters, symmetrised) — done; NCC unreg 0.63 -> reg 0.81
3. cache_v4 registered — done; fallback 0.1 %
4. gate: ROI-LR AUROC 0.902 (unregistered 0.852) — below the 0.92 target, accepted
5. tests: tests/test_v3.py, tests/test_infer_v3.py green; main_v3 --selftest OK with template
6. Kaggle dataset dat-parkinsons-cache-v4 (2.08 GB) uploaded; notebook re-injected (48 cells); queue running
## Next
- stack + pack (notebook cells 46-47) -> submission_v3.zip -> DrivenData smoke test -> real submission (slot 2 of 3)
- record OOF/site/public numbers in RESULTS_LOG.md
