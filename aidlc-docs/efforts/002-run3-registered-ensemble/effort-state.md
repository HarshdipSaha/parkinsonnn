# Effort 002 — run 3: registered frame + 3D/2.5D/LR ensemble
state: in-progress - queue 1 running on Kaggle (relaunched 06:50 UTC after the EMA fix; ETA ~12:45 UTC). Resume runbook: aidlc-docs/HANDOFF.md section 3; plan: docs/superpowers/plans/2026-09-06-run3-finish-and-run4.md Part A
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
6. Kaggle dataset dat-parkinsons-cache-v4 (2.08 GB) uploaded; notebook re-injected (48 cells); queue launched 06:20 UTC
7. HOTFIX 06:45 UTC: first B0 folds under-confident (AUROC 0.87/0.92 but logloss 0.58/0.55) -> EMA warm-start bug (0.999^1020 = 36 % random init in the average); step-adaptive decay + 0.998 (commit fc44f48); affected checkpoints dropped, queue relaunched
8. 07:15 UTC: EMA fix #1 changed nothing; diagnosis on OOF: p compressed to [0.3,0.75], T*=0.18-0.41, EMA val loss lags the raw model by ~15 epochs -> EMA WINDOW too long (0.998 = 15 of 30 epochs). Fix #2 (fc136d8): --ema 0.99 (2D) / 0.995 (3D), raw-model val metrics logged each epoch. Queue relaunched from folds 2/3.
   TODO before stacking: the cleanup's `pkill -f` killed its own shell before `rm`, so r2d_efficientnet_b0_f0_s0 and _f1_s0 are still the stale 0.998-EMA checkpoints -> delete them and re-run the runner cell (it retrains only those 2), then re-run stack/pack/e2e. Use `pkill -f "[s]cripts/train_v3.py"` in future.

9. 07:40 UTC: confirmed fix -> B0 fold 2/3: 0.279/0.294 log loss, AUROC 0.952/0.947 (vs run-2 best checkpoints ~0.27-0.41 with optimistic selection).

10. 08:00-08:40 UTC: B0 seeds 0/1 folds 0.258-0.326 (AUROC 0.935-0.963). Added inference-time z-shift TTA (-1/0/+1 x flip) in InputSpec.variants (shared by trainer OOF, reoof_v3.py and main_v3), coronal-slab option, full-data mode (fold -1), stack --extra; queue 2 = coronal B0 x5 folds, full-data B0 x3 + 3D x2. Notebook synced (push_cells.py) without touching the running queue.

## Next
- queue 1 ends ~12:45 UTC -> auto report/stack/pack/e2e (preliminary) -> delete stale B0 f0/f1 s0 -> runner (2 jobs) -> queue 2 -> reoof (TTA) -> stack/pack -> e2e -> split -> download parts -> join -> DrivenData smoke -> real submission (slot 2 of 3)
- record OOF/site/public numbers in RESULTS_LOG.md
