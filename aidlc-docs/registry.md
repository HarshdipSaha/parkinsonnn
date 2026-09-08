# Effort registry

Derived from `aidlc-docs/efforts/*/effort-state.md`; the per-effort files are the source of truth.

| # | Effort | State | Outcome so far |
|---|---|---|---|
| 001 | [run2-hotspot-pipeline](efforts/001-run2-hotspot-pipeline/effort-state.md) | complete (superseded) | Public 0.3391, rank 152 |
| 002 | [run3-registered-ensemble](efforts/002-run3-registered-ensemble/effort-state.md) | complete | Public **0.2817** / AUROC 0.9477 / **rank 66** (id-321263, Sept 6); OOF 0.2580; container bug fixed by local repack |
| 003 | [run4-toward-rank1](efforts/003-run4-toward-rank1/effort-state.md) | complete | Public **0.2667** / AUROC 0.9536 / **rank 39** (id-321298, Sept 7); nested 0.2491 |
| 004 | [run5-site-invariance](efforts/004-run5-site-invariance/effort-state.md) | complete / delivered | Nested **0.2433** (gain -0.0058); Smoke test **0.2388** (id-321496, Sept 8); crophist in (weight .375); cropinv out (+0.030 on site folds); stack OOF **0.2394** / AUROC 0.9672 |
| 005 | [run6-parallel-approaches](efforts/005-run6-parallel-approaches/effort-state.md) | delivered | A (SE crop-3D family), B (tri-planar fusion), C (116-descriptor feature arm + VICReg SSL) merged; three one-command Kaggle notebooks + HOWTO PDFs; runs pending |
| 006 | [run7-hemisphere-superstack](efforts/006-run7-hemisphere-superstack/effort-state.md) | built, launch pending | hemisphere-level member + bootstrapped-loss probe + super-stack over runs 3–7; one-command notebook; wayfinder T07 |
