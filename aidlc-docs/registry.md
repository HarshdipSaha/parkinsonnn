# Effort registry

Derived from `aidlc-docs/efforts/*/effort-state.md`; the per-effort files are the source of truth.

| # | Effort | State | Outcome so far |
|---|---|---|---|
| 001 | [run2-hotspot-pipeline](efforts/001-run2-hotspot-pipeline/effort-state.md) | complete (superseded) | Public 0.3391, rank 152 |
| 002 | [run3-registered-ensemble](efforts/002-run3-registered-ensemble/effort-state.md) | complete | Public **0.2817** / AUROC 0.9477 / **rank 66** (id-321263, Sept 6); OOF 0.2580; container bug fixed by local repack |
| 003 | [run4-toward-rank1](efforts/003-run4-toward-rank1/effort-state.md) | complete | Public **0.2667** / AUROC 0.9536 / **rank 39** (id-321298, Sept 7); nested 0.2491 |
| 004 | [run5-site-invariance](efforts/004-run5-site-invariance/effort-state.md) | complete / delivered | Nested **0.2433** (gain -0.0058); Smoke test **0.2388** (id-321496, Sept 8); crophist in (weight .375); cropinv out (+0.030 on site folds); stack OOF **0.2394** / AUROC 0.9672 |
| 005 | [run6-parallel-approaches](efforts/005-run6-parallel-approaches/effort-state.md) | complete / delivered | Unit B completed (`tri_cnx` OOF 0.2560, container smoke 0.2139, 25% stack weight); Unit C completed (VICReg SSL, `a3d_ssl` 38% stack weight, `featc` 12% weight, container smoke 0.2289); Unit A merged |
| 006 | [run7-hemisphere-superstack](efforts/006-run7-hemisphere-superstack/effort-state.md) | delivered / verified | Kaggle completed (32 jobs); stack OOF **0.2357** / AUROC 0.9684 (`a3d_hemi` 50% stack weight); DrivenData smoke test **id-321572 = 0.2706** for standalone `a3d_hemi` arm on hidden test set |
| 007 | [run8-distill-master-v3](efforts/007-run8-distill-master-v3/effort-state.md) | delivered / verified | **GATE PASSED** (nested **0.2289**, gain **+0.0034** over v2 0.2323); `submission_master_v3.zip` (847 MB, 71 ckpts, 7 arms) smoke verified (0.2266 local smoke loss); replaces v2 as primary slot-1 candidate |
