# Effort 006: run 7 — hemisphere-level members and the super-stack (wayfinder T07)

| | |
|---|---|
| **State** | built (Sept 7 04:48 UTC); launch pending (after run 5 on our account, or on the teammate's account) |
| **Spec** | [2026-09-07-run7-hemisphere-spec.md](../../../docs/superpowers/specs/2026-09-07-run7-hemisphere-spec.md) |
| **Map** | `docs/wayfinder/map.md` — the route to public 0.21–0.22; this effort resolves ticket T07 |
| **Deliverable** | `kaggle/run7_hemisphere/` (gen_run7.py → notebook.ipynb 72 cells + kernel-metadata.json; APPROACH.md; HOWTO.md + HOWTO.pdf); `datsc/models_h.py`; `--arch`/`--loss` on `scripts/train_v3.py`; tests `tests/test_run7.py` |
| **Launch** | `python kaggle/run7_hemisphere/gen_run7.py && kaggle kernels push -p kaggle/run7_hemisphere` |
| **Depth** | standard |

## Requirements delta (vs baseline)

| Id | Requirement |
|---|---|
| R17 | A hemisphere-level 3D member whose logit is invariant to mirroring the scan (unit-tested), trained on random and leave-site-out folds. |
| R18 | A label-noise-robust (soft bootstrapped target) loss selectable per job, gated by the nested estimate. |
| R19 | A super-stack cell that discovers every arm family of runs 3–7 in the checkpoint dataset and fits the nested combiner over the ten best by OOF log loss. |

## Construction log

| # | When (UTC) | Unit | Result |
|---|---|---|---|
| 1 | Sept 7 04:20–04:48 | `HemiNet3D` (shared encoder over mirrored half-crops, symmetric pair + max-side head), dispatch `arch=hemi`, `--arch`/`--loss` on the standard CLI, bootstrapped loss in the trainer, JOBS7 (32 jobs ≈ 2.3 h on 2×T4), super-stack cell (top-10 arms), HOWTO md + 1-page pdf, APPROACH | 66 tests green (3 new); selftest OK; notebook validated; real-cache CPU smoke of the member running |
