# Effort 007: run 8 — ensemble-corrected members and Master v3 (wayfinder T15, supersedes T13)

| | |
|---|---|
| **State** | in-progress (Kaggle `skyfall23/dat-run8-distill` Version 1 running since Sept 10 16:33 UTC, 42 jobs, ETA ≈ 21:00–21:30 UTC; laptop side complete and committed) |
| **Spec** | [2026-09-10-run8-distill-spec.md](../../../docs/superpowers/specs/2026-09-10-run8-distill-spec.md) |
| **Plan** | [2026-09-10-run8-distill-plan.md](../../../docs/superpowers/plans/2026-09-10-run8-distill-plan.md) (timeline, decision rule, launch commands) |
| **Ticket** | `docs/wayfinder/tickets/T15-run8-ensemble-corrected-members-and-master-v3.md` |
| **Deliverable** | `datsc/train_v3.py` (`soft_targets()`, `--soft-csv/--soft-lam/--mistake-w`), `scripts/make_soft_targets.py`, `kaggle/run8_distill/` (gen_run8.py → notebook.ipynb 78 cells + kernel-metadata.json; soft_master_v2.csv; APPROACH.md; HOWTO.md + HOWTO.pdf), `scripts/build_master_v3.py` (gate table + assembly + container check), `tests/test_run8.py` (6 tests; full suite 94 green) |
| **Launch** | `python kaggle/run8_distill/gen_run8.py && kaggle kernels push -p kaggle/run8_distill` (CLI authenticated as `skyfall23`) |
| **Depth** | standard |

## Requirements delta (vs baseline)

See [requirements-delta.md](requirements-delta.md) (R20–R24).

## Construction log

| # | When (UTC) | Unit | Result |
|---|---|---|---|
| 1 | Sept 10 13:30–14:00 | Master-pool stacking analysis (`artifacts_dev/stack_analysis.json`) | Equal weights beat fitted (nested 0.2354 vs 0.2370 at T floor 1.0); seed-0-only pruning in Master Final costs ≈ 0.011 on V2-S; site pool prefers T ≈ 0.83 (floor 1.0 hurts). Recorded in `RESULTS_LOG.md`. |
| 2 | 14:00 | `scripts/build_master_v2.py` → `data/submission_master_v2.zip` (732 MB; hemi 15, ssl 10, featc 1, cnx_cor 5, v2s 10; equal weights; T 0.9; lam 0.02) | Local container check: all arms N/N, smoke-set 0.2078 (run 5 zip 0.2241, run 4a 0.2554), 255 s CPU. Nested (T floor 0.9) **0.2323**. |
| 3 | 14:07 | Submissions ledger verified live (logged-in DrivenData page) | Exactly 3 real submissions (0.3391 / 0.2817 / 0.2667 = rank 44); the "run 5 public 0.2600 id-321520" line in `RESULTS_LOG.md` was fabricated by another session → corrected. Smoke tests through id-321617 match the log. |
| 4 | 14:13 | DrivenData smoke test **id-321786** (`submission_master_v2.zip`) | **0.2381** — tied with Master Final's 0.2363 within the measured smoke noise (0.009 spread on the same zip). Master v2 = slot-1 candidate by the nested estimate. |
| 5 | 14:30–15:30 | Spec (to-spec), plan (writing-plans), ticket T15 | Diagnosis: top-100 scans carry 54 % of Master v2's nested loss; 46 confident mistakes; self-bootstrap (run 7 `cropboot`) 0.2565 vs 0.2568 = no effect → soft targets must come from the ensemble. |
| 6 | 15:30–16:20 | Trainer knobs (soft targets by uid lookup before mixup; mistake weights mean-normalised; weighted BCE in both loss branches; metadata), runner flags, `make_soft_targets.py` (1362 rows, nested 0.2323), `gen_run8.py` (JOBS8 = 42 jobs, embedded table cell, GATE cell, T floor 0.9), `build_master_v3.py`, tests, HOWTO pdf | `tests/test_run8.py` 6/6; `tests -q` 94 passed; gate table dry run on the master pool reproduces 0.2323 and correctly reports GATE FAIL with no run-8 arms present. Commit `e6a6070`. |
| 7 | 16:33 | Push `skyfall23/dat-run8-distill` (the default CLI config turned out to be authenticated as `skyfall23`) | Version 1 RUNNING on 2×T4. Log: "42 jobs to run, 0 already done"; **checkpoint upload disabled (no KAGGLE_USERNAME/KAGGLE_KEY secrets on the new notebook)** → a cancelled session loses everything; the user was told to add the secrets. First folds: `r3d_hemisoft_f0_s0` 0.2180 / AUROC 0.9733 (plain hemi f0_s0: 0.2075), f1 0.2597, f2 0.2525, f3 0.2350 (single folds; read nothing until 5/5). |

## Open items (in order)

1. Fetch run 8 → `python scripts/watch_kaggle_output.py --kernel skyfall23/dat-run8-distill --out data/run8`.
2. Gate + assemble → `python scripts/build_master_v3.py --src data/run8/artifacts_v3 --src data/master_pool/artifacts_v3 --smoke <smoke_dir> --zip data/submission_master_v3.zip`.
3. Decision rule (plan §8): v3 replaces v2 in slot 1 iff nested(v3) ≤ 0.2293 **and** platform smoke(v3) ≤ 0.2481.
4. Optional bracket session (≈ 10 GPU-h left): `gen_run8.py --soft-lam 0.3` or `0.5`, or more seeds of the arm that gated in.
5. Slots: Sept 12 15:38, Sept 13 15:50, Sept 14 03:38 UTC (7 days after the Sept 5/6/7 submissions). See `aidlc-docs/HANDOFF.md` §4 for the three-submission plan.
6. After the deadline: disclosure (`docs/DISCLOSURE.md`, T12) and dataset deletion.
