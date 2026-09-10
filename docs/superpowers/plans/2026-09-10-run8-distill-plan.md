# Plan: Run 8 — ensemble-corrected members and Master v3 (effort 007, ticket T15)

*Written Sept 10 2026 ≈ 20:45 UTC. Spec: `docs/superpowers/specs/2026-09-10-run8-distill-spec.md`. Deadline Sept 16 23:59 UTC; slots Sept 12 15:38, Sept 13 15:11, Sept 14 ≈ 23:30 UTC (rolling 7-day window from the Sept 5/6/7 submissions). Teammate's account `skyfall23`, ≈ 20 GPU-hours.*

## Timeline

| When (UTC) | Step | Output |
|---|---|---|
| Sept 10 21:00–22:30 | Steps 1–6 below on the laptop (code, tests, notebook, HOWTO) | commit `run8: ...` |
| Sept 10 ≈ 22:30 | Step 7: push `skyfall23/dat-run8-distill` (session A, ≈ 5 h) | Kaggle Version 1 running |
| Sept 11 ≈ 04:00 | fetch outputs, Step 8 gate + Master v3 zip + container check, platform smoke test | `data/submission_master_v3.zip`, smoke id |
| Sept 11 daytime | optional session B (≈ 10 GPU-h): more seeds of whatever gated in, λ bracket 0.3/0.5 | second candidate for slot 2 |
| Sept 12 15:38 | slot 1 by the decision rule | real submission |
| Sept 13 15:11 / Sept 14 23:30 | slots 2–3: the other of v2/v3, then the λ 0.01 prior-mix bracket of the better one | |

## Step 1 — trainer knobs (`datsc/train_v3.py`)

`train_run(..., soft_csv=None, soft_lam=0.0, mistake_w=0.0)`:
- Load the table once: `uid → (p, loss)` from `soft_csv` (columns `uid,p,loss`; `loss` optional → 0).
- Build `soft_map = {uid: (1-λ)·y + λ·p}` for training uids present in the table (label otherwise) and `w_map = {uid: 1 + κ·loss}` normalised to mean 1 over the training uids (all 1 when κ = 0 or table missing).
- In the loop, right after `x, s, y` are moved to the device and before mixup: `y = tensor([soft_map.get(u, y_i)])`, `w = tensor([w_map.get(u, 1.0)])`; mixup mixes `w` with the same λ.
- Loss: per-sample BCE × w, mean — in both branches (plain and DANN/DRO; in DRO the weighted per-sample losses feed the group sums).
- Metadata gains `soft_csv`, `soft_lam`, `mistake_w`, `n_soft` (uids found). Validation stays on hard labels.

## Step 2 — runner flags (`scripts/train_v3.py`)

`--soft-csv PATH`, `--soft-lam 0.4`, `--mistake-w 0.0`; passed through verbatim; default tag unchanged (jobs use explicit `--tag`).

## Step 3 — soft-target table (`scripts/make_soft_targets.py`)

From the master pool (`--src` folders with OOF csvs): nested equal-weight Master v2 predictions (T refit per fold, floor 0.9, λ 0.02) → `artifacts/soft_master_v2.csv` (`uid,y,p,loss,fold`). Prints the nested loss (expect 0.2323) and the concentration table. Commit the csv (≈ 60 KB; it contains no scan data).

## Step 4 — tests (`tests/test_run8.py`)

- `test_soft_targets_and_weights`: a tiny table for the tiny cache; λ = 0 → targets equal labels; λ = 0.4 → blend; missing uid → label; κ = 2 → weights mean 1 and ∝ 1 + κ·loss; κ = 0 → all ones. (Exercise the helper that builds the maps.)
- `test_train_one_epoch_soft_mistake_weighted`: `train_run` on the tiny cache with `arch="hemi"`, `soft_csv`, `soft_lam=0.4`, `mistake_w=2.0`, `dann=0.1, dro=0.1` → finite loss, `.pt` + `.json` with the new fields, `n_soft` = number of training uids in the table, val log loss computed on hard labels (equals `logloss_np` of the oof csv).
- `test_run8_notebook_generates`: run the generator into a temp dir → nbformat validates, contains the soft-table cell, `--soft-csv cache_v4/soft_master_v2.csv` in every soft job, `device_count` in the runner, 42 jobs.
- `test_howto_pdf_one_page`.

## Step 5 — notebook (`kaggle/run8_distill/gen_run8.py`)

Reuse `gen_run7.py`'s pattern (exec of `gen_run3_inject.py`, `runner_cell`, `failsafe`). New pieces:
- `SOFT_CELL`: writes `cache_v4/soft_master_v2.csv` from an embedded base64-gzip string (generated from `artifacts/soft_master_v2.csv`).
- `JOBS8` (longest first, 42 jobs): hemisoft s0–2 random (15), hemisoft s0 site (5), hemimw s0 (5), hemi s3–4 (10), crophistsoft s0 (5), hemisoft full s0–1 (2).
- `ARMS_ALL` + `("a3d_hemisoft", "r3d_hemisoft_f?_s?", "r3d_hemisoft_f?_s?_site"), ("a3d_hemimw", "r3d_hemimw_f?_s?", None), ("a3d_crophistsoft", "r3d_crophistsoft_f?_s?", None)`; `EXTRAS_ALL` + `("a3d_hemisoft", "r3d_hemisoft_fall_s*")`.
- Stack cell: the run-7 super-stack plus a printed "gate table": nested (equal weights, T floor 0.9) for Master v2 pillars, pillars + each new arm, pillars + all new arms.
- `kernel-metadata.json` with `--owner skyfall23 --slug dat-run8-distill` as the default owner for this run.
- `APPROACH.md`, `HOWTO.md`, `make_howto_pdf.py` → `HOWTO.pdf` (one page).

## Step 6 — assembly (`scripts/build_master_v3.py`)

`python scripts/build_master_v3.py --src data/run8 [--src data/master_pool ...] --smoke <smoke_dir> --zip data/submission_master_v3.zip [--gate 0.003] [--T 0.9]`:
- pool OOFs; nested estimates (fitted/equal/shrunk) for: v2 pillars; + each new arm; + all new arms; prints the table.
- select the best set by equal-weight nested; if its gain over v2 < `--gate`, still build the zip but print `GATE: FAIL (keep Master v2 for slot 1)`.
- manifest: all seeds of every member (no pruning), equal weights, T 0.9, λ 0.02; drop lowest-weight arms' checkpoints until < 850 MB.
- pack from local artefact folders and the run-5/7 zips (as `build_master_v2.py`), then run `scripts/local_e2e_v3.py`.

## Step 7 — launch on the teammate's account

```
$env:KAGGLE_CONFIG_DIR = "C:\Users\HARSHDIP\.kaggle_skyfall23"     # holds the teammate's kaggle.json (never in the repo, never pasted in chat)
python kaggle/run8_distill/gen_run8.py
kaggle kernels push -p kaggle/run8_distill
```
Then on kaggle.com (the teammate or the shared login): notebook → Edit → GPU T4 x2, Internet On, Secrets `KAGGLE_USERNAME`/`KAGGLE_KEY` (already set on that account's earlier notebooks; they are per-notebook, so set them once more). Watch: `kaggle kernels status skyfall23/dat-run8-distill`; fetch: `python scripts/watch_kaggle_output.py --kernel skyfall23/dat-run8-distill --out data/run8` (with the same `KAGGLE_CONFIG_DIR`). Do not run other notebooks on that account concurrently.

## Step 8 — decision rule (written before the run)

- v3 replaces v2 in slot 1 iff nested(v3) ≤ nested(v2) − 0.003 **and** platform smoke(v3) ≤ smoke(v2) + 0.010 (0.2381 + 0.010 = 0.2391).
- Otherwise v2 in slot 1, v3 in slot 2 if nested(v3) < nested(v2) at all, else the λ 0.01 bracket of v2 in slot 2.
- Slot 3: the λ 0.01 bracket of whichever of v2/v3 went first.
- Record every number in `RESULTS_LOG.md`; smoke scores are read, never fitted on (ADR 0005).

## Risks

- Session cap / cancel: resume + upload every 4 jobs; re-push continues.
- Soft targets leak site information into the site-fold read-out: gate on random-fold nested only.
- Zip size: 42 new fp16 checkpoints ≈ 170 MB on top of 732 MB → drop `a2d_v2s` seed 1 first if over 850 MB.
- If the teammate's token is not on this laptop by 22:30 UTC, the push waits for it; everything else is done and committed.
