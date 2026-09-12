# forharsh — do everything yourself until Sept 16 (simple, in order)

You are at **public 0.2557, rank 23** (AUROC 0.9576, Master v3 evaluated Sept 12; climbed +21 spots from rank 44). Leader 0.2166, rank 10 = 0.2379. 4 real submissions used; next slots: **Sept 13 15:50, Sept 14 03:38 UTC** (plus rolling weekly slots). Deadline **Sept 16 23:59 UTC**. Smoke tests are free (3 per day, reset daily). The best submission on the private set counts automatically. Run every command from `H:\augsepthacks\parkinsonnn` in PowerShell or Git Bash.

## 0. What is already done and where it is

| Thing | Where | Number |
|---|---|---|
| Master v2 zip (5 pillars, equal weights, T 0.9, all seeds) — **diversification candidate** | `data\submission_master_v2.zip` (732 MB) | nested 0.2323, DrivenData smoke 0.2381 (id-321786) |
| Master v3 zip (7 arms, 71 ckpts, equal weights 1/7, T 0.9, lam 0.02) — **Slot 1 COMPLETED** | `data\submission_master_v3.zip` (847 MB) | **Public 0.2557, Rank #23**, AUROC 0.9576, nested 0.2289 |
| Master v3 bracket zip (7 arms, lam 0.01) — **Slot 2 candidate** | `data\submission_master_v3_lam01.zip` (847 MB) | nested **0.2266** (all-time lowest CV estimate), smoke 0.2474 |
| Smoke scans with labels (20, never train on them) | `data\smoke` (`niftis\`, `submission_format.csv`, `test_labels.csv`) | |
| All OOF files of runs 3–8 | `data\master_pool\artifacts_v3` | |
| Every number so far | `RESULTS_LOG.md` | |
| Full handoff, plan, spec | `aidlc-docs\HANDOFF.md`, `docs/superpowers/plans/2026-09-10-run8-distill-plan.md`, `docs/superpowers/specs/2026-09-10-run8-distill-spec.md` | |

The Kaggle CLI on this laptop is logged in as **skyfall23** (`kaggle kernels list --mine` shows their notebooks). DrivenData is your own account in the Playwright/Chrome window; uploads are done by hand there.

## 1. Tonight (Sept 10, ≈ 21:00–21:30 UTC): fetch run 8

1. **Wait for COMPLETE**: `kaggle kernels status skyfall23/dat-run8-distill` (every 30 min). The notebook page: https://www.kaggle.com/code/skyfall23/dat-run8-distill
2. **Download + verify**: `python scripts/watch_kaggle_output.py --kernel skyfall23/dat-run8-distill --out data/run8` (writes `data\run8\artifacts_v3\...` and `DONE.json`).
3. **If Kaggle shows ERROR or the session was cut** (it has no upload secrets, so nothing is saved until the end): push it again, it restarts from zero (≈ 4.5 h): `kaggle kernels push -p kaggle/run8_distill`. If you added the `KAGGLE_USERNAME` / `KAGGLE_KEY` secrets on the notebook (Edit → Add-ons → Secrets, skyfall23's own values), a re-push resumes instead.

## 2. The gate (one command decides everything)

```
python scripts/build_master_v3.py --src data/run8/artifacts_v3 --src data/master_pool/artifacts_v3 --smoke data/smoke --zip data/submission_master_v3.zip
```
It prints the **GATE TABLE** (nested estimate for: Master v2 as shipped, pillars + each run-8 arm, pillars + all, greedy best subset), then `GATE: PASS` or `GATE: FAIL`, builds the zip for the best set (all seeds, equal weights, T 0.9; trims to < 850 MB by itself), and runs the container check on `data\smoke` (must print `FORMAT OK` and every arm N/N). Copy the table into `RESULTS_LOG.md`.

- **PASS** (best set ≤ 0.2293): smoke-test `data\submission_master_v3.zip` on DrivenData (section 4). If its smoke score is **≤ 0.2481** (within 0.01 of v2's 0.2381), **v3 is the Sept 12 submission**. If the smoke score is worse than that, v2 stays first and v3 goes second.
- **FAIL**: Master v2 is the Sept 12 submission. If the table still shows v3 below 0.2323 at all, keep the zip as the second submission; otherwise build the bracket (section 5).

## 3. Optional: second GPU session (≈ 10 GPU-hours left on skyfall23, only if run 8 helped)

More seeds of the arm that gated in, resuming from run 8's own output (no secrets needed):
```
python kaggle/run8_distill/gen_run8.py --kernel-source skyfall23/dat-run8-distill --hemisoft-seeds 0 1 2 3 4 --skip r3d_hemimw r3d_crophistsoft
kaggle kernels push -p kaggle/run8_distill
```
(≈ 2.5 h: 10 new hemisoft jobs; everything already trained prints "already done".) Then fetch into `data\run8b` and re-run the gate with `--src data/run8b/artifacts_v3` added. Launch it by Sept 11 midday UTC so it is back before the Sept 12 slot; otherwise it feeds slot 2.

## 4. Uploading (by hand, DrivenData)

https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/submissions/ → **Make new submission** → choose the zip → private note (what it is + its nested number) → **Smoke test** (free) or **Normal submission** (a real slot) → Submit. Upload takes ≈ 12 min at 730 MB; scoring ≈ 20 min (smoke) to ≈ 1 h (normal). One code job runs at a time. Write every score into `RESULTS_LOG.md` next to its id. Smoke scores are read, never used for fitting (ADR 0005).

## 5. THE THREE SUBMISSIONS

| Slot (UTC) | Upload | Note to write | Status |
|---|---|---|---|
| **1 — Sept 12** | `submission_master_v3.zip` (7 arms, equal weights, T 0.9, lam 0.02) | "master v3: 7 arms, equal weights, T 0.9, nested 0.2289" | **COMPLETED**: Public **0.2557**, AUROC 0.9576, **Rank #23** (jumped +21 spots!) |
| **2 — Sept 13 15:50** | `data/submission_master_v3_lam01.zip` (nested 0.2266, sharper prior mix) or `submission_master_v2.zip` | "master v3 bracket lam 0.01 nested 0.2266" | Ready on disk (847 MB) |
| **3 — Sept 14 03:38** | `data/submission_master_v2.zip` (732 MB, nested 0.2323) or step-B zip (`v3b`) if trained | "master v2 diversification / v3b" | Ready on disk |

Bracket zips (verify each with the container check before uploading):
```
python scripts/build_master_v2.py --T 0.9 --lam 0.01 --out data/submission_master_v2_lam01.zip
python scripts/local_e2e_v3.py data/submission_master_v2_lam01.zip data/smoke
python scripts/build_master_v3.py --src data/run8/artifacts_v3 --src data/master_pool/artifacts_v3 --lam 0.01 --smoke data/smoke --zip data/submission_master_v3_lam01.zip
```
Never upload a zip whose container check did not print `FORMAT OK` with every arm N/N. Never re-submit the same zip twice.

## 5b. After each real submission: read the score, then run exactly this

The public score comes back ≈ 1 h after the upload. For the zip you submitted you know its nested estimate N (from the gate table; Master v2 = 0.2323). Compute the **gap G = public − N** and write both into the ladder table in `RESULTS_LOG.md`. Runs 3 / 4a had G = 0.042 / 0.018. Nothing else about the test set is visible, so G is the only feedback you get; use it as below. Start the GPU session of step B the same hour the slot-1 score arrives; do not wait for slot 2.

**After slot 1 (score ≈ Sept 12 16:40 UTC) — pick the first case that matches:**

1. **Public ≤ 0.2379 (top 10).** Change nothing. Slot 2 = the other zip (v2 or v3), slot 3 = the λ 0.01 bracket of the slot-1 zip (section 5). Still launch step B so slot 3 can use more seeds if they arrive in time.
2. **Public worse than 0.2667 (our old best).** The zip broke on the platform, not the model. On the submissions page click *View log* for that id and look for `checkpoints used N/N` per arm and any `Traceback`. Slot 2 = the zip that did NOT go first (v2 if v3 broke); if v2 itself broke, slot 2 = `data\submission_master_final.zip` (486 MB, platform smoke 0.2363, id-321617 — the proven fallback; the run-4a zip is no longer on disk). Fix the packaging before slot 3 (`python scripts/local_e2e_v3.py <zip> data/smoke` must print `FORMAT OK`).
3. **G ≤ 0.022 (gap like run 4a or smaller).** The nested estimate is trustworthy: chase it. Run step B; slot 2 = the step-B zip if its gate table beats the slot-1 zip by ≥ 0.002 nested and its smoke test is within 0.01 of the slot-1 zip's smoke; otherwise slot 2 = the other of v2/v3. Slot 3 = the λ 0.01 bracket of the better of the two public scores.
4. **G > 0.03 (shift dominates).** Confident mistakes are being punished; hedge the calibration. Slot 2 = the **shift hedge** of the slot-1 zip (same models, T 1.0, λ 0.03 — less confident):
   ```
   python scripts/build_master_v2.py --T 1.0 --lam 0.03 --out data/submission_master_v2_hedge.zip
   python scripts/local_e2e_v3.py data/submission_master_v2_hedge.zip data/smoke
   ```
   (or the same with `python scripts/build_master_v3.py --src data/run8/artifacts_v3 --src data/master_pool/artifacts_v3 --T 1.0 --lam 0.03 --smoke data/smoke --zip data/submission_master_v3_hedge.zip` if v3 went first). Smoke-test it first (free). Still run step B: more seeds also reduce variance under shift, and feed slot 3.

**Step B — the second GPU session (≈ 2.5–3 h on skyfall23; the quota bar on the notebook page shows what is left).** Choose ONE line according to run 8's gate table, then push:

```
# (i) the soft-target hemisphere arm gated in (a3d_hemisoft in the best set):
python kaggle/run8_distill/gen_run8.py --kernel-source skyfall23/dat-run8-distill --hemisoft-seeds 0 1 2 3 4 --skip r3d_hemimw r3d_crophistsoft
# (ii) the mistake-weighted arm gated in instead (a3d_hemimw in the best set):
python kaggle/run8_distill/gen_run8.py --kernel-source skyfall23/dat-run8-distill --hemimw-seeds 0 1 2 --skip r3d_crophistsoft
# (iii) nothing from run 8 gated in: more plain hemisphere seeds (the safe gain):
python kaggle/run8_distill/gen_run8.py --kernel-source skyfall23/dat-run8-distill --hemi-more-seeds 3 4 5 6 --skip r3d_hemisoft r3d_hemisoft_site r3d_hemimw r3d_crophistsoft r3d_hemisoft_full
kaggle kernels push -p kaggle/run8_distill
```
`--kernel-source` mounts run 8's own output, so every job already trained prints "already done" and only the new seeds run (10 jobs ≈ 2.5 h). Watch with `kaggle kernels status skyfall23/dat-run8-distill`. When COMPLETE:
```
python scripts/watch_kaggle_output.py --kernel skyfall23/dat-run8-distill --out data/run8b
python scripts/build_master_v3.py --src data/run8b/artifacts_v3 --src data/run8/artifacts_v3 --src data/master_pool/artifacts_v3 --smoke data/smoke --zip data/submission_master_v3b.zip
```
Read its GATE TABLE against the zip you already submitted (not against 0.2323): use `submission_master_v3b.zip` only if it is ≥ 0.002 nested better than that zip, and only after a free smoke test.

**After slot 2 (score ≈ Sept 13 16:50 UTC), 10 h before slot 3 — no training fits any more, only assembly:**

1. Write the score and G into `RESULTS_LOG.md`. You now have two public scores; the zip with the better one is the **base**.
2. If the step-B zip exists, was smoke-tested, and beat the base's nested by ≥ 0.002 → slot 3 = that zip.
3. Otherwise slot 3 = the λ 0.01 bracket of the base (`--lam 0.01`, section 5), unless case 4 applied above and the hedge scored better than its parent — then slot 3 = the hedge of the other zip (`--T 1.0 --lam 0.03`).
4. Container check (`FORMAT OK`, all arms N/N) before every upload; never the same zip twice; the best of the three on the private set counts.

**After slot 3 (Sept 14):** nothing more can be submitted before the deadline. Write the final ladder into `RESULTS_LOG.md`, then section 8.

## 6. What "good" looks like

- Master v2: nested 0.2323 → expected public ≈ 0.25 (the gap has been 0.018–0.05), rank ≈ 15–25.
- Run 8 success: nested 0.226–0.229 → public ≈ 0.245–0.250, rank ≈ 12–18. Top 10 needs public 0.2379. Rank 1 (0.2166) is out of reach with this data; nobody outside PPMI-style agreements has more DaT-SPECT scans.
- Smoke-to-public so far: 0.2567→0.3391, 0.2397→0.2817, 0.2176/0.2263→0.2667. The same zip scored 0.2176 and 0.2263 on two smoke tests: differences under 0.01 are noise.

## 7. If something breaks

- **Kaggle stuck RUNNING > 6 h** → open the notebook → ⋯ → Stop Session; push again.
- **Gate script says "missing/incomplete OOF"** → a run-8 arm has < 5 folds; it is skipped automatically. If everything is missing, `--src` points at the wrong folder (needs the folder that contains `oof_*.csv`).
- **Zip > 850 MB** → the script trims by itself (drops V2-S seed 1 first). If DrivenData still refuses, add `--max-mb 700`.
- **DrivenData "one active code job"** → wait for the running one to finish.
- **Watcher not downloading** → `kaggle kernels output skyfall23/dat-run8-distill -p data/run8` by hand, then move the files into `data\run8\artifacts_v3`.
- **Laptop** → keep it plugged in and awake (sleep is disabled).

## 8. After the deadline (Sept 17)

Write `docs\DISCLOSURE.md` (ImageNet torchvision weights BSD-3, Kaggle as training venue, no external data, no pseudo-labelling, the smoke set never trained on), keep the code MIT, delete the private Kaggle datasets (`dat-parkinsons-challenge-data`, `dat-parkinsons-cache-v4`, `dat-run3-checkpoints`) and the notebook version outputs on both accounts. Prize eligibility is verified by DrivenData after placement, silently; a clean disclosure is what makes any reshuffle above you count in your favour.

## 9. Where the thinking is

`aidlc-docs\HANDOFF.md` (full handoff), `RESULTS_LOG.md` (every number), `docs\wayfinder\map.md` + tickets T01–T15, `docs\superpowers\plans\` and `specs\`, each `kaggle\run*\APPROACH.md` + `HOWTO.pdf`.
