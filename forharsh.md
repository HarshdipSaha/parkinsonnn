# forharsh — do everything yourself until Sept 16 (simple, in order)

You are at **public 0.2667, rank 44**. Leader 0.2166, rank 10 = 0.2379. All three real submissions of this week are used; they come back at **Sept 12 15:38, Sept 13 15:50, Sept 14 03:38 UTC** (7 days after the Sept 5/6/7 submissions). Deadline **Sept 16 23:59 UTC**. Smoke tests are free (3 per day, reset daily). The best submission on the private set counts automatically. Run every command from `H:\augsepthacks\parkinsonnn` in PowerShell or Git Bash.

## 0. What is already done and where it is

| Thing | Where | Number |
|---|---|---|
| Master v2 zip (5 pillars, equal weights, T 0.9, all seeds) — **the safe Sept 12 submission** | `data\submission_master_v2.zip` (732 MB) | nested 0.2323, DrivenData smoke 0.2381 (id-321786) |
| Run 8 = soft-target / mistake-weighted hemisphere members + 2 extra seeds, 42 jobs | Kaggle `skyfall23/dat-run8-distill` (started Sept 10 16:33 UTC, ≈ 4.5 h) | gate: must beat 0.2323 by ≥ 0.003 |
| Smoke scans with labels (20, never train on them) | `data\smoke` (`niftis\`, `submission_format.csv`, `test_labels.csv`) | |
| All OOF files of runs 3–7 | `data\master_pool\artifacts_v3` | |
| Every number so far | `RESULTS_LOG.md` (the "run 5 public 0.2600" line was fabricated and is corrected there) | |
| Full handoff, plan, spec | `aidlc-docs\HANDOFF.md`, `docs\superpowers\plans\2026-09-10-run8-distill-plan.md`, `docs\superpowers\specs\2026-09-10-run8-distill-spec.md` | |

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

| Slot (UTC) | Upload | Note to write |
|---|---|---|
| **1 — Sept 12 15:38** | `submission_master_v3.zip` if the gate PASSED and its smoke ≤ 0.2481; otherwise `submission_master_v2.zip` | "master v3 (or v2): 5 pillars + run-8 arms, equal weights, T 0.9, nested X" |
| **2 — Sept 13 15:50** | the one that did not go first (v2 if v3 went first; v3 if it exists and is below 0.2323). If there is no v3 at all: the λ 0.01 bracket of v2 (below) | "second-best stack" / "bracket lam 0.01" |
| **3 — Sept 14 03:38** | the λ 0.01 bracket of whatever went in slot 1 (same models, sharper prior mix) — or, if the section-3 session produced a better-gated zip, that zip | "bracket lam 0.01" |

Bracket zips (verify each with the container check before uploading):
```
python scripts/build_master_v2.py --T 0.9 --lam 0.01 --out data/submission_master_v2_lam01.zip
python scripts/local_e2e_v3.py data/submission_master_v2_lam01.zip data/smoke
python scripts/build_master_v3.py --src data/run8/artifacts_v3 --src data/master_pool/artifacts_v3 --lam 0.01 --smoke data/smoke --zip data/submission_master_v3_lam01.zip
```
Never upload a zip whose container check did not print `FORMAT OK` with every arm N/N. Never re-submit the same zip twice.

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
