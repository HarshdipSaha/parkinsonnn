# Handoff: resume here

| | |
|---|---|
| **Last updated** | 2026-09-06 10:25 UTC (15:55 IST) |
| **Deadline** | 2026-09-16 23:59 UTC |
| **Goal (user)** | Rank 1 on the DrivenData DaT Parkinson's Challenge (public log loss < 0.219) |
| **Current public score** | 0.3391 log loss, AUROC 0.925, rank 152 (run 2, submission id-321156) |
| **In flight** | Run 3 as a **background commit** (Kaggle Version #2 "Save & Run All", started 09:50 UTC, ETA ~15:10 UTC for the zip; 8 B0 folds done by 10:20 UTC at 0.258–0.294). The interactive session died at ~09:00 UTC after a websocket drop and took 3.5 GPU-h of checkpoints with it. Check status with `kaggle kernels status harshdipsaha/dat-parkinsons-train`. |

Reading order:

1. This file.
2. [efforts/002-run3-registered-ensemble/effort-state.md](efforts/002-run3-registered-ensemble/effort-state.md): what is done and what is left.
3. [docs/superpowers/plans/2026-09-06-run3-finish-and-run4.md](../docs/superpowers/plans/2026-09-06-run3-finish-and-run4.md): step-by-step plan with expected outputs.
4. [efforts/003-run4-toward-rank1/effort-state.md](efforts/003-run4-toward-rank1/effort-state.md): the next effort.

---

## 1. Where things are

| Thing | Location |
|---|---|
| Repository (source of truth, pushed) | `H:\augsepthacks\parkinsonnn`, private GitHub `HarshdipSaha/parkinsonnn`, branch `master` |
| Competition | https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/ (submissions: `/submissions/`) |
| Kaggle training notebook | https://www.kaggle.com/code/harshdipsaha/dat-parkinsons-train/edit (GPU T4 x2, internet on). Session started 06:05 UTC Sept 6; 12-hour cap, dies ~18:05 UTC |
| Kaggle datasets (private) | `harshdipsaha/dat-parkinsons-challenge-data` (raw scans); `harshdipsaha/dat-parkinsons-cache-v4` (`cache_v4/` registered 2 mm cache, `assets_v4/` template + torchvision weights, `lr_v4/` ROI-LR arm) |
| Local caches | `cache_v3/` (CoM boxes), `cache_v4/` (registered), `artifacts/template.npz`, `pretrained/*.pt`, `artifacts_v3/` (LR arm), `data/upload_v4/*.zip` (what was uploaded) |
| Run-2 artefacts | `artifacts/run2_2026-09-05/` (zip, checkpoints, OOF csvs, and the canonical `folds.csv`) |
| Browser automation | Playwright MCP tool `plugin:playwright:playwright/browser_run_code_unsafe`; scripts in `.playwright-mcp/` (gitignored); Chrome profile is logged in to Google, Kaggle and DrivenData |
| Results and decisions | `RESULTS_LOG.md`, `docs/RUN3_BRAINSTORM.md`, `COUNCIL_VERDICT.md`, `aidlc-docs/audit.md` |

## 2. Kaggle notebook cell map (draft as of 10:25 UTC; the running commit executes a frozen copy saved at 09:50 UTC, without the resume/upload additions)

| Cell | Content | State |
|---|---|---|
| 0 | Markdown header | |
| 1 | Environment check | done |
| 2–43 | `%%writefile` for the 42 package files (one per repo file) | done; 7 updated cells queued to re-run |
| 44 | `touch scripts/__init__.py` + `main_v3.py --selftest` | done |
| 45 | Mount `cache_v4`, `pretrained/`, `artifacts/template.npz` from the dataset | done |
| 46 | Timing check (2 epochs per arm) | done |
| 47 | Stale-checkpoint cleanup (no-op on a fresh machine) | |
| 48 | Runner, queue 1: B0 5x3 seeds, 3D 5x3, B0 site-folds, V2-S 5x1; uploads checkpoints every 4 jobs when secrets exist | |
| 49 | Runner, queue 2: coronal B0 x5, full-data B0 x3 | |
| 50 | Re-OOF with test-time augmentation (fail-safe) | |
| 51 | Report + stack + pack `submission_v3.zip` (fail-safe) | |
| 52 | End-to-end check (fail-safe) | |
| 53 | Split the zip into < 95 MB parts (fail-safe) | |

Cell 45 (cache mount) also copies any `artifacts_v3/` or `out/` found under `/kaggle/input/` (a previous version's output or the checkpoint dataset) into the working `artifacts_v3/`, so a new commit resumes.

Files on the Kaggle machine: `artifacts_v3/<tag>.pt`, `<tag>.json`, `oof_<tag>.csv`; `log_<tag>.txt`; `cache_v4/` (symlinked memmap); `pretrained/`; `artifacts/template.npz`.

Draft-editor note: opening the notebook in a second browser while the automation tab holds the draft produces `ConcurrencyViolation … Sequence number must match Draft record`; it does not affect a running commit.

**Nothing in `/kaggle/working` survives a session death, and a session whose kernel is unreachable cannot be salvaged either** (Quick Save needs a connected editor and fails with `ConcurrencyViolation` when two editor tabs are open). Do not rely on interactive sessions for runs longer than ~1 h: use a background commit (section 3a).

## 3a. The commit run (current path)

Interactive sessions die when the browser connection drops for long; a commit ("Save & Run All") runs the saved notebook top to bottom on a fresh machine with no browser, keeps its outputs as a version, and is what is running now. Cell order in the saved draft: 0 header, 1 env, 2–43 writefile, 44 selftest, 45 cache mount, 46 timing, 47 stale cleanup (no-op on a fresh machine), 48 runner queue 1 (B0 x3 seeds, 3D x3 seeds, B0 site-folds, V2-S x1 seed), 49 runner queue 2 (coronal B0, full-data B0 x3), 50 re-OOF with TTA, 51 report/stack/pack, 52 end-to-end check, 53 split into parts. Cells 50–53 are wrapped in try/except so the commit succeeds even if a tail step fails.

Monitoring: https://www.kaggle.com/code/harshdipsaha/dat-parkinsons-train/log?scriptVersionId=347676112 (Version #2; poll with `.playwright-mcp/poll_commit.js`, peek with `peek_commit.js`), or View Active Events → kebab → "Open Logs in Viewer". Expected duration ~5.5 h. When it finishes, the version's Output tab lists `submission_v3.zip`, `parts/`, `out/`; download the parts from there (no in-editor size cap on the version output page) or attach the version output as a dataset input to a new session.

### Persistence rule (from now on)

Never run more than ~2 h of training in one interactive session or one commit without persisting. Two mechanisms:

1. **Chained commits (no credentials).** The cache-mount cell copies any `artifacts_v3/` or `out/` found under `/kaggle/input/` into the working `artifacts_v3/`, and both runners skip jobs whose `.pt` exists. So: commit N saves its outputs as a version → in the editor, Add Input → Your Work → that notebook version's output → commit N+1 resumes. Split `JOBS` into ≤ 2 h chunks per commit.
2. **Per-job upload (implemented, secrets present since Sept 6 10:45 UTC).** `KAGGLE_USERNAME` / `KAGGLE_KEY` are stored as notebook Secrets (Add-ons → Secrets) and attached; both runner cells push `artifacts_v3/` to the private dataset `harshdipsaha/dat-run3-checkpoints` after every 4 finished jobs and at the end (`push_checkpoints`, background thread, upgrades the kaggle client first). To resume from it: attach that dataset as an input; the cache-mount cell copies its files into `artifacts_v3/`.

**Kaggle CLI (local).** Credentials from `.env` (gitignored) are mirrored in `~/.kaggle/kaggle.json`; the fine-grained token only works with client >= 2.2 (`pip install -U kaggle`). Useful commands: `kaggle kernels status harshdipsaha/dat-parkinsons-train` (commit status), `kaggle kernels output harshdipsaha/dat-parkinsons-train -p data/kaggle_output` (download the latest version's output files, no size cap), `kaggle datasets version -p <dir> -m "<note>"` (push a dataset version).

### Automatic download of the commit output (no session needed)

`scripts/watch_kaggle_output.py` polls `kaggle kernels status harshdipsaha/dat-parkinsons-train`; when the status is no longer RUNNING/QUEUED it pages through the latest version's output via the Kaggle API into `data/kaggle_output/`, joins `submission_v3.zip.part*` if only parts exist, verifies the zip (md5, `testzip`, `main.py` present) and writes `data/kaggle_output/DONE.json` plus `watch.log`. It is registered as the Windows Task Scheduler job `DaT-KaggleOutputWatch` (every 5 min, `kaggle/register_watcher.ps1`, registered Sept 6 11:34 UTC) and deletes the task once the download is verified. The zip therefore lands on the laptop with no Claude session open. **What stays manual:** the DrivenData smoke test and real submission (`kaggle/gen_submit.py` + `.playwright-mcp/dd_submit.js`, or the website), and recording the score in `RESULTS_LOG.md`.

Check: `type data\kaggle_output\watch.log` and `schtasks /Query /TN DaT-KaggleOutputWatch`. Re-register after a reboot or for a new version with `powershell -File kaggle/register_watcher.ps1` (delete `data/kaggle_output/DONE.json` first if a newer version's output is wanted).

### Run 3 outcome and the container bug (Sept 6, 14:45 UTC) — **public 0.2817 / AUROC 0.9477 / rank 66** (id-321263). Next: effort 003, plan `docs/superpowers/plans/2026-09-06-run4-toward-0.21.md`

Version #2 completed: 56 jobs, no failures, stack OOF **0.2580** (AUROC 0.961; weights a3d .1 / b0 .3 / v2s .3 / cor .3 / lr 0; T = 1.0). Full numbers in `RESULTS_LOG.md`. The end-to-end cell then showed every 2D checkpoint failing with `FileNotFoundError`: the manifest's `model_kw` still carried `weights_path=pretrained/…` from training and `build_model` tried to load it inside the container. Fixed in commit `225d39d` (inference ignores `weights_path`, manifest strips it, e2e asserts every arm used all checkpoints). The Kaggle zip was repaired locally without retraining:

```
kaggle kernels output harshdipsaha/dat-parkinsons-train -p data/kaggle_output_zip --force --page-size 200 --file-pattern '^submission_v3\.zip$'
python scripts/repack_v3.py data/kaggle_output_zip/submission_v3.zip data/submission_v3_fixed.zip   # fresh code, same artifacts
python scripts/local_e2e_v3.py data/submission_v3_fixed.zip <smoke_dir>   # container-style run on the 20 smoke volumes (CPU, ~7 min)
```
Local result: all arms 43/43 checkpoints + LR, smoke log loss 0.2458 (run 2: 0.2567), 390 s on CPU. `data/submission_v3_fixed.zip` is the file submitted to DrivenData. Lesson: run `scripts/local_e2e_v3.py` on every zip before uploading; the Kaggle e2e cell now fails loudly on partial checkpoint use.

DrivenData upload gotchas: **zips above ~420 MB crash the Playwright MCP server** (the 844 MB run-4a zip killed it three times; upload those by hand or trim checkpoints). A 419 MB zip uploads for 10–20 min; the old `dd_submit.js` reloaded the page every 6 s, which raised `beforeunload` dialogs (accepting one aborts the upload). `kaggle/gen_submit.py` now waits for the upload dialog to close and auto-dismisses dialogs.

### Run 4a in flight (Sept 6 15:45 UTC → ~19:15 UTC) and how to finish it

Kaggle **Version 3** ("Save & Run All") trains the site-robust candidates of `JOBS3` in `kaggle/gen_run3_inject.py` (tiny slab net ×5 seeds with strong augmentation, V2-S seeds 1–2, strong-aug B0 on random + site folds, 224-px crop-64 B0, ConvNeXt-Tiny, coronal V2-S, striatal-crop 3D 48×48×32) on top of the 48 run-3 checkpoints resumed from the private dataset `harshdipsaha/dat-run3-checkpoints` (attached as an input; the cache cell also has an API-download fallback). The laptop watcher (`DaT-KaggleOutputWatch`, `-Out data/kaggle_output_v4`) downloads the version output when it ends.

The feature-engineered arm `feat` (`scripts/feat_arm_v4.py`, artefacts in `artifacts/featarm_v4/`) is fitted locally and is **not** in the Kaggle run; it is added at assembly time. Finish with:

```
python scripts/watch_kaggle_output.py --out data/kaggle_output_v4          # if DONE.json is not there yet
python scripts/run4_assemble.py --src data/kaggle_output_v4 --extra-dir artifacts/featarm_v4 --smoke <smoke_dir> --zip data/submission_v4.zip
python kaggle/gen_submit.py H:\augsepthacks\parkinsonnn\data\submission_v4.zip "run4a ..." smoke      # then .playwright-mcp/dd_submit.js
```
(`<smoke_dir>` = the extracted `data/smoke_test_data_*.tar.gz`: `niftis/`, `submission_format.csv`, `test_labels.csv`.) `run4_assemble.py` runs the nested stack over every arm with OOF files (grid 8), drops zero-weight arms, packs every referenced checkpoint and runs the container-style local check; it prints the nested estimate and per-site table — record them in `RESULTS_LOG.md`. Gate rule (R7/R10): keep a new member only if its weight ≥ 0.05 or the nested loss improves ≥ 0.003; compare `r2d_effb0_saug_*_site` with run 3's site-fold 0.2818.

Then: smoke test → real submission (slot 3) → record public score → run 4b (`docs/superpowers/plans/2026-09-06-run4-toward-0.21.md`, Task 7): seeds + full-data models for the kept members; upload the new checkpoints to the checkpoint dataset first (from inside the folder: `cd data/<dir> && kaggle datasets version -p . -m "..." --dir-mode skip`).

### Run 5 — ready to launch (Sept 7 00:20 UTC)

Code, tests and the notebook are done (effort 004; plan `docs/superpowers/plans/2026-09-06-run5-site-invariance-plan.md`). Launch either way:

- **Human path (no automation):** Kaggle → your notebook `dat-parkinsons-train` → File → Import Notebook → `kaggle/dat-parkinsons-train.ipynb` (or paste its cells), check inputs `dat-parkinsons-challenge-data`, `dat-parkinsons-cache-v4`, `dat-run3-checkpoints` (latest version — it must contain `template.npz` with `hist` landmarks and `feat_v4.npz`), GPU T4 x2, Internet on, Secrets `KAGGLE_USERNAME`/`KAGGLE_KEY`, then Save Version → Save & Run All. Queues 1–3 print "0 jobs to run"; queue 4 trains the run-5 jobs (≈ 3 h).
- **Automation path:** with the Playwright MCP connected, run `.playwright-mcp/push_cells.js` in the editor tab, then Save Version → Save & Run All.

Before launching make sure the stuck Version 3 is stopped (View Active Events → ⋯ → Stop Session) and `kaggle datasets status harshdipsaha/dat-run3-checkpoints` is `ready` after the Sept 7 upload (log `data/ckpt_upload_v5.log`).

After the run: `python scripts/run4_assemble.py --src <dir with artifacts_v3> --extra-dir artifacts/featarm_v4 --smoke <smoke_dir> --zip data/submission_v5.zip` (add `--typ` to the fit_stack call inside if wanted), gates per the plan Task 7, smoke test, real submission when slot 1 returns (Sept 12 15:38 UTC).

## 3b. Runbook to finish run 3 (interactive variant, kept for reference)

1. **Poll** with `.playwright-mcp/poll_runner.js` (25-minute wait) until the live runner ends with `all done in N min; failures: []` and cells 52/53 have output.
2. **Record** the cell 52 output (per-run table, `weights=… T=… stack OOF logloss=…`) and cell 53 (`FORMAT OK`, smoke-set log loss, extrapolated minutes) in `RESULTS_LOG.md`.
3. **Quick Save** the notebook version.
4. **Select cell 47 → `notebook:run-all-below`.** Order: cleanup → runner copy (retrains the 2 stale B0 checkpoints) → queue 2 (~60 min) → re-OOF with TTA → live-runner cell (prints "0 jobs to run") → report/stack/pack → e2e → split. If it is later than 14:30 UTC, first remove the `r3d_full` entry from cell 49 (saves ~25 min).
5. **Download** the parts with `.playwright-mcp/download_parts.js`. Files land in `data/parts_v3/` or in `.playwright-mcp/` (check both).
6. **Join and verify**: `python scripts/join_parts.py data/parts_v3 <md5 from cell 54> --out submission_v3.zip` → expect `md5 … OK` and `testzip: None`.
7. **Smoke test** (free, 3 per day): `python kaggle/gen_submit.py H:/augsepthacks/parkinsonnn/submission_v3.zip "<note>" smoke`, then run `.playwright-mcp/dd_submit.js`. Wait for `Completed`, open "View log", confirm every `arm …` line and `exit code: 0`.
8. **Real submission** (slot 2 of 3): same generator with `normal`. Record id, public log loss, AUROC and rank in `RESULTS_LOG.md`, effort 002 and memory.

### If the Kaggle session died

1. Click "Start session" (accelerator is already GPU T4 x2) and wait for the kernel to be idle.
2. Select cell 47 → `notebook:run-all-above` (runs cells 0–46: writes the package files, mounts the cache; ~5 min).
3. Continue from step 4 above. `artifacts_v3/` is gone, so the runner retrains everything (~5.5 h). If a Quick Save version exists, attach its output as a dataset input and copy the files back into `artifacts_v3/` first.

## 4. Gotchas (do not rediscover)

| Area | Gotcha | Do this instead |
|---|---|---|
| Shell | `pkill -f scripts/train_v3.py` kills its own shell before the rest of the command runs | `pkill -f "[s]cripts/train_v3.py"` |
| Kernel | Interrupting the kernel kills the training subprocesses (same process group) | Only interrupt when losing the running jobs is acceptable; the runner resumes by skipping tags with an existing `.pt` |
| Kernel | The runner cell blocks the kernel; other cells queue behind it | Read `log_<tag>.txt` from a second notebook, or interrupt |
| JupyterLab | Editing a queued cell's source before it runs takes effect | Safe to fix cells that are already queued |
| JupyterLab | `document.querySelectorAll('.jp-Cell')` is stale after inserting/deleting cells | Re-query before reading prompts |
| JupyterLab | `kernel:interrupt` does not exist; `notebook:run` needs the cell selected | `sessionContext.session.kernel.interrupt()`; `nb.deselectAll(); nb.select(nb.widgets[i])` then `notebook:run-cell` |
| Kaggle download | Contents API returns 404 through Kaggle's proxy; in-editor downloads fail silently above ~100 MB | Click `<a href=baseUrl+'files/<path>' download>` inside the frame (Kaggle's own downloader picks it up); split big files; small files via base64 printed in a cell |
| Playwright MCP | No `require`/`import`; long tool calls are backgrounded after 120 s but keep running; the result echoes the whole script | Pass code via `filename` under `.playwright-mcp/`; grep saved results |
| Windows shell | Heredocs mangle backslashes (`\\n` becomes a newline) | Generate JS from Python with `String.fromCharCode(10)`; validate with `node --check` |
| Training | EMA windows longer than ~3 epochs average in weights from when the model was still poor → compressed, under-confident logits | 0.99 for ~35 steps/epoch, 0.995 for ~70; always log raw-model validation metrics next to EMA metrics |
| Folds | Local `make_folds` output differs from Kaggle's | Canonical folds: `artifacts/run2_2026-09-05/out/folds.csv` (= `cache_v4/folds.csv`); site folds: `cache_v4/folds_site.csv` |
| Rules | 3 real submissions per rolling 7 days (1 used, Sept 5 15:38 UTC); smoke tests 3 per day; no pseudo-labelling; per-sample processing only | torchvision ImageNet weights are allowed (disclose); DINOv3 / non-commercial weights are not |

## 5. Skills and tools used so far

- Research: deep-research workflow (5 verified claims before quota), brainstorming (subagent, non-interactive), llm-council, arxiv / literature review, two EDA skills.
- Engineering: writing-plans, ai-dlc, pytest (`tests/test_v3.py`, `tests/test_infer_v3.py`, `tests/test_core.py`, `tests/test_infer.py`).
- Automation: Playwright MCP for Kaggle (JupyterLab model API) and DrivenData.
