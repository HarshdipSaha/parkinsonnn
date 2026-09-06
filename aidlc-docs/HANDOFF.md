# HANDOFF — resume here if the previous session is gone

Last updated: 2026-09-06 ~09:00 UTC (14:30 IST). Deadline 2026-09-16 23:59 UTC.
Goal stated by the user: **rank 1** on the DrivenData DaT Parkinson's Challenge (public log loss < 0.219). Current public: 0.3391 (rank 152, run 2). Run 3 is training now.

Read in this order: this file → `aidlc-docs/efforts/002-run3-registered-ensemble/effort-state.md` (what is done / what is left) → `docs/superpowers/plans/2026-09-06-run3-finish-and-run4.md` (step-by-step plan) → `aidlc-docs/efforts/003-run4-toward-rank1/effort-state.md`.

## 1. Where things physically are

| Thing | Location |
|---|---|
| Repo (source of truth, pushed) | `H:\augsepthacks\parkinsonnn`, private GitHub `HarshdipSaha/parkinsonnn`, branch `master` |
| Competition | https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/ (submissions page: `/submissions/`) |
| Kaggle notebook (training) | https://www.kaggle.com/code/harshdipsaha/dat-parkinsons-train/edit — GPU T4 x2, internet on; session started 06:05 UTC Sept 6 (12-h cap → dies ~18:05 UTC) |
| Kaggle datasets (private) | `harshdipsaha/dat-parkinsons-challenge-data` (raw), `harshdipsaha/dat-parkinsons-cache-v4` (registered 2 mm cache `cache_v4/`, `assets_v4/` = template + torchvision weights, `lr_v4/` = ROI-LR arm) |
| Local caches | `cache_v3/` (CoM boxes), `cache_v4/` (registered), `artifacts/template.npz`, `pretrained/*.pt`, `artifacts_v3/lr_v3.json + oof_lrarm*.csv + lrarm.json`, `data/upload_v4/*.zip` (what was uploaded) |
| Run-2 artefacts (submitted model) | `artifacts/run2_2026-09-05/` (submission_run2.zip, checkpoints, OOF csvs, Kaggle folds.csv = the canonical folds) |
| Browser automation | Playwright MCP (`plugin:playwright:playwright/browser_run_code_unsafe`), JS files under `.playwright-mcp/` (gitignored), Chrome profile already logged in to Google/Kaggle/DrivenData |
| Results & decisions | `RESULTS_LOG.md`, `docs/RUN3_BRAINSTORM.md`, `COUNCIL_VERDICT.md`, `aidlc-docs/audit.md` |

## 2. State of the Kaggle notebook right now (cell indices as of 09:00 UTC)

0 markdown · 1 env · 2–43 `%%writefile` (42 package files, one per repo file) · 44 `!touch scripts/__init__.py` + selftest · 45 cache/pretrained/template mount from the dataset · 46 timing check · **47 stale-checkpoint cleanup** (rm B0 f0/f1 seed 0) · **48 runner copy (queue 1)** · **49 runner (queue 2)** · **50 re-OOF with TTA** · **51 LIVE runner (queue 1, `[*]`)** · **52 report + stack + pack** (queued) · **53 end-to-end check** (queued) · **54 split zip into parts**.

Queue 1 (cell 51) = B0 5 folds x 3 seeds (done), 3D 5 x 3 (in progress), B0 site-folds, 3D site-folds, V2-S 5 x 2. ETA ~12:45 UTC. When it ends, cells 52–53 run automatically (preliminary zip). Seven `%%writefile` cells (new TTA code) are also queued and execute after that.

Files on the Kaggle box: `artifacts_v3/<tag>.pt|.json`, `artifacts_v3/oof_<tag>.csv`, `log_<tag>.txt`, `cache_v4/` (symlinked memmap), `pretrained/`, `artifacts/template.npz`.
**Nothing in `/kaggle/working` survives a session death.** Before any risky step: Kaggle "Save Version → Quick Save" keeps the current outputs as a version.

## 3. Exact runbook to finish run 3 (after queue 1 ends)

1. Poll: run `.playwright-mcp/poll_runner.js` (25-min wait) until the live runner prints `all done in N min; failures: []`, and cells 52/53 have output.
2. Read cell 52 output (per-run table, `fit_stack` line `weights=… T=… stack OOF … logloss=…`) and cell 53 (`FORMAT OK … smoke-test log loss`, `EXTRAPOLATED … min`). Record in `RESULTS_LOG.md`.
3. Quick Save the notebook version (insurance).
4. Select **cell 47** and `notebook:run-all-below`: cleanup → runner copy (retrains the 2 stale B0 checkpoints) → queue 2 (coronal B0 x5, full-data B0 x3, full-data 3D x2; ~60 min) → re-OOF with TTA → live-runner cell (prints "0 jobs to run") → report/stack/pack → e2e → split. If time is short, delete the `r3d_full` entry from cell 49 first (saves ~25 min).
5. Download the parts: `.playwright-mcp/download_parts.js` (Kaggle's own downloader serves `files/parts/<name>`; each < 95 MB). Files land in `data/parts_v3/` (or `.playwright-mcp/` if saveAs is "canceled" — check both). Then `python scripts/join_parts.py data/parts_v3 <md5 from cell 54> --out submission_v3.zip`.
6. Local check: `python -c "import zipfile; print(zipfile.ZipFile('submission_v3.zip').testzip())"`; optionally `DATSC_DATA_ROOT=data-demo DATSC_ARTIFACTS=<unzipped>/artifacts python main.py` on a few local scans.
7. DrivenData smoke test (free, 3/day): `python kaggle/gen_submit.py H:/augsepthacks/parkinsonnn/submission_v3.zip "run3 … note" smoke` then run `.playwright-mcp/dd_submit.js`. Wait for `Completed`, open "View log", confirm `arm a3d: checkpoints used N/N`, `arm a2d_b0 …`, `arm lr: LR on 20 volumes`, exit code 0, and the 20-scan score.
8. Real submission (slot 2 of 3): same generator with `normal`. Record id + public score/rank in `RESULTS_LOG.md`, `aidlc-docs/efforts/002-…/effort-state.md`, memory.

If the Kaggle session died: Start session (button top-right; accelerator already GPU T4 x2) → wait for kernel idle → select cell 47 → `notebook:run-all-above` (runs 0–46: writes files, mounts cache) → then step 4. Everything in `artifacts_v3/` is gone, so the runner retrains all 50 + queue 2 (~5.5 h). Alternative: if a Quick Save version exists, its output files can be attached as a dataset input and copied back into `artifacts_v3/`.

## 4. Gotchas learned the hard way (do not rediscover)

- **`pkill -f scripts/train_v3.py` kills its own shell first** → nothing after it runs. Use `pkill -f "[s]cripts/train_v3.py"`.
- **Interrupting the kernel kills the training subprocesses** (same process group). Only interrupt when you accept losing the running jobs; the runner is resumable (skips tags with an existing `.pt`).
- **The runner cell blocks the kernel**; extra cells queue behind it. Diagnostics need an interrupt, or read `log_<tag>.txt` from another notebook.
- Editing a *queued* cell's source before it executes takes effect (JupyterLab reads the source at execution time).
- `document.querySelectorAll('.jp-Cell')` snapshots are stale after inserting/deleting cells — re-query before reading prompts.
- Kaggle in-editor file download: use `<a href=baseUrl+'files/<path>' download>` inside the JupyterLab frame; Kaggle's own handler then downloads it (`Downloading file …` event). Works up to ~100 MB; larger files silently fail → split. The Jupyter contents API returns 404 on Kaggle's proxy. Small files: print base64 in a cell and read it via the notebook model (`.playwright-mcp/…b64…`).
- Playwright MCP sandbox: no `require`/`import`; pass code via `filename` under `.playwright-mcp/`; the tool backgrounds after 120 s but keeps running (long waits are fine, results arrive as task notifications); tool output echoes the whole JS (large scripts → result saved to a file; grep it).
- `notebook:run` needs the cell selected (`nb.deselectAll(); nb.select(nb.widgets[i])`); `notebook:run-cell`, `run-all-below`, `run-all-above` work; `kernel:interrupt` does not exist — use `sessionContext.session.kernel.interrupt()`.
- Windows/Git-Bash heredocs mangle backslashes (`\\n` → newline); generate JS with Python and use `String.fromCharCode(10)`; validate with `node --check`.
- EMA: decay must give a 2–3-epoch window (0.99 for ~35 steps/epoch, 0.995 for ~70). A long window (0.998/0.999) averages weights from when the model was still poor → compressed, under-confident logits (val 0.58 at AUROC 0.87). Always log raw-model val metrics next to EMA metrics.
- Kaggle folds: `artifacts/run2_2026-09-05/out/folds.csv` are canonical (local `make_folds` output differs). Pseudo-site folds: `cache_v4/folds_site.csv`.
- Rules: 3 real submissions per rolling 7 days (used 1 on Sept 5 15:38 UTC), smoke tests 3/day; no pseudo-labelling; per-sample processing only; torchvision ImageNet weights OK (disclose); DINOv3/NC weights not OK; Kaggle training is an accepted prize-eligibility risk.

## 5. Skills / tools used so far (for the record)

deep-research (Workflow; mostly failed on subagent quota, 5 verified claims kept), brainstorming (non-interactive subagent → `docs/RUN3_BRAINSTORM.md`), llm-council (`COUNCIL_VERDICT.md`), writing-plans (`docs/superpowers/plans/*`), arxiv/lit-review (`LITERATURE_REVIEW.md`), EDA skills (`EDA_REPORT.md`), ai-dlc (this folder), Playwright MCP for Kaggle/DrivenData, pytest (`tests/test_v3.py`, `tests/test_infer_v3.py`, `tests/test_core.py`, `tests/test_infer.py`).
