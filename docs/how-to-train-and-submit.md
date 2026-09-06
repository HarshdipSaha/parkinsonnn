# How to train run 3 on Kaggle and submit

Assumes the raw data is in `data/niftis/` + `data/train_labels*.csv` locally and the Kaggle datasets exist (see [aidlc-docs/HANDOFF.md](../aidlc-docs/HANDOFF.md) for names). Commands run from the repo root in Git Bash.

## 1. Build the caches locally (once, ~30 min on 12 CPUs)

```bash
python scripts/cache_volumes.py --out cache_v3 --workers 8              # 2 mm CoM boxes
python scripts/build_template.py --cache cache_v3 --out artifacts/template.npz --n-iter 2 --subset 400 --workers 10
python scripts/register_cache.py --src cache_v3 --out cache_v4 --template artifacts/template.npz --workers 10
cp artifacts/run2_2026-09-05/out/folds.csv cache_v3/folds_run2.csv
python scripts/pseudo_sites.py --cache cache_v3 --run2-folds cache_v3/folds_run2.csv   # folds.csv, folds_site.csv, sites.csv (copied into cache_v4 by register_cache)
python scripts/gate_v3.py --cache cache_v4 --template artifacts/template.npz --save --out artifacts_v3   # ROI-LR arm + gate
python scripts/fetch_pretrained.py pretrained
python -m pytest tests/test_v3.py tests/test_infer_v3.py -q
```

## 2. Upload the cache dataset (once per cache version)

1. Zip: `cache_v4/{vols,scalars,stats}.npy + csvs` → `cache_v4.zip`; `pretrained/*.pt + artifacts/template.npz` → `assets_v4.zip`; `artifacts_v3/{lr_v3.json,oof_lrarm*.csv,lrarm.json}` → `lr_v4.zip` (see `data/upload_v4/`).
2. Kaggle → Datasets → New Dataset → drop the three zips → title `dat-parkinsons-cache-v4` → Private → Create. Wait for "Version 1" and 19 files on the data tab.
3. In the notebook editor: Add Input → the new dataset (and the raw dataset).

## 3. Set up the notebook

1. Edit `JOBS` / `JOBS2` / `ARMS` in `kaggle/gen_run3_inject.py` if needed; `python kaggle/gen_run3_inject.py`.
2. Fresh notebook: run `.playwright-mcp/inject_run3.js` through the Playwright MCP tool (it wipes the notebook and inserts all cells).
   Existing notebook: `python kaggle/push_cells.py` then run `.playwright-mcp/push_cells.js` (updates changed cells, runs changed `%%writefile` cells, touches no runner).
3. Start the session (GPU T4 x2), wait for the kernel to be idle.
4. Select the first runner cell and execute `notebook:run-all-above` (writes files, mounts the cache, runs selftest and the timing check).

## 4. Train

1. Select the queue-1 runner cell → `notebook:run-all-below`. The runner starts two jobs at a time, one per GPU, skips tags that already have a `.pt`, and prints one line per finished job.
2. Poll with `.playwright-mcp/poll_runner.js` (25-minute wait per call).
3. When queue 1 is done: select the stale-checkpoint cleanup cell → `run-all-below` (queue 2 → re-OOF with TTA → report/stack/pack → end-to-end check → split).
4. Check the report cell: per-arm means, `weights=…`, `T=…`, `stack OOF … logloss=…`, and the e2e cell: `FORMAT OK`, extrapolated minutes < 180.

## 5. Get the zip off Kaggle

```bash
# after the split cell has printed the md5 and the part names
# run .playwright-mcp/download_parts.js (Playwright), then:
python scripts/join_parts.py data/parts_v3 <md5> --out submission_v3.zip
```

Parts smaller than ~100 MB download through Kaggle's own handler; the join script verifies the md5 and runs `testzip`.

## 6. Submit

```bash
python kaggle/gen_submit.py H:/augsepthacks/parkinsonnn/submission_v3.zip "run3: <what changed>" smoke
# run .playwright-mcp/dd_submit.js; wait for Completed; open View log; check every "arm …" line and exit code 0
python kaggle/gen_submit.py H:/augsepthacks/parkinsonnn/submission_v3.zip "run3: <what changed>" normal
# run .playwright-mcp/dd_submit.js  (uses one of the 3 real slots per rolling 7 days)
```

Record the id, public log loss, AUROC and rank in `RESULTS_LOG.md`.

## 7. Iterate

- Add ensemble members as entries in `JOBS2` (per-job `folds`, `seeds`, `args`); every new arm needs an entry in `ARMS` (name, OOF pattern, site pattern) and, for full-data models, in `EXTRAS`.
- Re-run steps 3.1–3.2 (`push_cells`), then the queue-2 cell and the cells below it.
- Keep a change only if the stacked OOF improves (see the plan for the nested-estimate gate).
