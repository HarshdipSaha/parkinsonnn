# Run 6b (tri-planar) — how to run it on Kaggle from this repo

**One command, from the repo root** (pushes the notebook and starts it as a Save & Run All commit on GPU T4 x2):

```
python kaggle/run6b_triplanar/gen_run6b.py          # optional: regenerate notebook.ipynb + kernel-metadata.json from the repo
kaggle kernels push -p kaggle/run6b_triplanar
```

One-time setup on the Kaggle account that runs it:
1. `~/.kaggle/kaggle.json` with that account's API token (Kaggle → Settings → API → Create New Token). Client 2.x: `pip install -U kaggle`.
2. The three input datasets must be visible to the account: `harshdipsaha/dat-parkinsons-cache-v4`, `harshdipsaha/dat-parkinsons-challenge-data`, `harshdipsaha/dat-run3-checkpoints` (private → add the account as a collaborator on each dataset, "can edit" on `dat-run3-checkpoints` so the per-job checkpoint upload works).
3. After the first push, open the notebook on kaggle.com → Add-ons → Secrets → add `KAGGLE_USERNAME` and `KAGGLE_KEY` (that account's token) and attach both; without them the run still completes but checkpoints are not uploaded every 4 jobs. Accelerator GPU T4 x2 and Internet are set by the metadata (`enable_gpu`, `enable_internet`).

Watch and fetch:
```
kaggle kernels status <owner>/dat-run6b-triplanar                              # RUNNING → COMPLETE (≈ 4 h)
python scripts/watch_kaggle_output.py --kernel <owner>/dat-run6b-triplanar --out data/run6b   # downloads + verifies the zip (or the Task-Scheduler watcher)
```
Assemble locally (nested stack over every arm with OOF files, gate, pack, container-style check on the 20 smoke scans):
```
python scripts/run4_assemble.py --src data/run6b --extra-dir artifacts/featarm_v4 --smoke <smoke_dir> --zip data/submission_run6b.zip
```
Then smoke-test the zip on DrivenData (`kaggle/gen_submit.py <zip> "<note>" smoke`, or the site) and, if its smoke score beats
run 4a's 0.2176–0.2263, submit it as a normal submission when a slot is free (Sept 12 15:38 UTC, then Sept 13).

**Second Kaggle account (teammate):** edit `kernel-metadata.json` → `"id": "<teammate>/dat-run6b-triplanar"` (or run
`python kaggle/run6b_triplanar/gen_run6b.py --owner <teammate>`), keep `dataset_sources` as they are if the datasets were
shared with the teammate, otherwise replace them with the teammate's re-uploaded copies (`kaggle datasets create -p data/upload_v4`,
`... -p data/ckpt_upload_v5`), and use the teammate's token in `~/.kaggle/kaggle.json` and in the notebook secrets.
The checkpoint-upload target inside the notebook is `harshdipsaha/dat-run3-checkpoints`; a teammate can only push to it as an
"edit" collaborator — otherwise the uploads print a permission error and the outputs are still saved in the version output.

What the notebook does: writes the package files, mounts the cache, resumes all existing checkpoints from the checkpoint
dataset, trains the 25 tri-planar jobs (`JOBS_B` in `gen_run6b.py`) on both GPUs, re-scores every checkpoint with the shared
TTA, fits the nested stack over 9 arms, packs `submission_v3.zip`, runs the container-style end-to-end check (per-arm smoke-set
log losses are printed — the 20 smoke scans are our out-of-distribution read-out), and splits the zip into < 95 MB parts.
