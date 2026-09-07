# Running the notebooks on a second Kaggle account (teammate)

Everything a notebook needs lives in three private Kaggle datasets owned by `harshdipsaha`, the repo, and two per-account secrets. Nothing has to be re-trained or re-uploaded if the datasets are *shared*; re-uploading is the fallback.

## 0. What the teammate needs

- A Kaggle account with phone verification (GPU + Internet require it) and its own API token: kaggle.com → Settings → API → *Create New Token* → `kaggle.json`. Put it in `C:\Users\<them>\.kaggle\kaggle.json` on the machine that will run the push commands (never commit it; our `.env` pattern is fine too).
- The Kaggle CLI 2.2.x on that machine: `pip install -U kaggle` (1.6 fails with 401/403), then `kaggle kernels list --user <them>` to confirm auth.
- A clone of this repo (`git clone <private repo>`), Python 3.12 with `nbformat` (only needed to regenerate notebooks).

## 1. Give the account the data (choose one)

**A. Share the private datasets (fastest, no upload).** From `harshdipsaha`'s browser, for each dataset → *Settings* → *Collaborators* → add the teammate's username:
- `dat-parkinsons-cache-v4` (registered 2 mm cache + ImageNet weights + LR arm) — *Can view*
- `dat-parkinsons-challenge-data` (raw competition data) — *Can view*
- `dat-run3-checkpoints` (all checkpoints, OOF files, landmark template, feature arm) — **Can edit** (the notebooks upload new checkpoints into it every 4 jobs; an editor collaborator can push versions with `kaggle datasets version`)
Shared datasets keep their `harshdipsaha/...` slugs, so `kernel-metadata.json` needs no dataset changes.

**B. Re-upload under the teammate's account** (if sharing is not wanted). From this repo on the teammate's machine with their token:
```
# raw competition data: zip data/niftis + data/train_labels_*.csv into a folder with dataset-metadata.json (id <them>/dat-parkinsons-challenge-data)
# cache + assets: a folder with cache_v4/*.npy + meta/folds/sites csv, pretrained/*.pt, artifacts/template.npz, artifacts_v3/lr_v3.json + oof_lrarm*.csv + lrarm.json
#                 (staging zips were deleted on Sept 7 to free disk; rebuild them from those folders)         -> id <them>/dat-parkinsons-cache-v4
# checkpoints: data/ckpt_ds_v4/artifacts_v3/ (the 337-file local copy) + artifacts/template.npz + artifacts/featarm_v4/*  -> id <them>/dat-run3-checkpoints
#              (copy them into one flat folder with a dataset-metadata.json; run the command from INSIDE the folder on Windows)
cd <folder> && kaggle datasets create -p . --dir-mode skip        # first time (private by default)
cd <folder> && kaggle datasets version -p . -m "sync" --dir-mode skip   # later updates
```
Then edit `dataset_sources` in each `kernel-metadata.json` to the `<them>/...` slugs, and set `CKPT_DS`/`CACHE_DS` in the generator (`kaggle/gen_run3_inject.py` and the run-6 generators) to the new slugs and regenerate.

## 2. Push a notebook (single command per notebook)

In each notebook folder, edit the `id` owner in `kernel-metadata.json` to the teammate's username (e.g. `"id": "<them>/dat-run6a-crop3d"`), then from the repo root:
```
kaggle kernels push -p kaggle/run6a_crop3d
kaggle kernels push -p kaggle/run6b_triplanar
kaggle kernels push -p kaggle/run6c_features_ssl
```
Each push creates (or updates) the private notebook under the teammate's account **and starts a run** ("Save & Run All") with GPU + Internet enabled from the metadata. Three notebooks × ≈ 5 h each fit the account's 30 GPU-hours/week; Kaggle runs at most one GPU session per account at a time, so push the second when the first has finished (`kaggle kernels status <them>/dat-run6a-crop3d`).

## 3. One-time settings on the Kaggle site (after the first push)

Open each notebook → *Edit*: Session options → Accelerator **GPU T4 x2** (the metadata only says "GPU"; if it landed on P100 or a single T4, change it and push again), Internet **On**; Add-ons → **Secrets**: `KAGGLE_USERNAME` = the teammate's username, `KAGGLE_KEY` = their API key (these let the runner upload checkpoints every 4 jobs; without them the run still completes but resumability is lost). Do not run cells interactively (that starts a draft GPU session and burns quota); `kaggle kernels push` again to re-run.

## 4. Watch, fetch, assemble, submit

```
kaggle kernels status <them>/dat-run6a-crop3d                       # RUNNING -> COMPLETE (fail-safe tail cells keep the version even if a tail step fails)
python scripts/watch_kaggle_output.py --kernel <them>/dat-run6a-crop3d --out data/run6a   # downloads + verifies submission_v3.zip (needs the teammate's token locally)
python scripts/run4_assemble.py --src data/run6a --extra-dir artifacts/featarm_v4 --smoke <smoke_dir> --zip data/submission_run6a.zip
```
If our token is the one on the local machine, the teammate can instead share the notebook with `harshdipsaha` (notebook → Share → add user), after which `kaggle kernels output <them>/dat-run6a-crop3d` works with our token too. Submissions go only through **our** DrivenData account (one team account); the zip from any notebook is submitted the same way as before (`kaggle/gen_submit.py` + the CDP upload, or by hand). Every zip gets a smoke test first.

## 5. Quota and rules

- GPU quota is per Kaggle account (30 h/week, resets weekly); the teammate's quota is independent of ours.
- The datasets stay private and are deleted after the competition (data-hygiene rule); no raw scan is written to notebook outputs.
- Only one team may submit; the teammate's account is a compute venue, not a second competition entry.
