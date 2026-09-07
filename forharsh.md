# forharsh — do everything yourself until Sept 16 (simple, in order)

You are at **public 0.2667, rank 39**. Leader 0.2192. Three real submissions left: **Sept 12 15:38 UTC, Sept 13 15:11 UTC, Sept 13 ≈ 23:30 UTC**. Deadline **Sept 16 23:59 UTC**. Smoke tests are free (3 per day). Everything below is copy-paste; run commands from `H:\augsepthacks\parkinsonnn` in PowerShell or Git Bash.

## 0. One-time checks (5 min)

```
git pull                                   # latest code
python -m pytest tests -q                  # should print "66 passed" (or more)
kaggle kernels status harshdipsaha/dat-parkinsons-train      # your Kaggle CLI works (RUNNING / COMPLETE)
```
Keep the laptop plugged in and awake (sleep is already disabled). Kaggle secrets `KAGGLE_USERNAME` / `KAGGLE_KEY` are set on the notebook; never paste the key anywhere else.

Where things are: all checkpoints = private Kaggle dataset `dat-run3-checkpoints` (also local copy `data/ckpt_ds_v4/artifacts_v3`); the 20 smoke scans with labels = `data/smoke_test_data_2gePzfM.tar.gz` (extract once to a folder, call it `<smoke_dir>`); the submitted zips = `data/submission_v4a_local.zip` (run 4a, 0.2667) and `data/submission_v3_fixed.zip` (run 3, 0.2817).

## 1. Today (Sept 7): finish run 5, then launch run 7

Run 5 (Kaggle Version 4, started 03:53 UTC, ≈ 3 h) is training the site-invariant and histogram-standardised crop-3D nets.

1. **Wait for COMPLETE**: `kaggle kernels status harshdipsaha/dat-parkinsons-train` every 30 min (or the watcher task `DaT-KaggleOutputWatch` downloads it by itself into `data/kaggle_output_v5`; check `type data\kaggle_output_v5\watch.log`).
2. **If the watcher did not download**: `python scripts/watch_kaggle_output.py --out data/kaggle_output_v5`
3. **If Kaggle shows RUNNING for more than 5 h**: something is stuck. Kaggle editor → View Active Events → ⋯ → Stop Session. Nothing is lost: everything trained is already in the checkpoint dataset. Then: `kaggle datasets download harshdipsaha/dat-run3-checkpoints -p data/ckpt_ds_v5 --unzip --force`, move all files into `data/ckpt_ds_v5/artifacts_v3/`, and use that folder as `--src` below.
4. **Assemble + verify** (the one command that stacks, gates, packs and checks):
   ```
   python scripts/run4_assemble.py --src data/kaggle_output_v5 --extra-dir artifacts/featcarm_v5 --smoke <smoke_dir> --zip data/submission_run5.zip
   ```
   Read three lines of its output: `nested combiner estimate` (was **0.2491** for run 4a; lower is better), `final weights` (which arms survived), and `FORMAT OK ... smoke-test log loss` (must print, all arms N/N).
5. **Read out run 5** (write the numbers into `RESULTS_LOG.md`): in the printed per-arm table compare `r3d_cropinv` (site-invariant) with `r3d_crop48x48x32` on the random folds, and `r3d_cropinv_*_site` vs `r3d_crop48x48x32_*_site` on the site folds; `r3d_crophist` vs `r3d_crop48x48x32`. Rule: a mechanism is "in" if it is ≥ 0.005 better on the site folds or the nested estimate improves ≥ 0.003.
6. **Launch run 7** (hemisphere members + super-stack, ≈ 3 h) as soon as Version 4 is finished:
   ```
   python kaggle/run7_hemisphere/gen_run7.py
   kaggle kernels push -p kaggle/run7_hemisphere
   ```
   Then on kaggle.com open notebook `dat-run7-hemisphere` → Edit → Session options: GPU **T4 x2**, Internet **On**; Add-ons → Secrets: add `KAGGLE_USERNAME` and `KAGGLE_KEY` (same values as on the other notebook). If the run started on the wrong GPU, fix the setting and push again. Watch with `kaggle kernels status harshdipsaha/dat-run7-hemisphere`; fetch with `python scripts/watch_kaggle_output.py --kernel harshdipsaha/dat-run7-hemisphere --out data/run7`; assemble with the same `run4_assemble.py` line using `--src data/run7 --zip data/submission_run7.zip`.

## 2. Teammate's account (any time from today): notebooks 6a, 6b, 6c

Guide: `docs/how-to-run-on-a-second-kaggle-account.md`. Short version:
1. Share the three private datasets with the teammate: on kaggle.com each dataset → Settings → Collaborators → add their username (`dat-run3-checkpoints` with **Can edit**, the other two Can view).
2. Teammate: API token in `~/.kaggle/kaggle.json`, `pip install -U kaggle`, clone the repo.
3. In each folder change the owner in `kernel-metadata.json` (`"id": "<teammate>/dat-run6a-crop3d"` etc.), then from the repo root:
   ```
   kaggle kernels push -p kaggle/run6b_triplanar        # ≈ 4 h
   kaggle kernels push -p kaggle/run6c_features_ssl     # ≈ 2.5 h   (one at a time: Kaggle runs one GPU notebook per account)
   kaggle kernels push -p kaggle/run6a_crop3d           # ≈ 4.7 h   (only after run 5 has finished on your account)
   ```
   Set GPU T4 x2 + Internet + their two secrets once on each notebook (same as step 1.6).
4. Fetch each output (`python scripts/watch_kaggle_output.py --kernel <teammate>/<slug> --out data/run6x`, with the teammate's token, or they share the notebook with you). Every finished checkpoint also lands in `dat-run3-checkpoints`, so the run-7 super-stack picks them up automatically.

## 3. Sept 8–11: build the final stack

Each day: fetch whatever finished, then one assembly over **everything**:
```
mkdir data\all && copy data\kaggle_output_v5\artifacts_v3\* data\all\artifacts_v3\   (and the same from data\run7, data\run6a, data\run6b, data\run6c)
python scripts/run4_assemble.py --src data/all --extra-dir artifacts/featcarm_v5 --smoke <smoke_dir> --zip data/submission_final.zip
```
(`run4_assemble.py` stacks every arm that has OOF files, drops zero-weight arms, packs, and runs the container-style check.) Keep the zip with the **lowest nested estimate**. Rules: ≤ 10 arms, zip ≤ 850 MB (DrivenData accepted 844 MB; if bigger, remove the lowest-weight arm's checkpoints from the folder and re-run), local check must say every arm N/N.

Smoke test the best zip once (free): DrivenData → Make new submission → file → note → **Smoke test** → Submit (upload takes ~1 min at 800 MB now). A sane smoke score is below 0.30; run 4a's was 0.2176/0.2263. If the site refuses the upload, use a smaller zip.

## 4. The three submissions

| When (UTC) | What to upload | Note to write |
|---|---|---|
| Sept 12 15:38 | best `submission_final.zip` (lowest nested estimate, smoke-tested) | "super-stack runs 3-6/7, nested X" |
| Sept 13 15:11 | run-7 stack if its nested estimate is better than slot 1's; otherwise the same zip with the alternative prior mix (see below) | |
| Sept 13 ≈ 23:30 | the bracket variant: same best stack, `--lam 0.01` instead of 0.02 (best private score counts automatically) | "bracket lam 0.01" |

Bracket variant: re-run the assembly with `--lam 0.01` added to the `fit_stack_v3.py` command inside `run4_assemble.py` (edit the string `--lam 0.02` → `--lam 0.01` in `scripts/run4_assemble.py`, run, then change it back).

Upload: DrivenData submissions page → Make new submission → choose the zip → note → **Normal submission** → Submit. The score appears after ~1 h. Write it into `RESULTS_LOG.md`.

## 5. What "good" looks like

- Nested estimate: 0.2491 (run 4a) → hope for 0.235–0.245 with runs 5–7 and 6a/6b/6c.
- Public ≈ nested + 0.018 (that has been the gap twice). 0.25 public ≈ rank 12; 0.24 ≈ top 10.
- Smoke-to-public so far: 0.2567→0.3391, 0.2397→0.2817, 0.2176→0.2667. A smoke score under 0.20 would be new territory.

## 6. If something breaks

- **Kaggle notebook stuck "RUNNING" with nothing happening** → Stop Session (View Active Events → ⋯); outputs are in the checkpoint dataset; assemble from a fresh download (step 1.3).
- **A training job crashed with NaN** → ignore it; the assembler drops arms with incomplete folds automatically.
- **Zip too big / upload refused** → drop the weakest arm's `.pt` files from the folder and re-assemble; keep ≤ 850 MB.
- **Watcher not downloading** → laptop asleep or the task removed itself; run `python scripts/watch_kaggle_output.py --kernel <owner>/<slug> --out <dir>` by hand.
- **Playwright/automation** → not needed; everything above is CLI + website.
- **Quota** → your Kaggle GPU week resets weekly (the editor shows "Quota"); the teammate's account is independent.

## 7. After the deadline (Sept 17)

Delete the private Kaggle datasets (`dat-parkinsons-challenge-data`, `dat-parkinsons-cache-v4`, `dat-run3-checkpoints`) and the notebook version outputs (data-hygiene promise); write `docs/DISCLOSURE.md` (ImageNet torchvision weights BSD-3, Kaggle as venue, no external data, no pseudo-labelling) for the organisers' open-source release; the code stays MIT.

## 8. Where the thinking is

`docs/wayfinder/map.md` (the route + open tickets; `python docs/wayfinder/frontier.py`), `aidlc-docs/HANDOFF.md` (full handoff), `RESULTS_LOG.md` (every number), `docs/superpowers/plans/` and `docs/superpowers/specs/` (plans and specs), each notebook folder's `APPROACH.md` + `HOWTO.pdf`.
