# Run 6a — striatal-crop 3D maximiser: how to run it on Kaggle (one command)

## 1. One-time setup on the Kaggle account that will run it
- API token: kaggle.com → Settings → API → Create New Token → put `kaggle.json` in `~/.kaggle/` (Windows: `C:\Users\<you>\.kaggle\`). `pip install -U kaggle` (needs ≥ 2.x).
- Datasets the kernel reads: `dat-parkinsons-challenge-data` (raw NIfTIs + smoke set), `dat-parkinsons-cache-v4` (2 mm registered cache + ImageNet weights), `dat-run3-checkpoints` (every checkpoint, OOF csv, template, feature arm). A second account needs them **shared** (each dataset → Settings → Collaborators → add the account; give **edit** rights on `dat-run3-checkpoints` so the kernel can upload its checkpoints) or **re-uploaded** under the new account (`data/upload_v4/*.zip`, `data/ckpt_upload_v5/`; then regenerate with `--cache-ds/--raw-ds/--ckpt-ds <account>/<slug>`).
- After the first push, open the kernel on the website once: Settings → Accelerator **GPU T4 x2**, Internet **on**, Add-ons → Secrets: `KAGGLE_USERNAME`, `KAGGLE_KEY` (the token of the running account; they drive the per-4-jobs checkpoint upload). Save & Run All if the CLI push did not already run with the GPU.

## 2. Run it (from the repo root)
```
python kaggle/run6a_crop3d/gen_run6a.py              # regenerates notebook.ipynb + kernel-metadata.json from the current repo code
kaggle kernels push -p kaggle/run6a_crop3d           # creates/updates the kernel AND starts a "Save & Run All" commit
```
Second account: `python kaggle/run6a_crop3d/gen_run6a.py --owner <account>` (plus the `--*-ds` flags above only if the datasets were re-uploaded), then the same push. Budget: 27 training jobs ≈ 4.7 h on two T4s (≈ 9.5 h on one) + 35 min of tail cells; the kernel resumes any checkpoint already in the checkpoint dataset, so a re-run after a crash only trains what is missing.

## 3. Watch, fetch, assemble, submit
```
kaggle kernels status <owner>/dat-run6a-crop3d                                  # RUNNING -> COMPLETE (log: website → Logs)
python scripts/watch_kaggle_output.py --kernel <owner>/dat-run6a-crop3d --out data/run6a        # downloads + verifies the output (idempotent)
python scripts/run4_assemble.py --src data/run6a --extra-dir artifacts/featarm_v4 --smoke <smoke_dir> --zip data/submission_run6a.zip
python kaggle/gen_submit.py <abs path>\data\submission_run6a.zip "run6a ..." smoke     # then normal; or upload by hand on DrivenData
```
The notebook already prints the nested stack, the per-arm **smoke-set log loss** (20 labelled smoke scans; an out-of-distribution check, never a fitting target — ladder so far: run 2 0.2567→0.3391 public, run 3 0.2397→0.2817, run 4a 0.2176→0.2667) and the container-style end-to-end check. Keep the zip under ~850 MB (all 3D nets together are < 60 MB; V2-S checkpoints dominate).

## 4. What the run trains (tags in the checkpoint dataset)
`r3d_crop64e60_f*_s0` (64×64×40 context crop, 60 ep) · `r3d_se48_f*_s0` (SE + stochastic depth, 32–256 widths, 60 ep) · `r3d_crop48e80_f*_s{1,2}` (run-4a net, 80 epochs, 2 seeds) · `r3d_cropinv_f*_s0` (DANN 0.1 + group-DRO 0.1; identical to run 5, reused if present) · `r3d_crop48e80_fall_s{0,1}` (full-data). Stack arms (≤ 10, nested, grid 8): the five 3D arms + V2-S, ConvNeXt, coronal B0, feature arm, LR; arms at weight 0 are dropped by `run4_assemble.py`.

## 5. If something breaks
Stuck for > 1 h after "all done": website → View Active Events → ⋯ → Stop Session; everything trained is already in the checkpoint dataset, so `python scripts/run4_assemble.py --src <download of the dataset with an artifacts_v3/ subfolder> ...` still produces the zip. `python -m pytest tests -q` must be green before any regeneration.
