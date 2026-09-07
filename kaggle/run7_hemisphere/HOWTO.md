# HOWTO — run 7 (hemisphere members + super-stack) in one command

**From the repo root (Kaggle CLI 2.2.x authenticated):**

```
python kaggle/run7_hemisphere/gen_run7.py            # regenerate notebook.ipynb + kernel-metadata.json from the current repo
kaggle kernels push -p kaggle/run7_hemisphere        # creates/updates the private notebook dat-run7-hemisphere and runs it
```

**One-time settings on the Kaggle site** (after the first push, open the notebook → Edit): Accelerator **GPU T4 x2** (the metadata only says "GPU"), Internet **On**, Add-ons → Secrets `KAGGLE_USERNAME` and `KAGGLE_KEY` (the runner uploads finished checkpoints to `dat-run3-checkpoints` every 4 jobs; without them the run still completes). Do not run cells interactively; push again to re-run. Inputs the metadata attaches: `dat-parkinsons-challenge-data`, `dat-parkinsons-cache-v4`, `dat-run3-checkpoints` (the latest version must contain `template.npz` with `hist` landmarks and the feature-arm files).

**What it does (≈ 3 h):** writes the package, mounts the cache, resumes every checkpoint already in the checkpoint dataset (queues of earlier runs print "0 jobs to run"), trains the hemisphere members (3 seeds, site folds, DANN/DRO variant), the bootstrapped-loss crop-3D probe and 2 full-data models, re-scores every checkpoint with TTA, ranks every arm family of runs 3–7 by OOF log loss, fits the nested combiner over the 10 best, packs `submission_v3.zip`, runs the container-style check with per-arm smoke-set scores, and splits the zip into < 95 MB parts.

**Watch / fetch / assemble / submit:**

```
kaggle kernels status <owner>/dat-run7-hemisphere
python scripts/watch_kaggle_output.py --kernel <owner>/dat-run7-hemisphere --out data/run7      # downloads + verifies the output
python scripts/run4_assemble.py --src data/run7 --extra-dir artifacts/featcarm_v5 --smoke <smoke_dir> --zip data/submission_run7.zip
python kaggle/gen_submit.py <abs path to zip> "<note>" smoke   # then .playwright-mcp/dd_submit.js, or upload by hand (smoke test first)
```

**Second Kaggle account:** `python kaggle/run7_hemisphere/gen_run7.py --owner <teammate>` (add `--cache-ds/--raw-ds/--ckpt-ds <teammate>/...` only if the datasets were re-uploaded instead of shared with edit rights on the checkpoint dataset), then the same push. Do not run it concurrently with another notebook that writes the same tags into the checkpoint dataset. Full guide: `docs/how-to-run-on-a-second-kaggle-account.md`.

**Gates to read after the run** (`RESULTS_LOG.md` conventions): hemisphere member vs crop-3D on random and site folds; bootstrapped loss vs plain (keep if nested ≥ 0.003 better); the super-stack's nested estimate vs run 4a's 0.2491 and its per-site table.
