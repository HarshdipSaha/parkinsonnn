# HOWTO — run 8 (ensemble-corrected members + Master v3 gate) in one command

**From the repo root, with the teammate's Kaggle CLI credentials active** (`$env:KAGGLE_CONFIG_DIR` pointing at a directory holding *their* `kaggle.json` — never the repo, never committed, never pasted in chat):

```
python kaggle/run8_distill/gen_run8.py                 # regenerate notebook.ipynb + kernel-metadata.json (owner defaults to skyfall23)
kaggle kernels push -p kaggle/run8_distill             # creates/updates the private notebook dat-run8-distill and runs it
```

**One-time settings on the Kaggle site** (after the first push, open the notebook → Edit): Accelerator **GPU T4 x2** (the metadata only says "GPU"), Internet **On**, Add-ons → Secrets `KAGGLE_USERNAME` and `KAGGLE_KEY` (paste from a password manager or clipboard, never typed where it could be logged — the runner uploads finished checkpoints to `dat-run3-checkpoints` every 4 jobs; without the secrets the run still completes but resumability is lost). Do not run cells interactively; push again to re-run.

**Inputs the metadata attaches:** `dat-parkinsons-challenge-data`, `dat-parkinsons-cache-v4`, `dat-run3-checkpoints` — all three must already be shared with this account with *Can view* (the first two) / *Can edit* (the checkpoint dataset), per `docs/how-to-run-on-a-second-kaggle-account.md`. The checkpoint dataset already holds every run 3–7 checkpoint, so run 8 resumes cleanly and its own new checkpoints land there too.

**What it does (≈ 4.5–5 h):** writes the package, mounts the cache, writes the embedded soft-target table to `cache_v4/soft_master_v2.csv`, resumes every checkpoint already present (a finished job prints "0 jobs to run"), trains 42 jobs (soft-target and mistake-weighted hemisphere members, two extra plain hemisphere seeds, a soft-target crop-hist probe, two full-data soft hemisphere models), re-scores every checkpoint with TTA, ranks every arm family of runs 3–8 by out-of-fold log loss, fits the nested combiner over the ten best, **prints the gate table** (pillars alone vs pillars + each new arm), packs `submission_v3.zip`, runs the container-style check with per-arm smoke-set scores, and splits the zip into < 95 MB parts.

**Watch / fetch / assemble / decide:**

```
kaggle kernels status skyfall23/dat-run8-distill
python scripts/watch_kaggle_output.py --kernel skyfall23/dat-run8-distill --out data/run8     # downloads + verifies the output
python scripts/build_master_v3.py --src data/run8/artifacts_v3 --src data/master_pool/artifacts_v3 --smoke <smoke_dir> --zip data/submission_master_v3.zip
```
Read the printed **GATE TABLE**: if the best set beats Master v2 by ≥ 0.003 nested, `submission_master_v3.zip` is the new slot-1 candidate (smoke-test it on DrivenData and compare to Master v2's 0.2381 — replace only if the platform score is within 0.01). Otherwise keep Master v2 for slot 1 and use v3 for slot 2 if it beats v2 at all.

**Second bracket session (optional, ≈ 10 GPU-h left):** `python kaggle/run8_distill/gen_run8.py --soft-lam 0.3` or `--soft-lam 0.5` (bracketing λ), or increase the seed counts in `JOBS8` inside `gen_run8.py`, then push again — the run resumes and only trains the new tags.

**Gates to read after the run** (`RESULTS_LOG.md` conventions): each new arm's nested gain over the pillars, its own OOF log loss and AUROC, its temperature on the site-fold pool if run, and the smoke-to-public gap once submitted. Full guide for the teammate's account: `docs/how-to-run-on-a-second-kaggle-account.md`.
