# Run 6c (approach C) — how to run it on Kaggle with one command

**What it is.** One Kaggle notebook (`notebook.ipynb`, generated from the repo by `gen_run6c.py`) that adds two members that are
maximally different from the ImageNet 2.5D nets, then stacks them with the run-4a survivors: (1) `featc` — 116 radiomic/shape
descriptors of the registered striatum + striatal-patch PCA → kernel (random-Fourier) logistic / MLP, numpy inference;
(2) `a3d_ssl` — the striatal-crop 3D net warm-started from a VICReg encoder pretrained on the *training* scans (labels unused).
Budget on Kaggle GPU T4 x2: ≈ 2–2.5 h (15 min CPU features, ≈ 30 min SSL pretraining, ≈ 60 min fine-tuning, ≈ 20 min tail).

## 1. One-time setup (any Kaggle account)

1. `pip install -U kaggle` (≥ 2.2) and put the account's API token at `~/.kaggle/kaggle.json` (Kaggle → Settings → API → Create token).
2. Make sure the three inputs are visible to that account: `dat-parkinsons-cache-v4`, `dat-parkinsons-challenge-data`, `dat-run3-checkpoints`
   (owner `harshdipsaha`). On a second account: either be added as a **collaborator** on those three private datasets (dataset → Settings →
   Collaborators; *edit* rights on `dat-run3-checkpoints` so the notebook can upload its checkpoints), or re-upload copies from
   `data/upload_v4/*.zip` and `data/ckpt_upload_v5/` and change the three ids in `kernel-metadata.json`.
3. Secrets (once, in the Kaggle editor after the first push: Add-ons → Secrets): `KAGGLE_USERNAME` and `KAGGLE_KEY` of the account
   that owns/edits `dat-run3-checkpoints`. Without them the run still works but checkpoints are only saved in the version output.

## 2. Run (single command, from the repo root)

```
python kaggle/run6c_features_ssl/gen_run6c.py            # optional: regenerate the notebook from the current code
kaggle kernels push -p kaggle/run6c_features_ssl          # uploads notebook + metadata and starts "Save & Run All" on GPU T4 x2
```

`kernel-metadata.json` already sets GPU, internet, private, and the three dataset inputs. **Second account:** change `"id"` to
`<your-username>/dat-run6c-features-ssl` in `kernel-metadata.json` (and the dataset ids if you re-uploaded), nothing else.
If Kaggle assigns a single GPU, the runner adapts (`GPUS = list(range(torch.cuda.device_count()))`).

## 3. Watch, fetch, assemble, verify

```
kaggle kernels status <owner>/dat-run6c-features-ssl                                   # RUNNING → COMPLETE
python scripts/watch_kaggle_output.py --kernel <owner>/dat-run6c-features-ssl --out data/run6c    # or the Task-Scheduler watcher
python scripts/run4_assemble.py --src data/run6c --extra-dir artifacts/featarm_v4 --smoke <smoke_dir> --zip data/submission_run6c.zip
```
The notebook's own tail already prints: per-arm OOF (random + leave-site-out), the nested stack estimate with weights, the container
check on the 20 smoke scans **per arm and for the stack**, and the 3-hour extrapolation. The smoke ladder to read it against:
run 2 0.2567 → 0.3391, run 3 0.2397 → 0.2817, run 4a 0.2176/0.2263 → 0.2667 (public ≈ smoke + 0.04–0.05).
Ship the zip only after `scripts/local_e2e_v3.py` shows every arm at N/N; keep zips < 850 MB.

## 4. Where things are

- Code: `datsc/features_c.py`, `datsc/featmodel_c.py`, `datsc/ssl_c.py`, `scripts/feat_arm_c.py`, `scripts/ssl_pretrain_c.py`,
  `scripts/train_v3.py --init`, tests in `tests/test_run6c.py`. Design and evidence: `APPROACH.md`.
- Outputs of a run (version output / checkpoint dataset): `artifacts_v3/featc_v5.npz`, `oof_featcarm*.csv`, `ssl_c_encoder.pt`,
  `r3d_ssl_f*_s*.pt`, `manifest.json`, `submission_v3.zip` (+ `parts/`), `out/smoke_scores.json`.
