# Reference: run-3 pipeline

Facts only; for the rationale see [concept-registered-frame.md](concept-registered-frame.md), for procedures see [how-to-train-and-submit.md](how-to-train-and-submit.md).

## Data flow

```
NIfTI --datsc/io.py (RAS, 2 mm)--> datsc/volume.prepare_volume
      --> head mask, p75-normalised 96x96x72 box centred on the head centre of mass
      --> datsc/register.register_and_normalise (when a template is given)
          similarity + affine registration to artifacts/template.npz, NCC QC (CoM fallback),
          reference-region normalisation -> "ref" units (SBR + 1)
      --> InputSpec (datsc/train_v3.py): 3D resize | 2.5D slabs (axial or coronal) | variants for TTA
      --> models (datsc/models_v3.py) | ROI features + LR (datsc/features_v3.py, datsc/lrmodel.py)
      --> combiner (datsc/infer_v3.combine): weighted logit average, temperature T, prior mix lambda
```

## Modules

| Module | Public functions / classes |
|---|---|
| `datsc/volume.py` | `prepare_volume(path, template=None) -> {vol fp16, mask, scalars[8], stats}`; `renormalise(vol, stats, mode)`; constants `VOL_SPACING_MM = 2.0`, `VOL_SHAPE = (96, 96, 72)`, `STAT_NAMES` (16) |
| `datsc/register.py` | `register(moving, template) -> (registered, params)`; `register_and_normalise(box, Template) -> (vol, reg_stats[8], params)`; `Template.load/save`; `derive_masks`; `symmetrize`; `ncc`; `REG_STAT_NAMES` |
| `datsc/data_v3.py` | `VolumeCache(root)`, `VolDataset(cache, uids, labels, norm)`, `to_3d_input(x, size)`, `to_slabs(x, n_slabs, slab_mm, hw, mode, z_center, plane)` |
| `datsc/models_v3.py` | `ResNet3D(n_scalars=0, widths, dropout, stem_stride)` (GroupNorm/SiLU); `Net2D(backbone, weights_path, n_scalars=0, n_in, dropout, in_scale=8)` (torchvision backbones: convnext_*, efficientnet_*, resnet*); `build_model(arm, **kw)` |
| `datsc/train_v3.py` | `InputSpec(arm, size, n_slabs, slab_mm, hw, slab_mode, z_center, plane, tta_z)`; `InputSpec.variants(x)`; `predict(model, loader, spec, device, tta_flip)`; `EMA(model, decay)` (step-adaptive); `train_run(...)` |
| `datsc/features_v3.py` | `striatal_features(vol, template) -> [17]`; `FEATURE_NAMES` |
| `datsc/infer_v3.py` | `preprocess_all(uids, nifti_dir, n_workers, deadline_s, t0, template_path)`; `run_arm(arm_cfg, artifacts_dir, vols, sc, st, ok, t0)`; `run_lr_arm`; `combine(arm_probs, weights, T, lam, pi)`; `LADDER_STOP_S = 150*60` |
| `main_v3.py` | Entrypoint. `--selftest` runs synthetic volumes through `prepare_volume` (with the template if present) |

## Cache layout (`cache_v3/`, `cache_v4/`)

| File | Shape / content |
|---|---|
| `vols.npy` | float16 `[N, 96, 96, 72]`; `cache_v3` = p75 units (CoM frame), `cache_v4` = ref units (registered frame) |
| `masks.npy` | uint8 head mask, same shape (`cache_v3` only) |
| `scalars.npy` | float32 `[N, 8]`, the run-2 scalar contract (not used by run-3 models) |
| `stats.npy` | float32 `[N, 16]` (`cache_v3`) or `[N, 24]` (`cache_v4` = 16 + `REG_STAT_NAMES`) |
| `meta.csv` | `uid, ok, err, <stats columns>, is_pathologic` (+ registration columns in `cache_v4`); row order = array order |
| `folds.csv` | canonical 5 random stratified folds (copied from the Kaggle run-2 output) |
| `folds_site.csv` | leave-pseudo-site-out folds |
| `sites.csv` | `uid, site, site_id`; site = spacing x native shape x intensity regime (< 25 scans merged into `other`) |

`STAT_NAMES`: `p75, p50, p99, mean, std, mask_ml, com_x, com_y, com_z, sp_x, sp_y, sp_z, shape_x, shape_y, shape_z, n_levels_native`. `REG_STAT_NAMES`: `ncc_init, ncc_reg, reg_ok, reg_scale, reg_rot_deg, reg_trans_mm, ref_mean, bg_level`.

## Template (`artifacts/template.npz`)

Keys: `box` (float16, 96x96x72, p75 units, L/R symmetric), `brain_mask`, `striatal_mask` (mirror-symmetric), `ref_mask` (brain eroded 8 mm minus striatum dilated 20 mm), `striatal_center` (voxels; x = 47.5, y ≈ 57.3, z ≈ 33.5), `ncc_thresh` (QC threshold), `spacing`.

## Training artefacts (`artifacts_v3/`)

| File | Content |
|---|---|
| `<tag>.pt` | fp16 state dict of the final EMA model |
| `<tag>.json` | `tag, fold, seed, epochs, select, best_val_logloss, val_logloss, val_auroc, spec, model_kw, norm, folds_csv, mixup, ema_decay, lr, backbone_lr, weight_decay, history[], params, sec` |
| `oof_<tag>.csv` | `uid, y, p` for the validation fold (with the spec's TTA) |
| `lr_v3.json`, `oof_lrarm.csv`, `oof_lrarm_site.csv`, `lrarm.json` | ROI-LR arm |
| `manifest.json` | see below |

Tags: `r3d_<size>_f<fold>_s<seed>`, `r2d_<backbone>[_cor]_f<fold>_s<seed>`, suffix `_site` for site folds, `fall` for full-data models (fold -1).

## `manifest.json`

```json
{"pi": 0.548, "T": 1.03, "T_random": 0.97, "T_site": 1.03, "lam": 0.02,
 "weights": {"a3d": 0.4, "a2d_b0": 0.4, "a2d_v2s": 0.1, "lr": 0.1},
 "arms": [{"name": "a3d", "arm": "3d", "spec": {InputSpec dict}, "model_kw": {...}, "norm": "ref",
           "checkpoints": ["artifacts/r3d_64x64x48_f0_s0.pt", "..."]},
          {"name": "lr", "arm": "lr", "spec": {"arm": "lr"}, "model_kw": {}, "norm": "ref",
           "checkpoints": ["artifacts/lr_v3.json"]}],
 "oof_logloss": 0.25, "oof_auroc": 0.96, "n": 1362}
```

Combination: `z = sum_a w_a * logit(p_a)` over arms that produced predictions (weights renormalised), `p = sigmoid(z / T)`, `p = (1 - lam) * p + lam * pi`, clipped to `[1e-4, 1 - 1e-4]`. Missing volumes get `pi`.

## Runtime environment variables (`main_v3.py`)

| Variable | Default | Meaning |
|---|---|---|
| `DATSC_DATA_ROOT` | `/code_execution/data` | contains `niftis/` and `submission_format.csv` |
| `DATSC_OUTPUT_CSV` | `submission.csv` | output path |
| `DATSC_ARTIFACTS` | `<zip root>/artifacts` | manifest, template, checkpoints |
| `OMP_NUM_THREADS` etc. | 4 (parent), 1 (workers) | thread pinning |

## Scripts and flags

| Script | Flags (defaults) |
|---|---|
| `scripts/cache_volumes.py` | `--out cache_v3`, `--workers <cpus-2>`, `--limit` |
| `scripts/build_template.py` | `--cache cache_v3`, `--out artifacts/template.npz`, `--n-iter 2`, `--subset 400`, `--workers`, `--seed 0` |
| `scripts/register_cache.py` | `--src cache_v3`, `--out cache_v4`, `--template artifacts/template.npz`, `--workers` |
| `scripts/pseudo_sites.py` | `--cache`, `--run2-folds cache/folds.csv`, `--min-site 25`, `--n-site-folds 5` |
| `scripts/gate_v3.py` | `--cache cache_v4`, `--template`, `--out artifacts_v3`, `--save`, `--C 0.3` |
| `scripts/train_v3.py` | `--arm {3d,2d}`, `--fold` (-1 = all data), `--seed 0`, `--cache cache_v4`, `--out artifacts_v3`, `--folds-csv folds.csv`, `--tag`, `--epochs 60`, `--batch-size 16`, `--lr 5e-4`, `--backbone-lr`, `--weight-decay 1e-2`, `--warmup 3`, `--mixup 0`, `--ema 0.995`, `--select {final,best}`, `--norm {ref,p75,median,zscore,p99}`, `--n-scalars 0`, `--no-aug`, `--workers 2`; 3D: `--size 96 96 72`, `--widths 24 48 96 192`, `--stem-stride 1`; 2D: `--backbone efficientnet_b0`, `--weights`, `--n-slabs 3`, `--slab-mm 8`, `--slab-mode {mean,max}`, `--hw 192`, `--z-center`, `--plane {axial,coronal}`, `--tta-z 0` |
| `scripts/reoof_v3.py` | `--art artifacts_v3`, `--cache cache_v4`, `--tta-z -1 0 1`, `--only "*"`, `--batch-size 32` |
| `scripts/fit_stack_v3.py` | `--out artifacts_v3`, `--arm NAME PATTERN` (repeat), `--site-arm NAME PATTERN`, `--extra NAME GLOB`, `--lam 0.02`, `--pi 0.548`, `--t-floor 1.0`, `--grid 10` |
| `scripts/pack_v3.py` | `--artifacts artifacts_v3`, `--out submission_v3.zip` |
| `scripts/join_parts.py` | positional: `parts_dir md5 [--out zip]` |
| `scripts/fetch_pretrained.py` | positional: output dir (downloads torchvision ImageNet / Kinetics weights) |

## Kaggle helpers (`kaggle/`)

| File | Purpose |
|---|---|
| `gen_run3_inject.py` | Defines `JOBS`, `JOBS2`, `ARMS`, `EXTRAS`; writes `.playwright-mcp/inject_run3.js` (wipes and rebuilds the notebook) and `update_runner.js` |
| `push_cells.py` | Writes `.playwright-mcp/push_cells.js`: sync cells by first line, insert missing ones, run changed `%%writefile` cells; never runs the job runners |
| `push_update.py <cleanup-cmd>` | Update trainer + runner cells, run a cleanup command, relaunch the runner (interrupts nothing by itself) |
| `split_zip_cell.py` | Notebook cell: split `submission_v3.zip` into < 95 MB parts with an md5 |
| `e2e_check_v3.py` | Notebook cell: run the packed zip like the container (smoke scans + 150-volume timing) |
| `gen_submit.py <zip> "<note>" smoke\|normal` | Writes `.playwright-mcp/dd_submit.js` for the DrivenData submissions page |
