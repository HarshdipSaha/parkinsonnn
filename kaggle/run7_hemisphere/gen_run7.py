"""Run 7: generate the self-contained Kaggle notebook + kernel-metadata.json for the hemisphere-level members and the super-stack.

    python kaggle/run7_hemisphere/gen_run7.py [--owner <kaggle user>] [--slug dat-run7-hemisphere]
    kaggle kernels push -p kaggle/run7_hemisphere            # one command: creates/updates the private notebook and runs it

Reuses the run-3 cell machinery (kaggle/gen_run3_inject.py): package %%writefile cells, cache mount + resume from the checkpoint
dataset (+ API download fallback), the resumable multi-GPU runner with per-4-jobs checkpoint uploads, re-OOF with TTA, the
fail-safe tail. New here: JOBS7 (hemisphere members on random + site folds, DANN/DRO and bootstrap-loss variants, full-data
models) and a SUPER-STACK cell that scores every arm of runs 3-7 present in artifacts_v3/, keeps the 10 best by OOF log loss and
fits the nested combiner over them (compositions enumeration, --typ, grid 8).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--owner", default="harshdipsaha")
ap.add_argument("--slug", default="dat-run7-hemisphere")
ap.add_argument("--cache-ds", default="harshdipsaha/dat-parkinsons-cache-v4")
ap.add_argument("--raw-ds", default="harshdipsaha/dat-parkinsons-challenge-data")
ap.add_argument("--ckpt-ds", default="harshdipsaha/dat-run3-checkpoints")
args = ap.parse_args()

# ---- reuse the run-3 generator's cells/functions ----
ns = {"__file__": str(ROOT / "kaggle" / "gen_run3_inject.py"), "__name__": "gen"}
src3 = (ROOT / "kaggle" / "gen_run3_inject.py").read_text(encoding="utf-8")
src3 = src3.replace('CKPT_DS = "harshdipsaha/dat-run3-checkpoints"', f'CKPT_DS = "{args.ckpt_ds}"') \
           .replace('CACHE_DS = "/kaggle/input/datasets/harshdipsaha/dat-parkinsons-cache-v4"', f'CACHE_DS = "/kaggle/input/datasets/{args.cache_ds}"') \
           .replace('RAW_DS = "/kaggle/input/datasets/harshdipsaha/dat-parkinsons-challenge-data"', f'RAW_DS = "/kaggle/input/datasets/{args.raw_ds}"')
exec(compile(src3, "gen", "exec"), ns)
runner_cell, failsafe = ns["runner_cell"], ns["failsafe"]
base_cells = [c for c in ns["cells"] if not c["source"].startswith(("# === RESUMABLE", "# report + stacking", "# === END-TO-END", "# === SPLIT", "# re-OOF", "# DaT Parkinson"))]

CROP = "--arm 3d --size 48 48 32 --crop3d 48 48 32 --center3d {XC} {YC} {ZC} --widths 24 48 96 192 --epochs 50 --batch-size 16 --lr 5e-4 --mixup 0.2 --ema 0.995"
HEMI = CROP + " --arch hemi"
JOBS7 = [   # ≈ 265 GPU-min on T4s → ≈ 2.3 h on two + ≈ 35 min tail; long jobs first
    {"name": "r3d_hemi", "seeds": [0, 1, 2], "folds_csv": "folds.csv", "tag": "r3d_hemi48_f{f}_s{s}", "args": HEMI},
    {"name": "r3d_hemi_site", "seeds": [0], "folds_csv": "folds_site.csv", "tag": "r3d_hemi48_f{f}_s{s}_site", "args": HEMI},
    {"name": "r3d_hemi_inv", "seeds": [0], "folds_csv": "folds.csv", "tag": "r3d_hemiinv_f{f}_s{s}", "args": HEMI + " --dann 0.1 --dro 0.1"},
    {"name": "r3d_crop_boot", "seeds": [0], "folds_csv": "folds.csv", "tag": "r3d_cropboot_f{f}_s{s}", "args": CROP + " --loss bootstrap"},
    {"name": "r3d_hemi_full", "seeds": [0, 1], "folds": [-1], "folds_csv": "folds.csv", "tag": "r3d_hemi48_fall_s{s}", "args": HEMI},
]
# every arm family of runs 3-7; the stack cell keeps the 10 best by OOF log loss among those with OOF files
ARMS_ALL = [
    ("a3d_hemi", "r3d_hemi48_f?_s?", "r3d_hemi48_f?_s?_site"), ("a3d_hemiinv", "r3d_hemiinv_f?_s?", None), ("a3d_cropboot", "r3d_cropboot_f?_s?", None),
    ("a3d_crop", "r3d_crop48x48x32_f?_s?", "r3d_crop48x48x32_f?_s?_site"), ("a3d_cropinv", "r3d_cropinv_f?_s?", "r3d_cropinv_f?_s?_site"), ("a3d_crophist", "r3d_crophist_f?_s?", None),
    ("a3d_crop80", "r3d_crop48e80_f?_s?", None), ("a3d_se48", "r3d_se48_f?_s?", None), ("a3d_crop64", "r3d_crop64e60_f?_s?", None), ("a3d_ssl", "r3d_ssl_f?_s?", "r3d_ssl_f?_s?_site"),
    ("a2d_v2s", "r2d_efficientnet_v2_s_f?_s?", None), ("a2d_cnx", "r2d_convnext_tiny_f?_s?", None), ("a2d_cnx_cor", "r2d_convnext_tiny_cor_f?_s?", None), ("a2d_cor", "r2d_efficientnet_b0_cor_f?_s?", None),
    ("a2d_b0", "r2d_efficientnet_b0_f?_s?", "r2d_efficientnet_b0_f?_s?_site"), ("a2t_cnx", "r2t_convnext_tiny_f?_s?", None), ("a2t_cnx_hist", "r2t_convnext_tiny_hist_f?_s?", None), ("a2t_v2s", "r2t_efficientnet_v2_s_f?_s?", None),
    ("featc", "featcarm", "featcarm_site"), ("feat", "featarm", "featarm_site"), ("lr", "lrarm", "lrarm_site"),
]
EXTRAS_ALL = [("a3d_hemi", "r3d_hemi48_fall_s*"), ("a3d_crop", "r3d_crop48x48x32_fall_s*"), ("a3d_crop80", "r3d_crop48e80_fall_s*"), ("a3d_ssl", "r3d_ssl_fall_s*"),
              ("a2d_v2s", "r2d_efficientnet_v2_s_fall_s*"), ("a2d_b0", "r2d_efficientnet_b0_fall_s*")]

STACK = '''# super-stack: score every arm family of runs 3-7 present in artifacts_v3/, keep the 10 best by OOF log loss, nested combiner
import glob, json, os, subprocess, pandas as pd, numpy as np
from sklearn.metrics import log_loss, roc_auc_score
ARMS = __ARMS__; EXTRAS = __EXTRAS__
rows = [{'run': os.path.basename(f)[:-5], **{k: json.load(open(f)).get(k) for k in ('val_logloss', 'val_auroc')}} for f in sorted(glob.glob('artifacts_v3/r*_f*_s*.json'))]
df = pd.DataFrame(rows); df['arm'] = df.run.str.replace(r'_f(\\d|all)_s\\d(_site)?$', '', regex=True); print(df.groupby('arm')[['val_logloss', 'val_auroc']].agg(['mean', 'count']).round(4))
scored = []
for n, p, q in ARMS:
    fs = glob.glob(f'artifacts_v3/oof_{p}.csv')
    if not fs: continue
    o = pd.concat([pd.read_csv(f, dtype={'uid': str}) for f in fs]).groupby('uid').agg(y=('y', 'first'), p=('p', 'mean'))
    if len(o) < 1300: print('skip (incomplete OOF)', n, len(o)); continue
    scored.append((log_loss(o.y, np.clip(o.p, 1e-6, 1 - 1e-6)), n, p, q, len(fs)))
scored.sort(); print('arms by OOF log loss:'); [print(f'  {ll:.4f} {n:<12} ckpts={k}') for ll, n, p, q, k in scored]
keep = scored[:10]
cmd = 'python scripts/fit_stack_v3.py --out artifacts_v3 ' + ' '.join(f'--arm {n} "{p}"' for _, n, p, q, _ in keep)
cmd += ' ' + ' '.join(f'--site-arm {n} "{q}"' for _, n, p, q, _ in keep if q and glob.glob(f'artifacts_v3/oof_{q}.csv'))
cmd += ' ' + ' '.join(f'--extra {n} "{g}"' for n, g in EXTRAS if glob.glob(f'artifacts_v3/{g}.pt') and n in [k[1] for k in keep]) + ' --lam 0.02 --t-floor 1.0 --nested --typ --grid 8 --cache cache_v4'
print(cmd); r = subprocess.run(cmd, shell=True, capture_output=True, text=True); print(r.stdout[-5000:]); print(r.stderr[-800:])
!python scripts/pack_v3.py --artifacts artifacts_v3 --out submission_v3.zip
!cp submission_v3.zip artifacts_v3/manifest.json artifacts_v3/oof_stack.csv out/ && cp artifacts_v3/*.json artifacts_v3/oof_*.csv cache_v4/folds*.csv cache_v4/sites.csv out/ 2>/dev/null; du -sh out submission_v3.zip'''

cells = [{"type": "markdown", "source": "# DaT Parkinson's Challenge — run 7: hemisphere-level members + super-stack over runs 3–7\n"
          "Shared-encoder hemisphere nets on the mirrored striatal half-crops (symmetric pair + max-side head), on random and leave-site-out folds, "
          "with DANN/DRO and bootstrapped-loss variants and full-data models; then a nested combiner over the 10 best arms of every run found in the checkpoint dataset.\n"
          "Inputs: raw competition dataset (private), `dat-parkinsons-cache-v4`, `dat-run3-checkpoints` (resume + upload every 4 jobs via the KAGGLE_USERNAME/KAGGLE_KEY secrets)."}]
cells += base_cells
run_src = runner_cell(JOBS7, "(run 7)").replace("for gpu in (0, 1):", "for gpu in range(max(1, __import__('torch').cuda.device_count())):")
assert "device_count" in run_src
cells.append({"type": "code", "source": run_src})
cells.append({"type": "code", "source": "# re-OOF every CV checkpoint with inference-time TTA (2D: z-shift -1/0/+1 x flip; 3D: flip) so the stack is fitted on what main.py produces\n!python scripts/reoof_v3.py --art artifacts_v3 --cache cache_v4 --tta-z -1 0 1 2>&1 | grep -v Warning"})
cells.append({"type": "code", "source": STACK.replace("__ARMS__", repr(ARMS_ALL)).replace("__EXTRAS__", repr(EXTRAS_ALL))})
cells.append({"type": "code", "source": (ROOT / "kaggle" / "run6a_crop3d" / "e2e_check_a.py").read_text(encoding="utf-8")})
cells.append({"type": "code", "source": (ROOT / "kaggle" / "split_zip_cell.py").read_text(encoding="utf-8")})
for c in cells:
    if c["source"].startswith(("# super-stack", "# === END-TO-END", "# === SPLIT", "# re-OOF")):
        c["source"] = failsafe(c["source"])

nb = nbformat.v4.new_notebook(); nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.cells = [nbformat.v4.new_markdown_cell(c["source"]) if c["type"] == "markdown" else nbformat.v4.new_code_cell(c["source"]) for c in cells]
nbformat.validate(nb); nbformat.write(nb, str(HERE / "notebook.ipynb"))
meta = {"id": f"{args.owner}/{args.slug}", "title": args.slug, "code_file": "notebook.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [args.cache_ds, args.raw_ds, args.ckpt_ds], "competition_sources": [], "kernel_sources": [], "model_sources": []}
(HERE / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
n_jobs = sum(len(j["seeds"]) * len(j.get("folds", range(5))) for j in JOBS7)
print(f"wrote {HERE / 'notebook.ipynb'} ({len(nb.cells)} cells, {n_jobs} training jobs) and kernel-metadata.json (id {meta['id']})")
