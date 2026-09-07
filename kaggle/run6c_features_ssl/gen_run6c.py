"""Approach C notebook generator: builds kaggle/run6c_features_ssl/notebook.ipynb + kernel-metadata.json by reusing the
run-3 cell machinery (kaggle/gen_run3_inject.py: package %%writefile cells, cache/resume cell, 2-GPU resumable runner with
per-job checkpoint upload, re-OOF, nested stack, pack, split) and adding the approach-C cells:

  1. richer feature arm (`featc`, CPU, scripts/feat_arm_c.py --grid)                                     ~15 min
  2. VICReg self-supervised pretraining of the striatal-crop encoder on the training scans (labels unused) ~30 min GPU
  3. runner: crop-3D arm fine-tuned from the SSL encoder (random folds x2 seeds, site folds, full-data)   ~60 min on 2xT4
  4. re-OOF (TTA) -> nested stack over {crop-3D, SSL crop-3D, V2-S, ConvNeXt, coronal B0, feat, featc, lr}
  5. pack -> end-to-end check with per-arm SMOKE-SET scores -> split

Run:  python kaggle/run6c_features_ssl/gen_run6c.py        (from the repo root; then `kaggle kernels push -p kaggle/run6c_features_ssl`)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

# --- reuse the run-3 generator's cells and helpers (its own JS side-effect file is harmless) ---
(ROOT / ".playwright-mcp").mkdir(exist_ok=True)          # the run-3 generator writes its Playwright script there
ns = {"__file__": str(ROOT / "kaggle" / "gen_run3_inject.py"), "__name__": "gen"}
exec(compile((ROOT / "kaggle" / "gen_run3_inject.py").read_text(encoding="utf-8"), "gen", "exec"), ns)
base_cells, runner_cell, failsafe = ns["cells"], ns["runner_cell"], ns["failsafe"]
CROP3D = ns["CROP3D"]; V2S = ns["V2S"]

KERNEL_ID = "harshdipsaha/dat-run6c-features-ssl"
DATASETS = ["harshdipsaha/dat-parkinsons-cache-v4", "harshdipsaha/dat-parkinsons-challenge-data", "harshdipsaha/dat-run3-checkpoints"]

SSL_ENC = "artifacts_v3/ssl_c_encoder.pt"
JOBS_C = [   # crop-3D arm warm-started from the self-supervised encoder; lower LR + shorter schedule than from scratch
    {"name": "r3d_ssl", "seeds": [0, 1], "folds_csv": "folds.csv", "tag": "r3d_ssl_f{f}_s{s}", "args": CROP3D.replace("--epochs 50", "--epochs 40").replace("--lr 5e-4", "--lr 3e-4") + f" --init {SSL_ENC}"},
    {"name": "r3d_ssl_site", "seeds": [0], "folds_csv": "folds_site.csv", "tag": "r3d_ssl_f{f}_s{s}_site", "args": CROP3D.replace("--epochs 50", "--epochs 40").replace("--lr 5e-4", "--lr 3e-4") + f" --init {SSL_ENC}"},
    {"name": "r3d_ssl_full", "seeds": [0, 1], "folds": [-1], "folds_csv": "folds.csv", "tag": "r3d_ssl_fall_s{s}", "args": CROP3D.replace("--epochs 50", "--epochs 40").replace("--lr 5e-4", "--lr 3e-4") + f" --init {SSL_ENC}"},
]
# arms of the final stack: the run-4a survivors + the two approach-C members (+ run-5 arms if their OOFs exist; missing ones are skipped)
ARMS_C = [("a3d_crop", "r3d_crop48x48x32_f?_s?", "r3d_crop48x48x32_f?_s?_site"), ("a3d_ssl", "r3d_ssl_f?_s?", "r3d_ssl_f?_s?_site"),
          ("a3d_cropinv", "r3d_cropinv_f?_s?", "r3d_cropinv_f?_s?_site"), ("a2d_v2s", "r2d_efficientnet_v2_s_f?_s?", None),
          ("a2d_cnx", "r2d_convnext_tiny_f?_s?", None), ("a2d_cor", "r2d_efficientnet_b0_cor_f?_s?", None),
          ("feat", "featarm", "featarm_site"), ("featc", "featcarm", "featcarm_site"), ("lr", "lrarm", "lrarm_site")]
EXTRAS_C = [("a3d_crop", "r3d_crop48x48x32_fall_s*"), ("a3d_ssl", "r3d_ssl_fall_s*"), ("a2d_v2s", "r2d_efficientnet_v2_s_fall_s*")]


def head(c):
    return c["source"].split("\n")[0]


def pick(prefix):
    return next(c for c in base_cells if head(c).startswith(prefix))


cells = []
cells.append({"type": "markdown", "source": "# DaT Parkinson's Challenge — run 6c: feature-engineering + self-supervised hybrid (approach C)\n"
              "Members that are maximally different from the ImageNet 2.5D arms: (1) `featc` — radiomic/shape descriptors of the registered striatum + "
              "striatal-patch PCA → kernel (RFF) logistic / MLP, numpy inference; (2) `a3d_ssl` — the striatal-crop 3D net warm-started from a VICReg "
              "encoder pretrained on the training scans (labels unused, test scans never touched). Stacked with the run-4a survivors "
              "(crop-3D, V2-S, ConvNeXt, coronal B0, feat) on a nested estimate; the e2e cell prints per-arm SMOKE-SET scores "
              "(run 2 0.2567→0.3391, run 3 0.2397→0.2817, run 4a 0.2176→0.2667 public).\n"
              "Source of truth: repo `parkinsonnn`, branch of approach C (`kaggle/run6c_features_ssl/`). Data hygiene: no raw scan is written to the output."})
cells.append(pick("!mkdir -p datsc scripts artifacts_v3 out"))
cells += [c for c in base_cells if head(c).startswith("%%writefile")]
cells.append(pick("!touch scripts/__init__.py"))
cells.append(pick("# cache_v4 (registered"))
cells.append({"type": "code", "source": "# quick timing probe (2 epochs of the crop-3D arm on fold 0; expect ~10 s/epoch on a T4)\n"
              "!CUDA_VISIBLE_DEVICES=0 python scripts/train_v3.py --arm 3d --fold 0 --seed 99 --epochs 2 --out artifacts_timing --cache cache_v4 "
              "--size 48 48 32 --crop3d 48 48 32 --center3d 47.5 57.3 33.5 --widths 24 48 96 192 2>&1 | grep -E 'params|ep0|done'\n!rm -rf artifacts_timing"})
cells.append({"type": "code", "source": "# === approach C, member 1: richer feature arm (CPU; ~15 min: 116 descriptors x 2 orientations for 1,362 scans, then a small model grid) ===\n"
              "!python scripts/feat_arm_c.py --cache cache_v4 --template artifacts/template.npz --out artifacts_v3 --grid 2>&1 | tail -40\n"
              "!cp artifacts_v3/featc_v5.npz artifacts_v3/oof_featcarm*.csv artifacts_v3/featcarm.json out/ 2>/dev/null; ls -la artifacts_v3/featc_v5.npz"})
cells.append({"type": "code", "source": "# === approach C, member 2a: VICReg self-supervised pretraining of the striatal-crop encoder on the TRAINING scans only ===\n"
              "# (labels unused; two physically augmented, crop-jittered views per scan; ~8 s/epoch on a T4). Skipped if the encoder already exists.\n"
              "import os, subprocess, time\n"
              "t0 = time.time()\n"
              f"r = subprocess.run('CUDA_VISIBLE_DEVICES=0 python scripts/ssl_pretrain_c.py --cache cache_v4 --out {SSL_ENC} --epochs 200 --batch-size 32 --workers 2 --probe', shell=True, capture_output=True, text=True)\n"
              "lines = [l for l in r.stdout.splitlines() if ('[ssl_c]' in l and (' ep0' in l or 'ep0' in l[:16] or 'saved' in l or 'scans' in l or 'probe' in l or l.split('ep')[-1][:3].rstrip(' ').endswith('00') or l.split('ep')[-1][:3].endswith('50')))]\n"
              "print('\\n'.join(lines[-30:])); print(r.stderr[-800:]); print(f'ssl pretraining cell: {(time.time()-t0)/60:.0f} min')\n"
              f"assert os.path.exists('{SSL_ENC}'), 'SSL encoder missing'\n"
              f"!cp {SSL_ENC} out/ 2>/dev/null; ls -la {SSL_ENC}"})
run_c = runner_cell(JOBS_C, "(queue C: crop-3D fine-tuned from the SSL encoder)")
run_c = run_c.replace("import subprocess, time, os, glob, json\n", "import subprocess, time, os, glob, json, torch\nGPUS = list(range(max(1, torch.cuda.device_count())))\n", 1)
run_c = run_c.replace("    for gpu in (0, 1):\n", "    for gpu in GPUS:\n", 1)
assert "GPUS = list(range" in run_c and "for gpu in GPUS:" in run_c
cells.append({"type": "code", "source": run_c})
cells.append(pick("# re-OOF every CV checkpoint"))
stack = dict(pick("# report + stacking + pack"))
stack["source"] = re.sub(r"ARMS = \[.*?\]; EXTRAS = \[.*?\]", "ARMS = " + repr(ARMS_C) + "; EXTRAS = " + repr(EXTRAS_C), stack["source"], count=1, flags=re.S)
assert "featcarm" in stack["source"] and "r3d_ssl_f?_s?" in stack["source"]
cells.append(stack)
cells.append({"type": "code", "source": (HERE / "e2e_c.py").read_text(encoding="utf-8")})
cells.append(pick("# === SPLIT submission_v3.zip"))
for c in cells:
    if c["source"].startswith(("# report + stacking", "# === END-TO-END", "# === SPLIT", "# re-OOF", "# === approach C, member 1", "# === approach C, member 2a")):
        if not c["source"].split("\n", 1)[1].startswith("try:"):
            c["source"] = failsafe(c["source"])

nb = nbformat.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.cells = [nbformat.v4.new_markdown_cell(c["source"]) if c["type"] == "markdown" else nbformat.v4.new_code_cell(c["source"]) for c in cells]
nbformat.validate(nb)
nbformat.write(nb, str(HERE / "notebook.ipynb"))
(HERE / "kernel-metadata.json").write_text(json.dumps({
    "id": KERNEL_ID, "title": "dat-run6c-features-ssl", "code_file": "notebook.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
    "dataset_sources": DATASETS, "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=1))
print(f"wrote {HERE/'notebook.ipynb'} ({len(nb.cells)} cells) + kernel-metadata.json (id {KERNEL_ID}); jobs: {sum(len(j['seeds'])*len(j.get('folds', range(5))) for j in JOBS_C)} fine-tune runs")
