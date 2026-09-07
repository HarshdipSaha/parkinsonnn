"""Run 6b -- tri-planar late-fusion member (approach B). Generates a self-contained Kaggle notebook + kernel metadata:

    python kaggle/run6b_triplanar/gen_run6b.py          # writes notebook.ipynb + kernel-metadata.json next to this file
    kaggle kernels push -p kaggle/run6b_triplanar        # pushes AND runs it (Save & Run All) on the account in ~/.kaggle/kaggle.json

Cells: markdown header, env, %%writefile for every package file (the repo is the source of truth), selftest, cache/resume cell
(resumes the 111 run-3/4a checkpoints from the private checkpoint dataset), the resumable multi-GPU runner with JOBS_B
(tri-planar ConvNeXt-Tiny / EfficientNet-V2-S members, ref and hist normalisation, random + site folds), re-OOF with the
shared TTA, nested stack over the kept arms + the new ones, pack, end-to-end check with per-arm smoke-set scores, split.
Reuses the cell machinery of kaggle/gen_run3_inject.py (executed as a module; its own .playwright-mcp side outputs are ignored).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
KERNEL_ID = "harshdipsaha/dat-run6b-triplanar"
DATASETS = ["harshdipsaha/dat-parkinsons-cache-v4", "harshdipsaha/dat-parkinsons-challenge-data", "harshdipsaha/dat-run3-checkpoints"]

# ---- jobs: 24 epochs at 160 px, 4 views through one backbone (~4x the images of a single-view job at 224 px x (160/224)^2 ~ 2x its cost)
TRI_CNX = ("--backbone convnext_tiny --weights pretrained/convnext_tiny_in1k.pt --n-slabs 3 --slab-mm 8 --hw 160 --epochs 24 --batch-size 16 "
           "--lr 5e-4 --backbone-lr 1e-4 --mixup 0.2 --ema 0.99 --z-jitter 1 --fusion attn --tta-z -1 0 1")
TRI_V2S = TRI_CNX.replace("convnext_tiny --weights pretrained/convnext_tiny_in1k.pt", "efficientnet_v2_s --weights pretrained/efficientnet_v2_s_in1k.pt")
JOBS_B = [
    {"name": "r2t_cnx", "seeds": [0], "folds_csv": "folds.csv", "tag": "r2t_convnext_tiny_f{f}_s{s}", "args": TRI_CNX},
    {"name": "r2t_cnx_hist", "seeds": [0], "folds_csv": "folds.csv", "tag": "r2t_convnext_tiny_hist_f{f}_s{s}", "args": TRI_CNX + " --norm hist"},
    {"name": "r2t_v2s", "seeds": [0], "folds_csv": "folds.csv", "tag": "r2t_efficientnet_v2_s_f{f}_s{s}", "args": TRI_V2S},
    {"name": "r2t_cnx_site", "seeds": [0], "folds_csv": "folds_site.csv", "tag": "r2t_convnext_tiny_f{f}_s{s}_site", "args": TRI_CNX},
    {"name": "r2t_cnx_s1", "seeds": [1], "folds_csv": "folds.csv", "tag": "r2t_convnext_tiny_f{f}_s{s}", "args": TRI_CNX},
]
# arms alive after the run-4a gate + the run-6b candidates (9 arms x grid 8 = 12,870 weight vectors per fit)
ARMS_B = [("a3d_crop", "r3d_crop48x48x32_f?_s?", "r3d_crop48x48x32_f?_s?_site"), ("a2d_v2s", "r2d_efficientnet_v2_s_f?_s?", None),
          ("a2d_cnx", "r2d_convnext_tiny_f?_s?", None), ("a2d_cor", "r2d_efficientnet_b0_cor_f?_s?", None), ("feat", "featarm", "featarm_site"),
          ("tri_cnx", "r2t_convnext_tiny_f?_s?", "r2t_convnext_tiny_f?_s?_site"), ("tri_cnx_hist", "r2t_convnext_tiny_hist_f?_s?", None),
          ("tri_v2s", "r2t_efficientnet_v2_s_f?_s?", None), ("lr", "lrarm", "lrarm_site")]
EXTRAS_B = [("a3d_crop", "r3d_crop48x48x32_fall_s*"), ("a2d_v2s", "r2d_efficientnet_v2_s_fall_s*")]

# ---- reuse the parent generator's cells and helpers
gen_path = ROOT / "kaggle" / "gen_run3_inject.py"
(ROOT / ".playwright-mcp").mkdir(exist_ok=True)              # the parent generator writes its Playwright side files there (gitignored)
ns = {"__file__": str(gen_path), "__name__": "gen_run3_inject"}
exec(compile(gen_path.read_text(encoding="utf-8"), str(gen_path), "exec"), ns)
pc = ns["cells"]


def find(prefix: str) -> dict:
    return next(c for c in pc if c["source"].startswith(prefix))


cells = [{"type": "markdown", "source": "# DaT Parkinson's Challenge — run 6b: tri-planar late-fusion member (approach B)\n"
          "One shared ImageNet backbone reads axial, coronal and both sagittal slab stacks of the registered 2 mm box; attention pooling fuses the views.\n"
          "Package files are written by the cells below (source of truth: the private GitHub repo `parkinsonnn`, branch of run 6b).\n"
          "Inputs: raw competition dataset (private), `dat-parkinsons-cache-v4` (registered cache + ImageNet weights), `dat-run3-checkpoints` (resume + upload target).\n"
          "Data-hygiene: no raw scan is written to the output; `out/` holds checkpoints, OOF csvs and the submission zip only."}]
cells.append(find("!mkdir -p datsc"))
cells += [c for c in pc if c["source"].startswith("%%writefile")]
cells.append(find("!touch scripts/__init__.py"))
cells.append(find("# cache_v4"))
runner = (ns["runner_cell"](JOBS_B, "(run 6b: tri-planar)")
          .replace("python scripts/train_v3.py", "python scripts/train_b.py")
          .replace("import subprocess, time, os, glob, json\n", "import subprocess, time, os, glob, json, torch\n")
          .replace("    for gpu in (0, 1):", "    for gpu in range(max(1, torch.cuda.device_count())):"))
assert "scripts/train_b.py" in runner and "device_count" in runner
cells.append({"type": "code", "source": runner})
cells.append(find("# re-OOF"))
report = find("# report + stacking")["source"].replace(repr(ns["ARMS"]), repr(ARMS_B)).replace(repr(ns["EXTRAS"]), repr(EXTRAS_B))
assert repr(ARMS_B) in report and repr(EXTRAS_B) in report
cells.append({"type": "code", "source": report})
cells.append({"type": "code", "source": ns["failsafe"]((HERE / "e2e_b.py").read_text(encoding="utf-8"))})
cells.append(find("# === SPLIT"))

# ---- write notebook + kernel metadata
import nbformat
nb = nbformat.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.cells = [nbformat.v4.new_markdown_cell(c["source"]) if c["type"] == "markdown" else nbformat.v4.new_code_cell(c["source"]) for c in cells]
nbformat.validate(nb)
nbformat.write(nb, str(HERE / "notebook.ipynb"))
owner = sys.argv[sys.argv.index("--owner") + 1] if "--owner" in sys.argv else KERNEL_ID.split("/")[0]
meta = {"id": f"{owner}/dat-run6b-triplanar", "title": "dat-run6b-triplanar", "code_file": "notebook.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true",
        "dataset_sources": DATASETS, "competition_sources": [], "kernel_sources": [], "model_sources": []}
(HERE / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
n_jobs = sum(len(j["seeds"]) * len(j.get("folds", range(5))) for j in JOBS_B)
print(f"wrote {HERE/'notebook.ipynb'} ({len(nb.cells)} cells) and kernel-metadata.json (id {meta['id']}); {n_jobs} training jobs")
