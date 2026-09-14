"""Generate the run-10 Kaggle notebook (2D Swin Transformer + soft distillation + Run 9 merge gate).

    python kaggle/run10_swin/gen_run10.py [--owner skyfall23] [--slug dat-run10-swin] [--out <dir>]
    kaggle kernels push -p kaggle/run10_swin

Reuses the run-8/run-3 cell machinery: package %%writefile cells, cache mount + resume from skyfall23/dat-run9-seeds,
Swin-T ImageNet-1K pretrained weights download, 2-GPU job runner for Swin-T standard & soft-distilled folds,
re-OOF with inference-time TTA, and GATE cell evaluating Run 9 vs Run 9 + Swin with strict improvement gating.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--owner", default="skyfall23")
ap.add_argument("--slug", default="dat-run10-swin")
ap.add_argument("--out", default=str(HERE))
ap.add_argument("--cache-ds", default="harshdipsaha/dat-parkinsons-cache-v4")
ap.add_argument("--raw-ds", default="harshdipsaha/dat-parkinsons-challenge-data")
ap.add_argument("--ckpt-ds", default="harshdipsaha/dat-run3-checkpoints")
ap.add_argument("--soft-lam", type=float, default=0.4)
ap.add_argument("--soft-table", default=str(ROOT / "kaggle" / "run8_distill" / "soft_master_v2.csv"))
ap.add_argument("--swin-seeds", type=int, nargs="+", default=[0], help="seeds of the Swin-T member")
ap.add_argument("--skip", nargs="*", default=[], help="job names to drop")
ap.add_argument("--kernel-source", action="append", default=["skyfall23/dat-run9-seeds"], help="notebook whose output is mounted under /kaggle/input")
args = ap.parse_args()
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)

# ---- reuse the run-3 generator's cells/functions ----
ns = {"__file__": str(ROOT / "kaggle" / "gen_run3_inject.py"), "__name__": "gen"}
src3 = (ROOT / "kaggle" / "gen_run3_inject.py").read_text(encoding="utf-8")
src3 = src3.replace('CKPT_DS = "harshdipsaha/dat-run3-checkpoints"', f'CKPT_DS = "{args.ckpt_ds}"') \
           .replace('CACHE_DS = "/kaggle/input/datasets/harshdipsaha/dat-parkinsons-cache-v4"', f'CACHE_DS = "/kaggle/input/datasets/{args.cache_ds}"') \
           .replace('RAW_DS = "/kaggle/input/datasets/harshdipsaha/dat-parkinsons-challenge-data"', f'RAW_DS = "/kaggle/input/datasets/{args.raw_ds}"')
exec(compile(src3, "gen", "exec"), ns)
runner_cell, failsafe = ns["runner_cell"], ns["failsafe"]
base_cells = [c for c in ns["cells"] if not c["source"].startswith(("# === RESUMABLE", "# report + stacking", "# === END-TO-END", "# === SPLIT", "# re-OOF", "# DaT Parkinson"))]
src7 = (ROOT / "kaggle" / "run7_hemisphere" / "gen_run7.py").read_text(encoding="utf-8")
STACK7 = src7.split("STACK = '''")[1].split("'''")[0]

SWIN = "--arm 2d --backbone swin_t --weights pretrained/swin_t_in1k.pt --n-slabs 3 --slab-mm 8 --hw 224 --epochs 30 --batch-size 24 --lr 5e-4 --backbone-lr 1e-4 --mixup 0.2 --ema 0.99 --z-center {ZC}"
SOFT = f" --soft-csv cache_v4/soft_master_v2.csv --soft-lam {args.soft_lam}"

JOBS10 = [
    {"name": "r2d_swin", "seeds": list(args.swin_seeds), "folds_csv": "folds.csv", "tag": "r2d_swin_t_f{f}_s{s}", "args": SWIN},
    {"name": "r2d_swinsoft", "seeds": list(args.swin_seeds), "folds_csv": "folds.csv", "tag": "r2d_swin_t_soft_f{f}_s{s}", "args": SWIN + SOFT},
]
JOBS10 = [j for j in JOBS10 if j["name"] not in set(args.skip)]

ARMS_ALL = [
    ("a2d_swin", "r2d_swin_t_f?_s?", None), ("a2d_swinsoft", "r2d_swin_t_soft_f?_s?", None),
    ("a3d_hemisoft", "r3d_hemisoft_f?_s?", "r3d_hemisoft_f?_s?_site"), ("a3d_hemimw", "r3d_hemimw_f?_s?", None), ("a3d_crophistsoft", "r3d_crophistsoft_f?_s?", None),
    ("a3d_hemi", "r3d_hemi48_f?_s?", "r3d_hemi48_f?_s?_site"), ("a3d_hemiinv", "r3d_hemiinv_f?_s?", None), ("a3d_cropboot", "r3d_cropboot_f?_s?", None),
    ("a3d_crop", "r3d_crop48x48x32_f?_s?", "r3d_crop48x48x32_f?_s?_site"), ("a3d_cropinv", "r3d_cropinv_f?_s?", "r3d_cropinv_f?_s?_site"), ("a3d_crophist", "r3d_crophist_f?_s?", None),
    ("a3d_crop80", "r3d_crop48e80_f?_s?", None), ("a3d_se48", "r3d_se48_f?_s?", None), ("a3d_crop64", "r3d_crop64e60_f?_s?", None), ("a3d_ssl", "r3d_ssl_f?_s?", "r3d_ssl_f?_s?_site"),
    ("a2d_v2s", "r2d_efficientnet_v2_s_f?_s?", None), ("a2d_cnx", "r2d_convnext_tiny_f?_s?", None), ("a2d_cnx_cor", "r2d_convnext_tiny_cor_f?_s?", None), ("a2d_cor", "r2d_efficientnet_b0_cor_f?_s?", None),
    ("a2d_b0", "r2d_efficientnet_b0_f?_s?", "r2d_efficientnet_b0_f?_s?_site"), ("a2t_cnx", "r2t_convnext_tiny_f?_s?", None), ("a2t_cnx_hist", "r2t_convnext_tiny_hist_f?_s?", None), ("a2t_v2s", "r2t_efficientnet_v2_s_f?_s?", None),
    ("featc", "featcarm", "featcarm_site"), ("feat", "featarm", "featarm_site"), ("lr", "lrarm", "lrarm_site"),
]
EXTRAS_ALL = [("a3d_hemisoft", "r3d_hemisoft_fall_s*"), ("a3d_hemi", "r3d_hemi48_fall_s*"), ("a3d_crop", "r3d_crop48x48x32_fall_s*"), ("a3d_crop80", "r3d_crop48e80_fall_s*"),
              ("a3d_ssl", "r3d_ssl_fall_s*"), ("a2d_v2s", "r2d_efficientnet_v2_s_fall_s*"), ("a2d_b0", "r2d_efficientnet_b0_fall_s*")]

table = Path(args.soft_table).read_bytes()
SOFT_CELL = ("# soft-target table (run 10): Master v2's NESTED out-of-fold probability + loss per training scan, embedded -> cache_v4/soft_master_v2.csv\n"
             "import base64, gzip, os, pandas as pd\n"
             f"_b = {base64.b64encode(gzip.compress(table)).decode()!r}\n"
             "os.makedirs('cache_v4', exist_ok=True); open('cache_v4/soft_master_v2.csv', 'wb').write(gzip.decompress(base64.b64decode(_b)))\n"
             "_t = pd.read_csv('cache_v4/soft_master_v2.csv', dtype={'uid': str}); print(len(_t), 'scans in soft table')\n")

SWIN_PRETRAINED_CELL = """# Swin-T pretrained weights setup (run 10):
import os, torch
os.makedirs('pretrained', exist_ok=True)
swin_pt = 'pretrained/swin_t_in1k.pt'
if not os.path.exists(swin_pt):
    try:
        import torchvision.models as tvm
        print('Downloading Swin-T ImageNet-1K pretrained weights from torchvision...')
        m = tvm.swin_t(weights=tvm.Swin_T_Weights.IMAGENET1K_V1)
        torch.save(m.state_dict(), swin_pt)
        print(f'Successfully downloaded and saved {swin_pt} ({os.path.getsize(swin_pt)/1e6:.1f} MB)')
    except Exception as e:
        print('Warning: Swin-T download failed:', e)
else:
    print(f'{swin_pt} already present ({os.path.getsize(swin_pt)/1e6:.1f} MB)')
"""

GATE10 = """# GATE (run 10): nested estimate of Run 9 baseline (7 arms) vs Run 9 + Swin arms (strict improvement gate)
import glob, numpy as np, pandas as pd
from scipy.optimize import minimize_scalar

RUN9_ARMS = {
    'a3d_hemi': 'r3d_hemi48_f?_s?',
    'a3d_ssl': 'r3d_ssl_f?_s?',
    'featc': 'featcarm',
    'a2d_cnx_cor': 'r2d_convnext_tiny_cor_f?_s?',
    'a2d_v2s': 'r2d_efficientnet_v2_s_f?_s[01]',
    'a3d_hemisoft': 'r3d_hemisoft_f?_s?',
    'a3d_crophistsoft': 'r3d_crophistsoft_f?_s?'
}
NEW_SWIN = {
    'a2d_swin': 'r2d_swin_t_f?_s?',
    'a2d_swinsoft': 'r2d_swin_t_soft_f?_s?'
}

def load(p):
    fs = [f for f in glob.glob(f'artifacts_v3/oof_{p}.csv') if '_site' not in f and '_fall' not in f]
    if not fs: return None
    o = pd.concat([pd.read_csv(f, dtype={'uid': str}) for f in fs]).groupby('uid').agg(y=('y', 'first'), p=('p', 'mean'))
    return o if len(o) >= 1300 else None

oofs = {n: o for n, o in ((n, load(p)) for n, p in {**RUN9_ARMS, **NEW_SWIN}.items()) if o is not None}
uids = sorted(set.intersection(*[set(o.index) for o in oofs.values()]))
y = next(iter(oofs.values())).loc[uids, 'y'].values.astype(float)
folds = pd.read_csv('cache_v4/folds.csv', dtype={'uid': str}).set_index('uid').loc[uids, 'fold'].values

lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sg = lambda z: 1 / (1 + np.exp(-z))
ll = lambda yy, pp: float(-np.mean(yy * np.log(np.clip(pp, 1e-7, 1)) + (1 - yy) * np.log(np.clip(1 - pp, 1e-7, 1))))

def nested(names, lam=0.01):
    Z = np.stack([lg(oofs[n].loc[uids, 'p'].values) for n in names]).mean(0)
    pn = np.zeros(len(y))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        T = max(minimize_scalar(lambda t: ll(y[tr], sg(Z[tr] / t)), bounds=(0.3, 5), method='bounded').x, 0.9)
        pn[va] = (1 - lam) * sg(Z[va] / T) + lam * 0.548
    return ll(y, pn)

r9_names = [n for n in RUN9_ARMS if n in oofs]
base_01 = nested(r9_names, lam=0.01)
base_02 = nested(r9_names, lam=0.02)
print(f'Run 9 baseline ({len(r9_names)} arms): nested CV = {base_01:.4f} (lam=0.01) / {base_02:.4f} (lam=0.02)')

best_set = list(r9_names)
best_score = base_01
for n in NEW_SWIN:
    if n in oofs:
        score_01 = nested(r9_names + [n], lam=0.01)
        diff = score_01 - base_01
        solo = ll(y, oofs[n].loc[uids, 'p'].values)
        print(f'  + {n:<16} nested {score_01:.4f} ({diff:+.4f}); arm alone {solo:.4f}')
        if score_01 < best_score:
            best_score = score_01
            best_set = r9_names + [n]

both = [n for n in NEW_SWIN if n in oofs]
if len(both) > 1:
    score_both = nested(r9_names + both, lam=0.01)
    diff_both = score_both - base_01
    print(f'  + all swin arms    nested {score_both:.4f} ({diff_both:+.4f})')
    if score_both < best_score:
        best_score = score_both
        best_set = r9_names + both

gain = base_01 - best_score
print('=' * 68)
if gain > 0:
    print(f'GATE VERDICT: PASS (Swin improves nested loss by {gain:+.4f})')
    print(f'Recommended set for v10: {best_set}')
    print('Action on laptop: run scripts/build_master_v3.py to assemble data/submission_v10.zip')
else:
    print(f'GATE VERDICT: FAIL (Swin did not improve nested loss; best diff {gain:+.4f})')
    print('Action: Preserve Master v9 (data/submission_v9.zip, smoke 0.2400) as final submission.')
print('=' * 68)
"""

cells = [{"type": "markdown", "source": "# DaT Parkinson's Challenge — run 10: 2D Swin Transformer + Run 9 Merge Gate\n"
          "Inherits all 91 checkpoints and artifacts from `skyfall23/dat-run9-seeds`.\n"
          "Trains 2D Swin Transformer (`swin_t`) across 5 folds (standard + soft-distilled).\n"
          "Re-OOFs with inference-time TTA, and evaluates strict improvement gate over Run 9."}]
cells += base_cells
cells.append({"type": "code", "source": SOFT_CELL})
cells.append({"type": "code", "source": SWIN_PRETRAINED_CELL})
run_src = runner_cell(JOBS10, "(run 10)").replace("for gpu in (0, 1):", "for gpu in range(max(1, __import__('torch').cuda.device_count())):")
assert "device_count" in run_src
cells.append({"type": "code", "source": run_src})
cells.append({"type": "code", "source": "# re-OOF every CV checkpoint with inference-time TTA\n!python scripts/reoof_v3.py --art artifacts_v3 --cache cache_v4 --tta-z -1 0 1 2>&1 | grep -v Warning"})
cells.append({"type": "code", "source": STACK7.replace("__ARMS__", repr(ARMS_ALL)).replace("__EXTRAS__", repr(EXTRAS_ALL)).replace("--t-floor 1.0", "--t-floor 0.9").replace("runs 3-7", "runs 3-10")})
cells.append({"type": "code", "source": GATE10})
cells.append({"type": "code", "source": (ROOT / "kaggle" / "run6a_crop3d" / "e2e_check_a.py").read_text(encoding="utf-8")})
cells.append({"type": "code", "source": (ROOT / "kaggle" / "split_zip_cell.py").read_text(encoding="utf-8")})
for c in cells:
    if c["source"].startswith(("# super-stack", "# GATE", "# === END-TO-END", "# === SPLIT", "# re-OOF")):
        c["source"] = failsafe(c["source"])

nb = nbformat.v4.new_notebook(); nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.cells = [nbformat.v4.new_markdown_cell(c["source"]) if c["type"] == "markdown" else nbformat.v4.new_code_cell(c["source"]) for c in cells]
nbformat.validate(nb); nbformat.write(nb, str(OUT / "notebook.ipynb"))

meta = {
    "id": f"{args.owner}/{args.slug}",
    "title": args.slug,
    "code_file": "notebook.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "dataset_sources": [args.cache_ds, args.raw_ds, args.ckpt_ds],
    "competition_sources": [],
    "kernel_sources": list(args.kernel_source),
    "model_sources": []
}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
n_jobs = sum(len(j["seeds"]) * len(j.get("folds", range(5))) for j in JOBS10)
print(f"wrote {OUT / 'notebook.ipynb'} ({len(nb.cells)} cells, {n_jobs} training jobs) and kernel-metadata.json (id {meta['id']}; kernel sources {meta['kernel_sources']})")
