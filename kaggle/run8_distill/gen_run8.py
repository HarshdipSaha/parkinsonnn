"""Generate the run-8 Kaggle notebook (ensemble-corrected members + Master v3 gate) and its kernel-metadata.json.

    python kaggle/run8_distill/gen_run8.py [--owner skyfall23] [--slug dat-run8-distill] [--out <dir>] [--soft-lam 0.4] [--mistake-w 2.0]
    kaggle kernels push -p kaggle/run8_distill          # with KAGGLE_CONFIG_DIR pointing at the teammate's kaggle.json

Reuses the run-3 cell machinery (kaggle/gen_run3_inject.py) exactly as run 7 does: package %%writefile cells, cache mount + resume from the
checkpoint dataset, the multi-GPU runner with checkpoint upload every 4 jobs, fail-safe tail. New here: SOFT_CELL (writes the embedded
soft-target table cache_v4/soft_master_v2.csv = Master v2's nested OOF probability and loss per training scan), JOBS8 (hemisphere members on
soft targets, 3 seeds + site folds; mistake-weighted variant; two extra plain hemisphere seeds; crop-hist on soft targets; full-data soft
hemispheres), the run-8 arm families in the super-stack, and a GATE cell printing the nested estimate of the Master v2 pillars vs pillars +
each run-8 arm (equal weights, T floor 0.9) so the slot decision can be read from the Kaggle log."""
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
ap.add_argument("--slug", default="dat-run8-distill")
ap.add_argument("--out", default=str(HERE))
ap.add_argument("--cache-ds", default="harshdipsaha/dat-parkinsons-cache-v4")
ap.add_argument("--raw-ds", default="harshdipsaha/dat-parkinsons-challenge-data")
ap.add_argument("--ckpt-ds", default="harshdipsaha/dat-run3-checkpoints")
ap.add_argument("--soft-lam", type=float, default=0.4)
ap.add_argument("--mistake-w", type=float, default=2.0)
ap.add_argument("--soft-table", default=str(HERE / "soft_master_v2.csv"))
ap.add_argument("--hemisoft-seeds", type=int, nargs="+", default=[0, 1, 2], help="seeds of the soft-target hemisphere member on random folds (bracket session: 0 1 2 3 4)")
ap.add_argument("--hemi-more-seeds", type=int, nargs="+", default=[3, 4], help="extra plain hemisphere seeds")
ap.add_argument("--hemimw-seeds", type=int, nargs="+", default=[0], help="seeds of the mistake-weighted hemisphere member")
ap.add_argument("--crophistsoft-seeds", type=int, nargs="+", default=[0], help="seeds of the soft-target crop-hist member")
ap.add_argument("--skip", nargs="*", default=[], help="job names to drop, e.g. r3d_hemimw r3d_crophistsoft")
ap.add_argument("--kernel-source", action="append", default=[], help="notebook whose output is mounted under /kaggle/input (e.g. skyfall23/dat-run8-distill): the cache cell copies its artifacts_v3 so the push resumes without secrets")
args = ap.parse_args()
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)

# ---- reuse the run-3 generator's cells/functions (as run 7) ----
ns = {"__file__": str(ROOT / "kaggle" / "gen_run3_inject.py"), "__name__": "gen"}
src3 = (ROOT / "kaggle" / "gen_run3_inject.py").read_text(encoding="utf-8")
src3 = src3.replace('CKPT_DS = "harshdipsaha/dat-run3-checkpoints"', f'CKPT_DS = "{args.ckpt_ds}"') \
           .replace('CACHE_DS = "/kaggle/input/datasets/harshdipsaha/dat-parkinsons-cache-v4"', f'CACHE_DS = "/kaggle/input/datasets/{args.cache_ds}"') \
           .replace('RAW_DS = "/kaggle/input/datasets/harshdipsaha/dat-parkinsons-challenge-data"', f'RAW_DS = "/kaggle/input/datasets/{args.raw_ds}"')
exec(compile(src3, "gen", "exec"), ns)
runner_cell, failsafe = ns["runner_cell"], ns["failsafe"]
base_cells = [c for c in ns["cells"] if not c["source"].startswith(("# === RESUMABLE", "# report + stacking", "# === END-TO-END", "# === SPLIT", "# re-OOF", "# DaT Parkinson"))]
src7 = (ROOT / "kaggle" / "run7_hemisphere" / "gen_run7.py").read_text(encoding="utf-8")
STACK7 = src7.split("STACK = '''")[1].split("'''")[0]          # the run-7 super-stack cell, verbatim

CROP = "--arm 3d --size 48 48 32 --crop3d 48 48 32 --center3d {XC} {YC} {ZC} --widths 24 48 96 192 --epochs 50 --batch-size 16 --lr 5e-4 --mixup 0.2 --ema 0.995"
HEMI = CROP + " --arch hemi"
SOFT = f" --soft-csv cache_v4/soft_master_v2.csv --soft-lam {args.soft_lam}"
MW = SOFT + f" --mistake-w {args.mistake_w}"
JOBS8 = [   # ≈ 12.4 GPU-min per hemisphere fold on a T4 → 42 jobs ≈ 8.7 GPU-h ≈ 4.5 h wall on two T4s; priority order (a cut session keeps the top)
    {"name": "r3d_hemisoft", "seeds": list(args.hemisoft_seeds), "folds_csv": "folds.csv", "tag": "r3d_hemisoft_f{f}_s{s}", "args": HEMI + SOFT},
    {"name": "r3d_hemisoft_site", "seeds": [0], "folds_csv": "folds_site.csv", "tag": "r3d_hemisoft_f{f}_s{s}_site", "args": HEMI + SOFT},
    {"name": "r3d_hemi_more", "seeds": list(args.hemi_more_seeds), "folds_csv": "folds.csv", "tag": "r3d_hemi48_f{f}_s{s}", "args": HEMI},
    {"name": "r3d_hemimw", "seeds": list(args.hemimw_seeds), "folds_csv": "folds.csv", "tag": "r3d_hemimw_f{f}_s{s}", "args": HEMI + MW},
    {"name": "r3d_crophistsoft", "seeds": list(args.crophistsoft_seeds), "folds_csv": "folds.csv", "tag": "r3d_crophistsoft_f{f}_s{s}", "args": CROP + " --norm hist" + SOFT},
    {"name": "r3d_hemisoft_full", "seeds": [0, 1], "folds": [-1], "folds_csv": "folds.csv", "tag": "r3d_hemisoft_fall_s{s}", "args": HEMI + SOFT},
]
JOBS8 = [j for j in JOBS8 if j["name"] not in set(args.skip)]
ARMS_ALL = [
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
SOFT_CELL = ("# soft-target table (run 8): Master v2's NESTED out-of-fold probability + loss per training scan, embedded (no scan data), -> cache_v4/soft_master_v2.csv\n"
             "import base64, gzip, os, pandas as pd\n"
             f"_b = {base64.b64encode(gzip.compress(table)).decode()!r}\n"
             "os.makedirs('cache_v4', exist_ok=True); open('cache_v4/soft_master_v2.csv', 'wb').write(gzip.decompress(base64.b64decode(_b)))\n"
             "_t = pd.read_csv('cache_v4/soft_master_v2.csv', dtype={'uid': str}); print(len(_t), 'scans; nested loss', round(_t.loss.mean(), 4), '; top-100 share', round(_t.loss.sort_values(ascending=False).iloc[:100].sum() / _t.loss.sum(), 3))\n")

GATE = '''# GATE (run 8): nested estimate (equal weights, T refit per fold with floor 0.9, lam 0.02) of the Master v2 pillars vs pillars + each run-8 arm
import glob, numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
PIL = {'a3d_hemi': 'r3d_hemi48_f?_s?', 'a3d_ssl': 'r3d_ssl_f?_s?', 'featc': 'featcarm', 'a2d_cnx_cor': 'r2d_convnext_tiny_cor_f?_s?', 'a2d_v2s': 'r2d_efficientnet_v2_s_f?_s[01]'}
NEW = {'a3d_hemisoft': 'r3d_hemisoft_f?_s?', 'a3d_hemimw': 'r3d_hemimw_f?_s?', 'a3d_crophistsoft': 'r3d_crophistsoft_f?_s?'}
def load(p):
    fs = [f for f in glob.glob(f'artifacts_v3/oof_{p}.csv') if '_site' not in f and '_fall' not in f]
    if not fs: return None
    o = pd.concat([pd.read_csv(f, dtype={'uid': str}) for f in fs]).groupby('uid').agg(y=('y', 'first'), p=('p', 'mean')); return o if len(o) >= 1300 else None
oofs = {n: o for n, o in ((n, load(p)) for n, p in {**PIL, **NEW}.items()) if o is not None}
uids = sorted(set.intersection(*[set(o.index) for o in oofs.values()])); y = next(iter(oofs.values())).loc[uids, 'y'].values.astype(float)
folds = pd.read_csv('cache_v4/folds.csv', dtype={'uid': str}).set_index('uid').loc[uids, 'fold'].values
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))); sg = lambda z: 1 / (1 + np.exp(-z))
ll = lambda yy, pp: float(-np.mean(yy * np.log(np.clip(pp, 1e-7, 1)) + (1 - yy) * np.log(np.clip(1 - pp, 1e-7, 1))))
def nested(names):
    Z = np.stack([lg(oofs[n].loc[uids, 'p'].values) for n in names]).mean(0); pn = np.zeros(len(y))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        T = max(minimize_scalar(lambda t: ll(y[tr], sg(Z[tr] / t)), bounds=(0.3, 5), method='bounded').x, 0.9)
        pn[va] = 0.98 * sg(Z[va] / T) + 0.02 * 0.548
    return ll(y, pn)
pil = [n for n in PIL if n in oofs]; base = nested(pil); print(f'pillars ({len(pil)} arms): nested {base:.4f}')
for n in NEW:
    if n in oofs: print(f'  + {n:<16} nested {nested(pil + [n]):.4f}  ({nested(pil + [n]) - base:+.4f});  arm alone {ll(y, oofs[n].loc[uids, "p"].values):.4f}')
new = [n for n in NEW if n in oofs]
if len(new) > 1: print(f'  + all run-8 arms   nested {nested(pil + new):.4f}  ({nested(pil + new) - base:+.4f})')
print('GATE: an arm is in if the gain is <= -0.003 (laptop: python scripts/build_master_v3.py --src data/run8/artifacts_v3 --src data/master_pool/artifacts_v3)')
'''

cells = [{"type": "markdown", "source": "# DaT Parkinson's Challenge — run 8: ensemble-corrected members (nested-OOF soft targets, mistake weights, extra seeds) + Master v3 gate\n"
          f"Hemisphere members trained on soft targets y* = (1-{args.soft_lam})·y + {args.soft_lam}·p_nested (Master v2's nested out-of-fold probability, embedded below), "
          f"a mistake-weighted variant (w = 1 + {args.mistake_w}·nested loss), two extra plain hemisphere seeds, crop-hist on soft targets, full-data soft hemispheres; "
          "then the run-7 super-stack over every arm family found in the checkpoint dataset and the gate table for Master v3.\n"
          "Inputs: raw competition dataset (private), `dat-parkinsons-cache-v4`, `dat-run3-checkpoints` (resume + upload every 4 jobs via the KAGGLE_USERNAME/KAGGLE_KEY secrets)."}]
cells += base_cells
cells.append({"type": "code", "source": SOFT_CELL})
run_src = runner_cell(JOBS8, "(run 8)").replace("for gpu in (0, 1):", "for gpu in range(max(1, __import__('torch').cuda.device_count())):")
assert "device_count" in run_src
cells.append({"type": "code", "source": run_src})
cells.append({"type": "code", "source": "# re-OOF every CV checkpoint with inference-time TTA (2D: z-shift -1/0/+1 x flip; 3D: flip) so the stack is fitted on what main.py produces\n!python scripts/reoof_v3.py --art artifacts_v3 --cache cache_v4 --tta-z -1 0 1 2>&1 | grep -v Warning"})
cells.append({"type": "code", "source": STACK7.replace("__ARMS__", repr(ARMS_ALL)).replace("__EXTRAS__", repr(EXTRAS_ALL)).replace("--t-floor 1.0", "--t-floor 0.9").replace("runs 3-7", "runs 3-8")})
cells.append({"type": "code", "source": GATE})
cells.append({"type": "code", "source": (ROOT / "kaggle" / "run6a_crop3d" / "e2e_check_a.py").read_text(encoding="utf-8")})
cells.append({"type": "code", "source": (ROOT / "kaggle" / "split_zip_cell.py").read_text(encoding="utf-8")})
for c in cells:
    if c["source"].startswith(("# super-stack", "# GATE", "# === END-TO-END", "# === SPLIT", "# re-OOF")):
        c["source"] = failsafe(c["source"])

nb = nbformat.v4.new_notebook(); nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.cells = [nbformat.v4.new_markdown_cell(c["source"]) if c["type"] == "markdown" else nbformat.v4.new_code_cell(c["source"]) for c in cells]
nbformat.validate(nb); nbformat.write(nb, str(OUT / "notebook.ipynb"))
meta = {"id": f"{args.owner}/{args.slug}", "title": args.slug, "code_file": "notebook.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [args.cache_ds, args.raw_ds, args.ckpt_ds], "competition_sources": [], "kernel_sources": list(args.kernel_source), "model_sources": []}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
n_jobs = sum(len(j["seeds"]) * len(j.get("folds", range(5))) for j in JOBS8)
print(f"wrote {OUT / 'notebook.ipynb'} ({len(nb.cells)} cells, {n_jobs} training jobs) and kernel-metadata.json (id {meta['id']}; kernel sources {meta['kernel_sources']})")
