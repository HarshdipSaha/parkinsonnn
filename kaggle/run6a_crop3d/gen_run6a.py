"""Approach A (run 6a): generate a self-contained Kaggle notebook + kernel-metadata.json for the striatal-crop 3D maximiser.

    python kaggle/run6a_crop3d/gen_run6a.py                      # writes kaggle/run6a_crop3d/{notebook.ipynb, kernel-metadata.json}
    kaggle kernels push -p kaggle/run6a_crop3d                    # creates/updates AND runs the kernel (GPU, internet, 3 datasets)

Second Kaggle account: python kaggle/run6a_crop3d/gen_run6a.py --owner <user> [--cache-ds <user>/dat-parkinsons-cache-v4
--raw-ds <user>/dat-parkinsons-challenge-data --ckpt-ds <user>/dat-run3-checkpoints]  (only if the datasets were re-uploaded
rather than shared as collaborator; the checkpoint upload target must be writable by the account that runs the kernel).

The notebook: writes the package from this repo (cells), mounts the 2 mm registered cache + ImageNet weights, resumes every
checkpoint from the private checkpoint dataset, trains the run-6a jobs on all visible GPUs (list(range(torch.cuda.device_count()))),
uploads checkpoints every 4 jobs, re-OOFs with TTA, fits the nested stack (compositions enumeration, <= 10 arms), packs the zip,
runs the container-style end-to-end check with the smoke-set log loss of EVERY arm, and splits the zip into < 95 MB parts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(HERE))
ap.add_argument("--owner", default="harshdipsaha", help="Kaggle username that will own/run the kernel")
ap.add_argument("--cache-ds", default="harshdipsaha/dat-parkinsons-cache-v4")
ap.add_argument("--raw-ds", default="harshdipsaha/dat-parkinsons-challenge-data")
ap.add_argument("--ckpt-ds", default="harshdipsaha/dat-run3-checkpoints", help="private dataset that holds every checkpoint; the runner uploads to it (needs write access)")
ap.add_argument("--slug", default="dat-run6a-crop3d")
args = ap.parse_args()

CACHE_DS = f"/kaggle/input/datasets/{args.cache_ds}"
RAW_DS = f"/kaggle/input/datasets/{args.raw_ds}"
CKPT_DS = args.ckpt_ds
CKPT_ROOT = f"/kaggle/input/datasets/{args.ckpt_ds}"

FILES = [*sorted(ROOT.glob("datsc/*.py")), *sorted(ROOT.glob("scripts/*.py")), ROOT / "main_v3.py", ROOT / "tests_synth.py"]
FILES = [p for p in FILES if p.name != "__init__.py" or p.parent.name == "datsc"]

# ---------------- the run-6a job list (GPU-minutes on a T4 in brackets; ~560 total -> ~4.7 h on two T4s + ~35 min tail) ----------------
CROP48 = "--crop3d 48 48 32 --size 48 48 32 --center3d {XC} {YC} {ZC} --batch-size 16 --lr 5e-4 --mixup 0.2 --ema 0.995"
JOBS_A = [
    # bigger context: 64x64x40 voxels = 128x128x80 mm around the striatum, base arch, 60 epochs (~27 min/fold)
    {"name": "r3d_crop64", "seeds": [0], "folds_csv": "folds.csv", "script": "scripts/train_a.py", "tag": "r3d_crop64e60_f{f}_s{s}",
     "args": "--crop3d 64 64 40 --size 64 64 40 --center3d {XC} {YC} {ZC} --batch-size 16 --lr 5e-4 --mixup 0.2 --ema 0.995 --epochs 60"},
    # SE + stochastic-depth family, wider (32-256), 60 epochs (~20 min/fold)
    {"name": "r3d_se48", "seeds": [0], "folds_csv": "folds.csv", "script": "scripts/train_a.py", "tag": "r3d_se48_f{f}_s{s}",
     "args": CROP48 + " --arch se --widths 32 64 128 256 --blocks 1 --drop-path 0.1 --epochs 60"},
    # the run-4a net trained longer (its validation loss was still falling at epoch 50): 80 epochs, two new seeds (~20 min/fold)
    {"name": "r3d_crop48e80", "seeds": [1, 2], "folds_csv": "folds.csv", "script": "scripts/train_a.py", "tag": "r3d_crop48e80_f{f}_s{s}",
     "args": CROP48 + " --epochs 80"},
    # site-invariant variant (gradient-reversal site head + group-DRO); same tag/args as run 5 so a finished copy is reused (~17 min/fold)
    {"name": "r3d_cropinv", "seeds": [0], "folds_csv": "folds.csv", "script": "scripts/train_a.py", "tag": "r3d_cropinv_f{f}_s{s}",
     "args": CROP48 + " --epochs 50 --dann 0.1 --dro 0.1"},
    # full-data models of the 80-epoch net (no OOF; appended to the arm at inference through EXTRAS)
    {"name": "r3d_crop48e80_full", "seeds": [0, 1], "folds": [-1], "folds_csv": "folds.csv", "script": "scripts/train_a.py", "tag": "r3d_crop48e80_fall_s{s}",
     "args": CROP48 + " --epochs 80"},
]
ARMS_A = [("a3d_crop", "r3d_crop48x48x32_f?_s?", "r3d_crop48x48x32_f?_s?_site"), ("a3d_crop80", "r3d_crop48e80_f?_s?", None),
          ("a3d_se48", "r3d_se48_f?_s?", None), ("a3d_crop64", "r3d_crop64e60_f?_s?", None), ("a3d_cropinv", "r3d_cropinv_f?_s?", "r3d_cropinv_f?_s?_site"),
          ("a2d_v2s", "r2d_efficientnet_v2_s_f?_s?", None), ("a2d_cnx", "r2d_convnext_tiny_f?_s?", None), ("a2d_cor", "r2d_efficientnet_b0_cor_f?_s?", None),
          ("feat", "featarm", "featarm_site"), ("lr", "lrarm", "lrarm_site")]
EXTRAS_A = [("a3d_crop80", "r3d_crop48e80_fall_s*"), ("a3d_crop", "r3d_crop48x48x32_fall_s*"), ("a2d_v2s", "r2d_efficientnet_v2_s_fall_s*")]

cells = []
cells.append({"type": "markdown", "source": "# DaT Parkinson's Challenge — run 6a: striatal-crop 3D maximiser\n"
              "Approach A: the 48×48×32 striatal-crop 3D net is the best and least-correlated member of the run-4a stack (OOF 0.2568); this run trains a\n"
              "stronger family of it (80-epoch seeds, SE/stochastic-depth net, 64×64×40 context crop, site-invariant variant, full-data models) and\n"
              "stacks it with the kept 2.5D arms and the feature arm (nested estimate, ≤ 10 arms). Every arm's smoke-set log loss is printed at the end.\n"
              "Package files are written by the cells below (source of truth: the `parkinsonnn` repo, `kaggle/run6a_crop3d/`).\n"
              "Data hygiene: no raw scan is written to the output; `out/` holds checkpoints, OOF csvs and the submission zip only."})
cells.append({"type": "code", "source": f"""!mkdir -p datsc scripts artifacts_v3 out
import os, torch, subprocess
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count(), 'cpus', os.cpu_count())
RAW = '{RAW_DS}'; CACHE_DS = '{CACHE_DS}'; CKPT_ROOT = '{CKPT_ROOT}'
print('raw:', os.listdir(RAW) if os.path.isdir(RAW) else 'MISSING -- attach the competition dataset')
print('cache ds:', os.listdir(CACHE_DS) if os.path.isdir(CACHE_DS) else 'MISSING -- attach dat-parkinsons-cache-v4')
print('checkpoint ds:', len(os.listdir(CKPT_ROOT)) if os.path.isdir(CKPT_ROOT) else 'MISSING (will try the API download)')
!nvidia-smi --query-gpu=name,memory.total --format=csv"""})
for p in FILES:
    rel = p.relative_to(ROOT).as_posix()
    cells.append({"type": "code", "source": f"%%writefile {rel}\n" + p.read_text(encoding="utf-8")})
cells.append({"type": "code", "source": "!touch scripts/__init__.py\n!python main_v3.py --selftest"})
cells.append({"type": "code", "source": f"""# cache_v4 (registered, reference-normalised 2 mm boxes) + template + ImageNet weights from the cache dataset; resume everything else
import os, shutil, glob, numpy as np
src = os.path.dirname(next(iter(glob.glob(CACHE_DS + '/**/vols.npy', recursive=True)), ''))
assert src, 'cache dataset not mounted -- attach dat-parkinsons-cache-v4'
os.makedirs('cache_v4', exist_ok=True); os.makedirs('pretrained', exist_ok=True); os.makedirs('artifacts', exist_ok=True)
for f in os.listdir(src):
    dst = os.path.join('cache_v4', f)
    if not os.path.exists(dst):
        if f.endswith('.npy') and os.path.getsize(os.path.join(src, f)) > 50_000_000:
            os.symlink(os.path.join(src, f), dst)     # big arrays stay on the read-only mount (memmap)
        else:
            shutil.copy(os.path.join(src, f), dst)
for f in glob.glob(CACHE_DS + '/**/*.pt', recursive=True):
    if not os.path.exists('pretrained/' + os.path.basename(f)): shutil.copy(f, 'pretrained/')
tpl = next(iter(glob.glob(CACHE_DS + '/**/template.npz', recursive=True)))
shutil.copy(tpl, 'artifacts/template.npz'); shutil.copy(tpl, 'artifacts_v3/template.npz')
for f in glob.glob(CKPT_ROOT + '*/template.npz') + glob.glob(CKPT_ROOT + '/template.npz'):   # template with hist landmarks (run 5)
    shutil.copy(f, 'artifacts/template.npz'); shutil.copy(f, 'artifacts_v3/template.npz'); print('template from the checkpoint dataset (landmarks):', 'hist' in np.load(f).files)
for pat in ('lr_v3.json', 'oof_lrarm.csv', 'oof_lrarm_site.csv', 'lrarm.json', 'feat_v4.npz', 'oof_featarm.csv', 'oof_featarm_site.csv', 'featarm.json'):
    for f in glob.glob(CACHE_DS + '/**/' + pat, recursive=True): shutil.copy(f, 'artifacts_v3/')
# RESUME every checkpoint / OOF / json from the checkpoint dataset (and any mounted notebook output)
n_res = 0
for src_art in glob.glob('/kaggle/input/**/artifacts_v3', recursive=True) + glob.glob('/kaggle/input/**/out', recursive=True) + glob.glob(CKPT_ROOT + '*'):
    for f in glob.glob(src_art + '/*'):
        if os.path.isfile(f) and f.endswith(('.pt', '.json', '.csv', '.npz')) and not os.path.exists('artifacts_v3/' + os.path.basename(f)):
            shutil.copy(f, 'artifacts_v3/'); n_res += 1
print('resumed artefacts copied from /kaggle/input:', n_res)
if n_res == 0:      # fallback: pull the private checkpoint dataset with the API (needs the KAGGLE_USERNAME/KAGGLE_KEY secrets + internet)
    try:
        from kaggle_secrets import UserSecretsClient
        _u = UserSecretsClient(); os.environ['KAGGLE_USERNAME'] = _u.get_secret('KAGGLE_USERNAME'); os.environ['KAGGLE_KEY'] = _u.get_secret('KAGGLE_KEY')
        import subprocess as _sp
        _sp.run('pip -q install -U kaggle', shell=True, capture_output=True)
        _r = _sp.run('kaggle datasets download {CKPT_DS} -p /kaggle/working/ckpt_ds --unzip --force', shell=True, capture_output=True, text=True)
        print('checkpoint dataset download:', (_r.stdout + _r.stderr).strip()[-300:])
        for f in glob.glob('/kaggle/working/ckpt_ds/**/*', recursive=True):
            if os.path.isfile(f) and f.endswith(('.pt', '.json', '.csv', '.npz')) and not os.path.exists('artifacts_v3/' + os.path.basename(f)):
                shutil.copy(f, 'artifacts_v3/'); n_res += 1
        print('resumed artefacts from the checkpoint dataset download:', n_res)
    except Exception as _e:
        print('checkpoint dataset download skipped:', type(_e).__name__, str(_e)[:200])
print('checkpoints present:', len(glob.glob('artifacts_v3/*.pt')), '| crop-3D:', sorted(os.path.basename(f) for f in glob.glob('artifacts_v3/r3d_crop*.pt'))[:8])"""})
cells.append({"type": "code", "source": """# timing check: 2 epochs of the base and the SE crop nets on fold 0 (expect ~15 s and ~20 s per epoch on a T4)
import numpy as np
_sc = np.load('artifacts/template.npz')['striatal_center']; XC, YC, ZC = [float(v) for v in _sc]
!CUDA_VISIBLE_DEVICES=0 python scripts/train_a.py --fold 0 --seed 99 --epochs 2 --out artifacts_timing --cache cache_v4 --crop3d 48 48 32 --size 48 48 32 --center3d {XC} {YC} {ZC} 2>&1 | grep -E "params|ep0|done"
!CUDA_VISIBLE_DEVICES=0 python scripts/train_a.py --fold 0 --seed 99 --epochs 2 --out artifacts_timing --cache cache_v4 --crop3d 48 48 32 --size 48 48 32 --center3d {XC} {YC} {ZC} --arch se --widths 32 64 128 256 2>&1 | grep -E "params|ep0|done"
!rm -rf artifacts_timing"""})

RUNNER = r'''# === RESUMABLE MULTI-GPU JOB RUNNER __LABEL__ === (skips jobs whose checkpoint already exists in artifacts_v3/)
import subprocess, time, os, glob, json, threading, shutil as _sh
import numpy as np, torch
LABEL = __LABEL_REPR__
JOBS = __JOBS__
GPUS = list(range(torch.cuda.device_count())) or [0]          # one job per visible GPU; a single-GPU allocation just runs serially
_sc = np.load('artifacts/template.npz')['striatal_center']; XC, YC, ZC = float(_sc[0]), float(_sc[1]), float(_sc[2]); print('template striatal x/y/z (vox):', XC, YC, ZC, '| gpus', GPUS)
def expand(jobs):
    out = []
    for j in jobs:
        for s in j['seeds']:
            for f in j.get('folds', range(5)):
                tag = j['tag'].format(f=f if f >= 0 else 'all', s=s)
                args = (j['args'].replace('{ZC}', f"{ZC:.1f}").replace('{YC}', f"{YC:.1f}").replace('{XC}', f"{XC:.1f}")
                        + f" --folds-csv {j['folds_csv']} --tag {tag}")
                out.append((j.get('script', 'scripts/train_a.py'), f, s, args, tag))
    return out
# --- persistence: push artifacts_v3 to the private dataset __CKPT_DS__ after every 4 finished jobs (needs Kaggle Secrets) ---
def _creds():
    try:
        from kaggle_secrets import UserSecretsClient
        c = UserSecretsClient(); return c.get_secret('KAGGLE_USERNAME'), c.get_secret('KAGGLE_KEY')
    except Exception:
        return None, None
_KU, _KK = _creds(); _UPLOAD_OK = bool(_KU and _KK)
print('checkpoint upload:', 'enabled' if _UPLOAD_OK else 'disabled (no KAGGLE_USERNAME/KAGGLE_KEY secrets)')
_up_lock = threading.Lock()
def push_checkpoints(note):
    if not _UPLOAD_OK or not _up_lock.acquire(blocking=False):
        return
    try:
        d = '/kaggle/working/ckpt_upload'; os.makedirs(d, exist_ok=True)
        for f in glob.glob('artifacts_v3/*'):
            if os.path.isfile(f) and not os.path.exists(os.path.join(d, os.path.basename(f))): _sh.copy(f, d)
        json.dump({'title': __CKPT_TITLE__, 'id': __CKPT_DS_REPR__, 'licenses': [{'name': 'other'}]}, open(os.path.join(d, 'dataset-metadata.json'), 'w'))
        env = {**os.environ, 'KAGGLE_USERNAME': _KU, 'KAGGLE_KEY': _KK}
        subprocess.run(['python', '-m', 'pip', 'install', '-q', '-U', 'kaggle'], capture_output=True, timeout=600)
        r = subprocess.run(['kaggle', 'datasets', 'version', '-p', d, '-m', note, '-q', '--dir-mode', 'skip'], env=env, capture_output=True, text=True, timeout=1800)
        print(f"[upload] {note}: rc={r.returncode} {(r.stdout or r.stderr).strip()[-160:]}", flush=True)
    except Exception as e:
        print('[upload] failed:', type(e).__name__, str(e)[:120], flush=True)
    finally:
        _up_lock.release()
queue = [(sc, f, s, a, t) for sc, f, s, a, t in expand(JOBS) if not os.path.exists(f"artifacts_v3/{t}.pt")]
print(f"{len(queue)} jobs to run, {len(expand(JOBS)) - len(queue)} already done")
running = {}; t0 = time.time(); done = []
while queue or running:
    for gpu in GPUS:
        if gpu not in running and queue:
            sc, f, s, a, tag = queue.pop(0)
            log = open(f"log_{tag}.txt", "w")
            cmd = f"python {sc} --fold {f} --seed {s} --out artifacts_v3 --cache cache_v4 {a}"
            running[gpu] = (tag, subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}))
            print(f"[gpu{gpu}] start {tag} at {time.time()-t0:.0f}s", flush=True)
    for gpu in list(running):
        tag, p = running[gpu]
        if p.poll() is not None:
            done.append((tag, p.returncode)); del running[gpu]
            last = subprocess.run(f"tail -n 1 log_{tag}.txt", shell=True, capture_output=True, text=True).stdout.strip()
            print(f"[gpu{gpu}] {tag} exit={p.returncode} at {time.time()-t0:.0f}s :: {last[-120:]}", flush=True)
            if len(done) % 4 == 0:
                threading.Thread(target=push_checkpoints, args=(f"{LABEL} {len(done)} jobs done",), daemon=True).start()
    time.sleep(10)
print(f"all done in {(time.time()-t0)/60:.0f} min; failures: {[t for t,r in done if r]}")
push_checkpoints(f"{LABEL} final ({len(done)} jobs)")'''


def runner_cell(jobs, label):
    return (RUNNER.replace("__LABEL__", label).replace("__LABEL_REPR__", repr(label)).replace("__JOBS__", json.dumps(jobs, indent=1))
            .replace("__CKPT_DS__", CKPT_DS).replace("__CKPT_DS_REPR__", repr(CKPT_DS)).replace("__CKPT_TITLE__", repr(CKPT_DS.split("/")[-1])))


cells.append({"type": "code", "source": runner_cell(JOBS_A, "(run 6a)")})
cells.append({"type": "code", "source": "# re-OOF every CV checkpoint with inference-time TTA (2D: z-shift -1/0/+1 x flip; 3D: flip) so the stack is fitted on what main.py produces\n!python scripts/reoof_v3.py --art artifacts_v3 --cache cache_v4 --tta-z -1 0 1 2>&1 | grep -v Warning"})
STACK = r'''# report + stacking + pack
import glob, json, os, subprocess, pandas as pd
rows = [{'run': os.path.basename(f)[:-5], **{k: json.load(open(f)).get(k) for k in ('best_val_logloss', 'val_logloss', 'val_auroc')}} for f in sorted(glob.glob('artifacts_v3/r*_f*_s*.json'))]
df = pd.DataFrame(rows); print(df.to_string(index=False))
df['arm'] = df.run.str.replace(r'_f(\d|all)_s\d', '', regex=True); print(df.groupby('arm')[['best_val_logloss', 'val_logloss', 'val_auroc']].mean().round(4))
lag = [(os.path.basename(f)[:-5], round(h[-1]['val_logloss'], 4), round(h[-1]['val_raw_logloss'], 4)) for f in sorted(glob.glob('artifacts_v3/r*_f*_s*.json')) for h in [json.load(open(f)).get('history', [])] if h and h[-1].get('val_raw_logloss') is not None and h[-1]['val_logloss'] > h[-1]['val_raw_logloss'] + 0.02]
print('EMA-lag guard:', 'OK' if not lag else f'WARNING {len(lag)} checkpoints lag the raw model by > 0.02: {lag[:5]}')
ARMS = __ARMS__; EXTRAS = __EXTRAS__
cmd = 'python scripts/fit_stack_v3.py --out artifacts_v3 ' + ' '.join(f'--arm {n} "{p}"' for n, p, _ in ARMS if glob.glob(f'artifacts_v3/oof_{p}.csv'))
cmd += ' ' + ' '.join(f'--site-arm {n} "{q}"' for n, _, q in ARMS if q and glob.glob(f'artifacts_v3/oof_{q}.csv'))
cmd += ' ' + ' '.join(f'--extra {n} "{g}"' for n, g in EXTRAS if glob.glob(f'artifacts_v3/{g}.pt')) + ' --lam 0.02 --t-floor 1.0 --nested --typ --grid 8 --cache cache_v4'
print(cmd); r = subprocess.run(cmd, shell=True, capture_output=True, text=True); print(r.stdout[-5000:]); print(r.stderr[-800:])
!python scripts/pack_v3.py --artifacts artifacts_v3 --out submission_v3.zip
!cp submission_v3.zip artifacts_v3/manifest.json artifacts_v3/oof_stack.csv out/ && cp artifacts_v3/*.json artifacts_v3/oof_*.csv cache_v4/folds*.csv cache_v4/sites.csv out/ 2>/dev/null; du -sh out submission_v3.zip'''
cells.append({"type": "code", "source": STACK.replace("__ARMS__", repr(ARMS_A)).replace("__EXTRAS__", repr(EXTRAS_A))})
cells.append({"type": "code", "source": (HERE / "e2e_check_a.py").read_text(encoding="utf-8")})
cells.append({"type": "code", "source": (ROOT / "kaggle" / "split_zip_cell.py").read_text(encoding="utf-8")})


def failsafe(src):
    NL = chr(10)
    body = NL.join("    " + l for l in src.split(NL))
    first = src.split(NL)[0]
    return first + NL + "try:" + NL + body + NL + "except Exception as _e:" + NL + "    import traceback; traceback.print_exc(); print('CELL FAILED (kept going):', _e)"


for c in cells:
    if c["source"].startswith(("# report + stacking", "# === END-TO-END", "# === SPLIT", "# re-OOF")):
        c["source"] = failsafe(c["source"])

import nbformat
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
nb = nbformat.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.cells = [nbformat.v4.new_markdown_cell(c["source"]) if c["type"] == "markdown" else nbformat.v4.new_code_cell(c["source"]) for c in cells]
nbformat.validate(nb); nbformat.write(nb, str(out / "notebook.ipynb"))
meta = {"id": f"{args.owner}/{args.slug}", "title": args.slug, "code_file": "notebook.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": "true", "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true",
        "dataset_sources": [args.cache_ds, args.raw_ds, args.ckpt_ds], "competition_sources": [], "kernel_sources": [], "model_sources": []}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
n_jobs = sum(len(j["seeds"]) * len(j.get("folds", range(5))) for j in JOBS_A)
print(f"wrote {out / 'notebook.ipynb'} ({len(nb.cells)} cells, {len(FILES)} package files, {n_jobs} training jobs) and kernel-metadata.json (id {meta['id']})")
