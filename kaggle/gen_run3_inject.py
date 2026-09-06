"""Generate .playwright-mcp/inject_run3.js: wipes the notebook and injects the run-3 cell set.

Cells: markdown header, env, %%writefile for every package file, selftest, cache/pretrained setup, timing,
resumable 2-GPU job runner (job list in JOBS), stack+pack, end-to-end smoke check. Edit JOBS then re-run:
    python kaggle/gen_run3_inject.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = [*sorted(ROOT.glob("datsc/*.py")), *sorted(ROOT.glob("scripts/*.py")), ROOT / "main_v3.py", ROOT / "tests_synth.py"]
FILES = [p for p in FILES if p.name != "__init__.py" or p.parent.name == "datsc"]

CACHE_DS = "/kaggle/input/datasets/harshdipsaha/dat-parkinsons-cache-v4"
CKPT_DS = "harshdipsaha/dat-run3-checkpoints"
RAW_DS = "/kaggle/input/datasets/harshdipsaha/dat-parkinsons-challenge-data"

# (tag_prefix, command args) -- one entry per (fold, seed) is expanded by the runner
JOBS = [
    {"name": "r2d_effb0", "seeds": [0, 1, 2], "folds_csv": "folds.csv", "args": "--arm 2d --backbone efficientnet_b0 --weights pretrained/efficientnet_b0_in1k.pt --n-slabs 3 --slab-mm 8 --hw 192 --epochs 30 --batch-size 32 --lr 5e-4 --backbone-lr 2e-4 --mixup 0.2 --ema 0.99 --z-center {ZC}"},
    {"name": "r3d", "seeds": [0, 1, 2], "folds_csv": "folds.csv", "args": "--arm 3d --size 64 64 48 --widths 24 48 96 192 --epochs 50 --batch-size 16 --lr 5e-4 --mixup 0.2 --ema 0.995"},
    {"name": "r2d_effb0_site", "seeds": [0], "folds_csv": "folds_site.csv", "args": "--arm 2d --backbone efficientnet_b0 --weights pretrained/efficientnet_b0_in1k.pt --n-slabs 3 --slab-mm 8 --hw 192 --epochs 30 --batch-size 32 --lr 5e-4 --backbone-lr 2e-4 --mixup 0.2 --ema 0.99 --z-center {ZC}"},
    {"name": "r2d_effv2s", "seeds": [0], "folds_csv": "folds.csv", "args": "--arm 2d --backbone efficientnet_v2_s --weights pretrained/efficientnet_v2_s_in1k.pt --n-slabs 3 --slab-mm 8 --hw 224 --epochs 30 --batch-size 24 --lr 5e-4 --backbone-lr 1e-4 --mixup 0.2 --ema 0.99 --z-center {ZC}"},
]
JOBS2 = [
    {"name": "r2d_effb0_cor", "seeds": [0], "folds_csv": "folds.csv", "args": "--arm 2d --backbone efficientnet_b0 --weights pretrained/efficientnet_b0_in1k.pt --n-slabs 3 --slab-mm 8 --hw 192 --epochs 30 --batch-size 32 --lr 5e-4 --backbone-lr 2e-4 --mixup 0.2 --ema 0.99 --plane coronal --z-center {YC}"},
    {"name": "r2d_effb0_full", "seeds": [0, 1, 2], "folds": [-1], "folds_csv": "folds.csv", "args": "--arm 2d --backbone efficientnet_b0 --weights pretrained/efficientnet_b0_in1k.pt --n-slabs 3 --slab-mm 8 --hw 192 --epochs 30 --batch-size 32 --lr 5e-4 --backbone-lr 2e-4 --mixup 0.2 --ema 0.99 --z-center {ZC}"},
]
B0 = "--arm 2d --backbone efficientnet_b0 --weights pretrained/efficientnet_b0_in1k.pt --n-slabs 3 --slab-mm 8 --hw 192 --epochs 30 --batch-size 32 --lr 5e-4 --backbone-lr 2e-4 --mixup 0.2 --ema 0.99 --z-center {ZC}"
V2S = "--arm 2d --backbone efficientnet_v2_s --weights pretrained/efficientnet_v2_s_in1k.pt --n-slabs 3 --slab-mm 8 --hw 224 --epochs 30 --batch-size 24 --lr 5e-4 --backbone-lr 1e-4 --mixup 0.2 --ema 0.99 --z-center {ZC}"
# run 4a (Sept 6): site-robust candidates. Public 0.2817 vs OOF 0.258 = test-site shift (site-fold B0 0.2818 == public), so every
# member here targets robustness: unrealistic augmentation + slab jitter (Hamburg 2024), tiny slab net, striatal 3D crop at 2 mm,
# and diversity (V2-S seeds, ConvNeXt, coronal V2-S, 224-px central crop). Long jobs first so both GPUs stay busy.
JOBS3 = [
    {"name": "r2d_effv2s", "seeds": [1, 2], "folds_csv": "folds.csv", "args": V2S},
    {"name": "r2d_cnx", "seeds": [0], "folds_csv": "folds.csv", "args": "--arm 2d --backbone convnext_tiny --weights pretrained/convnext_tiny_in1k.pt --n-slabs 3 --slab-mm 8 --hw 224 --epochs 30 --batch-size 24 --lr 5e-4 --backbone-lr 1e-4 --mixup 0.2 --ema 0.99 --z-center {ZC}"},
    {"name": "r2d_effv2s_cor", "seeds": [0], "folds_csv": "folds.csv", "args": V2S + " --plane coronal --z-center {YC}"},
    {"name": "r3d_crop", "seeds": [0], "folds_csv": "folds.csv", "tag": "r3d_crop48x48x32_f{f}_s{s}", "args": "--arm 3d --size 48 48 32 --crop3d 48 48 32 --center3d {XC} {YC} {ZC} --widths 24 48 96 192 --epochs 50 --batch-size 16 --lr 5e-4 --mixup 0.2 --ema 0.995"},
    {"name": "r2d_effb0_saug", "seeds": [0], "folds_csv": "folds.csv", "tag": "r2d_effb0_saug_f{f}_s{s}", "args": B0 + " --aug strong --z-jitter 1"},
    {"name": "r2d_effb0_crop64", "seeds": [0], "folds_csv": "folds.csv", "tag": "r2d_effb0_crop64_f{f}_s{s}", "args": B0.replace("--hw 192", "--hw 224") + " --crop 64"},
    {"name": "r2d_effb0_saug_site", "seeds": [0], "folds_csv": "folds_site.csv", "tag": "r2d_effb0_saug_f{f}_s{s}_site", "args": B0 + " --aug strong --z-jitter 1"},
    {"name": "r2d_tiny", "seeds": [0, 1, 2, 3, 4], "folds_csv": "folds.csv", "args": "--arm 2d --backbone tiny --n-slabs 1 --slab-mm 12 --hw 72 --crop 72 --epochs 40 --batch-size 64 --lr 1e-3 --weight-decay 1e-2 --mixup 0.2 --aug strong --z-jitter 1 --ema 0.99 --z-center {ZC}"},
]
ARMS = [("a3d", "r3d_64x64x48_f?_s?", "r3d_64x64x48_f?_s?_site"), ("a3d_crop", "r3d_crop48x48x32_f?_s?", None),
        ("a2d_b0", "r2d_efficientnet_b0_f?_s?", "r2d_efficientnet_b0_f?_s?_site"), ("a2d_b0_saug", "r2d_effb0_saug_f?_s?", "r2d_effb0_saug_f?_s?_site"),
        ("a2d_b0_crop", "r2d_effb0_crop64_f?_s?", None), ("a2d_v2s", "r2d_efficientnet_v2_s_f?_s?", None), ("a2d_v2s_cor", "r2d_efficientnet_v2_s_cor_f?_s?", None),
        ("a2d_cnx", "r2d_convnext_tiny_f?_s?", None), ("a2d_cor", "r2d_efficientnet_b0_cor_f?_s?", None), ("a2d_tiny", "r2d_tiny_f?_s?", None), ("lr", "lrarm", "lrarm_site")]
EXTRAS = [("a3d", "r3d_64x64x48_fall_s*"), ("a2d_b0", "r2d_efficientnet_b0_fall_s*")]

cells = []
cells.append({"type": "markdown", "source": "# DaT Parkinson's Challenge — run 3 (whole-brain 2 mm cache, 3D ResNet + 2.5D ImageNet arms, stacked)\n"
              "Package files are written by the cells below (source of truth: private GitHub repo `parkinsonnn`, commit in RESULTS_LOG.md).\n"
              "Inputs: raw competition dataset (private) + `dat-parkinsons-cache-v4` (2 mm whole-brain cache built locally from the same data, plus torchvision ImageNet weights).\n"
              "Data-hygiene: no raw scan or crop is written to the notebook output; `out/` holds checkpoints, OOF csvs and the submission zip only."})
cells.append({"type": "code", "source": f"""!mkdir -p datsc scripts artifacts_v3 out
import os, torch, subprocess
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count(), 'cpus', os.cpu_count())
RAW = '{RAW_DS}'; CACHE_DS = '{CACHE_DS}'
print('raw:', os.listdir(RAW) if os.path.isdir(RAW) else 'MISSING')
print('cache ds:', os.listdir(CACHE_DS) if os.path.isdir(CACHE_DS) else 'MISSING (will build cache in-session)')
!nvidia-smi --query-gpu=name,memory.total --format=csv"""})
for p in FILES:
    rel = p.relative_to(ROOT).as_posix()
    cells.append({"type": "code", "source": f"%%writefile {rel}\n" + p.read_text(encoding="utf-8")})
cells.append({"type": "code", "source": "!touch scripts/__init__.py\n!python main_v3.py --selftest"})
cells.append({"type": "code", "source": f"""# cache_v4 (registered, reference-normalised 2 mm boxes) + template + ImageNet weights come from the uploaded dataset
import os, shutil, glob
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
for pat in ('lr_v3.json', 'oof_lrarm.csv', 'oof_lrarm_site.csv', 'lrarm.json'):      # registered-ROI LR arm (fitted locally)
    for f in glob.glob(CACHE_DS + '/**/' + pat, recursive=True): shutil.copy(f, 'artifacts_v3/')
print('cache_v4:', sorted(os.listdir('cache_v4'))); print('pretrained:', sorted(os.listdir('pretrained')))
# RESUME: if a previous version's output (or any dataset) mounted under /kaggle/input carries artifacts_v3/, copy it back so the
# runners skip finished jobs (chained commits = persistence without credentials)
os.makedirs('artifacts_v3', exist_ok=True); n_res = 0
for src_art in glob.glob('/kaggle/input/**/artifacts_v3', recursive=True) + glob.glob('/kaggle/input/**/out', recursive=True) + glob.glob('/kaggle/input/datasets/harshdipsaha/dat-run3-checkpoints*'):
    for f in glob.glob(src_art + '/*'):
        if os.path.isfile(f) and (f.endswith('.pt') or f.endswith('.json') or f.endswith('.csv')) and not os.path.exists('artifacts_v3/' + os.path.basename(f)):
            shutil.copy(f, 'artifacts_v3/'); n_res += 1
print('resumed artefacts copied from /kaggle/input:', n_res)
!ls -la cache_v4 | head -20"""})
cells.append({"type": "code", "source": """# timing check: 2 epochs of each arm on fold 0 (expect ~20-40 s/epoch on a T4)
!CUDA_VISIBLE_DEVICES=0 python scripts/train_v3.py --arm 3d --fold 0 --seed 99 --epochs 2 --out artifacts_timing --cache cache_v4 --size 64 64 48 --widths 24 48 96 192 2>&1 | grep -E "params|ep0|done"
!CUDA_VISIBLE_DEVICES=1 python scripts/train_v3.py --arm 2d --backbone efficientnet_b0 --weights pretrained/efficientnet_b0_in1k.pt --fold 0 --seed 99 --epochs 2 --out artifacts_timing --cache cache_v4 --batch-size 32 --z-center 33.5 2>&1 | grep -E "params|ep0|done"
!rm -rf artifacts_timing"""})
def runner_cell(jobs, label):
    lab = label or 'queue 1'
    return f"""# === RESUMABLE 2-GPU JOB RUNNER {label} === (skips jobs whose checkpoint already exists in artifacts_v3/)
import subprocess, time, os, glob, json
LABEL = {lab!r}
JOBS = {json.dumps(jobs, indent=1)}
import numpy as np
_sc = np.load('artifacts/template.npz')['striatal_center']; XC, YC, ZC = float(_sc[0]), float(_sc[1]), float(_sc[2]); print('template striatal x/y/z (vox):', XC, YC, ZC)
def expand(jobs):
    out = []
    for j in jobs:
        for s in j['seeds']:
            for f in j.get('folds', range(5)):
                tag = j.get('tag', '').format(f=f if f >= 0 else 'all', s=s)
                args = (j['args'].replace('{{ZC}}', f"{{ZC:.1f}}").replace('{{YC}}', f"{{YC:.1f}}").replace('{{XC}}', f"{{XC:.1f}}")
                        + f" --folds-csv {{j['folds_csv']}}" + (f" --tag {{tag}}" if tag else ''))
                out.append((j['name'], f, s, args, tag))
    return out
def tag_for(name, f, s, args, tag=''):
    if tag:
        return tag
    if '--arm 3d' in args:
        size = 'x'.join(args.split('--size')[1].split()[:3]); tag = f"r3d_{{size}}_f{{f if f >= 0 else 'all'}}_s{{s}}"
    else:
        bb = args.split('--backbone')[1].split()[0]; tag = f"r2d_{{bb}}{{'_cor' if '--plane coronal' in args else ''}}_f{{f if f >= 0 else 'all'}}_s{{s}}"
    return tag + ('_site' if 'folds_site' in args else '')
# --- persistence: push artifacts_v3 to the private dataset {CKPT_DS} after every 4 finished jobs (needs Kaggle Secrets) ---
import threading, shutil as _sh
def _creds():
    try:
        from kaggle_secrets import UserSecretsClient
        c = UserSecretsClient(); return c.get_secret('KAGGLE_USERNAME'), c.get_secret('KAGGLE_KEY')
    except Exception as e:
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
        json.dump({{'title': 'dat-run3-checkpoints', 'id': '{CKPT_DS}', 'licenses': [{{'name': 'other'}}]}}, open(os.path.join(d, 'dataset-metadata.json'), 'w'))
        env = {{**os.environ, 'KAGGLE_USERNAME': _KU, 'KAGGLE_KEY': _KK}}
        subprocess.run(['python', '-m', 'pip', 'install', '-q', '-U', 'kaggle'], capture_output=True, timeout=600)   # fine-grained tokens need client >= 2.x
        r = subprocess.run(['kaggle', 'datasets', 'version', '-p', d, '-m', note, '-q', '--dir-mode', 'skip'], env=env, capture_output=True, text=True, timeout=1800)
        print(f"[upload] {{note}}: rc={{r.returncode}} {{(r.stdout or r.stderr).strip()[-160:]}}", flush=True)
    except Exception as e:
        print('[upload] failed:', type(e).__name__, str(e)[:120], flush=True)
    finally:
        _up_lock.release()
queue = [(n, f, s, a, t) for n, f, s, a, t in expand(JOBS) if not os.path.exists(f"artifacts_v3/{{tag_for(n,f,s,a,t)}}.pt")]
print(f"{{len(queue)}} jobs to run, {{len(expand(JOBS)) - len(queue)}} already done")
running = {{}}; t0 = time.time(); done = []
while queue or running:
    for gpu in (0, 1):
        if gpu not in running and queue:
            n, f, s, a, t = queue.pop(0); tag = tag_for(n, f, s, a, t)
            log = open(f"log_{{tag}}.txt", "w")
            cmd = f"python scripts/train_v3.py --fold {{f}} --seed {{s}} --out artifacts_v3 --cache cache_v4 {{a}}"
            running[gpu] = (tag, subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, env={{**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}}))
            print(f"[gpu{{gpu}}] start {{tag}} at {{time.time()-t0:.0f}}s", flush=True)
    for gpu in list(running):
        tag, p = running[gpu]
        if p.poll() is not None:
            done.append((tag, p.returncode)); del running[gpu]
            last = subprocess.run(f"tail -n 1 log_{{tag}}.txt", shell=True, capture_output=True, text=True).stdout.strip()
            print(f"[gpu{{gpu}}] {{tag}} exit={{p.returncode}} at {{time.time()-t0:.0f}}s :: {{last[-120:]}}", flush=True)
            if len(done) % 4 == 0:
                threading.Thread(target=push_checkpoints, args=(f"{{LABEL}} {{len(done)}} jobs done",), daemon=True).start()
    time.sleep(10)
print(f"all done in {{(time.time()-t0)/60:.0f}} min; failures: {{[t for t,r in done if r]}}")
push_checkpoints(f"{{LABEL}} final ({{len(done)}} jobs)")"""
cells.append({"type": "code", "source": runner_cell(JOBS, "")})
cells.append({"type": "code", "source": runner_cell(JOBS2, "(queue 2)")})
cells.append({"type": "code", "source": runner_cell(JOBS3, "(queue 3, run 4a)")})
cells.append({"type": "code", "source": '# re-OOF every CV checkpoint with inference-time TTA (z-shift -1/0/+1 x flip) so the stack is fitted on what main.py produces\n!python scripts/reoof_v3.py --art artifacts_v3 --cache cache_v4 --tta-z -1 0 1 2>&1 | grep -v Warning'})
cells.append({"type": "code", "source": '# report + stacking + pack\nimport glob, json, os, subprocess, pandas as pd\nrows = [{\'run\': os.path.basename(f)[:-5], **{k: json.load(open(f)).get(k) for k in (\'best_val_logloss\', \'val_logloss\', \'val_auroc\')}} for f in sorted(glob.glob(\'artifacts_v3/r*_f*_s*.json\'))]\ndf = pd.DataFrame(rows); print(df.to_string(index=False))\ndf[\'arm\'] = df.run.str.replace(r\'_f(\\d|all)_s\\d\', \'\', regex=True); print(df.groupby(\'arm\')[[\'best_val_logloss\', \'val_logloss\', \'val_auroc\']].mean().round(4))\nlag = [(os.path.basename(f)[:-5], round(h[-1][\'val_logloss\'], 4), round(h[-1][\'val_raw_logloss\'], 4)) for f in sorted(glob.glob(\'artifacts_v3/r*_f*_s*.json\')) for h in [json.load(open(f)).get(\'history\', [])] if h and h[-1].get(\'val_raw_logloss\') is not None and h[-1][\'val_logloss\'] > h[-1][\'val_raw_logloss\'] + 0.02]\nprint(\'EMA-lag guard:\', \'OK\' if not lag else f\'WARNING {len(lag)} checkpoints lag the raw model by > 0.02: {lag[:5]}\')\nARMS = __ARMS__; EXTRAS = __EXTRAS__\ncmd = \'python scripts/fit_stack_v3.py --out artifacts_v3 \' + \' \'.join(f\'--arm {n} "{p}"\' for n, p, _ in ARMS if glob.glob(f\'artifacts_v3/oof_{p}.csv\'))\ncmd += \' \' + \' \'.join(f\'--site-arm {n} "{q}"\' for n, _, q in ARMS if q and glob.glob(f\'artifacts_v3/oof_{q}.csv\'))\ncmd += \' \' + \' \'.join(f\'--extra {n} "{g}"\' for n, g in EXTRAS if glob.glob(f\'artifacts_v3/{g}.pt\')) + \' --lam 0.02 --t-floor 1.0 --nested --grid 8 --cache cache_v4\'\nprint(cmd); r = subprocess.run(cmd, shell=True, capture_output=True, text=True); print(r.stdout[-4000:]); print(r.stderr[-800:])\n!python scripts/pack_v3.py --artifacts artifacts_v3 --out submission_v3.zip\n!cp submission_v3.zip artifacts_v3/manifest.json artifacts_v3/oof_stack.csv out/ && cp artifacts_v3/*.json artifacts_v3/oof_*.csv cache_v4/folds*.csv cache_v4/sites.csv out/ 2>/dev/null; du -sh out submission_v3.zip'.replace("__ARMS__", repr(ARMS)).replace("__EXTRAS__", repr(EXTRAS))})
cells.append({"type": "code", "source": (ROOT / "kaggle" / "e2e_check_v3.py").read_text(encoding="utf-8")})

cells.append({"type": "code", "source": (ROOT / "kaggle" / "split_zip_cell.py").read_text(encoding="utf-8")})

def failsafe(src):
    NL = chr(10)
    body = NL.join('    ' + l for l in src.split(NL))
    first = src.split(NL)[0]
    return first + NL + 'try:' + NL + body + NL + 'except Exception as _e:' + NL + "    import traceback; traceback.print_exc(); print('CELL FAILED (kept going):', _e)"
for c in cells:
    if c['source'].startswith(('# report + stacking', '# === END-TO-END', '# === SPLIT', '# re-OOF')):
        c['source'] = failsafe(c['source'])

js = """async (page) => {
  const cells = %s;
  const frame = page.frames().find(f => f.url().includes('jupyterlab'));
  return await frame.evaluate(({cells}) => {
    const nb = window.jupyterapp.shell.currentWidget.content; const model = nb.model; const sm = model.sharedModel;
    const n0 = model.cells.length;
    sm.transact(() => {
      if (n0 > 0) sm.deleteCellRange(0, n0);
      cells.forEach((c, i) => sm.insertCell(i, {cell_type: c.type, source: c.source}));
    });
    return {removed: n0, inserted: model.cells.length};
  }, {cells});
}
""" % json.dumps(cells)
out = ROOT / ".playwright-mcp" / "inject_run3.js"
out.write_text(js, encoding="utf-8")
runner_src = next(c["source"] for c in cells if c["source"].startswith("# === RESUMABLE"))
upd = """async (page) => {
  const src = %s;
  const frame = page.frames().find(f => f.url().includes('jupyterlab'));
  return await frame.evaluate((src) => {
    const model = window.jupyterapp.shell.currentWidget.content.model;
    for (let i = 0; i < model.cells.length; i++) {
      const c = model.cells.get(i);
      if (c.sharedModel.getSource().startsWith('# === RESUMABLE')) { c.sharedModel.setSource(src); return {updated: i}; }
    }
    return {updated: -1};
  }, src);
}
""" % json.dumps(runner_src)
(ROOT / ".playwright-mcp" / "update_runner.js").write_text(upd, encoding="utf-8")
print(f"wrote {out} with {len(cells)} cells ({len(FILES)} package files)")
