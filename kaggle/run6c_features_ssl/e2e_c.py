# === END-TO-END CHECK of submission_v3.zip exactly as the DrivenData container runs it (+ per-arm smoke-set scores) ===
import json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import log_loss

DS = Path(RAW)
def build_data_root(dst, nifti_paths, fmt_df):
    dst = Path(dst); shutil.rmtree(dst, ignore_errors=True); (dst / "niftis").mkdir(parents=True)
    for p in nifti_paths:
        os.symlink(p, dst / "niftis" / p.name)
    fmt_df.to_csv(dst / "submission_format.csv", index=False); return dst
def run_main(zip_path, data_root, tag):
    SIM = Path(f"/kaggle/working/code_execution_{tag}"); shutil.rmtree(SIM, ignore_errors=True); SIM.mkdir()
    with zipfile.ZipFile(zip_path) as z: z.extractall(SIM)
    assert (SIM / "main.py").exists()
    env = {**os.environ, "DATSC_DATA_ROOT": str(data_root), "DATSC_OUTPUT_CSV": "submission.csv"}
    t0 = time.time(); r = subprocess.run(["python", "main.py"], cwd=SIM, env=env, capture_output=True, text=True); wall = time.time() - t0
    print(r.stdout[-2500:]); print(r.stderr[-1500:]); print(f"exit={r.returncode} wall={wall:.0f}s"); assert r.returncode == 0
    return SIM, wall, r.stdout
smoke_fmt = next(p for p in DS.rglob("submission_format.csv")); SRC = smoke_fmt.parent
niftis = sorted(p for p in SRC.rglob("*") if p.name.endswith((".nii", ".nii.gz")))
fmt = pd.read_csv(smoke_fmt, dtype={"uid": str})
ROOT_S = build_data_root("/kaggle/working/sim_data_smoke", niftis, fmt)
SIM, wall, out = run_main("submission_v3.zip", ROOT_S, "smoke_v3")
import re
used = re.findall(r"arm (\S+): checkpoints used (\d+)/(\d+)", out)
assert used and all(int(u) == int(m) for _, u, m in used), f"ARM CHECKPOINTS NOT ALL USED: {used}"
print("all arms used every checkpoint:", [(a, u) for a, u, _ in used])
sub = pd.read_csv(SIM / "submission.csv", dtype={"uid": str})
assert list(sub.columns) == ["uid", "is_pathologic"] and list(sub.uid) == list(fmt.uid) and sub.is_pathologic.between(0, 1).all()
lab = pd.read_csv(next(SRC.rglob("test_labels.csv")), dtype={"uid": str}).set_index("uid").loc[sub.uid]
y = lab.is_pathologic.values.astype(float); p = sub.is_pathologic.values
print(f"FORMAT OK {sub.shape} | smoke-test log loss = {log_loss(y, p):.4f} (prior-only {log_loss(y, np.full(len(y), 0.548)):.4f}) | acc@0.5 = {((p>0.5)==(y>0.5)).mean():.3f} | {wall:.0f}s of 360s")
# --- approach C: per-arm smoke-set log loss. The 20 labelled smoke scans are an OUT-OF-DISTRIBUTION check only (never fitted on);
#     the smoke -> public ladder so far: run 2 0.2567 -> 0.3391, run 3 0.2397 -> 0.2817, run 4a 0.2176/0.2263 -> 0.2667 ---
from datsc import infer_v3 as _inf
man = json.loads((SIM / "artifacts" / "manifest.json").read_text())
uids = list(fmt.uid); t0 = time.time()
vols, sc, st, ok = _inf.preprocess_all(uids, ROOT_S / "niftis", 4, deadline_s=900, t0=t0, template_path=SIM / "artifacts" / "template.npz")
probs, rows = {}, []
for a in man["arms"]:
    pa = _inf.run_arm(a, SIM / "artifacts", vols, sc, st, ok, t0)
    if pa is None:
        continue
    probs[a["name"]] = pa
    pa_ = np.clip(np.nan_to_num(pa, nan=0.548), 1e-6, 1 - 1e-6)
    rows.append((a["name"], float(man["weights"].get(a["name"], 0.0)), log_loss(y, pa_), float(((pa_ > 0.5) == (y > 0.5)).mean())))
print("smoke-set per arm (name, stack weight, log loss, acc):")
for r in rows: print(f"  {r[0]:<12} w={r[1]:.3f}  logloss={r[2]:.4f}  acc={r[3]:.2f}")
pst = _inf.combine(probs, man["weights"], float(man["T"]), float(man["lam"]), float(man.get("pi", 0.548)), clip=man.get("clip"), hedge=man.get("hedge"))
print(f"smoke-set stack logloss={log_loss(y, pst):.4f}  (nested OOF {man.get('oof_logloss', float('nan')):.4f}; expect public ~= smoke + 0.04..0.05)")
json.dump({"per_arm": rows, "stack": float(log_loss(y, pst))}, open("out/smoke_scores.json", "w"))
# 3-hour budget: 150 training volumes on this 4-CPU box, extrapolated to 2,000 on the 24-vCPU/A100 box
from datsc.config import find_data_dir, find_nifti_path
d = find_data_dir(); labels = pd.read_csv(d.labels_csv, dtype={"uid": str}).sample(n=150, random_state=1)
ROOT_T = build_data_root("/kaggle/working/sim_data_timing", [find_nifti_path(d.niftis_dir, u) for u in labels.uid], pd.DataFrame({"uid": labels.uid.values, "is_pathologic": 0.5}))
SIM_T, wall_t, out_t = run_main("submission_v3.zip", ROOT_T, "timing_v3")
m = re.search(r"preprocessed (\d+)/(\d+) volumes in (\d+)s", out_t); prep = float(m.group(3))
nw = max(1, min(20, (os.cpu_count() or 4) - 2)); cpu_s = prep * nw / 150
print(f"EXTRAPOLATED 2,000 volumes: prep ~{cpu_s*2000/20/60:.0f} min + models ~{(wall_t-prep)*2000/150/60:.0f} min (T4->A100 is faster) | limit 180 min")
sub_t = pd.read_csv(SIM_T / "submission.csv", dtype={"uid": str}); lab_t = labels.set_index("uid").loc[sub_t.uid]
print(f"in-sample log loss on 150 training volumes = {log_loss(lab_t.is_pathologic.values.astype(float), sub_t.is_pathologic.values):.4f}")
