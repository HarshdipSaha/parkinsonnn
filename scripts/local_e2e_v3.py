"""Run a submission zip locally exactly as the DrivenData container does, on the smoke-test set, and score it.
    python scripts/local_e2e_v3.py <submission.zip> <smoke_dir>   (smoke_dir holds niftis/ or *.nii.gz, submission_format.csv, test_labels.csv)"""
from __future__ import annotations

import os, re, shutil, subprocess, sys, tempfile, time, zipfile
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.metrics import log_loss

zip_path, smoke = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
fmt_csv = next(smoke.rglob("submission_format.csv")); src = fmt_csv.parent
niftis = sorted(p for p in src.rglob("*") if p.name.endswith((".nii", ".nii.gz")))
work = Path(tempfile.mkdtemp(prefix="e2e_")); data = work / "data"; (data / "niftis").mkdir(parents=True)
for p in niftis:
    shutil.copy(p, data / "niftis" / p.name)
shutil.copy(fmt_csv, data / "submission_format.csv")
sim = work / "code_execution"; sim.mkdir()
with zipfile.ZipFile(zip_path) as z:
    z.extractall(sim)
assert (sim / "main.py").exists()
env = {**os.environ, "DATSC_DATA_ROOT": str(data), "DATSC_OUTPUT_CSV": "submission.csv"}
t0 = time.time(); r = subprocess.run([sys.executable, "main.py"], cwd=sim, env=env, capture_output=True, text=True); wall = time.time() - t0
print(r.stdout[-3000:]); print(r.stderr[-1500:]); print(f"exit={r.returncode} wall={wall:.0f}s")
assert r.returncode == 0
used = re.findall(r"arm (\S+): checkpoints used (\d+)/(\d+)", r.stdout)
print("checkpoints used per arm:", used)
assert used and all(int(u) == int(m) for _, u, m in used), "ARM CHECKPOINTS NOT ALL USED"
fmt = pd.read_csv(fmt_csv, dtype={"uid": str}); sub = pd.read_csv(sim / "submission.csv", dtype={"uid": str})
assert list(sub.columns) == ["uid", "is_pathologic"] and list(sub.uid) == list(fmt.uid) and sub.is_pathologic.between(0, 1).all()
lab = pd.read_csv(next(src.rglob("test_labels.csv")), dtype={"uid": str}).set_index("uid").loc[sub.uid]
y = lab.is_pathologic.values.astype(float); p = sub.is_pathologic.values
print(f"FORMAT OK {sub.shape} | smoke-test log loss = {log_loss(y, p):.4f} (prior-only {log_loss(y, np.full(len(y), 0.548)):.4f}) | acc@0.5 = {((p > 0.5) == (y > 0.5)).mean():.3f} | {wall:.0f}s")
print("work dir:", work)
