# Cells appended to the `dat-parkinsons-train` Kaggle notebook after training + packing.
# Run them once training has finished (they use submission.zip produced by the pack cell).

# %% [E2E inference on the smoke-test set, exactly as the DrivenData container runs it]
# unzip submission.zip into a clean dir -> run `python main.py` there with the smoke-test data as the data root
# -> validate the CSV -> score it against test_labels.csv (20 scans drawn from the TRAINING set, so optimistic).
import os, shutil, subprocess, time, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import log_loss

DS = Path("/kaggle/input/datasets/harshdipsaha/dat-parkinsons-challenge-data")

def build_data_root(dst, nifti_paths, fmt_df):
    """Recreate the runtime layout: <dst>/niftis/<uid>.nii(.gz) + <dst>/submission_format.csv"""
    dst = Path(dst); shutil.rmtree(dst, ignore_errors=True); (dst / "niftis").mkdir(parents=True)
    for p in nifti_paths:
        os.symlink(p, dst / "niftis" / p.name)
    fmt_df.to_csv(dst / "submission_format.csv", index=False)
    return dst

def run_main(data_root, tag):
    SIM = Path(f"/kaggle/working/code_execution_{tag}"); shutil.rmtree(SIM, ignore_errors=True); SIM.mkdir()
    with zipfile.ZipFile("submission.zip") as z:
        z.extractall(SIM)
    assert (SIM / "main.py").exists(), "main.py must be at the zip root"
    env = {**os.environ, "DATSC_DATA_ROOT": str(data_root), "DATSC_OUTPUT_CSV": "submission.csv"}
    t0 = time.time()
    r = subprocess.run(["python", "main.py"], cwd=SIM, env=env, capture_output=True, text=True)
    wall = time.time() - t0
    print(r.stdout[-2500:]); print(r.stderr[-1500:])
    print(f"exit={r.returncode}  wall={wall:.0f}s")
    assert r.returncode == 0
    return SIM, wall, r.stdout

smoke_fmt = next(p for p in DS.rglob("submission_format.csv"))
SMOKE_SRC = smoke_fmt.parent
smoke_niftis = sorted(p for p in SMOKE_SRC.rglob("*") if p.name.endswith((".nii", ".nii.gz")))
fmt = pd.read_csv(smoke_fmt, dtype={"uid": str})
print("smoke-test source:", SMOKE_SRC, "| volumes:", len(smoke_niftis), "| rows in submission_format:", len(fmt))
ROOT_SMOKE = build_data_root("/kaggle/working/sim_data_smoke", smoke_niftis, fmt)

SIM, wall, out = run_main(ROOT_SMOKE, "smoke")
sub = pd.read_csv(SIM / "submission.csv", dtype={"uid": str})
assert list(sub.columns) == ["uid", "is_pathologic"], sub.columns
assert list(sub.uid) == list(fmt.uid), "uid order/set mismatch vs submission_format.csv"
assert sub.is_pathologic.notna().all() and sub.is_pathologic.between(0, 1).all()
print(f"FORMAT OK: {sub.shape}, min={sub.is_pathologic.min():.4f} max={sub.is_pathologic.max():.4f}")
lab_path = next(iter(SMOKE_SRC.rglob("test_labels.csv")), None)
if lab_path is not None:
    lab = pd.read_csv(lab_path, dtype={"uid": str}).set_index("uid").loc[sub.uid]
    y = lab.is_pathologic.values.astype(float); p = sub.is_pathologic.values
    print(f"smoke-test log loss = {log_loss(y, p):.4f}   (prior-only = {log_loss(y, np.full(len(y), 0.548)):.4f}, n={len(y)})")
    print(f"smoke-test accuracy@0.5 = {((p > 0.5) == (y > 0.5)).mean():.3f}")
print(sub.head(8).to_string(index=False))
os.makedirs("/kaggle/working/out", exist_ok=True)
shutil.copy(SIM / "submission.csv", "/kaggle/working/out/submission_smoke.csv")
print(f"\nSMOKE-TEST BUDGET: {wall:.0f}s of the 360s smoke-test limit (includes imports + 20 volumes)")

# %% [3-hour budget check: time main.py on N training volumes and extrapolate to 2,000 test volumes]
N = 150
from datsc.config import find_data_dir, find_nifti_path
d = find_data_dir()
labels = pd.read_csv(d.labels_csv, dtype={"uid": str}).sample(n=N, random_state=1)
paths = [find_nifti_path(d.niftis_dir, u) for u in labels.uid]
fmt_t = pd.DataFrame({"uid": labels.uid.values, "is_pathologic": 0.5})
ROOT_T = build_data_root("/kaggle/working/sim_data_timing", paths, fmt_t)
SIM_T, wall_t, out_t = run_main(ROOT_T, "timing")
import re
m = re.search(r"preprocessed (\d+)/(\d+) volumes in (\d+)s", out_t)
prep_s = float(m.group(3)) if m else float("nan")
n_workers_here = max(1, min(20, (os.cpu_count() or 4) - 2))
per_vol_cpu_s = prep_s * n_workers_here / N
est_prep_real = per_vol_cpu_s * 2000 / 20
est_rest = (wall_t - prep_s) * 2000 / N
print(f"\nhere: {N} volumes -> prep {prep_s:.0f}s with {n_workers_here} workers ({per_vol_cpu_s:.1f} CPU-s/volume), total wall {wall_t:.0f}s")
print(f"EXTRAPOLATED for 2,000 test volumes on the real box: prep ~{est_prep_real/60:.0f} min + models ~{est_rest/60:.0f} min "
      f"= ~{(est_prep_real+est_rest)/60:.0f} min  (limit 180 min; ladder cuts CNN stage at 130 min)")
sub_t = pd.read_csv(SIM_T / "submission.csv", dtype={"uid": str})
lab_t = labels.set_index("uid").loc[sub_t.uid]
print(f"sanity log loss on these {N} TRAINING volumes (in-sample, optimistic) = {log_loss(lab_t.is_pathologic.values.astype(float), sub_t.is_pathologic.values):.4f}")
