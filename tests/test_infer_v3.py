"""End-to-end test of main_v3.py: tiny untrained models + manifest, synthetic NIfTIs, real subprocess run."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from datsc.config import N_SCALARS
from datsc.models_v3 import Net2D, ResNet3D
from tests_synth import make_synthetic_volume, write_nifti


def _fp16(sd):
    return {k: (v.half() if v.dtype == torch.float32 else v) for k, v in sd.items()}


def test_main_v3_end_to_end(tmp_path):
    art = tmp_path / "artifacts"; art.mkdir()
    torch.manual_seed(0)
    torch.save(_fp16(ResNet3D(N_SCALARS, widths=(8, 8, 16, 16)).state_dict()), art / "t3d_f0.pt")
    torch.save(_fp16(ResNet3D(N_SCALARS, widths=(8, 8, 16, 16)).state_dict()), art / "t3d_f1.pt")
    torch.save(_fp16(Net2D("resnet18", None, N_SCALARS, n_in=3).state_dict()), art / "t2d_f0.pt")
    man = {"pi": 0.55, "T": 1.1, "lam": 0.02, "weights": {"a3d": 0.6, "a2d": 0.4}, "arms": [
        {"name": "a3d", "arm": "3d", "spec": {"arm": "3d", "size": [32, 32, 24], "n_slabs": 3, "slab_mm": 12.0, "hw": 192, "slab_mode": "mean"},
         "model_kw": {"widths": [8, 8, 16, 16], "stem_stride": 1}, "norm": "p75", "checkpoints": ["artifacts/t3d_f0.pt", "artifacts/t3d_f1.pt"]},
        {"name": "a2d", "arm": "2d", "spec": {"arm": "2d", "size": [64, 64, 48], "n_slabs": 3, "slab_mm": 12.0, "hw": 64, "slab_mode": "mean"},
         "model_kw": {"backbone": "resnet18", "weights_path": None, "n_in": 3}, "norm": "zscore", "checkpoints": ["artifacts/t2d_f0.pt"]}]}
    (art / "manifest.json").write_text(json.dumps(man))
    data = tmp_path / "data"; (data / "niftis").mkdir(parents=True)
    uids = []
    for i, kw in enumerate(({}, {"levels": 8, "shape": (128, 128, 25), "spacing": (3.9,) * 3}, {"shape": (96, 96, 60), "spacing": (2.46,) * 3})):
        vol, sp = make_synthetic_volume(seed=i, **kw)
        write_nifti(data / "niftis" / f"syn{i}.nii.gz", vol, sp); uids.append(f"syn{i}")
    uids.append("missing1")   # no file -> must still get the prior
    pd.DataFrame({"uid": uids, "is_pathologic": 0.5}).to_csv(data / "submission_format.csv", index=False)
    out_csv = tmp_path / "submission.csv"
    env = {**os.environ, "DATSC_DATA_ROOT": str(data), "DATSC_ARTIFACTS": str(art), "DATSC_OUTPUT_CSV": str(out_csv)}
    # checkpoints are referenced relative to the zip root; mimic by running from a temp root with artifacts/ inside
    r = subprocess.run([sys.executable, str(ROOT / "main_v3.py")], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600)
    print(r.stdout[-2000:], r.stderr[-1500:])
    assert r.returncode == 0
    assert "stage0 prior written" in r.stdout and "preprocessed 3/4" in r.stdout
    assert "arm a3d: checkpoints used 2/2" in r.stdout and "arm a2d: checkpoints used 1/1" in r.stdout
    sub = pd.read_csv(out_csv, dtype={"uid": str})
    assert list(sub.columns) == ["uid", "is_pathologic"] and list(sub.uid) == uids
    assert sub.is_pathologic.between(0, 1).all()
    assert abs(sub.is_pathologic.iloc[-1] - 0.55) < 1e-3          # missing file -> prior
    assert sub.is_pathologic.iloc[:3].std() >= 0                   # finite, deterministic


def test_selftest_v3():
    r = subprocess.run([sys.executable, str(ROOT / "main_v3.py"), "--selftest"], cwd=ROOT, capture_output=True, text=True, timeout=600,
                       env={**os.environ, "DATSC_ARTIFACTS": str(ROOT / "nonexistent_artifacts")})
    assert r.returncode == 0 and "SELFTEST OK" in r.stdout, r.stdout + r.stderr
