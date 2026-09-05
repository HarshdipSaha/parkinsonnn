import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from tests_synth import make_synthetic_volume, write_nifti

ROOT = Path(__file__).resolve().parents[1]


def _demo(tmp_path, n=4):
    d = tmp_path / "data"; (d / "niftis").mkdir(parents=True)
    uids = [f"t{i:07d}" for i in range(n)]
    for i, u in enumerate(uids):
        vol, sp = make_synthetic_volume(seed=i, levels=8 if i == 0 else None)
        write_nifti(d / "niftis" / f"{u}.nii.gz", vol, sp)
    pd.DataFrame({"uid": uids, "is_pathologic": 0.5}).to_csv(d / "submission_format.csv", index=False)
    return d, uids


def test_main_prior_path(tmp_path):
    d, uids = _demo(tmp_path)
    out = tmp_path / "submission.csv"
    env = {**os.environ, "DATSC_DATA_ROOT": str(d), "DATSC_OUTPUT_CSV": str(out),
           "DATSC_ARTIFACTS": str(tmp_path / "no_artifacts")}
    r = subprocess.run([sys.executable, "main.py"], env=env, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    sub = pd.read_csv(out, dtype={"uid": str})
    assert list(sub.columns) == ["uid", "is_pathologic"] and list(sub["uid"]) == uids
    assert ((sub["is_pathologic"] > 0) & (sub["is_pathologic"] < 1)).all()
    for u in uids:
        assert u not in r.stdout


def test_selftest():
    r = subprocess.run([sys.executable, "main.py", "--selftest"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "SELFTEST OK" in r.stdout
