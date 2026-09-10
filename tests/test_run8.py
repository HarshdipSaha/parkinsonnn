"""Run 8: ensemble-corrected members (nested-OOF soft targets, mistake weights), notebook generator, HOWTO."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from datsc.models_v3 import build_model
from datsc.train_v3 import InputSpec, logloss_np, soft_targets, train_run
from tests.test_v3 import tiny_cache  # noqa: F401  (fixture)


def _table(path: Path, uids, p, loss):
    pd.DataFrame({"uid": uids, "y": [0] * len(uids), "p": p, "loss": loss}).to_csv(path, index=False)
    return path


def test_soft_targets_and_weights(tmp_path):
    uids = ["u000", "u001", "u002", "u003"]; y = np.array([0.0, 1.0, 1.0, 0.0])
    t = _table(tmp_path / "soft.csv", ["u000", "u001", "u002"], [0.2, 0.9, 0.4], [0.1, 0.1, 1.9])   # u003 missing
    soft, w, n = soft_targets(t, uids, y, soft_lam=0.0, mistake_w=0.0); assert soft == {} and w == {} and n == 0
    soft, w, n = soft_targets(t, uids, y, soft_lam=0.4, mistake_w=0.0)
    assert n == 3 and "u003" not in soft and abs(soft["u000"] - 0.4 * 0.2) < 1e-9 and abs(soft["u001"] - (0.6 + 0.4 * 0.9)) < 1e-9
    soft, w, n = soft_targets(t, uids, y, soft_lam=0.4, mistake_w=2.0)
    raw = np.array([1 + 2 * 0.1, 1 + 2 * 0.1, 1 + 2 * 1.9, 1.0]); exp = raw / raw.mean()
    assert abs(np.mean([w[u] for u in uids]) - 1.0) < 1e-9 and np.allclose([w[u] for u in uids], exp)
    soft, w, n = soft_targets(None, uids, y, soft_lam=0.4, mistake_w=2.0); assert soft == {} and w == {}


def test_train_one_epoch_soft_mistake_weighted(tiny_cache, tmp_path):
    f = pd.read_csv(tiny_cache / "folds.csv", dtype={"uid": str})
    t = _table(tmp_path / "soft.csv", f.uid.tolist()[:-1], np.linspace(0.1, 0.9, len(f) - 1), np.linspace(0, 2, len(f) - 1))
    spec = InputSpec("3d", size=(48, 48, 32), crop3d=(48, 48, 32), center3d=(47.5, 47.5, 35.5))
    meta = train_run(tiny_cache, fold=0, seed=0, out_dir=tmp_path, spec=spec, model_kw=dict(arch="hemi", widths=(8, 8, 16, 16)), tag="t8",
                     epochs=1, batch_size=4, ema_decay=0.9, num_workers=0, device="cpu", mixup=0.2, dann=0.1, dro=0.1,
                     soft_csv=t, soft_lam=0.4, mistake_w=2.0)
    assert np.isfinite(meta["history"][0]["train_loss"]) and (tmp_path / "t8.pt").exists()
    j = json.loads((tmp_path / "t8.json").read_text())
    assert j["soft_lam"] == 0.4 and j["mistake_w"] == 2.0 and j["soft_csv"].endswith("soft.csv")
    n_train_in_table = len(set(f[f.fold != 0].uid) & set(pd.read_csv(t, dtype={"uid": str}).uid)); assert j["n_soft"] == n_train_in_table
    oof = pd.read_csv(tmp_path / "oof_t8.csv", dtype={"uid": str})            # validation is scored on the HARD labels
    assert set(oof.y.unique()) <= {0.0, 1.0} and abs(logloss_np(oof.y.values, oof.p.values) - j["val_logloss"]) < 1e-6
    m = build_model("3d", **{k: v for k, v in j["model_kw"].items() if k != "n_scalars"}); m.load_state_dict(torch.load(tmp_path / "t8.pt")); m.eval()
    with torch.no_grad():
        assert m(torch.rand(1, 1, 48, 48, 32).float()).shape == (1,)


def test_train_plain_path_unchanged(tiny_cache, tmp_path):
    """No table, kappa 0: the loss is the plain mean BCE (weights all 1), metadata records the defaults."""
    spec = InputSpec("3d", size=(48, 48, 32), crop3d=(48, 48, 32), center3d=(47.5, 47.5, 35.5))
    meta = train_run(tiny_cache, fold=0, seed=0, out_dir=tmp_path, spec=spec, model_kw=dict(arch="hemi", widths=(8, 8, 16, 16)), tag="t8p",
                     epochs=1, batch_size=4, ema_decay=0.9, num_workers=0, device="cpu")
    assert meta["soft_csv"] is None and meta["soft_lam"] == 0.0 and meta["mistake_w"] == 0.0 and meta["n_soft"] == 0


def test_run8_notebook_generates(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "kaggle" / "run8_distill" / "gen_run8.py"), "--out", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    import nbformat
    nb = nbformat.read(str(tmp_path / "notebook.ipynb"), as_version=4); nbformat.validate(nb)
    src = "\n".join(c.source for c in nb.cells)
    assert "soft_master_v2.csv" in src and "--soft-csv cache_v4/soft_master_v2.csv --soft-lam 0.4" in src and "--mistake-w 2.0" in src
    assert "device_count" in src and "r3d_hemisoft_f{f}_s{s}" in src and "r3d_crophistsoft" in src
    assert "42 training jobs" in r.stdout
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text()); assert meta["id"] == "skyfall23/dat-run8-distill" and meta["enable_gpu"]


def test_howto_pdf_one_page():
    pdf = ROOT / "kaggle" / "run8_distill" / "HOWTO.pdf"
    if not pdf.exists():
        pytest.skip("HOWTO.pdf not built")
    from pypdf import PdfReader
    assert len(PdfReader(str(pdf)).pages) == 1
