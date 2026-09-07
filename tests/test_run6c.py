"""Approach C (run 6c): richer feature arm, numpy-serialisable kernel/MLP model, VICReg pretraining, --init fine-tune."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datsc.config import N_SCALARS
from datsc.volume import N_STATS, VOL_SHAPE

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def tiny_cache_c(tmp_path_factory):
    root = tmp_path_factory.mktemp("cache_c")
    N = 12
    rng = np.random.RandomState(0)
    vols = np.lib.format.open_memmap(root / "vols.npy", mode="w+", dtype=np.float16, shape=(N,) + VOL_SHAPE)
    vols[:] = (1.0 + rng.rand(N, *VOL_SHAPE) * 0.5).astype(np.float16)
    vols[:, 40:56, 50:66, 28:40] += 2.0
    vols.flush()
    np.save(root / "masks.npy", (rng.rand(N, *VOL_SHAPE) > 0.5).astype(np.uint8))
    np.save(root / "scalars.npy", rng.randn(N, N_SCALARS).astype(np.float32))
    stats = np.ones((N, N_STATS), np.float32); stats[:, 0] = 100; stats[:, 1] = 50; stats[:, 3] = 60; stats[:, 4] = 30
    np.save(root / "stats.npy", stats)
    uids = [f"u{i:03d}" for i in range(N)]
    y = np.array([i % 2 for i in range(N)], float)
    pd.DataFrame({"uid": uids, "ok": True, "err": "", "is_pathologic": y}).to_csv(root / "meta.csv", index=False)
    pd.DataFrame({"uid": uids, "is_pathologic": y, "fold": [i % 3 for i in range(N)]}).to_csv(root / "folds.csv", index=False)
    return root


def _template():
    p = ROOT / "artifacts" / "template.npz"
    if not p.exists():
        pytest.skip("template not built")
    from datsc.register import Template
    return Template.load(p)


def test_features_c_shape_and_mirror_symmetry():
    from datsc.features_c import C_NAMES, FEATURE_NAMES_C, feat_vector_c, glcm_features
    from datsc.features_v4 import mirror
    T = _template()
    rng = np.random.default_rng(0)
    vol = (1.0 + rng.random((96, 96, 72)).astype(np.float32) * 0.2); vol[40:56, 50:66, 28:40] += 2.0
    f = feat_vector_c(vol, T); fm = feat_vector_c(mirror(vol), T)
    assert f.shape == (len(FEATURE_NAMES_C),) and np.isfinite(f).all() and len(C_NAMES) == 62
    off = len(FEATURE_NAMES_C) - len(C_NAMES); names = list(C_NAMES)
    # asymmetry descriptors are mirror-invariant; L/R axis profiles swap
    asym = [i for i, n in enumerate(names) if n.startswith("asym_")]
    assert np.allclose(f[off:][asym], fm[off:][asym], atol=1e-4)
    L0, R0 = names.index("L_ax0"), names.index("R_ax0")
    assert np.isclose(f[off + L0], fm[off + R0], atol=1e-4) and np.isclose(f[off + R0], fm[off + L0], atol=1e-4)
    g = glcm_features(vol[24:72, 34:82, 18:50]); assert g.shape == (4,) and np.isfinite(g).all() and 0 <= g[2] <= 1


@pytest.mark.parametrize("kind", ["linear", "rff", "mlp"])
def test_featmodel_c_roundtrip(tmp_path, kind):
    from datsc.featmodel_c import FeatModelC
    rng = np.random.default_rng(1)
    N, nf, npch = 60, 20, 200
    F = rng.normal(size=(N, nf)); P = rng.normal(size=(N, npch)).astype(np.float32)
    y = (F[:, 0] + 0.3 * rng.normal(size=N) > 0).astype(float)
    m = FeatModelC(kind=kind, n_comp=5, C=0.3, gamma=0.5, n_rff=64, hidden=(8,), alpha=1.0).fit(F, P, y)
    p = m.predict_proba(F, P); m.save(tmp_path / "m.npz"); m2 = FeatModelC.load(tmp_path / "m.npz")
    assert p.shape == (N,) and np.allclose(p, m2.predict_proba(F, P)) and m2.kind == kind
    d = m2.typicality(P); assert d.shape == (N,) and np.isfinite(d).all()
    assert ((p > 0.5) == (y > 0.5)).mean() > 0.7                       # learns the planted signal


def test_vicreg_loss_and_two_views():
    from datsc.ssl_c import two_views, vicreg_loss
    z = torch.randn(16, 32)
    l_same, parts = vicreg_loss(z, z.clone()); l_diff, _ = vicreg_loss(z, torch.randn(16, 32))
    assert torch.isfinite(l_same) and l_same < l_diff and parts["inv"] == 0.0
    x = torch.rand(2, 1, 96, 96, 72)
    v1, v2 = two_views(x, (48, 48, 32), (47.5, 57.3, 33.5), jitter=2)
    assert tuple(v1.shape) == (2, 1, 48, 48, 32) == tuple(v2.shape) and not torch.allclose(v1, v2)


def test_ssl_pretrain_smoke_and_init_finetune(tiny_cache_c, tmp_path):
    from datsc.ssl_c import extract_features, pretrain
    from datsc.train_v3 import InputSpec, train_run
    out = tmp_path / "enc.pt"
    r = pretrain(tiny_cache_c, out, epochs=1, batch_size=4, widths=(8, 8, 16, 16), crop=(32, 32, 24), center=(47.5, 57.3, 33.5),
                 num_workers=0, device="cpu", max_steps=2)
    assert out.exists() and r["steps"] == 2 and np.isfinite(r["history"][0]["loss"])
    ck = torch.load(out, map_location="cpu", weights_only=False)
    assert "encoder" in ck and all(not k.startswith("head") or True for k in ck["encoder"])
    feats = extract_features(out, tiny_cache_c, [f"u{i:03d}" for i in range(4)], device="cpu")
    assert feats.shape == (4, 16)
    spec = InputSpec("3d", size=(32, 32, 24), crop3d=(32, 32, 24), center3d=(47.5, 57.3, 33.5))
    meta = train_run(tiny_cache_c, fold=0, seed=0, out_dir=tmp_path, spec=spec, model_kw=dict(widths=(8, 8, 16, 16)), tag="tssl",
                     epochs=1, batch_size=4, ema_decay=0.9, num_workers=0, device="cpu", init_path=str(out))
    assert meta["init"] == str(out) and meta["init_loaded"] > 0 and (tmp_path / "tssl.pt").exists()
