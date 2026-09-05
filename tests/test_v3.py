"""CPU smoke tests for the run-3 stack: cache format, input derivation, both model arms, one training epoch."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datsc.config import N_SCALARS
from datsc.data_v3 import VolDataset, VolumeCache, to_3d_input, to_slabs
from datsc.models_v3 import Net2D, ResNet3D, count_params
from datsc.train_v3 import InputSpec, train_run
from datsc.volume import N_STATS, VOL_SHAPE, prepare_volume, renormalise
from tests_synth import make_synthetic_volume, write_nifti


@pytest.fixture(scope="module")
def tiny_cache(tmp_path_factory):
    root = tmp_path_factory.mktemp("cache_v3")
    N = 12
    rng = np.random.RandomState(0)
    vols = np.lib.format.open_memmap(root / "vols.npy", mode="w+", dtype=np.float16, shape=(N,) + VOL_SHAPE)
    vols[:] = rng.rand(N, *VOL_SHAPE).astype(np.float16) * 2
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


def test_prepare_volume_shapes(tmp_path):
    vol, sp = make_synthetic_volume(seed=1, shape=(128, 128, 25), spacing=(3.9,) * 3, levels=8)
    p = write_nifti(tmp_path / "s.nii.gz", vol, sp)
    r = prepare_volume(p)
    assert r["vol"].shape == VOL_SHAPE and r["vol"].dtype == np.float16
    assert r["mask"].shape == VOL_SHAPE and r["mask"].dtype == bool and r["mask"].any()
    assert r["scalars"].shape == (N_SCALARS,) and np.isfinite(r["scalars"]).all()
    assert r["stats"].shape == (N_STATS,) and r["stats"][0] > 0
    z = renormalise(r["vol"], r["stats"], "zscore"); assert np.isfinite(z).all()
    m = renormalise(r["vol"], r["stats"], "median"); assert m.max() >= r["vol"].astype(np.float32).max()


def test_dataset_and_inputs(tiny_cache):
    c = VolumeCache(tiny_cache)
    ds = VolDataset(c, c.meta["uid"], c.meta["is_pathologic"], "p75")
    x, s, y, u = ds[0]
    assert x.shape == (1,) + VOL_SHAPE and s.shape == (N_SCALARS,) and u == "u000"
    xb = x[None]
    assert to_3d_input(xb, (64, 64, 48)).shape == (1, 1, 64, 64, 48)
    assert to_3d_input(xb, VOL_SHAPE).shape == (1, 1) + VOL_SHAPE
    img = to_slabs(xb, n_slabs=3, slab_mm=12, hw=96)
    assert img.shape == (1, 3, 96, 96)
    # slab 1 == mean over the 6 central z-slices (12 mm / 2 mm), transposed to [Y, X]
    z0 = VOL_SHAPE[2] // 2 - 9
    ref = xb[0, 0, :, :, z0 + 6:z0 + 12].mean(-1).T
    ref = torch.nn.functional.interpolate(ref[None, None], size=(96, 96), mode="bilinear", align_corners=False)[0, 0]
    assert torch.allclose(img[0, 1], ref, atol=1e-5)
    assert to_slabs(xb, n_slabs=5, slab_mm=8, hw=64, mode="max").shape == (1, 5, 64, 64)


def test_models_forward():
    m3 = ResNet3D(n_scalars=N_SCALARS, widths=(8, 8, 16, 16))
    out = m3(torch.randn(2, 1, 32, 32, 24), torch.randn(2, N_SCALARS))
    assert out.shape == (2,) and count_params(m3) > 0
    m2 = Net2D("resnet18", None, n_scalars=N_SCALARS, n_in=3)
    assert m2(torch.rand(2, 3, 64, 64) * 3, torch.randn(2, N_SCALARS)).shape == (2,)
    m5 = Net2D("resnet18", None, n_scalars=N_SCALARS, n_in=5)
    assert m5(torch.rand(2, 5, 64, 64), torch.randn(2, N_SCALARS)).shape == (2,)
    m2.eval(); a = m2(torch.rand(1, 3, 64, 64), torch.zeros(1, N_SCALARS)); assert torch.isfinite(a).all()


@pytest.mark.parametrize("arm", ["3d", "2d"])
def test_train_one_epoch(tiny_cache, tmp_path, arm):
    if arm == "3d":
        spec = InputSpec("3d", size=(32, 32, 24)); kw = dict(widths=(8, 8, 16, 16))
    else:
        spec = InputSpec("2d", n_slabs=3, slab_mm=12, hw=64); kw = dict(backbone="resnet18", weights_path=None, n_in=3)
    meta = train_run(tiny_cache, fold=0, seed=0, out_dir=tmp_path, spec=spec, model_kw=kw, tag=f"t_{arm}", epochs=2,
                     batch_size=4, mixup=0.4, ema_decay=0.9, num_workers=0, device="cpu")
    assert (tmp_path / f"t_{arm}.pt").exists() and (tmp_path / f"oof_t_{arm}.csv").exists()
    oof = pd.read_csv(tmp_path / f"oof_t_{arm}.csv")
    assert len(oof) == 4 and oof["p"].between(0, 1).all()
    j = json.loads((tmp_path / f"t_{arm}.json").read_text())
    assert len(j["history"]) == 2 and j["spec"]["arm"] == arm
    sd = torch.load(tmp_path / f"t_{arm}.pt")
    assert all(v.dtype != torch.float32 for v in sd.values() if v.is_floating_point())   # saved fp16
