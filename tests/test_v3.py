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


def test_augment_presets_shape_and_strength():
    from datsc.augment import PRESETS, train_augment
    x = torch.rand(2, 1, 96, 96, 72)
    torch.manual_seed(0); a = train_augment(x, "base")
    torch.manual_seed(0); b = train_augment(x, "strong")
    assert a.shape == x.shape == b.shape
    assert PRESETS["strong"]["blur_mm"] > PRESETS["base"]["blur_mm"] and PRESETS["strong"]["noise"] > PRESETS["base"]["noise"]
    assert torch.isfinite(a).all() and torch.isfinite(b).all() and (a >= 0).all()


def test_3d_crop_centred_on_striatum():
    x = torch.zeros(1, 1, 96, 96, 72); x[0, 0, 47, 57, 33] = 1.0          # a voxel at the template striatal centre
    spec = InputSpec("3d", size=(48, 48, 32), crop3d=(48, 48, 32), center3d=(47.5, 57.3, 33.5))
    inp = spec(x)
    assert tuple(inp.shape) == (1, 1, 48, 48, 32) and float(inp.sum()) == 1.0
    assert tuple(torch.nonzero(inp)[0, 2:].tolist()) == (23, 23, 15)          # crop spans x 24..71, y 34..81, z 18..49 (centre 47.5/57.5/33.5)
    d = spec.to_dict(); assert d["crop3d"] == [48, 48, 32] and InputSpec(**d).crop3d == (48, 48, 32)
    assert spec.flip(inp).shape == inp.shape


def test_train_one_epoch_with_z_jitter_and_strong_aug(tiny_cache, tmp_path):
    spec = InputSpec("2d", n_slabs=3, slab_mm=12, hw=32, z_center=36.0, tta_z=[-1, 0, 1])
    meta = train_run(tiny_cache, fold=0, seed=0, out_dir=tmp_path, spec=spec, model_kw=dict(backbone="tiny", n_in=3), tag="tj",
                     epochs=1, batch_size=4, ema_decay=0.9, num_workers=0, device="cpu", z_jitter=1, aug="strong")
    assert (tmp_path / "tj.pt").exists() and meta["z_jitter"] == 1 and meta["aug"] == "strong"


def test_feat_arm_features_and_model_roundtrip(tmp_path):
    from datsc.featmodel import FeatModel
    from datsc.features_v4 import FEATURE_NAMES, N_PATCH, feat_vector, mirror, striatal_patch
    from datsc.register import Template
    tpl_path = Path(__file__).resolve().parent.parent / "artifacts" / "template.npz"
    if not tpl_path.exists():
        pytest.skip("template not built")
    T = Template.load(tpl_path)
    rng = np.random.default_rng(0)
    vol = (1.0 + rng.random((96, 96, 72)).astype(np.float32) * 0.2)
    vol[40:56, 50:66, 28:40] += 2.0                                       # a bright striatal blob
    f = feat_vector(vol, T); p = striatal_patch(vol)
    assert f.shape == (len(FEATURE_NAMES),) and np.isfinite(f).all() and p.shape == (N_PATCH,)
    fm = feat_vector(mirror(vol), T)
    assert np.allclose(f[-2:], fm[-2:], atol=1e-4)                        # asymmetry features are mirror-invariant
    N = 40
    F = np.stack([f + rng.normal(0, 0.05, f.shape) for _ in range(N)]); P = np.stack([p + rng.normal(0, 0.05, p.shape) for _ in range(N)])
    y = (rng.random(N) > 0.5).astype(float); F[y == 1, 0] += 1.0
    m = FeatModel(n_comp=5, C=0.1).fit(F, P, y); m.save(tmp_path / "feat.npz")
    m2 = FeatModel.load(tmp_path / "feat.npz")
    assert np.allclose(m.predict_proba(F, P), m2.predict_proba(F, P)) and m2.predict_proba(F, P).shape == (N,)


def test_features_and_site_head():
    from datsc.models_v3 import Net2D, ResNet3D, SiteHead
    m3 = ResNet3D(widths=(8, 8, 16, 16)); x3 = torch.rand(2, 1, 32, 32, 24)
    h = m3.features(x3); assert h.shape == (2, 16) and m3.feat_dim == 16 and m3(x3).shape == (2,)
    m2 = Net2D(backbone="tiny", n_in=3); x2 = torch.rand(2, 3, 32, 32)
    h2 = m2.features(x2); assert h2.shape == (2, m2.feat_dim) and m2(x2).shape == (2,)
    sh = SiteHead(16, 5); out = sh(h, 0.5); assert out.shape == (2, 5)
    out.sum().backward()                                                 # gradient reversal must be differentiable
    assert all(p.grad is not None for p in m3.layers.parameters())


def test_train_one_epoch_dann_dro(tiny_cache, tmp_path):
    spec = InputSpec("3d", size=(32, 32, 24))
    meta = train_run(tiny_cache, fold=0, seed=0, out_dir=tmp_path, spec=spec, model_kw=dict(widths=(8, 8, 16, 16)), tag="td",
                     epochs=1, batch_size=4, ema_decay=0.9, num_workers=0, device="cpu", dann=0.1, dro=0.1)
    assert meta["dann"] == 0.1 and meta["dro"] == 0.1 and meta["n_sites"] >= 1 and np.isfinite(meta["history"][0]["train_loss"])
    sd = torch.load(tmp_path / "td.pt"); assert not any(k.startswith("site_head") for k in sd)

