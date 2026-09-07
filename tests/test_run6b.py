"""Run-6b: tri-planar input spec, model, spec factory, trainer/inference contract."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datsc.data_b import hemisphere_centres, to_slabs_sagittal
from datsc.models_b import TriPlanarNet
from datsc.models_v3 import build_model
from datsc.train_b import InputSpecB, make_spec
from datsc.train_v3 import InputSpec, train_run
from tests.test_v3 import tiny_cache  # noqa: F401  (module-scoped synthetic cache fixture)


def test_sagittal_slabs_and_hemisphere_centres():
    x = torch.zeros(1, 1, 96, 96, 72); x[0, 0, 36, 57, 33] = 1.0                    # left striatal voxel
    img = to_slabs_sagittal(x, n_slabs=3, slab_mm=8.0, hw=96, x_center=36.8)      # 3 slabs of 4 voxels centred on x=37: 31..42
    assert tuple(img.shape) == (1, 3, 96, 96) and float(img.sum()) > 0
    w = img[0, 1]; assert float(w.sum()) > 0 and float(img[0, 0].sum()) == 0 and float(img[0, 2].sum()) == 0   # middle slab only
    rr, cc = torch.meshgrid(torch.arange(96.0), torch.arange(96.0), indexing="ij")
    assert abs(float((w * rr).sum() / w.sum()) - 33 * 96 / 72) < 1.5 and abs(float((w * cc).sum() / w.sum()) - 57) < 1.5   # rows=z (resized 72->96), cols=y
    m = np.zeros((96, 96, 72), bool); m[30:46, 50:65, 28:40] = True; m[50:66, 50:65, 28:40] = True
    xl, xr = hemisphere_centres(m, 47.5); assert 37 < xl < 38 and 57 < xr < 58


def test_tri_spec_shapes_flip_and_roundtrip():
    spec = InputSpecB(n_slabs=3, slab_mm=8.0, hw=32, z_center=33.5, y_center=57.3, x_left=36.8, x_right=58.2, tta_z=[-1, 0, 1])
    x = torch.rand(2, 1, 96, 96, 72)
    inp = spec(x); assert tuple(inp.shape) == (2, 4, 3, 32, 32)
    f = spec.flip(inp)
    assert torch.allclose(f[:, 0], inp[:, 0].flip(dims=[-1])) and torch.allclose(f[:, 2], inp[:, 3]) and torch.allclose(f[:, 3], inp[:, 2])
    xm = x.flip(dims=[2])                                                             # mirrored volume == flipped input (exactly, up to resize)
    assert torch.allclose(spec(xm)[:, 2], f[:, 2], atol=1e-4) and torch.allclose(spec(xm)[:, 0], f[:, 0], atol=1e-4)
    assert len(list(spec.variants(x))) == 6
    d = spec.to_dict(); s2 = make_spec(d); assert isinstance(s2, InputSpecB) and s2.to_dict() == d and d["arm"] == "tri"
    assert isinstance(make_spec(InputSpec("2d", hw=32).to_dict()), InputSpec)


def test_tri_model_forward_and_build_model():
    m = TriPlanarNet(backbone="tiny", n_in=3, n_views=4, fusion="attn")
    x = torch.rand(2, 4, 3, 32, 32)
    assert m(x).shape == (2,) and m.features(x).shape == (2, m.feat_dim) and m.view_attention(x).shape == (2, 4)
    assert torch.allclose(m.view_attention(x).sum(1), torch.ones(2), atol=1e-5)
    mc = TriPlanarNet(backbone="tiny", n_in=3, n_views=4, fusion="concat"); assert mc(x).shape == (2,) and mc.feat_dim == 4 * 128
    mb = build_model("tri", backbone="tiny", n_in=3, n_views=4, fusion="attn", weights_path=None)
    assert isinstance(mb, TriPlanarNet) and any(n.startswith("backbone.") for n, _ in mb.named_parameters())


def test_train_one_epoch_tri(tiny_cache, tmp_path):
    spec = InputSpecB(n_slabs=3, slab_mm=12.0, hw=32, z_center=36.0, y_center=36.0, x_left=30.0, x_right=42.0, tta_z=[0])
    meta = train_run(tiny_cache, fold=0, seed=0, out_dir=tmp_path, spec=spec, model_kw=dict(backbone="tiny", n_in=3, n_views=4, fusion="attn"),
                     tag="t_tri", epochs=1, batch_size=4, ema_decay=0.9, num_workers=0, device="cpu", z_jitter=1)
    assert (tmp_path / "t_tri.pt").exists() and meta["spec"]["arm"] == "tri" and meta["spec"]["views"] == list(spec.views)
    oof = pd.read_csv(tmp_path / "oof_t_tri.csv"); assert len(oof) == 4 and oof["p"].between(0, 1).all()
    # inference contract: run_arm rebuilds the model from the meta via make_spec/build_model
    from datsc import infer_v3 as infer
    from datsc.data_v3 import VolumeCache
    c = VolumeCache(tiny_cache); vols = np.asarray(c.vols[:4], np.float32); st = c.stats[:4]; sc = c.scalars[:4]
    cfg = {"name": "tri", "arm": "tri", "spec": meta["spec"], "model_kw": meta["model_kw"], "norm": "ref", "checkpoints": ["artifacts/t_tri.pt"]}
    p = infer.run_arm(cfg, tmp_path, vols, sc, st, np.ones(4, bool), t0=__import__("time").time())
    assert p is not None and p.shape == (4,) and np.isfinite(p).all()
