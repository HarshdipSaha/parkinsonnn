"""Run 7: hemisphere-level member + bootstrapped-target loss."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datsc.models_h import HemiNet3D
from datsc.models_v3 import build_model
from datsc.train_v3 import InputSpec, train_run
from tests.test_v3 import tiny_cache  # noqa: F401  (fixture)


def test_hemi_forward_and_dispatch():
    m = HemiNet3D(widths=(8, 8, 16, 16)); x = torch.rand(3, 1, 48, 48, 32)
    z = m(x); assert z.shape == (3,) and torch.isfinite(z).all()
    h, fl, fr = m.features(x); assert h.shape == (3, 32) and fl.shape == (3, 16) and m.feat_dim == 32
    m2 = build_model("3d", arch="hemi", widths=(8, 8, 16, 16)); assert isinstance(m2, HemiNet3D)


def test_hemi_logit_is_mirror_invariant():
    torch.manual_seed(0)
    m = HemiNet3D(widths=(8, 8, 16, 16)).eval(); x = torch.rand(2, 1, 48, 48, 32)
    with torch.no_grad():
        z = m(x); zf = m(x.flip(dims=[2]))                   # whole-scan mirror == swapping the hemispheres
    assert torch.allclose(z, zf, atol=1e-5)


def test_train_one_epoch_hemi_bootstrap(tiny_cache, tmp_path):
    spec = InputSpec("3d", size=(48, 48, 32), crop3d=(48, 48, 32), center3d=(47.5, 47.5, 35.5))
    meta = train_run(tiny_cache, fold=0, seed=0, out_dir=tmp_path, spec=spec, model_kw=dict(arch="hemi", widths=(8, 8, 16, 16)), tag="th",
                     epochs=1, batch_size=4, ema_decay=0.9, num_workers=0, device="cpu", loss="bootstrap", dann=0.1, dro=0.1)
    assert meta["loss"] == "bootstrap" and np.isfinite(meta["history"][0]["train_loss"]) and (tmp_path / "th.pt").exists()
    j = json.loads((tmp_path / "th.json").read_text()); assert j["model_kw"]["arch"] == "hemi"
    # the container path rebuilds the model from the meta and loads the fp16 checkpoint
    m = build_model("3d", **{k: v for k, v in j["model_kw"].items() if k != "n_scalars"}); m.load_state_dict(torch.load(tmp_path / "th.pt")); m.eval()
    with torch.no_grad():
        assert m(torch.rand(1, 1, 48, 48, 32).float()).shape == (1,)
