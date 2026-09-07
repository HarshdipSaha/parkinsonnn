"""Approach A (run 6a): SE / stochastic-depth striatal-crop family, build_model dispatch, CLI tag scheme, one-epoch training."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from datsc.models_a import DropPath, SEResNet3D, build_3d_a          # noqa: E402
from datsc.models_v3 import ResNet3D, build_model, count_params       # noqa: E402
from datsc.train_v3 import InputSpec, train_run                       # noqa: E402
from tests.test_v3 import tiny_cache                                   # noqa: E402,F401  (module-scoped synthetic cache fixture)


def test_se_resnet3d_forward_features_and_dispatch():
    m = SEResNet3D(widths=(8, 8, 16, 16), blocks=2, drop_path=0.2)
    x = torch.rand(2, 1, 48, 48, 32)
    assert m(x).shape == (2,) and m.features(x).shape == (2, 16) and m.feat_dim == 16
    m2 = build_model("3d", arch="se", widths=(8, 8, 16, 16), blocks=1, se=False, drop_path=0.0)
    assert isinstance(m2, SEResNet3D) and m2(x).shape == (2,)
    assert isinstance(build_model("3d", widths=(8, 8, 16, 16)), ResNet3D)          # no arch key -> unchanged run-4a path
    assert isinstance(build_3d_a(widths=(8, 8, 16, 16), stem_stride=1, extra_ignored=1), SEResNet3D)
    assert count_params(SEResNet3D(widths=(32, 64, 128, 256), blocks=1)) > count_params(ResNet3D(widths=(24, 48, 96, 192)))


def test_drop_path_identity_in_eval_and_state_dict_round_trip(tmp_path):
    dp = DropPath(0.5).eval(); x = torch.rand(4, 3, 2, 2, 2)
    assert torch.equal(dp(x), x)
    m = SEResNet3D(widths=(8, 8, 16, 16)).eval(); x = torch.rand(1, 1, 48, 48, 32)
    sd = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in m.state_dict().items()}   # fp16 save as the trainer does
    torch.save(sd, tmp_path / "m.pt")
    m2 = build_model("3d", arch="se", widths=(8, 8, 16, 16)); m2.load_state_dict(torch.load(tmp_path / "m.pt", weights_only=True)); m2.eval()
    assert torch.allclose(m(x), m2(x), atol=2e-2)


def test_train_a_cli_tags():
    from scripts.train_a import default_tag
    class A: pass
    a = A(); a.crop3d = [48, 48, 32]; a.size = [48, 48, 32]; a.arch = "base"; a.blocks = 1; a.epochs = 80; a.dann = 0; a.dro = 0; a.fold = 0; a.seed = 1; a.folds_csv = "folds.csv"
    assert default_tag(a) == "r3d_crop48e80_f0_s1"
    a.epochs = 50; assert default_tag(a) == "r3d_crop48x48x32_f0_s1"                         # == the run-4a tag family
    a.dann = 0.1; assert default_tag(a) == "r3d_cropinv_f0_s1"                                # == the run-5 tag family
    a.dann = 0; a.arch = "se"; assert default_tag(a) == "r3d_se48_f0_s1"
    a.blocks = 2; assert default_tag(a) == "r3d_se48b2_f0_s1"
    a.blocks = 1; a.arch = "base"; a.epochs = 60; a.crop3d = [64, 64, 40]; a.size = [64, 64, 40]; assert default_tag(a) == "r3d_crop64e60_f0_s1"
    a.fold = -1; a.epochs = 80; a.crop3d = [48, 48, 32]; assert default_tag(a) == "r3d_crop48e80_fall_s1"
    a.fold = 0; a.folds_csv = "folds_site.csv"; a.epochs = 50; a.dro = 0.1; assert default_tag(a) == "r3d_cropinv_f0_s1_site"


def test_train_one_epoch_se_arch(tiny_cache, tmp_path):
    spec = InputSpec("3d", size=(32, 32, 24), crop3d=(32, 32, 24), center3d=(47.5, 47.5, 35.5))
    meta = train_run(tiny_cache, fold=0, seed=0, out_dir=tmp_path, spec=spec, model_kw=dict(arch="se", widths=(8, 8, 16, 16), blocks=1, drop_path=0.1),
                     tag="ta", epochs=1, batch_size=4, ema_decay=0.9, num_workers=0, device="cpu", dann=0.1, dro=0.1)
    assert (tmp_path / "ta.pt").exists() and meta["model_kw"]["arch"] == "se" and np.isfinite(meta["history"][0]["train_loss"])
    j = json.loads((tmp_path / "ta.json").read_text())
    m = build_model("3d", **j["model_kw"]); m.load_state_dict(torch.load(tmp_path / "ta.pt", weights_only=True))   # inference rebuild from meta


def test_gen_run6a_writes_valid_notebook(tmp_path):
    import nbformat
    r = subprocess.run([sys.executable, str(ROOT / "kaggle" / "run6a_crop3d" / "gen_run6a.py"), "--out", str(tmp_path)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-800:]
    nb = nbformat.read(str(tmp_path / "notebook.ipynb"), as_version=4); nbformat.validate(nb)
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["code_file"] == "notebook.ipynb" and meta["enable_gpu"] and "harshdipsaha/dat-run3-checkpoints" in meta["dataset_sources"]
    src = "\n".join(c.source for c in nb.cells)
    assert "scripts/train_a.py" in src and "range(torch.cuda.device_count())" in src and "smoke-set log loss per arm" in src
