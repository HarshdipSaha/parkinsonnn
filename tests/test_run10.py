"""Tests for Run 10: Swin Transformer integration into datsc/models_v3."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datsc.models_v3 import Net2D, build_model, count_params


def test_build_model_swin_t_architecture():
    """Verify that build_model instantiates Swin-T as a 2D arm and exposes the expected head/dim."""
    model = build_model("2d", backbone="swin_t", n_in=3, n_scalars=0)
    assert isinstance(model, Net2D)
    assert model.feat_dim == 768
    params = count_params(model)
    assert 25_000_000 < params < 35_000_000, f"Expected ~28M params for Swin-T, got {params}"


def test_swin_t_forward_pass_and_backward():
    """Verify forward and backward pass on 224x224 input."""
    model = build_model("2d", backbone="swin_t", n_in=3, n_scalars=0)
    model.train()
    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    out = model(x)
    assert out.shape == (2,)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == (2, 3, 224, 224)


def test_swin_t_with_scalars():
    """Verify forward pass when scalar features are provided."""
    model = build_model("2d", backbone="swin_t", n_in=3, n_scalars=4)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    scalars = torch.randn(2, 4)
    with torch.no_grad():
        out = model(x, scalars=scalars)
    assert out.shape == (2,)


def test_swin_t_state_dict_roundtrip(tmp_path):
    """Verify checkpoint saving and loading roundtrip."""
    m1 = build_model("2d", backbone="swin_t", n_in=3, n_scalars=0)
    ckpt_path = tmp_path / "swin_test.pt"
    torch.save(m1.state_dict(), ckpt_path)

    m2 = build_model("2d", backbone="swin_t", n_in=3, n_scalars=0)
    sd = torch.load(ckpt_path, weights_only=True)
    m2.load_state_dict(sd)

    m1.eval()
    m2.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y1 = m1(x)
        y2 = m2(x)
    assert torch.allclose(y1, y2, atol=1e-5)


def test_unknown_backbone_raises_value_error():
    """Verify that unsupported backbone names still raise ValueError."""
    with pytest.raises(ValueError, match="nonexistent_backbone"):
        build_model("2d", backbone="nonexistent_backbone")
