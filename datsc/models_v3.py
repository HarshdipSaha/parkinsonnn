"""Run-3 model zoo: a width-configurable 3D ResNet and torchvision ImageNet backbones for 2.5D slabs.

Both expose forward(x, scalars) -> logits[B]. 2D backbones are built WITHOUT downloading anything;
pretrained state_dicts (scripts/fetch_pretrained.py) are loaded from a local file when given.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Block


class ResNet3D(nn.Module):
    def __init__(self, n_scalars: int = 8, widths=(24, 48, 96, 192), dropout: float = 0.3, stem_stride: int = 1):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, widths[0], 3, stem_stride, 1, bias=False), nn.BatchNorm3d(widths[0]), nn.ReLU())
        layers, cin = [], widths[0]
        for i, w in enumerate(widths):
            layers.append(Block(cin, w, stride=1 if i == 0 else 2)); cin = w
        self.layers = nn.Sequential(*layers)
        self.scalar_bn = nn.BatchNorm1d(n_scalars)
        self.head = nn.Sequential(nn.Linear(cin + n_scalars, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x, scalars):
        h = F.adaptive_avg_pool3d(self.layers(self.stem(x)), 1).flatten(1)
        return self.head(torch.cat([h, self.scalar_bn(scalars)], 1)).squeeze(1)


def _tv_backbone(name: str, weights_path: str | Path | None):
    import torchvision.models as tvm
    ctor = getattr(tvm, name)
    m = ctor(weights=None)
    if weights_path:
        sd = torch.load(str(weights_path), map_location="cpu")
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not unexpected and all(k.startswith(("fc.", "classifier.")) for k in missing), (missing, unexpected)
    if name.startswith("convnext"):
        m.classifier[2] = nn.Identity(); dim = m.classifier[0].normalized_shape[0]
    elif name.startswith("efficientnet"):
        dim = m.classifier[1].in_features; m.classifier[1] = nn.Identity()
    elif name.startswith("resnet"):
        dim = m.fc.in_features; m.fc = nn.Identity()
    else:
        raise ValueError(name)
    return m, dim


class Net2D(nn.Module):
    """ImageNet backbone on n_in-channel slab images + scalar head. Channel-count mismatch (n_in != 3)
    is handled by averaging the pretrained stem filters (Wightman's timm trick)."""
    MEAN, STD = 0.45, 0.225

    def __init__(self, backbone: str = "efficientnet_b0", weights_path=None, n_scalars: int = 8, n_in: int = 3,
                 dropout: float = 0.3, in_scale: float = 3.0):
        super().__init__()
        self.backbone, dim = _tv_backbone(backbone, weights_path)
        if n_in != 3:
            self._adapt_stem(n_in)
        self.in_scale = in_scale
        self.scalar_bn = nn.BatchNorm1d(n_scalars)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim + n_scalars, 1))

    def _adapt_stem(self, n_in: int):
        conv = None
        for mod in self.backbone.modules():
            if isinstance(mod, nn.Conv2d) and mod.in_channels == 3:
                conv = mod; break
        w = conv.weight.data
        new = nn.Conv2d(n_in, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=conv.bias is not None)
        new.weight.data = w.mean(1, keepdim=True).repeat(1, n_in, 1, 1) * (3.0 / n_in)
        if conv.bias is not None:
            new.bias.data = conv.bias.data.clone()
        # replace in parent
        for parent in self.backbone.modules():
            for name, child in parent.named_children():
                if child is conv:
                    setattr(parent, name, new); return

    def forward(self, img, scalars):
        img = (img.clamp(0, self.in_scale) / self.in_scale - self.MEAN) / self.STD
        h = self.backbone(img)
        return self.head(torch.cat([h, self.scalar_bn(scalars)], 1)).squeeze(1)


def build_model(arm: str, n_scalars: int = 8, **kw) -> nn.Module:
    if arm == "3d":
        return ResNet3D(n_scalars=n_scalars, **{k: v for k, v in kw.items() if k in ("widths", "dropout", "stem_stride")})
    if arm == "2d":
        return Net2D(n_scalars=n_scalars, **{k: v for k, v in kw.items() if k in ("backbone", "weights_path", "n_in", "dropout", "in_scale")})
    raise ValueError(arm)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
