"""Run-3 model zoo: a GroupNorm/SiLU 3D ResNet and torchvision ImageNet backbones for 2.5D slabs.

Both expose forward(x, scalars) -> logits[B]. By default NO scalar features are used (n_scalars=0): the run-2
scalars fingerprinted the acquisition site (97% site-ID accuracy) and biased predictions under site shift.
2D backbones are built with weights=None; pretrained state_dicts are loaded from a local file when given.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(c: int, groups: int = 8) -> nn.GroupNorm:
    return nn.GroupNorm(min(groups, c), c)


class Block3D(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False); self.n1 = _gn(cout)
        self.c2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False); self.n2 = _gn(cout)
        self.skip = None
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False), _gn(cout))

    def forward(self, x):
        y = F.silu(self.n1(self.c1(x)))
        y = self.n2(self.c2(y))
        return F.silu(y + (x if self.skip is None else self.skip(x)))


class ScalarHead(nn.Module):
    """Optional image-only scalar side-input; disabled when n_scalars == 0."""

    def __init__(self, n_scalars: int):
        super().__init__()
        self.n = n_scalars
        self.norm = nn.LayerNorm(n_scalars) if n_scalars else None

    def forward(self, h, scalars):
        if not self.n:
            return h
        return torch.cat([h, self.norm(scalars[:, : self.n])], 1)


class ResNet3D(nn.Module):
    def __init__(self, n_scalars: int = 0, widths=(24, 48, 96, 192), dropout: float = 0.3, stem_stride: int = 1):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, widths[0], 3, stem_stride, 1, bias=False), _gn(widths[0]), nn.SiLU())
        layers, cin = [], widths[0]
        for i, w in enumerate(widths):
            layers.append(Block3D(cin, w, stride=1 if i == 0 else 2)); cin = w
        self.layers = nn.Sequential(*layers)
        self.scalars = ScalarHead(n_scalars)
        self.feat_dim = cin
        self.head = nn.Sequential(nn.Linear(cin + n_scalars, 128), nn.SiLU(), nn.Dropout(dropout), nn.Linear(128, 1))

    def features(self, x):
        """Pooled features before the head (used by the DANN site head)."""
        return F.adaptive_avg_pool3d(self.layers(self.stem(x)), 1).flatten(1)

    def forward(self, x, scalars=None):
        return self.head(self.scalars(self.features(x), scalars)).squeeze(1)


class TinyNet2D(nn.Module):
    """~200k-parameter ResNet-style 2D net trained from scratch (the Hamburg DaT-SPECT recipe: small nets on a
    single striatal slab generalise across cameras). GroupNorm, SiLU, global average pooling."""

    def __init__(self, n_in: int = 3, widths=(16, 32, 64, 128)):
        super().__init__()
        def block(cin, cout, stride):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.GroupNorm(min(8, cout), cout), nn.SiLU(),
                                 nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.GroupNorm(min(8, cout), cout), nn.SiLU())
        layers, cin = [], n_in
        for i, w in enumerate(widths):
            layers.append(block(cin, w, 1 if i == 0 else 2)); cin = w
        self.features = nn.Sequential(*layers)
        self.out_dim = cin

    def forward(self, x):
        return F.adaptive_avg_pool2d(self.features(x), 1).flatten(1)


def _tv_backbone(name: str, weights_path: str | Path | None):
    if name == "tiny":
        m = TinyNet2D(); return m, m.out_dim
    import torchvision.models as tvm
    if not hasattr(tvm, name):
        raise ValueError(name)
    m = getattr(tvm, name)(weights=None)
    if weights_path:
        sd = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not unexpected and all(k.startswith(("fc.", "classifier.", "head.")) for k in missing), (missing, unexpected)
    if name.startswith("convnext"):
        m.classifier[2] = nn.Identity(); dim = m.classifier[0].normalized_shape[0]
    elif name.startswith("efficientnet"):
        dim = m.classifier[1].in_features; m.classifier[1] = nn.Identity()
    elif name.startswith("resnet"):
        dim = m.fc.in_features; m.fc = nn.Identity()
    elif name.startswith("swin"):
        dim = m.head.in_features; m.head = nn.Identity()
    else:
        raise ValueError(name)
    return m, dim


class Net2D(nn.Module):
    """ImageNet backbone on n_in-channel slab images (+ optional scalars). Channel-count mismatch (n_in != 3) is
    handled by averaging the pretrained stem filters. Inputs are in SBR+1 / p75 units; in_scale maps [0, in_scale]
    to [0, 1] before ImageNet-style standardisation (8 keeps the striatal range: p99 of the central box > 3 in 93%
    of normals)."""
    MEAN, STD = 0.45, 0.225

    def __init__(self, backbone: str = "efficientnet_b0", weights_path=None, n_scalars: int = 0, n_in: int = 3,
                 dropout: float = 0.3, in_scale: float = 8.0):
        super().__init__()
        if backbone == "tiny":
            self.backbone = TinyNet2D(n_in=n_in); dim = self.backbone.out_dim
        else:
            self.backbone, dim = _tv_backbone(backbone, weights_path)
            if n_in != 3:
                self._adapt_stem(n_in)
        self.in_scale = in_scale
        self.scalars = ScalarHead(n_scalars)
        self.feat_dim = dim
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim + n_scalars, 1))

    def _adapt_stem(self, n_in: int):
        conv = next(mod for mod in self.backbone.modules() if isinstance(mod, nn.Conv2d) and mod.in_channels == 3)
        w = conv.weight.data
        new = nn.Conv2d(n_in, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=conv.bias is not None)
        new.weight.data = w.mean(1, keepdim=True).repeat(1, n_in, 1, 1) * (3.0 / n_in)
        if conv.bias is not None:
            new.bias.data = conv.bias.data.clone()
        for parent in self.backbone.modules():
            for name, child in parent.named_children():
                if child is conv:
                    setattr(parent, name, new); return

    def features(self, img):
        """Pooled backbone features before the head (used by the DANN site head)."""
        img = (img.clamp(0, self.in_scale) / self.in_scale - self.MEAN) / self.STD
        return self.backbone(img)

    def forward(self, img, scalars=None):
        return self.head(self.scalars(self.features(img), scalars)).squeeze(1)


class GradReverse(torch.autograd.Function):
    """Identity forward, reversed (and scaled) gradient backward: the DANN trick."""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = float(lam)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class SiteHead(nn.Module):
    """Pseudo-site classifier on pooled features behind a gradient-reversal layer. Trained jointly, never saved with the
    model: the backbone is pushed toward features that do not identify the camera/protocol."""

    def __init__(self, dim: int, n_sites: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, n_sites))

    def forward(self, h, lam: float = 1.0):
        return self.net(GradReverse.apply(h, lam))


def build_model(arm: str, n_scalars: int = 0, **kw) -> nn.Module:
    if arm == "3d":
        if kw.get("arch") == "se":                       # run 6a: SE / stochastic-depth striatal-crop family (datsc.models_a)
            from .models_a import build_3d_a
            return build_3d_a(n_scalars=n_scalars, **kw)
        if kw.get("arch") == "hemi":                     # run 7: hemisphere-level shared encoder (datsc.models_h)
            from .models_h import build_3d_h
            return build_3d_h(n_scalars=n_scalars, **kw)
        return ResNet3D(n_scalars=n_scalars, **{k: v for k, v in kw.items() if k in ("widths", "dropout", "stem_stride")})
    if arm == "2d":
        return Net2D(n_scalars=n_scalars, **{k: v for k, v in kw.items() if k in ("backbone", "weights_path", "n_in", "dropout", "in_scale")})
    if arm == "tri":                                        # run 6b: tri-planar late fusion (datsc.models_b)
        from .models_b import TriPlanarNet
        return TriPlanarNet(n_scalars=n_scalars, **{k: v for k, v in kw.items() if k in ("backbone", "weights_path", "n_in", "n_views", "fusion", "dropout", "in_scale", "hidden")})
    raise ValueError(arm)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
