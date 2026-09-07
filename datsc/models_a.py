"""Approach A: a stronger striatal-crop 3D family (run 6a).

Why: the 48x48x32 striatal-crop ResNet3D (2.0M params, 1 block per stage) is the best single member of the run-4a stack
(OOF 0.2568 / AUROC 0.960) and the least correlated with the 2.5D arms (logit r 0.93 vs 0.97 between the 2D arms), yet it
is under-trained (validation loss still falling between epochs 40 and 50) and shallow. This module adds a configurable
family used through `datsc.models_v3.build_model(arm="3d", arch="se", ...)`:

  * `SEResNet3D`: GroupNorm/SiLU basic blocks with channel squeeze-and-excitation (Hu et al. 2018) and stochastic depth
    (Huang et al. 2016) per residual branch, `blocks` residual blocks per stage, widths configurable. With the default
    (32, 64, 128, 256) x 1 block it is ~3.5x the FLOPs of the run-4a net at the same 48x48x32 input.
  * Inference needs nothing new: the manifest carries `model_kw` (arch/widths/blocks/se/drop_path) and `build_model`
    dispatches on `arch`; stochastic depth is identity in eval mode.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(c: int, groups: int = 8) -> nn.GroupNorm:
    return nn.GroupNorm(min(groups, c), c)


class SE3D(nn.Module):
    """Squeeze-and-excitation channel gate (global average pool -> 2-layer MLP -> sigmoid)."""

    def __init__(self, c: int, r: int = 8):
        super().__init__()
        h = max(4, c // r)
        self.fc1 = nn.Linear(c, h); self.fc2 = nn.Linear(h, c)

    def forward(self, x):
        s = F.adaptive_avg_pool3d(x, 1).flatten(1)
        s = torch.sigmoid(self.fc2(F.silu(self.fc1(s))))
        return x * s.view(s.shape[0], s.shape[1], 1, 1, 1)


class DropPath(nn.Module):
    """Stochastic depth on the residual branch (per sample); identity in eval mode."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if not self.training or self.p <= 0:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class SEBlock3D(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, se: bool = True, drop_path: float = 0.0):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False); self.n1 = _gn(cout)
        self.c2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False); self.n2 = _gn(cout)
        self.se = SE3D(cout) if se else nn.Identity()
        self.dp = DropPath(drop_path)
        self.skip = None
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False), _gn(cout))

    def forward(self, x):
        y = F.silu(self.n1(self.c1(x)))
        y = self.se(self.n2(self.c2(y)))
        return F.silu(self.dp(y) + (x if self.skip is None else self.skip(x)))


class SEResNet3D(nn.Module):
    """Striatal-crop 3D classifier: stem -> 4 stages (stride 1, 2, 2, 2) of `blocks` SE residual blocks -> GAP -> head.
    Exposes the same interface as ResNet3D (features(), feat_dim, head, scalars) so the trainer's DANN/DRO path works."""

    def __init__(self, n_scalars: int = 0, widths=(32, 64, 128, 256), blocks: int = 1, se: bool = True,
                 drop_path: float = 0.1, dropout: float = 0.3, stem_stride: int = 1):
        super().__init__()
        from .models_v3 import ScalarHead
        widths = tuple(int(w) for w in widths); n_blocks = int(blocks)
        self.stem = nn.Sequential(nn.Conv3d(1, widths[0], 3, stem_stride, 1, bias=False), _gn(widths[0]), nn.SiLU())
        layers, cin = [], widths[0]
        total = len(widths) * n_blocks; k = 0
        for i, w in enumerate(widths):
            for b in range(n_blocks):
                dp = drop_path * k / max(1, total - 1); k += 1            # linearly increasing stochastic depth
                layers.append(SEBlock3D(cin, w, stride=(2 if (i > 0 and b == 0) else 1), se=se, drop_path=dp)); cin = w
        self.layers = nn.Sequential(*layers)
        self.scalars = ScalarHead(n_scalars)
        self.feat_dim = cin
        self.head = nn.Sequential(nn.Linear(cin + n_scalars, 128), nn.SiLU(), nn.Dropout(dropout), nn.Linear(128, 1))

    def features(self, x):
        return F.adaptive_avg_pool3d(self.layers(self.stem(x)), 1).flatten(1)

    def forward(self, x, scalars=None):
        return self.head(self.scalars(self.features(x), scalars)).squeeze(1)


def build_3d_a(n_scalars: int = 0, **kw) -> nn.Module:
    """Factory used by datsc.models_v3.build_model when model_kw['arch'] == 'se'."""
    keys = ("widths", "blocks", "se", "drop_path", "dropout", "stem_stride")
    return SEResNet3D(n_scalars=n_scalars, **{k: v for k, v in kw.items() if k in keys})
