"""Compact 3D ResNet (~0.9M params) with a scalar-feature head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm3d(cout)
        self.c2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm3d(cout)
        self.skip = None
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False), nn.BatchNorm3d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(y + (x if self.skip is None else self.skip(x)))


class SmallResNet3D(nn.Module):
    def __init__(self, n_scalars: int = 8, widths=(16, 32, 64, 128), dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, widths[0], 3, 1, 1, bias=False), nn.BatchNorm3d(widths[0]), nn.ReLU())
        layers, cin = [], widths[0]
        for i, w in enumerate(widths):
            layers.append(Block(cin, w, stride=1 if i == 0 else 2)); cin = w
        self.layers = nn.Sequential(*layers)
        self.scalar_bn = nn.BatchNorm1d(n_scalars)
        self.head = nn.Sequential(nn.Linear(cin + n_scalars, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x, scalars):
        h = self.layers(self.stem(x))
        h = F.adaptive_avg_pool3d(h, 1).flatten(1)
        h = torch.cat([h, self.scalar_bn(scalars)], dim=1)
        return self.head(h).squeeze(1)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
