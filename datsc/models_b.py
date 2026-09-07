"""Run-6b model: tri-planar late-fusion network on a shared ImageNet backbone.

Input [B, V, C, H, W]: V view images (axial / coronal / sagittal-left / sagittal-right slab stacks of the registered box),
each C = n_slabs channels. All views go through ONE backbone (weight sharing = 4x the training signal per parameter and
the same texture features across planes), get a learned view embedding, and are fused by softmax attention pooling
(or concatenation) before the head. `features()` / `head` / `scalars` / `feat_dim` follow the run-3 model contract so the
existing trainer (mixup, EMA-final, DANN site head, group-DRO) and inference work unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models_v3 import Net2D, ScalarHead


class TriPlanarNet(nn.Module):
    MEAN, STD = Net2D.MEAN, Net2D.STD

    def __init__(self, backbone: str = "convnext_tiny", weights_path=None, n_in: int = 3, n_views: int = 4,
                 fusion: str = "attn", dropout: float = 0.3, in_scale: float = 8.0, n_scalars: int = 0, hidden: int = 256):
        super().__init__()
        body = Net2D(backbone=backbone, weights_path=weights_path, n_in=n_in, dropout=dropout, in_scale=in_scale)
        self.backbone = body.backbone                      # named `backbone` so the trainer's backbone-lr param split applies
        dim = body.feat_dim
        self.in_scale, self.n_views, self.fusion = float(in_scale), int(n_views), fusion
        self.view_emb = nn.Parameter(torch.zeros(n_views, dim))
        if fusion == "attn":
            self.score = nn.Sequential(nn.Linear(dim, 128), nn.SiLU(), nn.Linear(128, 1))
            self.feat_dim = dim
        elif fusion == "concat":
            self.feat_dim = dim * n_views
        else:
            raise ValueError(fusion)
        self.scalars = ScalarHead(n_scalars)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feat_dim + n_scalars, hidden), nn.SiLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """[B,V,C,H,W] -> [B, feat_dim]."""
        B, V = x.shape[:2]
        img = (x.flatten(0, 1).clamp(0, self.in_scale) / self.in_scale - self.MEAN) / self.STD
        h = self.backbone(img).view(B, V, -1) + self.view_emb[:V]
        if self.fusion == "attn":
            w = torch.softmax(self.score(h).squeeze(-1), dim=1)          # [B, V]
            return (w.unsqueeze(-1) * h).sum(1)
        return h.flatten(1)

    def view_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Softmax view weights [B, V] (attn fusion only) -- for reporting which planes the model trusts."""
        B, V = x.shape[:2]
        img = (x.flatten(0, 1).clamp(0, self.in_scale) / self.in_scale - self.MEAN) / self.STD
        h = self.backbone(img).view(B, V, -1) + self.view_emb[:V]
        return torch.softmax(self.score(h).squeeze(-1), dim=1)

    def forward(self, x, scalars=None):
        return self.head(self.scalars(self.features(x), scalars)).squeeze(1)
