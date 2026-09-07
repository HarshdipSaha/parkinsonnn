"""Run 7: hemisphere-level striatal member (arm "3d", arch "hemi").

The registered striatal crop [B,1,48,48,32] is split at the midline (x = 24) into a left and a right half-crop of
[B,1,24,48,32]; the right half is mirrored along x so both halves sit in the same orientation, and ONE shared encoder
(the run-3 GroupNorm/SiLU ResNet3D blocks) embeds each. The head is symmetric in the two hemispheres by construction:
  z = pair_head([f_L + f_R, |f_L - f_R|]) + max(side_head(f_L), side_head(f_R))
so a one-sided dopaminergic deficit (the typical onset) is an explicit signal, and mirroring the whole scan (the
label-preserving flip used in training augmentation and in TTA) leaves the logit unchanged.

Why: in every run so far the left/right asymmetry descriptors were among the strongest single features, and the
whole-crop nets have to discover asymmetry through convolutions that span the midline. The half-crop split with
mirroring doubles the effective sample count for the encoder (each scan contributes two hemispheres in one
orientation) and gives the head the asymmetry directly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models_v3 import Block3D, ScalarHead, _gn


class HemiEncoder3D(nn.Module):
    def __init__(self, widths=(24, 48, 96, 192), stem_stride: int = 1):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, widths[0], 3, stem_stride, 1, bias=False), _gn(widths[0]), nn.SiLU())
        layers, cin = [], widths[0]
        for i, w in enumerate(widths):
            layers.append(Block3D(cin, w, stride=1 if i == 0 else 2)); cin = w
        self.layers = nn.Sequential(*layers)
        self.out_dim = cin

    def forward(self, x):
        return F.adaptive_avg_pool3d(self.layers(self.stem(x)), 1).flatten(1)


class HemiNet3D(nn.Module):
    """Shared encoder over the two mirrored striatal half-crops; symmetric pair + max-side head."""

    def __init__(self, n_scalars: int = 0, widths=(24, 48, 96, 192), dropout: float = 0.3, stem_stride: int = 1, split: int | None = None):
        super().__init__()
        self.encoder = HemiEncoder3D(widths, stem_stride)
        d = self.encoder.out_dim
        self.feat_dim = 2 * d
        self.split = split                                     # x index of the midline; None = half of the crop width
        self.scalars = ScalarHead(n_scalars)
        self.pair_head = nn.Sequential(nn.Linear(2 * d + n_scalars, 128), nn.SiLU(), nn.Dropout(dropout), nn.Linear(128, 1))
        self.side_head = nn.Sequential(nn.Linear(d, 64), nn.SiLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def halves(self, x):
        """[B,1,X,Y,Z] -> (left [B,1,X/2,Y,Z], right mirrored to the left orientation)."""
        X = x.shape[2]; m = self.split if self.split is not None else X // 2
        left, right = x[:, :, :m], x[:, :, m:]
        right = right.flip(dims=[2])
        if left.shape[2] != right.shape[2]:                   # odd widths: trim to the common size
            k = min(left.shape[2], right.shape[2]); left, right = left[:, :, :k], right[:, :, :k]
        return left, right

    def features(self, x):
        """Symmetric pooled features [f_L + f_R, |f_L - f_R|] (used by the DANN site head)."""
        left, right = self.halves(x)
        B = x.shape[0]
        f = self.encoder(torch.cat([left, right], 0))          # one pass for both hemispheres
        fl, fr = f[:B], f[B:]
        return torch.cat([fl + fr, (fl - fr).abs()], 1), fl, fr

    def forward(self, x, scalars=None):
        h, fl, fr = self.features(x)
        z_pair = self.pair_head(self.scalars(h, scalars)).squeeze(1)
        z_side = torch.maximum(self.side_head(fl).squeeze(1), self.side_head(fr).squeeze(1))
        return z_pair + z_side


def build_3d_h(n_scalars: int = 0, **kw) -> nn.Module:
    return HemiNet3D(n_scalars=n_scalars, **{k: v for k, v in kw.items() if k in ("widths", "dropout", "stem_stride", "split")})
