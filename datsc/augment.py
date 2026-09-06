"""Augmentations applied to [B,1,X,Y,Z] tensors on the training device (GPU-side on Kaggle).

Physically motivated for multi-centre SPECT: blur (resolution differences), noise (counts),
quantisation (the ultra-low-count regime), reference-scale jitter (p75 normalisation error),
small affine (positioning), L/R mirror (the label is symmetric under mirroring).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .config import SPACING_MM


def _gauss1d(sigma: float, device):
    r = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / max(sigma, 1e-3)) ** 2)
    return k / k.sum()


def blur(x: torch.Tensor, sigma_vox: float) -> torch.Tensor:
    if sigma_vox < 0.05:
        return x
    k = _gauss1d(sigma_vox, x.device)
    n = k.numel(); pad = n // 2
    for dim in (2, 3, 4):
        shape = [1, 1, 1, 1, 1]; shape[dim] = n
        kk = k.view(*shape)
        p = [0] * 6; p[2 * (4 - dim)] = pad; p[2 * (4 - dim) + 1] = pad
        x = F.conv3d(F.pad(x, p, mode="replicate"), kk)
    return x


def random_blur(x, max_sigma_mm: float = 4.0):
    out = torch.empty_like(x)
    for i in range(x.shape[0]):
        s = float(torch.rand(1)) * max_sigma_mm / SPACING_MM
        out[i:i + 1] = blur(x[i:i + 1], s)
    return out


def random_noise(x, max_frac: float = 0.15):
    std = x.flatten(1).std(dim=1).view(-1, 1, 1, 1, 1)
    frac = torch.rand(x.shape[0], 1, 1, 1, 1, device=x.device) * max_frac
    return (x + torch.randn_like(x) * std * frac).clamp_min(0)


def quantize(x, k: int):
    m = x.flatten(1).amax(dim=1).view(-1, 1, 1, 1, 1).clamp_min(1e-6)
    return torch.round(x / m * (k - 1)) / (k - 1) * m


def random_quantize(x, levels=(4, 6, 8, 16, 64, 256), p: float = 0.5):
    out = x.clone()
    for i in range(x.shape[0]):
        if float(torch.rand(1)) < p:
            out[i:i + 1] = quantize(x[i:i + 1], int(levels[int(torch.randint(len(levels), (1,)))]))
    return out


def random_scale_jitter(x, frac: float = 0.2):
    s = 1.0 + (torch.rand(x.shape[0], 1, 1, 1, 1, device=x.device) * 2 - 1) * frac
    return x * s


def random_affine(x, max_deg: float = 10.0, max_scale: float = 0.10, max_shift_vox: float = 3.0):
    B = x.shape[0]
    theta = torch.zeros(B, 3, 4, device=x.device)
    for i in range(B):
        a = torch.deg2rad((torch.rand(3) * 2 - 1) * max_deg)
        cx, cy, cz = torch.cos(a); sx, sy, sz = torch.sin(a)
        Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        S = torch.diag(1.0 + (torch.rand(3) * 2 - 1) * max_scale)
        M = Rz @ Ry @ Rx @ S
        shift = (torch.rand(3) * 2 - 1) * max_shift_vox * 2.0 / torch.tensor(x.shape[2:], dtype=torch.float32)
        theta[i, :, :3] = M.to(x.device); theta[i, :, 3] = shift.to(x.device)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def random_flip(x, p: float = 0.5):
    out = x.clone()
    for i in range(x.shape[0]):
        if float(torch.rand(1)) < p:
            out[i] = x[i].flip(dims=[1])   # dim 1 of [1,X,Y,Z] is X
    return out


PRESETS = {
    # run-3 defaults (physically plausible multi-centre perturbations)
    "base":   dict(blur_mm=4.0, noise=0.15, deg=10.0, scale=0.10, shift=3.0, quant_p=0.5, scale_jitter=0.2),
    # deliberately unrealistic (Buddenkotte & Buchert 2024): pushes the net to ignore the camera's resolution/count statistics
    "strong": dict(blur_mm=8.0, noise=0.30, deg=15.0, scale=0.15, shift=4.0, quant_p=0.6, scale_jitter=0.3),
}


def train_augment(x: torch.Tensor, strength: str = "base") -> torch.Tensor:
    c = PRESETS[strength]
    x = random_flip(x)
    x = random_affine(x, c["deg"], c["scale"], c["shift"])
    x = random_blur(x, c["blur_mm"])
    x = random_scale_jitter(x, c["scale_jitter"])
    x = random_quantize(x, p=c["quant_p"])
    x = random_noise(x, c["noise"])
    return x
