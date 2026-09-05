"""Download permissively-licensed ImageNet/Kinetics torchvision weights once, for bundling in the submission.

All torchvision weights below are BSD-3 (torchvision) and trained on ImageNet-1k / Kinetics-400 (CC-BY).
Saved as plain state_dicts so main.py never needs network access.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torchvision.models as tvm
from torchvision.models import video as tvv

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "pretrained")
OUT.mkdir(parents=True, exist_ok=True)

SPECS = {
    "convnext_tiny_in1k.pt": lambda: tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    "efficientnet_b0_in1k.pt": lambda: tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "efficientnet_v2_s_in1k.pt": lambda: tvm.efficientnet_v2_s(weights=tvm.EfficientNet_V2_S_Weights.IMAGENET1K_V1),
    "resnet18_in1k.pt": lambda: tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50_in1k_v2.pt": lambda: tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2),
    "r3d_18_kinetics.pt": lambda: tvv.r3d_18(weights=tvv.R3D_18_Weights.KINETICS400_V1),
}

for name, ctor in SPECS.items():
    p = OUT / name
    if p.exists():
        print("exists", name); continue
    m = ctor()
    torch.save(m.state_dict(), p)
    print(f"saved {name}: {p.stat().st_size/1e6:.0f} MB, {sum(x.numel() for x in m.parameters())/1e6:.1f}M params")
