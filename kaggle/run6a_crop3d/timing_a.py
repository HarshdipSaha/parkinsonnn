"""Relative cost of the run-6a 3D candidates (CPU forward+backward on a batch of 4) vs the run-4a crop net, to size the GPU budget."""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from datsc.models_a import SEResNet3D
from datsc.models_v3 import ResNet3D, count_params

torch.set_num_threads(4)


def cost(model, shape, reps=3):
    x = torch.rand(4, 1, *shape); model.train()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    for _ in range(1):
        loss = model(x).sum(); loss.backward(); opt.zero_grad()
    t = time.time()
    for _ in range(reps):
        loss = model(x).sum(); loss.backward(); opt.zero_grad()
    return (time.time() - t) / reps


base = cost(ResNet3D(widths=(24, 48, 96, 192)), (48, 48, 32))
rows = [("run4a crop48 base (24-192, 1 block)", ResNet3D(widths=(24, 48, 96, 192)), (48, 48, 32))]
rows += [("se48 (32-256, 1 block, SE)", SEResNet3D(widths=(32, 64, 128, 256), blocks=1), (48, 48, 32)),
         ("se48 (24-192, 2 blocks, SE)", SEResNet3D(widths=(24, 48, 96, 192), blocks=2), (48, 48, 32)),
         ("se48 (32-256, 2 blocks, SE)", SEResNet3D(widths=(32, 64, 128, 256), blocks=2), (48, 48, 32)),
         ("base crop64x64x40 (24-192, 1 block)", ResNet3D(widths=(24, 48, 96, 192)), (64, 64, 40)),
         ("se crop64x64x40 (32-256, 1 block)", SEResNet3D(widths=(32, 64, 128, 256), blocks=1), (64, 64, 40))]
for name, m, shape in rows:
    c = cost(m, shape)
    print(f"{name:<42} params {count_params(m)/1e6:5.2f}M  rel cost x{c/base:4.2f}  ({c*1000:.0f} ms / batch of 4 on CPU)")
