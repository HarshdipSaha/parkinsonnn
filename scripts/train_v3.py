"""Run-3 training CLI.

python scripts/train_v3.py --arm 3d --fold 0 --seed 0 [--size 96 96 72] [--widths 24 48 96 192]
python scripts/train_v3.py --arm 2d --backbone efficientnet_b0 --weights pretrained/efficientnet_b0_in1k.pt --fold 0 --seed 0 --z-center 40.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datsc.train_v3 import InputSpec, train_run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["3d", "2d"], required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="cache_v4")
    ap.add_argument("--out", default="artifacts_v3")
    ap.add_argument("--folds-csv", default="folds.csv")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--backbone-lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--select", default="final", choices=["final", "best"])
    ap.add_argument("--norm", default="ref", choices=["ref", "p75", "median", "zscore", "p99"])
    ap.add_argument("--n-scalars", type=int, default=0)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    # 3d
    ap.add_argument("--size", type=int, nargs=3, default=[96, 96, 72])
    ap.add_argument("--widths", type=int, nargs=4, default=[24, 48, 96, 192])
    ap.add_argument("--stem-stride", type=int, default=1)
    # 2d
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--n-slabs", type=int, default=3)
    ap.add_argument("--slab-mm", type=float, default=8.0)
    ap.add_argument("--slab-mode", default="mean", choices=["mean", "max"])
    ap.add_argument("--hw", type=int, default=192)
    ap.add_argument("--z-center", type=float, default=None, help="voxel centre of the slab stack along the slab axis (template striatal_center[2] axial / [1] coronal)")
    ap.add_argument("--plane", default="axial", choices=["axial", "coronal"])
    ap.add_argument("--tta-z", type=int, nargs="+", default=[0], help="slab-centre offsets (voxels) averaged at test time, e.g. -1 0 1")
    ap.add_argument("--crop", type=int, default=None, help="central crop (voxels) of the slab image before resizing, e.g. 72")
    a = ap.parse_args()
    if a.arm == "3d":
        spec = InputSpec("3d", size=a.size)
        model_kw = dict(widths=tuple(a.widths), stem_stride=a.stem_stride, n_scalars=a.n_scalars)
        tag = a.tag or f"r3d_{'x'.join(map(str, a.size))}_f{a.fold if a.fold >= 0 else 'all'}_s{a.seed}"
    else:
        spec = InputSpec("2d", n_slabs=a.n_slabs, slab_mm=a.slab_mm, hw=a.hw, slab_mode=a.slab_mode, z_center=a.z_center,
                         plane=a.plane, tta_z=a.tta_z, crop=a.crop)
        model_kw = dict(backbone=a.backbone, weights_path=a.weights, n_in=a.n_slabs, n_scalars=a.n_scalars)
        tag = a.tag or f"r2d_{a.backbone}{'_cor' if a.plane == 'coronal' else ''}_f{a.fold if a.fold >= 0 else 'all'}_s{a.seed}"
    if a.folds_csv != "folds.csv":
        tag += "_" + Path(a.folds_csv).stem.replace("folds_", "")
    train_run(a.cache, a.fold, a.seed, a.out, spec, model_kw, tag, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
              backbone_lr=a.backbone_lr, weight_decay=a.weight_decay, warmup_epochs=a.warmup, mixup=a.mixup,
              ema_decay=a.ema or None, norm=a.norm, folds_csv=a.folds_csv, num_workers=a.workers, augment=not a.no_aug,
              select=a.select)
