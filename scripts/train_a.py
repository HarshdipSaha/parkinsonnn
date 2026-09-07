"""Run-6a training CLI: the striatal-crop 3D family (approach A). Same trainer as scripts/train_v3.py (datsc.train_v3.train_run),
plus the SE / stochastic-depth architecture switch and a tag scheme that keeps the variants apart in the checkpoint dataset.

python scripts/train_a.py --fold 0 --seed 0 --crop3d 48 48 32 --center3d 47.5 57.3 33.5 --epochs 80                 # base arch, 80 epochs
python scripts/train_a.py --fold 0 --seed 0 --crop3d 48 48 32 --center3d 47.5 57.3 33.5 --arch se --widths 32 64 128 256
python scripts/train_a.py --fold 0 --seed 0 --crop3d 64 64 40 --center3d 47.5 57.3 33.5 --size 64 64 40 --epochs 60
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datsc.train_v3 import InputSpec, train_run


def default_tag(a) -> str:
    crop = "x".join(map(str, a.crop3d)) if a.crop3d else "x".join(map(str, a.size))
    if a.arch == "se":
        base = f"r3d_se{a.crop3d[0] if a.crop3d else a.size[0]}" + (f"b{a.blocks}" if a.blocks > 1 else "")
    elif a.epochs != 50:
        base = f"r3d_crop{a.crop3d[0]}e{a.epochs}" if a.crop3d else f"r3d_{crop}e{a.epochs}"
    else:
        base = f"r3d_crop{crop}" if a.crop3d else f"r3d_{crop}"
    if a.dann > 0 or a.dro > 0:      # the standard config keeps the run-5 tag family so a finished copy is reused by the runner
        base = "r3d_cropinv" if (crop == "48x48x32" and a.epochs == 50 and a.arch == "base") else base + "inv"
    tag = f"{base}_f{a.fold if a.fold >= 0 else 'all'}_s{a.seed}"
    if a.folds_csv != "folds.csv":
        tag += "_" + Path(a.folds_csv).stem.replace("folds_", "")
    return tag


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="cache_v4")
    ap.add_argument("--out", default="artifacts_v3")
    ap.add_argument("--folds-csv", default="folds.csv")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--mixup", type=float, default=0.2)
    ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--norm", default="ref", choices=["ref", "p75", "median", "zscore", "p99", "hist"])
    ap.add_argument("--aug", default="base", choices=["base", "strong"])
    ap.add_argument("--dann", type=float, default=0.0)
    ap.add_argument("--dro", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--size", type=int, nargs=3, default=[48, 48, 32], help="model input size (== crop3d for no resize)")
    ap.add_argument("--crop3d", type=int, nargs=3, default=[48, 48, 32])
    ap.add_argument("--center3d", type=float, nargs=3, default=None, help="template striatal_center x y z (voxels)")
    ap.add_argument("--arch", default="base", choices=["base", "se"], help="base = run-4a ResNet3D; se = SE + stochastic-depth family (datsc.models_a)")
    ap.add_argument("--widths", type=int, nargs=4, default=None, help="default: base 24 48 96 192, se 32 64 128 256")
    ap.add_argument("--blocks", type=int, default=1, help="se: residual blocks per stage")
    ap.add_argument("--drop-path", type=float, default=0.1, help="se: max stochastic-depth rate (linear over blocks)")
    ap.add_argument("--no-se", action="store_true", help="se arch without the channel gates")
    ap.add_argument("--tta-affine", type=int, default=0)
    a = ap.parse_args()
    widths = tuple(a.widths) if a.widths else ((32, 64, 128, 256) if a.arch == "se" else (24, 48, 96, 192))
    spec = InputSpec("3d", size=a.size, crop3d=a.crop3d, center3d=a.center3d, tta_affine=a.tta_affine)
    if a.arch == "se":
        model_kw = dict(arch="se", widths=widths, blocks=a.blocks, se=not a.no_se, drop_path=a.drop_path, n_scalars=0)
    else:
        model_kw = dict(widths=widths, stem_stride=1, n_scalars=0)
    tag = a.tag or default_tag(a)
    train_run(a.cache, a.fold, a.seed, a.out, spec, model_kw, tag, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
              weight_decay=a.weight_decay, warmup_epochs=a.warmup, mixup=a.mixup, ema_decay=a.ema or None, norm=a.norm,
              folds_csv=a.folds_csv, num_workers=a.workers, aug=a.aug, dann=a.dann, dro=a.dro)
