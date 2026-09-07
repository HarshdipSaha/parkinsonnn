"""Run-6b training CLI: tri-planar late-fusion member (shared ImageNet backbone over axial / coronal / sagittal-L / sagittal-R
slab stacks of the registered box). Same protocol as run 3 (fixed epochs, EMA-final, mixup, base augmentation), plus
slab-centre jitter and optional site-invariant training / hist normalisation.

python scripts/train_b.py --fold 0 --seed 0 --backbone convnext_tiny --weights pretrained/convnext_tiny_in1k.pt [--norm hist]
Tags: r2t_<backbone>[_hist]_f<f>_s<s>[_site]   (r2t = registered, 2.5D, tri-planar)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from datsc.data_b import hemisphere_centres
from datsc.train_b import DEFAULT_VIEWS, InputSpecB
from datsc.train_v3 import train_run


def template_centres(path: str):
    """(z_center, y_center, x_left, x_right) from the study template; None -> InputSpecB defaults."""
    p = Path(path)
    if not p.exists():
        return None
    z = np.load(p); c = z["striatal_center"]
    xl, xr = hemisphere_centres(z["striatal_mask"].astype(bool), float(c[0]))
    return float(c[2]), float(c[1]), xl, xr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="cache_v4")
    ap.add_argument("--out", default="artifacts_v3")
    ap.add_argument("--folds-csv", default="folds.csv")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--template", default="artifacts/template.npz")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--backbone-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--mixup", type=float, default=0.2)
    ap.add_argument("--ema", type=float, default=0.99)
    ap.add_argument("--norm", default="ref", choices=["ref", "hist", "p75"])
    ap.add_argument("--aug", default="base", choices=["base", "strong"])
    ap.add_argument("--z-jitter", type=int, default=1)
    ap.add_argument("--dann", type=float, default=0.0)
    ap.add_argument("--dro", type=float, default=0.0)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--backbone", default="convnext_tiny")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--n-slabs", type=int, default=3)
    ap.add_argument("--slab-mm", type=float, default=8.0)
    ap.add_argument("--hw", type=int, default=160)
    ap.add_argument("--views", nargs="+", default=list(DEFAULT_VIEWS))
    ap.add_argument("--fusion", default="attn", choices=["attn", "concat"])
    ap.add_argument("--z-center", type=float, default=None)
    ap.add_argument("--y-center", type=float, default=None)
    ap.add_argument("--x-left", type=float, default=None)
    ap.add_argument("--x-right", type=float, default=None)
    ap.add_argument("--tta-z", type=int, nargs="+", default=[-1, 0, 1])
    ap.add_argument("--tta-affine", type=int, default=0)
    ap.add_argument("--crop", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=0.3)
    a = ap.parse_args()
    tc = template_centres(a.template)
    zc, yc, xl, xr = (a.z_center, a.y_center, a.x_left, a.x_right)
    if tc is not None:
        zc = zc if zc is not None else tc[0]; yc = yc if yc is not None else tc[1]
        xl = xl if xl is not None else tc[2]; xr = xr if xr is not None else tc[3]
    spec = InputSpecB(n_slabs=a.n_slabs, slab_mm=a.slab_mm, hw=a.hw, views=a.views, tta_z=a.tta_z, crop=a.crop, tta_affine=a.tta_affine,
                      **{k: v for k, v in dict(z_center=zc, y_center=yc, x_left=xl, x_right=xr).items() if v is not None})
    model_kw = dict(backbone=a.backbone, weights_path=a.weights, n_in=a.n_slabs, n_views=len(a.views), fusion=a.fusion,
                    dropout=a.dropout, n_scalars=0)
    tag = a.tag or f"r2t_{a.backbone}{'_hist' if a.norm == 'hist' else ''}_f{a.fold if a.fold >= 0 else 'all'}_s{a.seed}"
    if a.folds_csv != "folds.csv" and not a.tag:
        tag += "_" + Path(a.folds_csv).stem.replace("folds_", "")
    train_run(a.cache, a.fold, a.seed, a.out, spec, model_kw, tag, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
              backbone_lr=a.backbone_lr, weight_decay=a.weight_decay, warmup_epochs=a.warmup, mixup=a.mixup,
              ema_decay=a.ema or None, norm=a.norm, folds_csv=a.folds_csv, num_workers=a.workers, augment=not a.no_aug,
              aug=a.aug, z_jitter=a.z_jitter, dann=a.dann, dro=a.dro)
