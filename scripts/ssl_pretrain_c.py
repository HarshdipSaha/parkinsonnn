"""Approach C: VICReg pretraining of the striatal-crop 3D encoder on the training scans (labels unused), then an optional
linear-probe report. Skips the pretraining when --out already exists (resume-friendly).

python scripts/ssl_pretrain_c.py --cache cache_v4 --out artifacts_v3/ssl_c_encoder.pt --epochs 200 [--probe] [--max-steps N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd

from datsc.ssl_c import extract_features, linear_probe_oof, pretrain

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache_v4")
    ap.add_argument("--out", default="artifacts_v3/ssl_c_encoder.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--widths", type=int, nargs=4, default=[24, 48, 96, 192])
    ap.add_argument("--center3d", type=float, nargs=3, default=None, help="default: template striatal_center from artifacts/template.npz")
    ap.add_argument("--jitter", type=int, default=2)
    ap.add_argument("--aug", default="base", choices=["base", "strong"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=None, help="smoke tests: stop after N optimisation steps")
    ap.add_argument("--probe", action="store_true", help="after pretraining: linear-probe OOF on frozen features (random + site folds)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    center = a.center3d
    if center is None:
        tpl = next((p for p in (Path("artifacts/template.npz"), Path(a.cache) / "template.npz", Path("artifacts_v3/template.npz")) if p.exists()), None)
        center = [float(v) for v in np.load(tpl)["striatal_center"]] if tpl else [47.5, 57.3, 33.5]
    if out.exists() and not a.force:
        print(f"[ssl_c] {out} exists -- skipping pretraining (use --force to redo)")
    else:
        pretrain(a.cache, out, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, widths=tuple(a.widths), center=tuple(center),
                 jitter=a.jitter, aug=a.aug, num_workers=a.workers, max_steps=a.max_steps)
    if a.probe:
        from sklearn.metrics import log_loss, roc_auc_score
        cache = Path(a.cache); meta = pd.read_csv(cache / "meta.csv", dtype={"uid": str})
        y = meta["is_pathologic"].values.astype(float)
        feats = extract_features(out, cache, meta["uid"], num_workers=0)
        rep = {}
        for name, fc in (("random", "folds.csv"), ("site", "folds_site.csv")):
            if not (cache / fc).exists():
                continue
            folds = pd.read_csv(cache / fc, dtype={"uid": str}).set_index("uid").loc[meta.uid, "fold"].values
            p = linear_probe_oof(feats, y, folds)
            rep[name] = {"auroc": float(roc_auc_score(y, p)), "logloss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))}
            print(f"[ssl_c] linear probe ({name} folds): AUROC={rep[name]['auroc']:.4f} logloss={rep[name]['logloss']:.4f}", flush=True)
        out.with_suffix(".probe.json").write_text(json.dumps(rep))
