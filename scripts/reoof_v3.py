"""Recompute OOF predictions for every CV checkpoint in --art with a given TTA (so stacking weights are fitted on
exactly what main.py produces). Updates oof_<tag>.csv, and val_logloss/val_auroc + spec.tta_z in <tag>.json.

python scripts/reoof_v3.py --art artifacts_v3 --cache cache_v4 --tta-z -1 0 1 [--only "r2d_*"]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datsc.data_v3 import VolDataset, VolumeCache
from datsc.models_v3 import build_model
from datsc.train_v3 import InputSpec, auroc_np, logloss_np, predict

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default="artifacts_v3")
    ap.add_argument("--cache", default="cache_v4")
    ap.add_argument("--tta-z", type=int, nargs="+", default=[-1, 0, 1])
    ap.add_argument("--tta-affine", type=int, default=None, help="override the spec's affine-TTA count (None = keep the checkpoint's)")
    ap.add_argument("--only", default="*")
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()
    art, cache = Path(a.art), VolumeCache(a.cache)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for jp in sorted(glob.glob(str(art / f"{a.only}.json"))):
        meta = json.loads(Path(jp).read_text())
        if "spec" not in meta or meta.get("fold", -1) < 0 or meta["spec"].get("arm") not in ("2d", "3d"):
            continue
        tag = meta["tag"]; ck = art / f"{tag}.pt"
        if not ck.exists():
            continue
        spec_d = dict(meta["spec"]); spec_d["tta_z"] = a.tta_z if spec_d["arm"] == "2d" else [0]
        if a.tta_affine is not None:
            spec_d["tta_affine"] = a.tta_affine
        spec = InputSpec(**spec_d)
        folds = pd.read_csv(Path(a.cache) / meta.get("folds_csv", "folds.csv"), dtype={"uid": str})
        va = folds[(folds["fold"] == meta["fold"]) & folds["uid"].isin(cache.index)]
        dl = DataLoader(VolDataset(cache, va["uid"], va["is_pathologic"], meta.get("norm", "ref")), batch_size=a.batch_size, num_workers=2)
        model = build_model(spec.arm, **meta["model_kw"]).to(device)
        model.load_state_dict(torch.load(ck, map_location=device, weights_only=True)); model.eval()
        p, uids = predict(model, dl, spec, device, tta_flip=True)
        y = va.set_index("uid").loc[uids, "is_pathologic"].values.astype(float)
        old_ll = meta.get("val_logloss")
        meta["val_logloss"], meta["val_auroc"], meta["spec"] = logloss_np(y, p), auroc_np(y, p), spec.to_dict()
        pd.DataFrame({"uid": uids, "y": y, "p": p}).to_csv(art / f"oof_{tag}.csv", index=False)
        Path(jp).write_text(json.dumps(meta))
        print(f"{tag}: logloss {old_ll if old_ll is None else round(old_ll, 4)} -> {meta['val_logloss']:.4f}  auroc {meta['val_auroc']:.4f}  tta_z={spec.tta_z} tta_affine={spec.tta_affine}", flush=True)
