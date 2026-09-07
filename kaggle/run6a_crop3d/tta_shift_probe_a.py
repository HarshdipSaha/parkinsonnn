"""Probe: does integer-voxel shift TTA (roll +-1 in x / y of the 48x48x32 crop, x flip) help the run-4a crop-3D checkpoint?
Uses the fold-0 checkpoint on its own validation fold (273 scans, CPU). Prints log loss for flip-only vs flip+shift variants.
python kaggle/run6a_crop3d/tta_shift_probe_a.py [--art ...] [--cache ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datsc.data_v3 import VolDataset, VolumeCache
from datsc.models_v3 import build_model
from datsc.train_v3 import InputSpec, auroc_np, logloss_np

ap = argparse.ArgumentParser()
ap.add_argument("--art", default="H:/augsepthacks/parkinsonnn/data/ckpt_ds_v4/artifacts_v3")
ap.add_argument("--cache", default="H:/augsepthacks/parkinsonnn/cache_v4")
ap.add_argument("--tag", default="r3d_crop48x48x32_f0_s0")
a = ap.parse_args()
meta = json.loads(Path(a.art, a.tag + ".json").read_text()); spec = InputSpec(**meta["spec"])
cache = VolumeCache(a.cache)
folds = pd.read_csv(Path(a.cache) / meta.get("folds_csv", "folds.csv"), dtype={"uid": str})
va = folds[(folds["fold"] == meta["fold"]) & folds["uid"].isin(cache.index)]
dl = DataLoader(VolDataset(cache, va["uid"], va["is_pathologic"], meta.get("norm", "ref")), batch_size=32, num_workers=0)
model = build_model("3d", **meta["model_kw"]); model.load_state_dict(torch.load(Path(a.art, a.tag + ".pt"), map_location="cpu", weights_only=True)); model.eval()
SHIFTS = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
Z = {s: [] for s in SHIFTS}; Zf = {s: [] for s in SHIFTS}; ys = []
with torch.no_grad():
    for x, s_, y, u in dl:
        inp = spec(x)
        for sh in SHIFTS:
            v = torch.roll(inp, shifts=sh, dims=(2, 3, 4)) if any(sh) else inp
            Z[sh].append(model(v).float()); Zf[sh].append(model(spec.flip(v)).float())
        ys.append(y)
y = torch.cat(ys).numpy().astype(float)
def ll(zs):
    z = torch.stack([torch.cat(t) for t in zs]).mean(0); p = torch.sigmoid(z).numpy(); return logloss_np(y, p), auroc_np(y, p)
print("no TTA          : logloss %.4f auroc %.4f" % ll([Z[(0, 0, 0)]]))
print("flip            : logloss %.4f auroc %.4f" % ll([Z[(0, 0, 0)], Zf[(0, 0, 0)]]))
print("flip + xy shifts: logloss %.4f auroc %.4f" % ll([Z[s] for s in SHIFTS[:5]] + [Zf[s] for s in SHIFTS[:5]]))
print("flip + xyz shift: logloss %.4f auroc %.4f" % ll([Z[s] for s in SHIFTS] + [Zf[s] for s in SHIFTS]))
