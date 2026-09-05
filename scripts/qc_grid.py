"""PNG grids of cached crops for eyeballing. Usage: python scripts/qc_grid.py --cache cache --out qc"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def grid(cache: Path, uids, title, out_png: Path, ncol=6):
    n = len(uids)
    if n == 0:
        return
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.2 * ncol, 2.4 * nrow))
    for ax, uid in zip(np.ravel(axes), list(uids) + [None] * (nrow * ncol - n)):
        ax.axis("off")
        if uid is None:
            continue
        z = np.load(cache / f"{uid}.npz")
        crop = z["crop"].astype(np.float32)
        ax.imshow(crop[:, :, crop.shape[2] // 2].T, cmap="hot", origin="lower", vmin=0, vmax=max(crop.max(), 1e-3))
        ax.set_title(uid, fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="qc")
    a = ap.parse_args()
    cache, out = Path(a.cache), Path(a.out)
    out.mkdir(exist_ok=True)
    meta = pd.read_csv(cache / "meta.csv", dtype={"uid": str})
    ok = meta[meta["ok"]]
    rng = np.random.default_rng(0)
    grid(cache, rng.choice(ok["uid"], min(30, len(ok)), replace=False), "random", out / "random.png")
    grid(cache, ok[ok["n_levels"] < 12]["uid"], "ultra-low grey levels (<12)", out / "ultralow.png")
    grid(cache, ok[ok["z_extent_mm"] < 120]["uid"], "short z extent (<120mm)", out / "short_z.png")
    grid(cache, ok[ok["fallback"]]["uid"], "anchor fallback", out / "fallback.png")
    print(meta[["ok", "fallback"]].agg(["sum", "mean"]))
    print("ultra-low:", int((ok["n_levels"] < 12).sum()), "short-z:", int((ok["z_extent_mm"] < 120).sum()))
