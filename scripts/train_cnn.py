"""python scripts/train_cnn.py --fold 0 --seed 0 [--control] [--epochs 40] [--cache cache] [--out artifacts]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datsc.train import train_fold

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--control", action="store_true", help="train on plain CoM crops (baseline)")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    train_fold(a.cache, a.fold, a.seed, a.out, epochs=a.epochs, batch_size=a.batch_size,
               num_workers=a.workers, control=a.control)
