"""DrivenData submission entrypoint. Writes a prior-only submission.csv immediately, then overwrites it
stage-wise (LR -> CNN ensemble) under a wall-clock ladder so any failure still leaves a valid file."""
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "4")
import sys
import time
from pathlib import Path

T0 = time.time()
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from datsc import config, infer
from datsc.calibrate import Calibration

ARTIFACTS = Path(os.environ.get("DATSC_ARTIFACTS", ROOT / "artifacts"))


def selftest():
    import tempfile
    from tests_synth import make_synthetic_volume, write_nifti
    from datsc.preprocess import preprocess
    with tempfile.TemporaryDirectory() as td:
        for i, kw in enumerate(({}, {"levels": 8, "shape": (128, 128, 25), "spacing": (3.9,) * 3},
                                {"shape": (256, 256, 256), "spacing": (2.46,) * 3})):
            vol, sp = make_synthetic_volume(seed=i, **kw)
            p = write_nifti(Path(td) / f"s{i}.nii.gz", vol, sp)
            r = preprocess(p)
            assert r["crop"].shape == config.CROP_SHAPE and np.isfinite(r["scalars"]).all()
    print("SELFTEST OK")


def main():
    fmt = pd.read_csv(config.RUNTIME_SUBMISSION_FORMAT, dtype={"uid": str})
    uids = fmt["uid"].astype(str).tolist()
    calib = Calibration.load(ARTIFACTS / "calib.json") if (ARTIFACTS / "calib.json").exists() else \
        Calibration(w=0.0, T=1.0, lam=0.0, pi=config.DEFAULT_PRIOR)
    pi = calib.pi
    infer.write_submission(uids, np.full(len(uids), pi), config.RUNTIME_OUTPUT_CSV)   # stage 0: prior
    print(f"stage0 prior written for {len(uids)} rows at {time.time()-T0:.0f}s", flush=True)

    n_workers = max(1, min(20, (os.cpu_count() or 4) - 2))
    crops, crops_com, sc, rad, ok = infer.preprocess_all(uids, config.RUNTIME_NIFTI_DIR, n_workers,
                                                         deadline_s=infer.LADDER_SKIP_CNN_S, t0=T0)
    p_lr = np.full(len(uids), pi)
    if (ARTIFACTS / "lr.json").exists():
        p_lr = infer.run_lr(ARTIFACTS / "lr.json", rad, sc, ok, pi)
        # LR arm alone: T was fitted on the CNN/LR blend, so apply only the prior-mix hedge here
        lr_only = Calibration(w=0.0, T=1.0, lam=max(calib.lam, 0.02), pi=pi)
        infer.write_submission(uids, np.where(ok, lr_only.apply(p_lr, p_lr), pi), config.RUNTIME_OUTPUT_CSV)
        print(f"stage1 LR written at {time.time()-T0:.0f}s", flush=True)

    all_w = sorted(ARTIFACTS.glob("cnn_f*_s*.pt"))
    hot_w = [p for p in all_w if "_com" not in p.name]
    com_w = [p for p in all_w if "_com" in p.name]
    if all_w and time.time() - T0 < infer.LADDER_SKIP_CNN_S:
        p_cnn = infer.run_cnn([(hot_w, crops), (com_w, crops_com)], sc, ok, T0)
        if p_cnn is not None:
            final = calib.apply(np.where(np.isnan(p_cnn), p_lr, p_cnn), p_lr)
            final = np.where(ok, final, pi)
            infer.write_submission(uids, final, config.RUNTIME_OUTPUT_CSV)
            print(f"stage2 CNN ensemble written at {time.time()-T0:.0f}s", flush=True)
    print(f"done in {time.time()-T0:.0f}s; failed volumes: {int((~ok).sum())}", flush=True)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
        sys.stdout.flush()
        os._exit(0)   # submission.csv is already on disk; never let a hung worker process keep us alive
