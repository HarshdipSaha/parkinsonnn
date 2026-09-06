"""DrivenData submission entrypoint (run 3). Prior CSV first, then a manifest-driven multi-arm ensemble
written stage-wise under a wall-clock ladder so any failure still leaves a valid submission.csv."""
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

from datsc import config, infer_v3 as infer

ARTIFACTS = Path(os.environ.get("DATSC_ARTIFACTS", ROOT / "artifacts"))


def selftest():
    import tempfile
    from tests_synth import make_synthetic_volume, write_nifti
    from datsc.volume import VOL_SHAPE, prepare_volume
    from datsc.register import load_template
    tpl = load_template(ARTIFACTS / "template.npz")
    with tempfile.TemporaryDirectory() as td:
        for i, kw in enumerate(({}, {"levels": 8, "shape": (128, 128, 25), "spacing": (3.9,) * 3},
                                {"shape": (256, 256, 256), "spacing": (2.46,) * 3})):
            vol, sp = make_synthetic_volume(seed=i, **kw)
            r = prepare_volume(write_nifti(Path(td) / f"s{i}.nii.gz", vol, sp), template=tpl)
            assert r["vol"].shape == VOL_SHAPE and np.isfinite(r["scalars"]).all() and np.isfinite(r["stats"]).all()
    print(f"selftest: template {'used' if tpl is not None else 'absent'}")
    if (ARTIFACTS / "manifest.json").exists():
        m = infer.load_manifest(ARTIFACTS / "manifest.json")
        for a in m["arms"]:
            for ck in a["checkpoints"]:
                assert (ARTIFACTS / Path(ck).name).exists(), ck
    print("SELFTEST OK")


def main():
    fmt = pd.read_csv(config.RUNTIME_SUBMISSION_FORMAT, dtype={"uid": str})
    uids = fmt["uid"].astype(str).tolist()
    man = infer.load_manifest(ARTIFACTS / "manifest.json") if (ARTIFACTS / "manifest.json").exists() else \
        {"pi": config.DEFAULT_PRIOR, "T": 1.0, "lam": 0.0, "weights": {}, "arms": []}
    pi = float(man.get("pi", config.DEFAULT_PRIOR))
    infer.write_submission(uids, np.full(len(uids), pi), config.RUNTIME_OUTPUT_CSV)
    print(f"stage0 prior written for {len(uids)} rows at {time.time()-T0:.0f}s", flush=True)

    n_workers = max(1, min(20, (os.cpu_count() or 4) - 2))
    tpl = ARTIFACTS / "template.npz"
    print(f"template: {'yes' if tpl.exists() else 'NO (CoM frame)'}; workers {n_workers}", flush=True)
    vols, sc, st, ok = infer.preprocess_all(uids, config.RUNTIME_NIFTI_DIR, n_workers, deadline_s=infer.LADDER_STOP_S, t0=T0,
                                            template_path=tpl if tpl.exists() else None)
    arm_probs = {}
    for a in man["arms"]:
        if time.time() - T0 > infer.LADDER_STOP_S:
            print("ladder: skipping remaining arms"); break
        try:
            arm_probs[a["name"]] = infer.run_arm(a, ARTIFACTS, vols, sc, st, ok, T0)
        except Exception as e:  # noqa: BLE001
            print(f"arm {a['name']} failed: {type(e).__name__}", flush=True); arm_probs[a["name"]] = None
        p = infer.combine(arm_probs, man["weights"], float(man["T"]), float(man["lam"]), pi, clip=man.get("clip"), hedge=man.get("hedge"))
        if p is not None:
            infer.write_submission(uids, np.where(ok & np.isfinite(p), p, pi), config.RUNTIME_OUTPUT_CSV)
            print(f"stage written after arm {a['name']} at {time.time()-T0:.0f}s", flush=True)
    print(f"done in {time.time()-T0:.0f}s; failed volumes: {int((~ok).sum())}", flush=True)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
        sys.stdout.flush()
        os._exit(0)
