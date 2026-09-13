"""Sweep the combiner's temperature floor and prior mix over the nested pool, next to the tail risk each lam buys.

python scripts/sweep_calibration.py [--src data/run8/artifacts_v3 --src data/master_pool/artifacts_v3]
                                    [--arms a3d_hemi@all,a3d_ssl@all,...] [--T 0.8,0.9,1.0] [--lam 0,0.005,0.01,0.02]

Why this script exists: the nested pool is built from random folds and therefore contains no site shift, so its
log loss is monotone decreasing in lam and it will always recommend lam -> 0. That recommendation is only safe if
the hidden test holds no scans the ensemble is confidently wrong about. lam is the insurance against exactly those:
a maximally confident wrong scan costs -log(lam*pi) nats, against -log(EPS) = 9.21 at lam = 0 (datsc/calibrate.py:11,48).
The WORST column prices that insurance, and BREAK-EVEN says how many such scans make the cheaper lam the wrong choice.
Read both columns before moving lam. The smoke set is never fitted on (ADR 0005).
"""
import argparse, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

EPS = 1e-4  # datsc/calibrate.py:11 — the floor the shipped combiner clips to


def worst_case_nats(lam, pi, q_lo=0.0, q_hi=1.0):
    """Loss of a single confidently wrong scan at this lam.

    Defaults give the theoretical bound (the model predicting 0 or 1 exactly). Pass the pool's observed
    extremes instead for the realistic figure: this ensemble never reaches 0 or 1, so the theoretical bound
    overstates what lam=0 actually risks (4.0 nats of extra exposure against a real ~3.0).
    """
    p_when_truth_is_1 = max((1 - lam) * q_lo + lam * pi, EPS)
    p_when_truth_is_0 = min((1 - lam) * q_hi + lam * pi, 1 - EPS)
    return -np.log(p_when_truth_is_1), -np.log(1 - p_when_truth_is_0)


def main():
    import build_master_v3 as bm  # deferred: pulls in scipy/sklearn via fit_stack_v3, so importing this module stays cheap

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", default=None)
    ap.add_argument("--cache", default="cache_v4")
    ap.add_argument("--arms", default=None, help="comma-separated pool keys; default = pillars@all + every run-8 arm present")
    ap.add_argument("--T", default="0.80,0.85,0.90,0.95,1.00")
    ap.add_argument("--lam", default="0,0.0025,0.005,0.0075,0.01,0.015,0.02,0.03")
    ap.add_argument("--mode", default="equal", choices=["equal", "shrunk", "fitted"])
    ap.add_argument("--n-test", type=int, default=2000, help="assumed hidden test size, for the break-even count")
    a = ap.parse_args()

    srcs = a.src or ["data/run8/artifacts_v3", "data/master_pool/artifacts_v3"]
    Ts = [float(x) for x in a.T.split(",")]
    lams = [float(x) for x in a.lam.split(",")]

    pool = bm.Pool(srcs, a.cache, Ts[0], lams[0])
    if a.arms:
        names = [n.strip() for n in a.arms.split(",")]
        missing = [n for n in names if n not in pool.oofs]
        if missing:
            raise SystemExit(f"not in pool: {missing}\navailable: {sorted(pool.oofs)}")
    else:
        names = [n + "@all" for n in bm.PILLARS if n + "@all" in pool.oofs] + [n for n in bm.NEW if n in pool.oofs]
    print(f"pool: {len(pool.uids)} scans, prevalence {pool.y.mean():.4f}; arms ({len(names)}): {', '.join(names)}")

    grid = {}
    print(f"\nNESTED LOG LOSS ({a.mode} weights)")
    print("   T    " + "".join(f"lam={l:<9.4f}" for l in lams))
    for T in Ts:
        pool.T_floor = T
        row = f"{T:>5.2f}  "
        for lam in lams:
            pool.lam = lam
            grid[(T, lam)] = pool.nested(names, a.mode)
            row += f"{grid[(T, lam)]:<13.5f}"
        print(row)

    (bT, bl), bll = min(grid.items(), key=lambda kv: kv[1])
    print(f"\nnested optimum: T={bT:.2f} lam={bl:.4f} -> {bll:.5f}   (NO SHIFT IN THIS POOL — see the header)")

    # Price the tail against what this ensemble actually predicts, not against 0 and 1 which it never reaches.
    Zs = pool.Z(names); weq = np.ones(len(names)) / len(names)
    q_obs = 1.0 / (1.0 + np.exp(-(weq[:, None] * Zs).sum(0) / bT))
    q_lo, q_hi = float(q_obs.min()), float(q_obs.max())

    print(f"\nTAIL RISK AT THE NESTED-OPTIMAL T={bT:.2f}")
    print(f"observed combined-probability range: [{q_lo:.6f}, {q_hi:.6f}] "
          f"(theoretical bound would use [0, 1] and overstate lam=0's exposure by ~1 nat)")
    print(f"{'lam':>8} {'nested':>10} {'vs lam=.01':>11} {'WORST y=1':>10} {'WORST y=0':>10} {'BREAK-EVEN vs lam=.01':>23}")
    pool.T_floor = bT
    ref = grid[(bT, 0.01)] if (bT, 0.01) in grid else None
    w1_ref, _ = worst_case_nats(0.01, pool.pi, q_lo, q_hi)
    for lam in lams:
        ll = grid[(bT, lam)]
        w1, w0 = worst_case_nats(lam, pool.pi, q_lo, q_hi)
        cell = ""
        if ref is not None and lam != 0.01:
            gain, extra = ref - ll, w1 - w1_ref
            cell = "safer" if gain < 0 or extra <= 0 else f"{gain * a.n_test / extra:.0f} scans of {a.n_test}"
        print(f"{lam:>8.4f} {ll:>10.5f} {(ref - ll) if ref is not None else float('nan'):>+11.5f} {w1:>10.2f} {w0:>10.2f} {cell:>23}")
    print(f"\nBREAK-EVEN = how many confidently-wrong scans in {a.n_test} erase that lam's nested gain over lam=0.01.")
    print("If you would not bet against that many, keep lam=0.01.")


if __name__ == "__main__":
    main()
