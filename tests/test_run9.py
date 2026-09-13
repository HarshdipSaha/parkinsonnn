"""Run 9: the endgame guards — what the prior mix can and cannot buy, and the two build checks that were blind."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from sweep_calibration import EPS, worst_case_nats  # noqa: E402

POOL = ROOT / "kaggle" / "run8_distill" / "soft_master_v2.csv"   # Master v2 nested OOF, as shipped (T 0.9, lam 0.02)
SHIPPED_LAM = 0.02


def _logloss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _pool():
    d = pd.read_csv(POOL, dtype={"uid": str})
    y, p = d.y.values.astype(float), d.p.values.astype(float)
    pi = y.mean()
    q = np.clip((p - SHIPPED_LAM * pi) / (1 - SHIPPED_LAM), 1e-9, 1 - 1e-9)   # undo the shipped prior mix
    return y, q, pi


def test_worst_case_nats_prices_the_clip():
    """lam is insurance: it caps what one confidently-wrong scan can cost. At lam=0 only EPS caps it."""
    w1_0, w0_0 = worst_case_nats(0.0, 0.5485)
    assert abs(w1_0 - (-np.log(EPS))) < 1e-9 and abs(w0_0 - (-np.log(EPS))) < 1e-9   # 9.21 nats, the bare clip
    w1_01, _ = worst_case_nats(0.01, 0.5485)
    assert abs(w1_01 - (-np.log(0.01 * 0.5485))) < 1e-9
    assert w1_01 < w1_0 - 3.5, "dropping lam 0.01 -> 0 must cost >3.5 nats on a confident miss"
    worst = [worst_case_nats(l, 0.5485)[0] for l in (0.0, 0.005, 0.01, 0.02, 0.03)]
    assert worst == sorted(worst, reverse=True), "worst case must fall monotonically as lam rises"


def test_pool_reproduces_the_shipped_master_v2_number():
    y, q, pi = _pool()
    assert len(y) == 1362 and abs(pi - 0.5485) < 1e-3
    assert abs(_logloss(y, (1 - SHIPPED_LAM) * q + SHIPPED_LAM * pi) - 0.2323) < 5e-4


def test_lambda_only_gain_is_bounded():
    """Guard against the run-8 misreading: lam 0.02 -> 0.01 was credited with +0.0084 public.

    With the arms fixed, lam is a pure squash of the probabilities, so its gain is capped no matter what the
    data looks like. Anything larger came from the ensemble changing (e.g. the ladder dropping arms), not lam.
    """
    y, q, pi = _pool()
    L = lambda lam: _logloss(y, (1 - lam) * q + lam * pi)   # noqa: E731

    one, zero = np.array([1.0]), np.array([0.0])
    best = lambda lam: (1 - lam) * 1.0 + lam * pi   # noqa: E731  a maximally confident CORRECT scan
    ceiling = max(
        _logloss(one, np.array([best(0.02)])) - _logloss(one, np.array([best(0.01)])),
        _logloss(zero, np.array([0.02 * pi])) - _logloss(zero, np.array([0.01 * pi])),
    )
    assert ceiling < 0.006, f"analytic ceiling on a lam 0.02->0.01 move, got {ceiling:.5f}"

    observed = L(0.02) - L(0.01)
    assert 0.0015 < observed < 0.0030, f"realistic lam 0.02->0.01 gain on the real pool, got {observed:.5f}"
    assert observed < ceiling, "no dataset can beat the ceiling"

    # The other side of the same coin: on a confident MISS, the lower lam is strictly worse.
    miss = _logloss(one, np.array([0.01 * pi])) - _logloss(one, np.array([0.02 * pi]))
    assert miss > 0.6, f"lam 0.02->0.01 costs {miss:.2f} nats on each confident miss"

    assert L(0.01) - L(0.0) < 0.0025, "lam 0.01 -> 0 buys very little"
    lams = [0.0, 0.005, 0.01, 0.02, 0.03]
    assert [L(x) for x in lams] == sorted(L(x) for x in lams), "no-shift pool is monotone in lam; it cannot price tail risk"


def test_build_master_v3_refuses_an_oversized_zip():
    """DROP_ORDER can be exhausted with the zip still over the cap; it used to be written anyway."""
    s = (ROOT / "scripts" / "build_master_v3.py").read_text(encoding="utf-8")
    i = s.index("for arm_name, key in DROP_ORDER")
    tail = s[i:s.index('print(f"wrote {out}', i)]
    assert "raise SystemExit" in tail or "assert" in tail, "no size guard after the trim loop"


def test_build_master_v3_refuses_a_silently_trimmed_zip():
    """A *successful* trim is the subtler bug: the gate table scored the untrimmed set, and dropping an arm
    also renormalises the weights, so the zip that ships is not the one that was measured."""
    s = (ROOT / "scripts" / "build_master_v3.py").read_text(encoding="utf-8")
    assert "trimmed and not allow_trim" in s, "a successful trim must be fatal by default"
    assert '"--allow-trim"' in s and "a.allow_trim" in s, "--allow-trim must exist and be wired through"


def test_local_e2e_catches_a_silent_arm():
    """featc/LR arms never print 'checkpoints used N/N', so the old assertion could not see them die."""
    s = (ROOT / "scripts" / "local_e2e_v3.py").read_text(encoding="utf-8")
    assert "feat model|LR" in s and "SILENT IN OUTPUT" in s
    assert "ladder: " in s and "DEGRADED RUN" in s, "a ladder-truncated run must fail the gate"
    assert 'man.get("typ") is None' in s, "typ must stay off: it prints test-set statistics"
