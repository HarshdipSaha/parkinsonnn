"""Is the hard core irreducible label noise, or a shared preprocessing failure? One command decides.

    python scripts/diagnose_hard_core.py
    python scripts/diagnose_hard_core.py --pool kaggle/run8_distill/soft_master_v2.csv --cache cache_v4 --top 88

The endgame analysis found 88 scans (6.5 % of the pool) carrying 46.9 % of all loss and wrong under every
representation tried. That was read as irreducible label noise. But every arm shares the SAME upstream
preprocessing -- the registered 2 mm frame, the head box, the reference-region scaling -- so "wrong everywhere"
has a second explanation: one shared upstream defect. The published check only showed the errors do not cluster
by FOLD, and folds are random, so it could not have detected a site effect.

This script tests the alternative directly: do the hard-core scans concentrate in a pseudo-site, an intensity
regime, a voxel spacing, or the scans where registration fell back to the CoM frame? Enrichment is scored with
a label-permutation test, so no scipy is needed and the multiple-comparison problem is handled by permuting the
max statistic rather than testing each group separately.

Reads only cached artefacts. No GPU, no training, no submission. Needs numpy + pandas.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(0)


def permutation_max_enrichment(flag, groups, n_perm=20000):
    """p-value for 'some group is enriched in flag', correcting for the number of groups by permuting the max."""
    codes, _ = pd.factorize(groups)
    k = codes.max() + 1
    n = np.bincount(codes, minlength=k).astype(float)
    obs_counts = np.bincount(codes, weights=flag, minlength=k)
    base = flag.mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        obs = np.nanmax(np.where(n >= 10, obs_counts / n / base, np.nan))
    hits, f = 0, flag.copy()
    for _ in range(n_perm):
        RNG.shuffle(f)
        c = np.bincount(codes, weights=f, minlength=k)
        with np.errstate(divide="ignore", invalid="ignore"):
            m = np.nanmax(np.where(n >= 10, c / n / base, np.nan))
        hits += m >= obs
    return float(obs), (hits + 1) / (n_perm + 1)


def report_group(name, df, col, n_perm):
    if col not in df.columns or df[col].nunique() < 2:
        print(f"\n[{name}] column '{col}' absent or constant -- skipped")
        return None
    g = df[col].astype(str).fillna("?")
    base = df.hard.mean()
    tab = df.groupby(g).agg(scans=("hard", "size"), hard=("hard", "sum"), mean_loss=("loss", "mean"))
    tab["rate"] = tab.hard / tab.scans
    tab["enrich"] = tab["rate"] / base
    tab = tab.sort_values("enrich", ascending=False)
    print(f"\n[{name}]  overall hard-core rate {base:.3f}")
    print(f"  {'group':<22} {'scans':>6} {'hard':>5} {'rate':>7} {'enrich':>7} {'mean loss':>10}")
    for k, r in tab.iterrows():
        mark = "  <-- " if r.enrich >= 1.5 and r.scans >= 10 else ""
        print(f"  {k[:22]:<22} {int(r.scans):>6} {int(r.hard):>5} {r.rate:>7.3f} {r.enrich:>7.2f} {r.mean_loss:>10.4f}{mark}")
    obs, p = permutation_max_enrichment(df.hard.values.astype(float), g.values, n_perm)
    print(f"  max enrichment {obs:.2f}x over a group with >=10 scans; permutation p = {p:.4f}")
    return obs, p, tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="kaggle/run8_distill/soft_master_v2.csv", help="nested OOF with uid,y,p[,loss]")
    ap.add_argument("--cache", default="cache_v4", help="folder holding sites.csv / folds.csv")
    ap.add_argument("--top", type=int, default=88, help="hard core = this many worst-loss scans")
    ap.add_argument("--n-perm", type=int, default=20000)
    a = ap.parse_args()

    pool = Path(a.pool if Path(a.pool).is_absolute() else ROOT / a.pool)
    df = pd.read_csv(pool, dtype={"uid": str})
    if "loss" not in df.columns:
        p = np.clip(df.p.values, 1e-15, 1 - 1e-15)
        df["loss"] = -(df.y * np.log(p) + (1 - df.y) * np.log(1 - p))
    df = df.sort_values("loss", ascending=False).reset_index(drop=True)
    df["hard"] = (np.arange(len(df)) < a.top).astype(int)
    share = df.loc[df.hard == 1, "loss"].sum() / df.loss.sum()
    print(f"pool {pool.name}: {len(df)} scans; hard core = worst {a.top} "
          f"({100 * a.top / len(df):.1f} % of scans, {100 * share:.1f} % of total loss)")

    cache = Path(a.cache if Path(a.cache).is_absolute() else ROOT / a.cache)
    sites = cache / "sites.csv"
    if not sites.exists():
        print(f"\nFATAL: {sites} not found. This script must run on the training machine, where cache_v4/ exists.")
        print("Without it the site question cannot be answered and the 'irreducible noise' reading stays unverified.")
        raise SystemExit(2)
    s = pd.read_csv(sites, dtype={"uid": str})
    df = df.merge(s, on="uid", how="left")
    print(f"joined {sites.name}: columns {[c for c in s.columns if c != 'uid']}")

    results = {}
    for label, col in [("PSEUDO-SITE", "site"), ("INTENSITY REGIME", "regime"), ("VOXEL SPACING", "spacing"),
                       ("GRID SHAPE", "shape"), ("REGISTRATION FALLBACK", "fallback")]:
        for cand in (col, col + "_id", "pseudo_" + col, col.upper()):
            if cand in df.columns:
                r = report_group(label, df, cand, a.n_perm)
                if r:
                    results[label] = r
                break
        else:
            print(f"\n[{label}] no column matching '{col}' in sites.csv -- skipped")

    print("\n" + "=" * 78)
    sig = {k: v for k, v in results.items() if v[1] < 0.05 and v[0] >= 1.5}
    if sig:
        print("VERDICT: CONCENTRATED -- the hard core is NOT uniformly spread.")
        for k, (obs, p, _) in sig.items():
            print(f"  {k}: {obs:.2f}x enrichment, p = {p:.4f}")
        print("""
This is the good outcome. It means a slice of the data is being preprocessed badly rather than being
inherently ambiguous, and that is fixable. Next, in this order:
  1. Pull 10 hard-core scans from the most enriched group and 10 easy scans from the same group, and look at
     them with scripts/qc_grid.py. You are looking for a registration that put the striatum in the wrong
     place, a clipped field of view, or a reference region landing on skull or background.
  2. If registration is the fault: the per-scan fix is already in the codebase as the registration fallback.
     Check whether these scans took the fallback and whether the fallback is worse than the registration it
     replaced for this group.
  3. Only if 1-2 show a real defect, and only if it is fixable without touching datsc/ in a way that ships
     untested inference code, is this worth a rebuild before the deadline. Re-cache that group, re-run the
     affected arms, re-gate. Anything less than +0.003 nested is inside the pool's +/-0.013 noise.
  4. If the defect is real but too large to fix safely in the time left, write it up as the first item of the
     next run. It is the most valuable thing you have found.""")
    else:
        print("VERDICT: UNIFORM -- no group carries the hard core.")
        print("""
The 'irreducible label noise' reading survives. These scans are genuinely ambiguous, which matches the ~7.8 %
discrepant-expert-read rate for DaT-SPECT in the literature. Consequences:
  1. Stop looking for accuracy. More capacity, more seeds and more members will not move these scans.
  2. Stop looking at calibration too -- it was measured at 0.0002 of headroom, and both 'never be confident'
     and hedge-on-disagreement were shown to lose outright.
  3. Ship the best gated ensemble you already have and spend the remaining time on verification.
  4. The ceiling is real. Record it so the next run does not re-litigate it.""")
    print("=" * 78)


if __name__ == "__main__":
    main()
