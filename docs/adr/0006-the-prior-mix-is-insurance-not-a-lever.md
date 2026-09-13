# 6. The prior mix is insurance, not a lever

Date: 2026-09-13

## Status

Accepted

## Context

Slot 1 shipped Master v3 at λ = 0.02 and scored 0.2557. Slot 2 shipped the same seven arms at λ = 0.01 and
scored 0.2473. `RESULTS_LOG.md` recorded the difference as the prior mix "living up to its nested estimate"
and the plan for the last slot became: push λ lower again.

That reading cannot be right. With the arms fixed, λ is a pure squash applied after the temperature:
`p = (1 − λ)·σ(z/T) + λ·π`. Its effect on log loss is therefore bounded by arithmetic, independent of the data.
Evaluated at π = 0.5485, the largest improvement a move from λ = 0.02 to λ = 0.01 can produce on any scan is
0.0055 nats, and that is reached only by a scan the ensemble is already maximally confident *and* correct about.
On the real nested pool (`kaggle/run8_distill/soft_master_v2.csv`, 1,362 scans) the move is worth **0.0020**.
The recorded public gain of 0.0084 is 4× the realistic effect and above the hard ceiling.

So λ did not buy 0.0084. Something else differed between the two submissions.

The obvious suspect was the wall-clock ladder (`datsc/infer_v3.py:24`, `main_v3.py:68`), which drops whole arms
once `LADDER_STOP_S` passes while `combine()` renormalises over the survivors — silently. **It has been ruled
out.** Slots 1 and 2 uploaded the same 847 MB / 71-checkpoint zip, so their runtime is identical; and from the
one platform-measured anchor a 71-checkpoint run costs ≈ 24 min against the 9000 s ladder, so truncation would
need on the order of 1,580 checkpoints. Neither run can have been truncated.

That leaves the live hypothesis: **one of the recorded numbers is simply wrong.** Neither real submission
carries a DrivenData id, though `aidlc-docs/HANDOFF.md` requires one after an earlier fabrication incident;
slot 2's rank is written as "Estimated"; and the entry was byte-corrupted by a script rather than transcribed.
There is a free test that settles it: AUROC is on the leaderboard and λ is strictly monotone, so it cannot
change any ranking — **two genuine λ-only twins must report identical AUROC (0.9576)**. Slot 2's AUROC was
never recorded. Reading it decides the question at zero cost.

The nested pool cannot settle where λ should sit, because it is built from random folds and contains no site
shift. Its log loss is monotone decreasing in λ, so it always recommends λ → 0. That recommendation prices
only the upside. λ's real job is the downside: it floors a confidently wrong scan at −log(λ·π), which is 5.2
nats at λ = 0.01 and 9.21 nats at λ = 0 (where only `EPS = 1e-4` remains, `datsc/calibrate.py:11,48`).

## Decision

**Ship T = 0.90, λ = 0.01, `clip = null`, unchanged.**

The deep-research pass (`docs/research/dat-endgame/results/calibration.json`) strengthened this from a cautious
default into a measured one, and corrected two things along the way.

- **λ turns on at G ≈ 0.020, not G ≈ 0.05.** Applying the measured per-scan leave-pseudo-site-out logit
  displacement (same 1,362 uids; mean +0.034, sd 0.849; 14 scans moving > 2 nats toward the wrong class) to the
  shipped ensemble gives λ\*(G) = 0 at G ≤ 0.018, 0.0055 at G = 0.020, and 0.0165 at G = 0.028. At the observed
  nested→public gap of 0.0207, λ\* ≈ 0.006–0.010 — so **λ = 0.01 sits inside the bracket on evidence**, not
  merely as insurance.
- **The whole (T, λ) question is worth ≤ 0.0018 nats.** Across four shift levels the shipped cell has max
  regret 0.00182 against 0.00088 for the minimax cell (T = 0.90, λ = 0.005) — a 0.0009 edge smaller than the
  shift model's own uncertainty. λ = 0 is the second-worst cell tested (0.00260). Not worth the last slot.
- **T = 0.90 is already the right hedge.** The five-pillar pool's own nested optimum is T = 0.870 (not 0.815,
  which is run 4a's manifest value), and T = 0.90 costs 0.00015 against it. Under shift the temperature
  *flattens* — T\* rises 1.137 → 1.281 (featarm), 1.220 → 1.296 (featcarm) — so T = 0.90 errs in the safe
  direction. **`RESULTS_LOG.md`:321 reads this backwards**: its "T_site 0.826 vs T_random ~1.0" compares a
  fitted site value against a *floored* one (line 313 sets the floor to 1.0), while
  `artifacts/run4a_stack/manifest.json:4` records the unfloored T_random = 0.8148.
- **`clip` is fully subsumed by λ = 0.01**, which already bounds p to [0.005485, 0.995485] against EPS's
  [1e-4, 1 − 1e-4]; EPS binds on 0 of 1,362 scans. No clip setting beats plain λ = 0.01 by more than 0.0004 at
  any shift level.

λ = 0 is not shipped: it removes the last cap on a confident miss to buy at most 0.0018 nested. The exposure is
**3.01 extra nats** per confidently-wrong scan, not the 4.00 a p = 0 bound would suggest — this ensemble's most
extreme prediction is 0.999714, never 0 or 1 — but the break-even still lands under one scan in 2,000.

No public-leaderboard movement is attributed to a calibration constant again without first checking that the
two submissions ran the same arms — read `checkpoints used N/N` and the absence of `ladder:` in the DrivenData
submission log before drawing any conclusion from a score difference.

## Consequences

- The last slot is not spent on another λ bracket; the remaining headroom there is ~0.002 at best and the
  downside is unbounded.
- `tests/test_run9.py::test_lambda_only_gain_is_bounded` pins the ceiling and the realistic gain, so the same
  misreading cannot be re-derived later.
- `scripts/sweep_calibration.py` reports the tail-risk and break-even columns beside the nested number, because
  the nested number alone will always argue for less insurance than is wise.
- The ladder becomes a first-class thing to verify rather than an implementation detail: if it has been firing,
  the arms that justified Master v3 over Master v2 are the ones being dropped, since they run last.
