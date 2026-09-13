# Spec: Run 9 — the last submission (effort 008, wayfinder ticket T16)

Written 2026-09-13 09:05 UTC. Deadline 2026-09-16 23:59 UTC (87 h). One real submission remains.

## Problem Statement

Five real submissions are spent. The sixth and last slot opens **Sept 14 03:38 UTC** and, because the next
expiry after that is Sept 19, it is the only one left before the deadline. The standing plan (`forharsh.md` §5,
`HANDOFF.md` §4) spends it at 03:38 on `submission_master_v2.zip` and then declares "no training fits any more,
only assembly."

Three things in that plan are wrong.

1. **The slot opens; it does not expire.** It stays open until Sept 16 23:59. Submitting at 03:38 throws away
   ~60 h of runway and spends the last option on a zip whose nested estimate (0.2323) is *worse* than the
   0.2266 already banked. The plan also assumes ≈10 GPU-hours remain; the real figure is 30+.
2. **The λ conclusion it chases is arithmetically impossible.** See ADR 0006. λ 0.02 → 0.01 is capped at
   0.0055 nats and is worth 0.0020 on the real nested pool; it was credited with 0.0084 on the public board.
   The plan's next step — push λ lower again — buys at most 0.0018 more and removes the last cap on a
   confident miss (9.21 nats at λ=0 against 5.2 at λ=0.01).
3. **The binding constraint is zip size, not GPU.** `build_master_v3.py --max-mb` defaults to 850 and the
   shipped zip is 847 MB with 71 checkpoints: ~3 MB of headroom, a quarter of one checkpoint. Any new seed
   silently triggers `DROP_ORDER`, whose **first** action is to delete `a2d_v2s` seed 1 — a pruning the repo
   itself measures at ≈0.01 nats, five times any gain the new seed could bring. Training more seeds without
   first raising the ceiling makes the submission worse.

Underneath all three sits an unverified ledger: neither claimed real submission (0.2557, 0.2473) carries a
DrivenData id, though the repo's own rule (`HANDOFF.md`:126) requires one, and the entry recording the second
was byte-corrupted. The same ambiguity has a mechanical cause worth ruling out: the wall-clock ladder can drop
whole arms and `combine()` renormalises over the survivors, so two 847 MB / 71-checkpoint zips can ship two
different ensembles with nothing in the logs read back.

## Solution

Spend the last slot as late as is safe, on the best ensemble that fits inside a deliberately raised size
envelope, with every silent-degradation path closed first.

1. **Verify before deciding.** Read the submissions page and the two `View log` outputs. Settle: how many slots
   actually remain, what the two real scores and their AUROCs were, and — decisively — whether every arm ran
   (`checkpoints used N/N` seven times, no `ladder:` line). If the ladder fired, the arms that justified Master
   v3 are the ones being dropped, because they run last.
2. **Close the blind spots in the build tools** so the final zip cannot degrade unnoticed: a hard size assert,
   an arm-coverage check that can see the non-torch arms, and a refusal to ship a manifest that would print
   test-set statistics.
3. **Choose the calibration on evidence, not on the nested curve alone.** `scripts/sweep_calibration.py`
   prints nested log loss beside the worst-case cost and the break-even count, on the real master pool.
   Default: stay at λ = 0.01, T = 0.9.
4. **Spend GPU inside the size envelope.** Raise `--max-mb` deliberately, add seeds only to arms already in the
   greedy best subset, and re-gate. Every added checkpoint must be paid for by a nested gain larger than what
   `DROP_ORDER` would cost if the zip overflows.
5. **Ship late, verified.** Freeze Sept 16 ~12:00, smoke-test the exact zip, upload by ~16:00, with ~8 h margin.

## User Stories

**As Harshdip, I want to know what actually shipped in slots 1 and 2**, so that a score difference tells me
something about the model rather than about which arms happened to finish.
*Acceptance:* the submissions page rows are transcribed into `RESULTS_LOG.md` with ids; for each of the two
real submissions, `checkpoints used N/N` per arm and the presence or absence of `ladder:` is recorded.

**As Harshdip, I want the build to refuse a zip that is not what the gate table measured**, so that I cannot
upload a silently trimmed or silently truncated ensemble on my only remaining slot.
*Acceptance:* `build_master_v3.py` exits non-zero when the zip is still over `--max-mb` after `DROP_ORDER`;
`local_e2e_v3.py` fails when any manifest arm produced no output, when a `ladder:` or failure line appears,
or when `typ` is set.

**As Harshdip, I want the prior mix chosen against its downside**, so that I do not trade 0.002 of nested gain
for an unbounded tail on a shifted private set.
*Acceptance:* `sweep_calibration.py` runs on the real master pool and prints, for each λ, the nested loss, the
worst-case nats, and how many confidently-wrong scans erase that λ's gain over λ = 0.01.

**As Harshdip, I want to use the remaining GPU without making the submission worse**, so that extra seeds are
a gain rather than a trade against `a2d_v2s` seed 1.
*Acceptance:* no zip is built whose checkpoint set differs from the set its gate table scored; `--max-mb` is
set explicitly and the build reports the final size and per-arm counts.

**As Harshdip, I want the last upload to be recoverable**, so that a broken zip does not end the competition.
*Acceptance:* the zip is frozen and smoke-tested before the real upload, with the fallback zip on disk and
≥ 6 h between the real upload and the deadline.

## Implementation Decisions

- **λ stays 0.01, T stays 0.9** unless the break-even column says otherwise (ADR 0006). λ = 0 is not shipped.
- **`--max-mb` is raised explicitly, not by editing `DROP_ORDER`.** A **1421 MB / 39-checkpoint zip already
  smoke-scored successfully** (`id-321496`), and a 1.8 GB zip was once rejected, so the real upload limit lies
  in [1421, 1815) MB and 1,050 MB is comfortably conservative. `DROP_ORDER`'s first entry stays untouched
  precisely because triggering it is the failure being guarded against, not a trim strategy — and triggering
  it *successfully* is now fatal by default, because the gate table scored the untrimmed set and dropping an
  arm renormalises the weights.
- **Checkpoint size is per-arm, not an average.** fp16 3D checkpoints are ≈3.83 MB; `a2d_v2s` ≈37.1 MB;
  `a2d_cnx_cor` ≈51.2 MB. The naive 847/71 ≈ 11.9 MB is the wrong unit and understates 3D headroom threefold.
- **Seeds go only to arms in the greedy best subset**, and the gate is re-run over the enlarged pool so the
  shipped set is the set that was scored.
- **The smoke test is a format and liveness check, not a ranking signal.** It runs 20 scans under a 6-minute
  cap. The fixed cost there is checkpoint *count*, not megabytes — +17 checkpoints adds only ~5–17 s — and a
  1421 MB / 39-checkpoint zip has already passed, so size is not the hazard it appears to be. Pass criterion:
  within ~0.01 of **0.2374** (`id-321850`), the most recent smoke score carrying a real submission id. It is
  never fitted on (ADR 0005).
- **Nothing under `datsc/` or `main_v3.py` is edited.** `build_master_v3.py:126-128` packs the working tree, so
  an edit there ships untested inference code. All changes live in `scripts/`, which is not packed.

## Testing Decisions

- `tests/test_run9.py` pins the arithmetic the endgame turns on: the λ ceiling, the realistic λ gain on the
  committed nested pool, the worst-case cost of each λ, and the presence of both new build guards.
- The λ tests run against `kaggle/run8_distill/soft_master_v2.csv`, which is committed, so they need only
  numpy and pandas and cannot silently stop testing anything.
- The guard tests assert on source, matching the existing convention in `tests/test_run8.py` for artefacts that
  cannot be executed in the test environment.
- The full suite (94 tests + 5) must be green before the final build.

## Out of Scope

- New architectures or input representations. Integration risk on the only remaining slot is not worth it.
- Refitting combiner weights: equal weights beat fitted ones in nested CV and that policy holds.
- Isotonic/Platt recalibration on the site pool (ADR 0003 already rejected learned meta-calibration).
- Averaging the two previous submissions: mixing λ = 0.02 and λ = 0.01 outputs merely reproduces λ ≈ 0.015.
- Training on the smoke set, or fitting anything on it (ADR 0005).
- Resolving the Kaggle prize-eligibility gray area; it is a disclosure matter for after the deadline (T12).

## Further Notes

The competition's own numbers bound the ambition. Observed nested→public gaps are +0.018, +0.021, +0.027.
Top 10 needs public ≤ 0.2379, i.e. nested ≈ 0.211–0.220 — between 0.005 and 0.016 below the current 0.2266,
where the entire run-8 campaign bought 0.0034. Top 15 needs roughly 0.002. The realistic target for this last
submission is to hold or slightly improve the banked position, not to reach the podium; the honest ceiling
with this data was set out in `forharsh.md` §6 and has not moved.

Because the best private-set submission counts automatically, a worse final entry cannot displace what is
already banked — but a *crashed* one still consumes the slot. That asymmetry is the whole argument for
spending the remaining effort on verification rather than on novelty.
