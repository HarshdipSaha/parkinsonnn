# 5. The smoke set is validation-only

Date: 2026-09-07

## Status

Accepted

## Context

DrivenData's smoke test scores a submission on twenty labelled scans whose labels we also hold locally, and three smoke tests per day are free. The three real submissions so far ordered the same way on the smoke set and on the public leaderboard (0.2567 → 0.3391, 0.2397 → 0.2817, 0.2176 → 0.2667). It is tempting to fit combiner weights, temperatures or member selection on those twenty scans, or to train on them.

## Decision

The twenty smoke scans are used only as an out-of-distribution sanity check (per-arm and stacked scores are printed by every notebook's end-to-end cell) and as one coarse factor when two candidates are otherwise tied. Nothing — no weight, temperature, prior, member gate or training example — is ever fitted on them.

## Consequences

- Candidate selection stays anchored to the nested out-of-fold estimate and the leave-site-out pool, which have 1,362 scans behind them; twenty scans have a log-loss standard error of about 0.05 and would be fitted noise.
- The smoke-to-public track stays an honest read-out of transfer, because the smoke score is never optimised.
- We forgo whatever small gain training on twenty more scans might have given.
