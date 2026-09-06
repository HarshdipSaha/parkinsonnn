---
status: accepted (2026-09-06)
---

# Combine arms with a few fitted numbers and never sharpen

The arms are combined by a weighted average of log-odds (weights on a coarse simplex), one temperature and one prior mix, all fitted on out-of-fold predictions; the temperature is fitted on leave-pseudo-site-out predictions when they exist and is floored at 1.0. We rejected a learned meta-model (logistic stacking) because with 1,362 scans and an out-of-fold noise floor of about 0.014 nats it fits noise and sharpens the boundary, and because over-confidence under site shift is unbounded in log loss while under-confidence is bounded: a temperature 10–20 % too high costs a few thousandths, one too low can cost the competition.

**Consequences**: any new arm needs out-of-fold predictions on the canonical folds (and ideally on site folds) before it can be weighted; a model that would benefit from sharpening must fix its training instead (this is how the EMA-window bug in run 3 was caught rather than masked).
