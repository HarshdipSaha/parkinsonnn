# LLM Council Verdict — the last submission (Sept 13, 2026)

Convened 2026-09-13 09:05 UTC, 87 hours before the deadline, with one real submission remaining and 30+ GPU-hours
unspent. Six members were briefed independently and did not see each other's replies: **A** leaderboard game
theory, **B** calibration and the zero-GPU levers, **C** how to spend 30 GPU-hours, **D** packaging and the
inference container, **E** where the residual error lives, **F** red team. A, B, D and F reported; C and E were
still running when this was written and are marked where their input is missing.

The council was told to assume the recorded standing (public 0.2473, rank ~#16, five submissions used) is real.
It is recorded here that F disputes that premise on evidence; see *Blind Spots*.

> **Superseded in part by the deep-research pass** (`docs/research/dat-endgame/report.md`, 7 items, one agent
> each, schema-validated). It corrected four conclusions below and sharpened the recommendation from "spend the
> GPU well" to **"verify, don't train"**:
>
> - Checkpoint size is **per-arm**, not the 11.9 MB zip average: 3D ≈ 3.83 MB, `a2d_v2s` ≈ 37.1, `a2d_cnx_cor` ≈ 51.2.
> - **Neither size nor the ladder binds.** A 1421 MB zip already smoke-scored (`id-321496`); a 71-checkpoint
>   run uses ~24 min of the 150-minute ladder. The cap is *diminishing returns* — `a3d_hemi` seeds 3→5 bought
>   0.0001 nats for 10 checkpoints.
> - **The ladder cannot explain the λ anomaly**: slots 1 and 2 uploaded the identical 71-checkpoint zip. The
>   live hypothesis is that a recorded number is wrong, and slot 2's unrecorded AUROC is the free test.
> - **The residual is predominantly irreducible**: 88 scans (6.46 %) carry 46.9 % of the loss and are wrong
>   under all three independent representations, against 0.04 expected under independence. Calibration has
>   0.0002 of headroom; "never be confident" and hedge-on-disagreement both strictly lose.
>
> The λ dispute between B and F is settled by measurement rather than by caution: λ\* turns on at G ≈ 0.020,
> and at the observed G = 0.0207 the optimum is λ ≈ 0.006–0.010, so λ = 0.01 sits inside the bracket.

## Where the Council Agrees

1. **One slot remains, and it opens rather than expires.** Sept 7 03:38 frees the Sept 14 03:38 slot; the next
   expiry is Sept 19, past the deadline. A and F derived this independently. But the slot is live until
   Sept 16 23:59, so the standing plan — submit at 03:38, "no training fits any more" — throws away ~60 hours
   and spends the last option on `master_v2` (nested 0.2323), which is *worse* than the banked 0.2266.
2. **The λ lever is exhausted, and the recorded gain from it is impossible.** A, B and F each computed the
   ceiling independently (0.0055 / 0.00499 / 0.005) against a recorded public gain of 0.0084. On the real
   nested pool the move is worth 0.0020. λ 0.01 → 0 buys at most 0.0018 more.
3. **Nobody should chase the nested λ optimum to zero.** The nested pool is built from random folds, contains
   no shift, and is monotone in λ — it can only price the upside. λ's job is the downside.
4. **Verification, not novelty, is where the remaining effort belongs.** Because the best private submission
   counts automatically, a worse final entry cannot displace what is banked — but a *crashed* one still burns
   the slot. The asymmetry argues for spending the endgame on making sure the zip is what the gate measured.
5. **No new architectures.** Integration risk on the only remaining slot is not worth a speculative arm.

## Where the Council Clashes

**λ = 0.005 or λ = 0.01?** B ran shift simulations on a validated proxy arm and found that at the observed
shift magnitude (G ≈ 0.021) the optimal λ is still 0 — insurance only starts paying at G ≈ 0.05 — and
recommends λ = 0.005 as a compromise that captures half the gain while keeping a 5.9-nat cap. F argues the
opposite: λ is U-shaped, they have two points on one side of the minimum and are extrapolating through it, and
five extra catastrophic scans in the private set would cost more than the entire gain. **Resolved for λ = 0.01
(ADR 0006):** the disputed quantity is ~0.001, the downside is unbounded, and B's own simulation is built on a
proxy arm rather than the real master pool. `scripts/sweep_calibration.py` exists to settle it on real data —
if its break-even column says otherwise, move.

**Is the ladder actually firing?** A and B both nominate it as the explanation for the impossible 0.0084. D
measured the budget and concluded it probably is *not* binding: ~45–60 min of real work against a 150-minute
deadline. Unresolved, and cheap to settle — it is the first thing Step 1 reads out of `View log`.

**How much does the size ceiling really allow?** D puts the headroom at ~3 MB (a quarter of one checkpoint) and
concludes he cannot ship more seeds at all without raising `--max-mb`. C, who would have costed the GPU spend
against that envelope, had not reported. The plan therefore makes the envelope an explicit measurement (Step 3)
before any training is launched.

## Blind Spots the Council Caught

- **The 850 MB ceiling is the binding constraint, not GPU hours** (D). Every earlier plan reasoned about GPU
  budget; none noticed that `DROP_ORDER`'s first action deletes `a2d_v2s` seed 1, a pruning the repo itself
  prices at ≈0.01 — five times what any single new seed could return. Training more seeds without raising the
  ceiling makes the submission strictly worse.
- **`build_master_v3.py` would write an oversized zip with no assert** once `DROP_ORDER` was exhausted (D).
- **`local_e2e_v3.py` could not see the `featc`/LR arms** (D): they print a different line, so the
  `checkpoints used N/N` assertion never covered them and a silently dead arm still passed the gate.
- **`typ` must stay `None`** (D): enabling it makes inference print aggregate statistics of the hidden test
  set, which the rules list as grounds for disqualification.
- **The smoke test punishes large zips far more than the real run does** (rules, §"Runtime"): 6 minutes for 20
  scans, where checkpoint loading is a fixed cost, against 3 hours for ~2,000. A bigger zip can fail smoke and
  still be fine on the real set — a trap that would cause a good candidate to be discarded.
- **Neither real submission carries a DrivenData id** (F), though `HANDOFF.md`:126 requires one after a prior
  fabrication incident; slot 1's score is recorded ~13 h *before* its own slot opened; the rank is written as
  "Estimated"; and the entry was byte-corrupted (independently confirmed: 3 NUL bytes, a stray cp1252 `0x97`,
  two collapsed lines — now repaired). F would bet the standing is not a real platform reading. The council was
  instructed to proceed as if it is; Step 1 costs 15 minutes and settles it either way.
- **The two G's are conflated** (F). `HANDOFF.md`:71 and `forharsh.md`:66 quote "G = 0.042 / 0.018", mixing run
  3's *smoke→public* gap with run 4a's *nested→public* gap. The whole decision tree branches on that number.

## The Recommendation

Hold the slot. Verify first, calibrate honestly, spend GPU only inside a measured size envelope, ship late.

1. **Verify (free, 15 min).** Transcribe every submissions-page row with its id, and both `View log` outputs:
   seven `stage written after arm …` lines, `checkpoints used N/N` per torch arm, an `arm featc:` line, and no
   `ladder:`. This settles the slot count and whether the ensemble that scored is the ensemble that was gated.
2. **Sweep the combiner on the real master pool**, reading the break-even column, not only the nested column.
   Default λ = 0.01, T = 0.9.
3. **Measure the size envelope** at `--max-mb 1050` before launching any GPU. Checkpoint size is per-arm, not
   an average: fp16 3D checkpoints are ≈3.83 MB, `a2d_v2s` ≈37.1 MB, `a2d_cnx_cor` ≈51.2 MB. 1,050 MB buys
   ~52 more 3D checkpoints but only ~5 v2s. Neither size nor the ladder actually binds — a 1421 MB zip already
   smoke-scored (`id-321496`), and a 71-checkpoint run uses ~24 min of the 150-minute ladder. **Diminishing
   returns is the real cap**: `a3d_hemi` seeds 3→5 bought 0.0001 nats for 10 checkpoints. Add ~20 checkpoints
   where seeds still pay, and stop.
4. **Spend GPU on seeds of arms already in the greedy best subset**, then re-gate against **0.2266**.
5. **Freeze Sept 16 12:00, smoke-test the exact frozen zip, upload ~16:00**, leaving ~8 h of margin.

Expected outcome, stated honestly: this defends the banked position and is worth perhaps 0.002–0.004 nested.
Top 10 needs nested ≈0.211–0.220 against 0.2266 now, where the entire run-8 campaign bought 0.0034 — A puts
top 10 at 5–10% and top 15 at 45–60%, the latter mostly a race against other teams still improving.

## Dated build order

| When (UTC) | Step |
|---|---|
| Sept 13 09:30 | Verify the ledger and both submission logs |
| Sept 13 10:00 | `sweep_calibration.py` on the real master pool |
| Sept 13 10:30 | Size envelope probe at `--max-mb 1050` |
| Sept 13 12:00 | GPU session A — seeds for best-subset arms only |
| Sept 14 03:38 | Slot opens. **Do not submit.** |
| Sept 14 08:00 | Fetch, re-gate against 0.2266 |
| Sept 15 18:00 | Last moment any training may start |
| Sept 16 12:00 | **Freeze**: build, `pytest -q`, `FORMAT OK`, all arms N/N |
| Sept 16 14:00 | Smoke-test the exact frozen zip |
| Sept 16 16:00 | **Real upload** (22:30 drop-dead) |

## The One Thing to Do First

Open the DrivenData submissions page and read it. Not because the recorded numbers are assumed wrong, but
because every remaining decision — how many slots exist, whether λ explained anything, whether the ladder is
eating the arms that justify Master v3 — is answered by rows that are already sitting there, for free, in
fifteen minutes. Everything else in this verdict is downstream of that page.

---

# Appendix — the briefs

**A (leaderboard game theory).** Derived the slot table from the five recorded submissions: R3 (Sept 7 03:38)
frees Sept 14 03:38; spending it puts R4/R5/R6 in the window with the next expiry Sept 19. Holding the slot
buys 60.4 wall-clock hours at ~2% risk with an 8-hour margin. Computed the λ ceiling at 0.004551 (y=1) /
0.005525 (y=0) and flagged the observed 0.0084 as 1.68× the maximum. Noted the smoke set actually *preferred*
λ=0.02 (0.2374 vs 0.2474), consistent with λ-only twins and inconsistent with the recorded narrative. Target
maths: top 10 needs nested 0.211–0.220; P ≈ 5–10%. Flagged `forharsh.md`:105 and the "10 GPU-h" figure as stale.

**B (calibration).** Located the shipped transform at `infer_v3.py:187-194`; λ mixes in probability space,
π = 747/1362 = 0.5485 (`config.py:35`). Could not load the master pool (gitignored), so validated against
`artifacts/run4a_stack/oof_stack.csv`, inverting to log-odds with 1.1e-16 error and reproducing the manifest's
0.24572; its λ 0.02→0.01 delta (+0.00231) matches master v3's real delta (0.0023). Nested optimum λ=0, monotone.
Shift simulations tuned to G≈0.021 still give λ*=0; λ turns on near G≈0.05. Nested-optimal T = 0.815, but every
shift model pushes T *up*, so T=0.90 is on the right side. Verdicts: rank-averaging a trap (0.2950→0.3147);
probability-space averaging mildly better but not worth a code change; averaging the two submissions pointless
(reproduces λ≈0.015); isotonic/Platt rejected by ADR 0003.

**D (packaging).** Traced entrypoint `main_v3.py:81-87` → prior CSV at `:49` → preprocess → per-arm loop `:67-77`,
rewriting the CSV after every arm. Found **no disqualification risk and no crash bug**: all normalisation is
per-volume, models run under `eval()`, `combine()` reduces over the arm axis only. One live exposure: `main_v3.py:64`
would print test-set typicality statistics, gated on `man["typ"]`, which `build_master_v3.py:111` hard-sets to
`None` — keep it that way. Ladder is one flat `LADDER_STOP_S = 9000 s` checked before each arm and each
checkpoint, with the run-8 arms ordered **last**; estimated real work 45–60 min, so probably not binding.
Size: 3 MB headroom; `DROP_ORDER` drops `a2d_v2s` seed 1 first and can be exhausted without an assert.
`local_e2e_v3.py` blind to `featc`, to arm count vs manifest, and to CUDA/container parity.

**F (red team).** Neither real submission carries a DrivenData id; the last real id in the repo is `id-321850`,
a Sept 11 *smoke* test. Slot 1's score is recorded at Sept 12 01:40 with its commit at 01:57, against a slot the
ledger says opened at 15:38. Same-id-two-ranks (`id-321298` = #39 and #44), same-zip-two-nested-values (0.2354
vs 0.2323), same-zip-two-sizes (463.92 vs 486 MB). Priced the λ tail: at λ=0.01 a confident miss caps at 5.40
nats, at λ=0 at 9.21 — five such scans in the private set ≈ +0.019, more than the whole claimed gain. Noted
AUROC is a free consistency test: λ is strictly monotone, so both zips must show identical AUROC; none was
recorded for slot 2.

**C (GPU budget) and E (residual error)** had not reported when this verdict was written. C's question — how to
spend 30 GPU-hours — is partly answered by D's size finding: the envelope, not the quota, is the constraint,
which is why the plan measures it before launching anything.

**My own computation** (numpy/pandas, on the committed `kaggle/run8_distill/soft_master_v2.csv`) reproduces the
repo's recorded numbers exactly — nested 0.232252 vs 0.2323, top-100 loss share 54.2% vs 54%, 94 scans over
1 nat, capped-at-2 ceiling 0.220130 vs 0.2201 — confirming the inversion is sound, and gives the λ ladder:
0.02 → 0.232252, 0.01 → 0.230255, 0.005 → 0.229320, 0 → 0.228437.
