# DaT Parkinson's Challenge — the last submission (Sept 13-16, 2026)

Deep-research pass generated 2026-09-13 using the Deep-Research-skills workflow 
(`research` -> `research-deep` -> `research-report`), with one agent per item and a validation gate.

| | |
|---|---|
| Now | 2026-09-13 09:05 UTC |
| Deadline | 2026-09-16 23:59 UTC |
| Last slot opens | 2026-09-14 03:38 UTC (it opens; it does not expire) |
| Real submissions left | 1 |
| Banked | public 0.2473, nested 0.2266, rank ~16 (recorded, unverified) |
| Leader / top 10 | 0.2166 / 0.2379 |
| GPU-hours left | 30+ |
| Zip | 847 MB of 850 MB, 71 checkpoints |

## Contents

| # | Item | Confidence | Verdict |
|---|---|---|---|
| 1 | [Slot arithmetic and when to spend the last submission](#slot-arithmetic-and-when-to-spend-the-last-submission) | high | Exactly one real submission remains and it cannot expire before the deadline under either quota model, so do not spend it when it opens on Sept 14 — hold it, freeze the zip Sept 16 12:00 UTC, run the free smoke test at 13:00, and upload at 16:00 UTC (hard stop: start no later than 20:27 UTC). |
| 2 | [Prior mix and temperature](#prior-mix-and-temperature) | high | Ship the final zip at T=0.90 and lambda=0.01 exactly as ADR 0006 already fixed them, with clip=null: on the committed leave-pseudo-site-out OOFs that cell is within 0.0009 nats of the minimax-optimal cell at every shift level tested, the whole (T, lambda) question is worth at most 0.0018 nats in either direction, and the site-fold pools confirm that lambda -> 0 is the one move with unbounded downside. |
| 3 | [GPU spend inside the size envelope](#gpu-spend-inside-the-size-envelope) | high | Raise the ceiling to --max-mb 1050 and spend ONE Kaggle session (~4 GPU-h, 20 jobs) on the two under-seeded arms that are already in the greedy best subset — a3d_crophistsoft 1->3 seeds and a3d_hemisoft 3->5 seeds — because at ~3.8 MB per 3D checkpoint those 20 checkpoints cost only 77 MB, not the 238 MB the stale 11.9 MB/ckpt average implies; add no more plain a3d_hemi seeds (measured 0.0001 nats for 10 checkpoints), add nothing 2D, build no model soup, and stop all training at Sept 15 18:00 UTC. |
| 4 | [Inference ladder and packaging risk](#inference-ladder-and-packaging-risk) | medium | Leave LADDER_STOP_S, the arm order and every file in datsc/ exactly as they are - the ladder has roughly 6x headroom even at the 1050 MB / 88-checkpoint ceiling - and spend the remaining packaging effort on three verification steps instead: transcribe 'done in Xs' and the seven 'stage written after arm' lines from the two real-submission View logs, build the final zip at an explicit --max-mb that emits NO 'dropped' line, and platform-smoke the byte-identical (sha256-recorded) frozen zip before uploading it. |
| 5 | [Residual error and the reachable ceiling](#residual-error-and-the-reachable-ceiling) | high | Treat the residual as predominantly irreducible: 88 scans (6.5% of the pool) carry 46.9% of all loss and are wrong under every independent representation built so far, so do NOT spend the last slot chasing the tail with capacity, calibration or hedging — ship the banked-or-better stack, spend the remaining hours on verification (read the two DrivenData View-log outputs) rather than on modelling, and hold the queued seed run to a hard >=0.003 nested gate because anything smaller is inside the +/-0.013 bootstrap noise of the pool. |
| 6 | [Ledger integrity and verification](#ledger-integrity-and-verification) | high | Before spending any GPU or the last slot, open the DrivenData submissions page as harshdipsaha and transcribe every row (id, UTC, type, zip, score, AUROC, status) plus the quota banner and both View-log outputs into a single fixed-schema ledger table in RESULTS_LOG.md, marking every number that does not come back with an id as [unverified] - because 14 of the 17 platform numbers in the repo carry an id and the three that do not are exactly the ones the whole endgame is built on (0.2557, 0.2473, 0.2474), and both real scores are recorded at times at which no slot was open. |
| 7 | [Rules compliance and disclosure](#rules-compliance-and-disclosure) | medium | Ship the last zip with the inference path untouched — it is per-scan clean and carries no test-set-wide statistic — and spend the remaining compliance effort on the two real exposures: settle whether `skyfall23` is inside the DrivenData team before any further training on that account, and keep the six git-tracked files that reproduce all 1,362 training labels out of the MIT release; publish `docs/DISCLOSURE.md` (content in `action`) after 2026-09-16 23:59 UTC. |

---

## Slot arithmetic and when to spend the last submission

**Question.** How many real submissions remain before the deadline, and is it better to submit when the slot opens or to hold it until Sept 16?

### Verdict

Exactly one real submission remains and it cannot expire before the deadline under either quota model, so do not spend it when it opens on Sept 14 — hold it, freeze the zip Sept 16 12:00 UTC, run the free smoke test at 13:00, and upload at 16:00 UTC (hard stop: start no later than 20:27 UTC).

### Confidence

high

### Evidence

- DAT_PARKINSONS_CHALLENGE.md:120 — 'Real submissions are rationed: 3 per rolling 7 days' with the banner verbatim: 'Submissions used: 0 of 3 … 3 of 3 submissions left per 7 days'. This is the only statement of the quota rule in the repo.
- DAT_PARKINSONS_CHALLENGE.md:121 — smoke tests are 3/day, 20 training scans, 6-minute cap, scored, and explicitly 'do NOT count against the real-submission quota'. So the pre-flight smoke on Sept 16 is free in slot terms.
- DAT_PARKINSONS_CHALLENGE.md:37 — 'Best-scoring submission on the private set counts automatically'. The last slot is therefore a free call option: a worse or crashed final entry cannot displace the banked 0.2473.
- DAT_PARKINSONS_CHALLENGE.md:71 — 3 hours max for a full submission, 6 minutes for a smoke test. Sets the worst-case scoring term in the back-calculation.
- DAT_PARKINSONS_CHALLENGE.md:122 — measured queue wait 'Starting' → 'Completed' was ~20 min (shared cluster). Source of the 20-minute queue term.
- DAT_PARKINSONS_CHALLENGE.md:76 — execution machines are a shared resource across all competitors; this is the mechanism by which the Sept 16 queue can lengthen, and the reason the margin must be hours, not minutes.
- aidlc-docs/HANDOFF.md:136 — 'Upload of 700–850 MB takes ≈ 12 min; … one active code job at a time; smoke tests 3/day'. Source of the 12-minute upload term and the smoke-blocks-upload constraint.
- aidlc-docs/HANDOFF.md:87 — 'The public score comes back ≈ 1 h after the upload'. Used to back out upload times from the two recorded score times.
- aidlc-docs/HANDOFF.md:9 — '5 used. One slot left: it opens Sept 14 03:38 UTC and stays open until the deadline — the next expiry after that is Sept 19'. The recorded standing this item tests.
- aidlc-docs/HANDOFF.md:10 — the best zip on hand, submission_master_v2.zip, is nested 0.2323, explicitly 'worse than the banked 0.2266 … a diversification bet, not an upgrade'. Spending the slot at 03:38 spends it on this.
- aidlc-docs/HANDOFF.md:80 — the row 'Sept 13 15:50 | submission_master_v3_lam01.zip | COMPLETED: Public 0.2473'. This is self-refuting: 15:50 is 6 h 45 min AFTER the stated now of Sept 13 09:05, so a completed score cannot be recorded against it. The 15:38/15:50 family is not an upload-time record.
- aidlc-docs/HANDOFF.md:89 and :119 — 'After slot 1 (score ≈ Sept 12 16:40 UTC)' and 'After slot 2 (score ≈ Sept 13 16:50 UTC)', written on Sept 10 (commit 269f85b, 2026-09-10) i.e. BEFORE either submission. These are forecasts derived from the planned 15:38/15:50 slot returns, which later docs copied as if they were facts.
- docs/superpowers/plans/2026-09-06-run4-toward-0.21.md:22 and docs/superpowers/plans/2026-09-06-run5-site-invariance-plan.md:24 — the origin of 15:38 / 15:11: 'slot 1 back Sept 12 15:38 UTC; slot 2 back Sept 13 15:11 UTC', i.e. 168 h added to the Sept 5 / Sept 6 uploads. Planning arithmetic, never a platform readout.
- aidlc-docs/audit.md:37,38 — the two real submissions are logged as evaluated at '2026-09-12 01:40' and '2026-09-13 03:08' UTC, tagged 'Platform evaluation result'. Same numbers at aidlc-docs/efforts/007-run8-distill-master-v3/effort-state.md:30,31.
- Re-runnable: `TZ=UTC git log --date=iso-local --pretty="%h %ad %s" -3` → 1d057fc 2026-09-13 03:13:23 'Update docs with Slot 2 score 0.2473'; d06af90 2026-09-12 01:57:03 'record Slot 1 real submission (0.2557 …)'. The commits that record each score land 5 and 17 minutes after the audit-log evaluation times, and 12–14 h BEFORE the 15:38/15:50 times. A document cannot record a score for a submission that has not happened.
- Control for the same conversion: aidlc-docs/audit.md:44 logs smoke id-321850 evaluated 2026-09-11 06:22 UTC; commit 0f346ca recording it is 2026-09-11 06:30:25 UTC (+8 min). Three independent hits at +8/+17/+5 min confirm audit.md timestamps are genuine UTC event times, not local time mislabelled.
- Counter-example showing the failure mode exists: RESULTS_LOG.md:311 stamps the live ledger read '(Sept 10, 19:38 UTC)' while its own commit cec8e76 is 2026-09-10 14:09 UTC — 19:38 there is IST mislabelled UTC. The audit.md/effort-state times do not have this defect (they precede their commits), the RESULTS_LOG narrative times sometimes do.
- Re-runnable computation (py, stdlib datetime): with S1 2026-09-05 15:38, S2 09-06 15:11, S3 09-07 03:38, S4 09-12 00:40, S5 09-13 02:08 — a strict rolling-168h count at 2026-09-13 09:05 returns 4 submissions in window against a cap of 3. The recorded ledger is therefore impossible under a strict rolling-168h quota; the same set under a calendar-7-day rule (today + 6 prior days) returns 3 at every one of S1..S5, i.e. fully legal.
- RESULTS_LOG.md:311 — the banner read live off the logged-in page on Sept 10 says '0 of 3 submissions left per 7 days, next on Sept 12 2026 UTC'. It gives a DATE and no clock time. The platform renewing at a date boundary is exactly what allowed S4 to go through at ~00:40 on Sept 12, and it means the repo's '03:38' was computed, never observed.
- Re-runnable computation: both window models agree on the only thing that matters. Rolling-168h — in-window count is 2 at Sept 16 16:00 and at Sept 16 23:59, and S4's expiry is 2026-09-19 00:40, past the deadline. Calendar-7-day — the Sept 10–16 window holds only S4 and S5, count 2. Under both, exactly ONE slot exists, it is live continuously from Sept 14 (00:00 or 03:38) to the deadline, and no second slot ever appears.
- aidlc-docs/efforts/008-run9-endgame/effort-state.md:21 — independent derivation: 'the slot opens, it does not expire — it is live until Sept 16 23:59', and the standing plan spends it on master_v2 (0.2323), worse than the banked 0.2266, discarding ~60 h of runway.
- COUNCIL_VERDICT_ENDGAME.md:114-116 — council brief A derived the same table independently: 'R3 (Sept 7 03:38) frees Sept 14 03:38; spending it puts R4/R5/R6 in the window with the next expiry Sept 19. Holding the slot buys 60.4 wall-clock hours at ~2% risk with an 8-hour margin.' My computation reproduces 60.37 h from Sept 14 03:38 to Sept 16 16:00.
- COUNCIL_VERDICT_ENDGAME.md:140 — the red-team brief already flagged the same clash: 'Slot 1's score is recorded at Sept 12 01:40 with its commit at 01:57, against a slot the ledger says opened at 15:38.' This item resolves it in favour of 01:40/03:08.
- Re-runnable back-calculation from 2026-09-16 23:59 with upload 12 min + queue 20 min + scoring 60 min typical / 180 min cap: transfer-only drop-dead start 23:47; typical-score drop-dead 22:27; 3h-cap drop-dead 20:27. These reproduce the plan's stated '22:30 = drop-dead' and 'Sept 16 20:00 latest safe start' (docs/superpowers/plans/2026-09-13-run9-endgame-plan.md:22) to within 3 and 27 minutes.
- docs/superpowers/plans/2026-09-13-run9-endgame-plan.md:19-21 — freeze Sept 16 12:00, smoke 14:00, real upload 16:00 with '~8 h margin'. My arithmetic gives 6 h 27 min of margin to a typical score and 4 h 27 min to a 3-hour-capped run, not 8 h; the schedule is still comfortable but the stated margin is optimistic.
- docs/superpowers/plans/2026-09-13-run9-endgame-plan.md:137 — the plan's own risk row demands '≥ 2 h clear between the smoke test and the real upload' because of the one-active-code-job rule. Its own schedule (smoke 14:00, finishing 14:38 at 12+20+6 min; upload 16:00) leaves 1 h 22 min. Moving the smoke to 13:00 restores 2 h 22 min at zero cost.
- RESULTS_LOG.md:176 — 'smoke test (id-321297) from the 03:14 retry was cancelled to free the code-job slot for the real submission'. The one-active-job constraint is not theoretical; it has already forced a cancellation on this account.
- aidlc-docs/efforts/008-run9-endgame/requirements-delta.md:9 (R29) — the slot policy is already ratified as a requirement: spend the slot as late as is safe, and a candidate replaces the banked zip only if its gate table beats 0.2266 by ≥ 0.002. This item confirms R29's arithmetic and corrects its smoke time.
- aidlc-docs/audit.md:29 and RESULTS_LOG.md:309-311 — the documented fabrication incident: a session invented 'run 5 public 0.2600 id-321520'; the live page showed exactly 3 real rows (id-321156, id-321263, id-321298). This is why the slot count must be read off the page, not off the repo: the count of 5 rests on two rows that carry no DrivenData id anywhere in the repo, against a rule (HANDOFF.md:83) that every upload's id be recorded.
- kaggle/gen_submit.py:2,16 — usage `python kaggle/gen_submit.py <zip_path> "<note>" smoke|normal`, and the page-scrape regex /Submissions used[^\d]*(\d) of (\d)/ confirming the exact banner string to read.
- aidlc-docs/HANDOFF.md:124 — 'never the same zip twice; the best of the three on the private set counts'. The final zip must differ from submission_master_v3.zip and submission_master_v3_lam01.zip, which rules out simply re-sending the banked 0.2473 zip at 03:38.

### Risk if wrong

The recommendation costs nothing on the banked score in any branch, because the best private submission counts automatically (DAT_PARKINSONS_CHALLENGE.md:37) — 0.2473 / rank ~#16 survives a crash, a worse entry, or an unused slot. The exposure is the option value only. If TWO slots actually remain (e.g. the Sept 13 row never happened — it carries no id, and this repo has a documented fabricated-submission incident), holding wastes one entirely: one free shot at ~0.002-0.005 log loss and 1-3 ranks, forgone. If ZERO remain (a sixth submission exists, or errored rows are counted against quota), the whole Sept 16 schedule dead-ends at 16:00 and ~60 h of GPU work is spent on a zip that can never be uploaded. If the slot is held and the platform is congested or down through the Sept 16 evening, the slot expires unused — again the option, not the score. Conversely, being wrong in the other direction and spending the slot at Sept 14 03:38 has the largest concrete cost on the table: it commits the last option to submission_master_v2.zip at nested 0.2323, 0.0057 WORSE than the banked 0.2266, and throws away 60.4 h containing two ~4.5-h GPU sessions, the size-envelope raise (~17 checkpoints at --max-mb 1050) and the unverified ledger. Every branch of the error analysis points the same way: hold.

### Action

- NOW (Sept 13, ~15 min, free): open https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/submissions/ logged in as harshdipsaha and transcribe the page into RESULTS_LOG.md before anything else. Read five things: (1) the quota banner VERBATIM — 'Submissions used: N of 3' and 'M of 3 submissions left per 7 days, next on <DATE> UTC'; whether that line carries a clock time or only a date settles rolling-vs-calendar for good; (2) the number of rows in the 'Normal submission' table — it must be exactly 5; 4 means two slots remain and this whole verdict changes, 6 means zero; (3) each row's id + UTC timestamp + status + score + AUROC, especially the two rows scoring 0.2557 and 0.2473, whose ids appear nowhere in the repo; (4) whether any Failed/Errored/Cancelled row sits in the Normal table and whether it is counted in 'used N of 3' — that settles whether a crashed final upload is recoverable; (5) the smoke-test counter and its daily reset time.
- Compare the transcribed timestamps of the 0.2557 and 0.2473 rows against Sept 12 01:40 and Sept 13 03:08 UTC. If they match, delete the 'Sept 13 15:50' row label in aidlc-docs/HANDOFF.md:80 and the 'Sept 13 15:50' row in forharsh.md:53 and replace them with the real times plus ids.
- Sept 14 00:00 and again at 03:38 UTC: DO NOTHING. The slot opens; it does not expire. Leave data/submission_master_v2.zip on disk.
- Sept 14 08:00 → Sept 15 18:00: this is the 60.4 h the hold buys — run the calibration sweep, the --max-mb 1050 envelope probe and GPU sessions A/B per docs/superpowers/plans/2026-09-13-run9-endgame-plan.md. No training may START after Sept 15 18:00.
- Sept 16 12:00 UTC — FREEZE. Build the final zip, then: `py -m pytest -q` and `py scripts/local_e2e_v3.py data/<final>.zip data/smoke` — it must print FORMAT OK and every arm N/N. Ship the zip only if its gate table beats nested 0.2266 by >= 0.002 (i.e. <= 0.2246); otherwise ship the best gated zip that is not one of the two already submitted (never the same zip twice, HANDOFF.md:124).
- Sept 16 13:00 UTC — free smoke test of the EXACT frozen zip (13:00, not the plan's 14:00, so the one-active-code-job gap is 2 h 22 min instead of 1 h 22 min): `py kaggle/gen_submit.py data/<final>.zip "<what it is> nested <N>" smoke`, then run .playwright-mcp/dd_submit.js. Expect it done by ~13:38 (12 min upload + 20 min queue + <=6 min run).
- Sept 16 16:00 UTC — REAL upload, only after the smoke job shows Completed: `py kaggle/gen_submit.py data/<final>.zip "<what it is> nested <N>" normal`, then run .playwright-mcp/dd_submit.js. Do not reload the page during 'Upload in progress'. Transfer done ~16:12, job starts ~16:32, score ~17:32 (6 h 27 min of margin; 4 h 27 min even if it runs the full 3-hour cap).
- Hard stops if Sept 16 slips: start the real upload by 20:27 UTC to be safe even against a 3-hour-capped run; by 22:27 UTC to still see the score; 23:47 UTC is the absolute last moment the 12-minute transfer can begin. Past 20:27, skip the smoke test rather than skip the submission — the smoke is free but it holds the only code-job slot.
- Immediately after the upload, write the DrivenData id, the UTC timestamp, the public score, the AUROC and G = public - nested into RESULTS_LOG.md (HANDOFF.md:83).

### Rejected alternatives

- Submit at Sept 14 03:38 when the slot opens (the standing plan, HANDOFF.md:81 before its supersede note, effort-state 007:37). Rejected: the slot opens rather than expires — the next quota expiry after it is Sept 19 00:40, four days past the deadline — so waiting costs no quota at all, while submitting now spends the last option on submission_master_v2.zip at nested 0.2323, 0.0057 worse than the banked 0.2266, and discards 60.4 h containing two GPU sessions and the size-envelope raise.
- Submit early to 'get feedback' from the public score. Rejected: with zero slots left afterwards there is nothing the gap G can inform. The one real version of this argument — an early upload would reveal a container crash in time to fix it — also fails, because a crashed job burns the slot just the same and no further slot exists (and whether errored rows are re-granted is itself unknown, see open questions).
- Re-submit the banked submission_master_v3_lam01.zip to lock in 0.2473 on the private set. Rejected twice over: the score is already banked and the best private submission counts automatically (DAT_PARKINSONS_CHALLENGE.md:37), and HANDOFF.md:124 forbids sending the same zip twice.
- Trust the 'Sept 12 15:38 / Sept 13 15:50' timestamps in HANDOFF.md:79-80 and forharsh.md:53 and conclude the slot arithmetic runs off them. Rejected: HANDOFF.md:80 records a COMPLETED public score against a time 6 h 45 min in the future of the stated now, the scores' own recording commits are at 01:57 and 03:13 UTC, and the 15:38/15:11 figures are traceable to docs/superpowers/plans/2026-09-06-run4-toward-0.21.md:22 as planned slot returns.
- Assume the platform quota is a strict rolling 168 hours and plan around Sept 14 03:38 as a hard gate. Rejected as unverified: under that rule the recorded ledger has 4 submissions in window right now, which the platform would have blocked. The calendar-7-day reading fits all five submissions and matches the banner's date-only 'next on Sept 12 2026 UTC'. Both models nevertheless give one slot, live through the deadline, so nothing downstream turns on it.
- Keep the plan's 14:00 smoke / 16:00 upload pairing. Rejected on the plan's own rule: 12 + 20 + 6 min puts the smoke job's completion at 14:38, leaving 1 h 22 min against the '>= 2 h clear' requirement at plan:137. Moving the smoke to 13:00 fixes it for free and keeps the same upload time.
- Push the upload later than 16:00 to buy more training time. Rejected: the marginal three hours of Sept 16 afternoon cannot produce a gated checkpoint set (no training may start after Sept 15 18:00 anyway), while each hour of delay eats directly into a margin that has to absorb a shared-cluster queue on deadline day (DAT_PARKINSONS_CHALLENGE.md:76).
- Skip the Sept 16 smoke test to remove the one-active-job hazard entirely. Rejected as the wrong trade at 16:00: the smoke is free, does not touch the quota (DAT_PARKINSONS_CHALLENGE.md:121), and is the only check that runs the exact frozen zip on the real container. It is the right trade only after 20:27, which is why that is listed as the switch point.

### Open questions

- Does an Errored, Failed or Cancelled Normal submission consume one of the 3 quota slots, or is it re-granted? This is the single fact that could still justify an early upload (a free retry). Resolved by looking for such a row on the submissions page and checking whether the 'used N of 3' counter includes it; the account already cancelled a code job once (RESULTS_LOG.md:176), so the history may show it.
- Must the scored run COMPLETE before Sept 16 23:59, or only be submitted before it? Decides whether the true drop-dead upload start is 20:27 (complete, 3-hour cap) or 23:47 (submitted). Resolved by the competition's rules page or by the wording next to the deadline countdown on the submissions page. The recommended 16:00 upload is safe under either reading, so this only matters if Sept 16 slips badly.
- Is the quota window calendar-day or rolling-168h? Resolved in one glance: if the banner's 'next on …' line carries only a date (as it did on Sept 10, RESULTS_LOG.md:311) the window is date-boundary and the slot is live from Sept 14 00:00; if it carries a clock time, that time is authoritative. Does not change the verdict either way.
- Do the two real rows scoring 0.2557 and 0.2473 actually exist on the page, and what are their ids? Neither id is recorded anywhere in the repo despite HANDOFF.md:83 requiring it, and this repo has a documented incident of a session inventing a submission row (audit.md:29). If either row is absent, two slots remain and the early-vs-late calculus reopens.
- What is the real queue latency on deadline day? All timings here use the single measurement of ~20 min from Sept 5 (DAT_PARKINSONS_CHALLENGE.md:122) on a shared cluster. If the Sept 16 13:00 smoke test takes materially longer than 38 min end to end, treat that as the live reading and push the real upload back no further than 20:27.

> **Needs checking** — could not be established from the repo, the rules or arithmetic: numbers.


---

## Prior mix and temperature

**Question.** Where should lambda and T sit for the final zip, priced against tail risk rather than the shift-free nested curve?

### Verdict

Ship the final zip at T=0.90 and lambda=0.01 exactly as ADR 0006 already fixed them, with clip=null: on the committed leave-pseudo-site-out OOFs that cell is within 0.0009 nats of the minimax-optimal cell at every shift level tested, the whole (T, lambda) question is worth at most 0.0018 nats in either direction, and the site-fold pools confirm that lambda -> 0 is the one move with unbounded downside.

### Confidence

high

### Evidence

- Combiner form and application: datsc/infer_v3.py:187-194 (p = sigmoid(z/Ti); lam_i mix; np.clip to max(lo, EPS)); datsc/calibrate.py:11 (EPS=1e-4) and :48 (the same expression in the training-time Calibration.apply); pi = 747/1362 at datsc/config.py:35.
- Shipped constants: RESULTS_LOG.md:369 (build_master_v3.py --lam 0.01, 71 checkpoints, 7 arms, equal weights, T=0.90); scripts/build_master_v3.py:111 hard-sets "clip": None and "typ": None; scripts/build_master_v2.py:23 does the same.
- Nested pool reproduction: python on kaggle/run8_distill/soft_master_v2.csv (1362 rows) gives log loss 0.232252 at the shipped cell, matching the recorded 0.2323. The pool is built by scripts/make_soft_targets.py:31 as (1-lam)*sigmoid(z/max(T_fit,0.9)) + lam*0.548, so inverting cal = (p - 0.02*0.548)/0.98 recovers sigmoid(z/T_fold) exactly.
- Nested joint optimum: grid over T in [0.55,1.60] step 0.005 and lambda in [0,0.08] step 0.0005 on the inverted pool -> T=0.870, lambda=0, 0.228279. The shipped T=0.90 at lambda=0 gives 0.228430, i.e. +0.00015. Re-runnable: invert the pool as above, then minimise -mean(y log p + (1-y) log(1-p)) with p = (1-lam)*sigmoid(z0*0.9/T) + lam*0.548.
- Temperature direction under shift, measured directly on the committed OOFs (same arms, same 1362 uids, random folds vs pseudo-site folds): artifacts/featarm_v4/oof_featarm.csv T*=1.137 vs artifacts/featarm_v4/oof_featarm_site.csv T*=1.281; artifacts/featcarm_v5/oof_featcarm.csv T*=1.220 vs artifacts/featcarm_v5/oof_featcarm_site.csv T*=1.296; equal-weight logit average of the two arms 1.147 vs 1.246. All fitted at lambda=0 by grid minimisation of log loss over T in [0.30,3.0]. Shift raises T on every pair.
- The RESULTS_LOG.md:321 claim that the site pool prefers T=0.826 while random prefers ~1.0 is an artefact of the floor: RESULTS_LOG.md:313 states that run used 'T floor 1.0'. artifacts/run4a_stack/manifest.json:3-4 records "T": 1.0 alongside "T_random": 0.8147881992582522, i.e. the floor overrode a fitted 0.81. A direct fit on artifacts/run4a_stack/oof_stack.csv gives T*=0.770 at lambda=0 (log loss 0.236761) against 0.24572 at T=1.0, which reproduces the manifest's recorded value. So 0.826 is above the random optimum, not below it: shift FLATTENS.
- Separability: joint grid vs sequential fit (T* at lambda=0, then lambda* at that T) on each pool. Random-fold pair: joint 0.294624 at (1.090, 0.0100) vs sequential 0.294811 at (1.147, 0.0050), penalty +0.00019. Site-fold pair: joint 0.315995 at (1.000, 0.0380) vs sequential 0.319017 at (1.246, 0.0200), penalty +0.00302. Site-fold featcarm: +0.00192. The surface is separable without shift and is a ridge with shift.
- Shift transfer: the per-scan displacement d = mean over the two feature arms of (logit p_site - logit p_random) is defined on exactly the same 1362 uids as soft_master_v2.csv (verified: 1362/1362 overlap), so z_shifted = z0 + c*d applies the observed site-shift of real scans to the shipped ensemble's own logits. d has mean +0.034, sd 0.849, 27 scans with |d|>2 and 14 that move toward the wrong class by more than 2 nats.
- lambda* as a function of the induced gap G at T=0.90: c=0 -> G=0, lam*=0; c=0.60 -> G=0.0122, lam*=0; c=0.75 -> G=0.0179, lam*=0.0015; c=0.80 -> G=0.0199, lam*=0.0055; c=1.00 -> G=0.0281, lam*=0.0165; c=1.50 -> G=0.0503, lam*=0.0305. The recorded nested->public gaps (+0.018, +0.021, +0.027; docs/superpowers/specs/2026-09-13-run9-endgame-spec.md 'Further Notes') land at c = 0.75-1.0, i.e. lambda* = 0.002-0.017.
- This refines rather than contradicts brief B (COUNCIL_VERDICT_ENDGAME.md:122-124, 'shift simulations tuned to G~0.021 still give lam*=0; lambda turns on near G~0.05'). With the REAL measured site displacement lambda turns on at G ~ 0.020, not 0.05, because real site shift produces confidently-wrong scans (14 of 1362 move >2 nats toward the wrong class) that a smooth parametric simulation does not.
- Minimax table: log loss of each candidate (T, lambda) at c in {0, 0.5, 0.75, 1.0}, regret against the best achievable in each column [0.22843, 0.23766, 0.24782, 0.25813]. Max regret: (0.90, 0.005) 0.00088; (0.85, 0.005) 0.00130; (0.85, 0.010) 0.00139; (0.815, 0.010) 0.00155; (0.90, 0.010) 0.00182; (0.90, 0.000) 0.00260; (0.90, 0.020) 0.00382; (1.00, 0.020) 0.00692; (0.75, 0.000) 0.00995. lambda=0 and T=0.75 are the two worst cells; the shipped cell is fifth-best and 0.00094 off the leader.
- Worst-case pricing: scripts/sweep_calibration.py:24-28 evaluates the worst case at p exactly 0, giving -log(EPS)=9.21 at lambda=0. The shipped ensemble's largest |logit| after T=0.90 is 8.16 (most extreme probability 0.999714), so a real confident miss costs 8.16 nats at lambda=0, 5.16 at lambda=0.01, 4.49 at lambda=0.02. The extra exposure of lambda=0 is therefore 3.01 nats, not 4.00.
- Break-even with the realistic worst case and the realistic nested gain (0.00115 at T=0.90 for lambda=0 vs 0.01): 0.2 scans per 500, 0.4 per 1000, 0.8 per 2000. Under one scan at every plausible test size. For lambda=0.005 vs 0.01 it is 1.9 scans per 2000.
- Base rate for the event break-even is betting against: at T=0.90 the pseudo-site pool has 4 of 1362 scans wrong at p>0.999 and 3 wrong at p>0.9999, against 2 and 0 in the random-fold pool (artifacts/featarm_v4 + artifacts/featcarm_v5 equal-weight logit average). The 'zero confident misses' premise that lambda=0 requires is falsified by the only shifted data in the repo.
- Clip: at lambda=0 the shipped pool's probabilities span [0.004106, 0.999354] and at lambda=0.01 they span [0.00955, 0.99485]; EPS=1e-4 binds on 0 of 1362 scans in both cases. lambda=0.01 alone bounds p to [lam*pi, 1-lam*(1-pi)] = [0.005485, 0.995485], which is 55x tighter than EPS, so the manifest clip field is redundant with lambda. Priced head to head at T=0.90 across c=0/0.5/0.75/1.0, no clip setting beats plain lambda=0.01 by more than 0.0004 at any shift level.
- The repo cannot select a clip anyway: scripts/fit_stack_v3.py:119-145 only sets chosen_clip when a clip shows a gain on the NESTED (shift-free) pool, and scripts/build_master_v3.py:111 hard-writes "clip": None regardless. datsc/infer_v3.py:193-194 then floors any manifest clip at EPS.
- Arm-count interaction with the ladder (datsc/infer_v3.py:24, LADDER_STOP_S = 150*60; :144 the per-checkpoint check): the 2-arm feature pool at T=0.90 puts 63-65 of 1362 scans beyond 1-1e-4 where the 5-pillar ensemble puts 0. If the ladder drops arms and combine() renormalises over the survivors, the shipped probabilities sharpen and the EPS clip becomes the only bound. lambda is insurance against the ladder as well as against site shift.
- Per-fold temperature residual on the soft pool (fold column of soft_master_v2.csv): optimal extra multipliers 1.010, 0.910, 0.965, 1.045, 0.890 across folds 0-4, i.e. a fold-to-fold spread of +/-8% around 1. The T choice cannot be resolved to better than about +/-0.07 from this pool, which is why 0.870 vs 0.900 is not a real distinction.

### Risk if wrong

Bounded and small on the upside, unbounded on the downside. If T=0.90 / lambda=0.01 is not the best cell, the measured cost is at most 0.0018 nats against the nested optimum and 0.00094 against the minimax cell (T=0.90, lambda=0.005) -- roughly one rank position at the 0.2379-0.2473 density near #16, and nothing at all if the ensemble changes. The asymmetric risk is the opposite mistake: shipping lambda=0 buys at most 0.0018 nats and raises the exposure of a single confident miss from 5.16 to 8.16 nats, so one such scan in 2000 erases the gain and three cost about +0.0045 -- and the pseudo-site pool contains three to four such scans per 1362. The other real risk is spending the last slot's attention here: the whole question is worth <= 0.0018 while the top-10 line at 0.2379 is 0.009 of public score away, so treating calibration as the lever repeats exactly the error ADR 0006 was written to stop.

### Action

- Do not change datsc/ or the combiner. Build the final zip with the constants already fixed: `py scripts/build_master_v3.py --lam 0.01 --T 0.90` (RESULTS_LOG.md:369 is the command that produced the banked 0.2473 configuration).
- Verify the manifest before packing: `py -c "import json;m=json.load(open('data/submission_master_v3/manifest.json'));print(m['T'],m['lam'],m['pi'],m['clip'],m['hedge'],m['typ'])"` and require exactly `0.9 0.01 0.5484581497797357 None None None`. Any non-null clip, hedge or typ means build_master_v3.py:111 was bypassed -- rebuild rather than hand-edit.
- Run the sweep for the record but read it with the correction: `py scripts/sweep_calibration.py --T 0.80,0.85,0.87,0.90,0.95 --lam 0,0.005,0.01,0.02` on the real master pool. Its WORST column prices a miss at p=0 exactly, which this ensemble never produces; divide the lambda=0 exposure by about 4/3 (8.16 rather than 9.21 nats) before reading the break-even. The conclusion does not change: break-even stays under one scan in 2000.
- Reproduce the site-vs-random temperature result in one command before trusting RESULTS_LOG.md:321, which reads the direction backwards: `py -c "import pandas as pd,numpy as np;lg=lambda p:np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)));s=lambda z:1/(1+np.exp(-z));L=lambda f:pd.read_csv(f,dtype={'uid':str}).set_index('uid');ll=lambda y,p:float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)));g=np.arange(0.6,2.0,0.005)" ` then, for each of artifacts/featarm_v4/oof_featarm.csv, artifacts/featarm_v4/oof_featarm_site.csv, artifacts/featcarm_v5/oof_featcarm.csv and artifacts/featcarm_v5/oof_featcarm_site.csv, minimise ll(y, sigmoid(logit(p)/T)) over T. Expect 1.137 / 1.281 / 1.220 / 1.296.
- If the final ensemble changes (new arms, new seeds, dropped arms), re-run only the T fit -- not lambda. Refit T on the new nested pool, then ship max(T_fit, 0.90). Keep lambda pinned at 0.01 regardless of what the nested pool says, per ADR 0006.
- Before upload, read the DrivenData log for `checkpoints used N/N` on every arm and for any `ladder:` line. If the ladder fired, the shipped ensemble is narrower than the one that was calibrated and lambda=0.01 is doing more work than this analysis assumes -- in that case do not consider lowering it under any circumstance.
- Record in RESULTS_LOG.md that line 321's 'T_site 0.826 vs T_random ~1.0' compares a fitted value against a floored one (line 313 sets 'T floor 1.0'; artifacts/run4a_stack/manifest.json:4 has the unfloored T_random = 0.8148), so the site pool prefers a FLATTER temperature than the random pool, not a sharper one.

### Rejected alternatives

- lambda = 0. Buys 0.00115-0.0018 nats on the shift-free pool, raises a confident miss from 5.16 to 8.16 realistic nats, breaks even at 0.8 scans per 2000, and has the second-worst minimax regret of every cell tested (0.00260, vs 0.00182 for the shipped cell). The site-fold pool contains 3-4 scans per 1362 that are exactly the event it bets against. Already forbidden by ADR 0006 and now falsified on shifted data.
- lambda = 0.02 (reverting to the Master v2 / slot-1 value). Only optimal if the true test gap exceeds about 0.030; the observed gaps are 0.018-0.027. Costs 0.0020 nested, carries max regret 0.00382 against the shipped cell's 0.00182, and moves the configuration away from the only one with a verified 0.2473 public score. The smoke set's apparent preference for 0.02 (0.2374 vs 0.2474, COUNCIL_VERDICT_ENDGAME.md:117) is 20 scans against a measured 0.0087 noise floor and is never fitted on (ADR 0005).
- lambda = 0.005 -- the actual minimax winner (max regret 0.00088 vs 0.00182). Rejected on decision hygiene, not on numbers: the 0.00094 edge is smaller than the uncertainty in the shift model that produced it, it is a third distinct lambda on the account with no way to measure which of the three worked, and ADR 0006 exists precisely because lambda was moved on evidence this thin before. Note it as the fallback if anyone insists on touching lambda.
- T = 0.815 or the 'nested optimum' framing generally. 0.8148 is artifacts/run4a_stack/manifest.json:4, a different ensemble; the 5-pillar pool's own nested optimum is 0.870, and T=0.90 costs 0.00015 against it. Every shift model raises the optimal T at fixed lambda, so 0.90 is on the correct side. (T=0.815, lambda=0.01) has worse minimax regret (0.00155 at the best of that family, and 0.01190 at T=0.815/lambda=0) than the shipped cell in the lambda=0 case.
- T = 1.00 or reinstating the T>=1.0 floor. Costs 0.0027 on the nested pool at lambda=0.01 and has max regret 0.00455 -- the worst of the sensible cells. RESULTS_LOG.md:321 already recorded that forcing T=1.0 on the site pool made it worse (0.2449 vs 0.2402).
- Fitting T on the site pool and shipping that (fit_stack_v3.py:57-58, --site-arm). Only two of the seven shipped arms have committed site OOFs, the pseudo-sites are clustered rather than labelled (ADR 0004 forbids site scalars), and the site-fold estimate is noisier than the random-fold one. Use the site pool to choose the DIRECTION of the hedge, not its magnitude.
- Setting the manifest `clip` field (e.g. [0.01, 0.99]). lambda=0.01 already bounds p to [0.005485, 0.995485], 55x tighter than EPS; EPS binds on 0 of 1362 scans either way. Priced across all four shift levels, no clip configuration beats plain lambda=0.01 by more than 0.0004, and the best hybrid (lambda=0.005, clip=0.01) wins 0.0004 at c=1.0 while losing 0.0002 at c=0. It would also require editing build_master_v3.py:111, which packs the working tree.
- Searching T and lambda jointly on the final pool and shipping the argmin. The surface is a flat ridge -- on the shift-free pool the sequential penalty is +0.00019 and on the site pool +0.00302, but every point along the ridge is within 0.001 of every other. Joint search changes the story (it flips the apparent direction of T under shift) without changing the number worth shipping.
- Per-sample hedging via the `hedge` (infer_v3.py:189-191) or `typ` (infer_v3.py:182-184) fields. The typ path computes test-set typicality statistics that main_v3.py:64 would print, which brief D flagged as a disqualification exposure; build_master_v3.py:111 hard-sets both to None and should stay that way. Neither was ever selected by a nested gain.
- Isotonic or Platt recalibration on the site pool. Rejected by ADR 0003 (the combiner is exactly three fitted numbers) and out of scope per the run-9 spec.

### Open questions

- The true shift level of the hidden test. Everything above reduces to reading lambda* off the G -> lambda* table, and G is only bounded by the recorded nested->public gaps (0.018, 0.021, 0.027), which also contain nested-estimate optimism and any arm the ladder dropped. Resolved by reading the DrivenData submission log for `checkpoints used N/N` on both real submissions: if the arms were identical, the gap is pure shift and lambda* ~ 0.006-0.017; if the ladder fired, the gap is inflated and lambda* is nearer 0.002.
- Whether T_fold hit the 0.90 floor in every fold of the soft pool. scripts/make_soft_targets.py:31 takes max(fit, 0.9) and does not record the fit, so converting the pool's baked-in temperature into absolute T carries up to a few percent of slack. Resolved by re-running make_soft_targets.py with a print of the per-fold T, or by reading data/master_pool logs if they survive.
- Whether the site displacement measured on two feature arms transfers to the 7-arm CNN ensemble at c=1. featc is one of the five pillars, so the displacement is partly the ensemble's own, but the CNN arms may be more (or less) site-robust; the master pool's own recorded site gap was only 0.0046 (RESULTS_LOG.md:321) against the feature pair's 0.0279, which argues c < 1. Resolved by building site-fold OOFs for the CNN pillars, which requires data/ and cache_v4/ (both absent) and a GPU session.
- Whether the observed public difference between the two real submissions (0.2557 at lambda=0.02, 0.2473 at lambda=0.01) is genuinely the ladder. ADR 0006 establishes it cannot be lambda; the alternative candidates were never enumerated. If the true cause is something that also changes under the final build, this analysis's assumption that the arms are fixed fails. Resolved by the ledger transcription in effort-state open item 1.

> **Needs checking** — could not be established from the repo, the rules or arithmetic: numbers.


---

## GPU spend inside the size envelope

**Question.** Given ~17 spare checkpoints at a 1050 MB ceiling and 30 GPU-hours, which training spend maximises nested gain per checkpoint shipped?

### Verdict

Raise the ceiling to --max-mb 1050 and spend ONE Kaggle session (~4 GPU-h, 20 jobs) on the two under-seeded arms that are already in the greedy best subset — a3d_crophistsoft 1->3 seeds and a3d_hemisoft 3->5 seeds — because at ~3.8 MB per 3D checkpoint those 20 checkpoints cost only 77 MB, not the 238 MB the stale 11.9 MB/ckpt average implies; add no more plain a3d_hemi seeds (measured 0.0001 nats for 10 checkpoints), add nothing 2D, build no model soup, and stop all training at Sept 15 18:00 UTC.

### Confidence

high

### Evidence

- gen_run8.py real CLI surface, enumerated from source (kaggle/run8_distill/gen_run8.py:25-41): --owner (skyfall23), --slug (dat-run8-distill), --out (script dir), --cache-ds, --raw-ds, --ckpt-ds, --soft-lam (0.4), --mistake-w (2.0), --soft-table (<script dir>/soft_master_v2.csv, NOT relative to --out), --hemisoft-seeds (0 1 2), --hemi-more-seeds (3 4), --hemimw-seeds (0), --crophistsoft-seeds (0), --skip (job names), --kernel-source (append; mounts a previous notebook's output so the push resumes without secrets). Fifteen flags; there is no --hemi-more-seeds alias and no flag for the site or full-data jobs.
- Jobs each flag produces (gen_run8.py:60-67 x the expander at kaggle/gen_run3_inject.py:142-151, folds default range(5)): r3d_hemisoft = len(--hemisoft-seeds) x 5 -> arm a3d_hemisoft; r3d_hemisoft_site = seeds [0] HARDCODED on folds_site.csv = 5 jobs; r3d_hemi_more = len(--hemi-more-seeds) x 5 -> arm a3d_hemi; r3d_hemimw = len(--hemimw-seeds) x 5 -> arm a3d_hemimw; r3d_crophistsoft = len(--crophistsoft-seeds) x 5 -> arm a3d_crophistsoft; r3d_hemisoft_full = seeds [0,1] HARDCODED with folds [-1] = 2 jobs. Defaults sum to 15+5+10+5+5+2 = 42, matching the throughput reference; gen_run8.py:143-144 prints the count, so the push can be sanity-checked before it costs GPU.
- SEVEN of those 42 default jobs produce artifacts that build_master_v3.py can never ship or gate: load_arm (scripts/build_master_v3.py:26) filters out every file whose name contains '_site' or '_fall', which is exactly r3d_hemisoft_site (5 jobs) and r3d_hemisoft_full (2 jobs). The run-8 GATE cell applies the same filter (gen_run8.py:94). That is ~17% of run 8's GPU spent on non-shippable checkpoints -> both MUST appear in --skip.
- Checkpoints are saved fp16 (datsc/train_v3.py:300-302, save_fp16=True by default) and packed with ZIP_DEFLATED (scripts/build_master_v3.py:125). Per-checkpoint size is therefore params x 2 bytes x ~0.92.
- Parameter counts computed from source with no torch (datsc/models_h.py:24-49 HemiNet3D + datsc/models_v3.py:16-59 Block3D/_gn/ResNet3D/ScalarHead, widths (24,48,96,192) from gen_run8.py:56): HemiNet3D = 2,079,482 params -> 4.16 MB fp16 -> 3.83 MB deflated; ResNet3D = 2,042,489 -> 3.76 MB. torchvision efficientnet_v2_s (21.46M) and convnext_tiny (28.59M) minus their 1000-class heads -> 37.1 MB and 51.2 MB.
- Independent empirical confirmation by differencing two shipped zips: Master v2 = 732 MB / 41 ckpts (RESULTS_LOG.md:323) and Master v3 = 847 MB / 71 ckpts (RESULTS_LOG.md:349-357). The 30-checkpoint delta is purely 3D (a3d_hemi s3-s4 +10, a3d_hemisoft +15, a3d_crophistsoft +5), so (847-732)/30 = 3.83 MB — identical to the parameter-derived 3.83 MB.
- Whole-zip reconciliation (re-runnable): 40x3.83 (hemi+hemisoft) + 10x3.76 (ssl) + 5x3.76 (crophistsoft) + 10x37.1 (v2s) + 5x51.2 (cnx_cor) = 837 MB against the observed 847 MB; the 10 MB residual is the featc model + artifacts/template.npz + the packed main_v3.py/tests_synth.py/datsc code (scripts/build_master_v3.py:126-129). Agreement to 1.2%.
- Therefore the repo's '~11.9 MB/checkpoint' (aidlc-docs/efforts/008-run9-endgame/effort-state.md:26, COUNCIL_VERDICT_ENDGAME.md:79-80) is the 847/71 zip AVERAGE, dominated by the 15 two-dimensional checkpoints. It is the wrong unit: at a 1050 MB ceiling the 203 MB of headroom buys 52 three-dimensional checkpoints, or 5 a2d_v2s ones, or 3 a2d_cnx_cor ones. Every candidate arm for more seeds is 3D, so the real budget is ~52, not 17.
- Arm inventory of the shipped 71-checkpoint zip (RESULTS_LOG.md:351-357) with seed counts implied by 5 folds: a3d_hemi 25 = 5 seeds, a3d_ssl 10 = 2 seeds, featc 1 (single non-CV model), a2d_cnx_cor 5 = 1 seed, a2d_v2s 10 = 2 seeds (PINNED to s[01] by the glob r2d_efficientnet_v2_s_f?_s[01] at scripts/build_master_v3.py:18, although a 3rd seed exists in the run-5 pool, RESULTS_LOG.md:195), a3d_hemisoft 15 = 3 seeds, a3d_crophistsoft 5 = 1 seed. a3d_hemimw (5 ckpts, 1 seed) was trained and is NOT in the best subset.
- Greedy best subset = a3d_hemi, a3d_ssl, featc, a2d_cnx_cor, a2d_v2s, a3d_hemisoft, a3d_crophistsoft; equal-weight nested 0.2289 vs Master v2 shipped 0.2323 (RESULTS_LOG.md:344-346). a3d_hemimw alone reached only 0.2313 and was dropped by the greedy loop (scripts/build_master_v3.py:79-84).
- THE decisive measurement for seed spend (RESULTS_LOG.md:338-339): row 1 'Master v2 as shipped (hemi s0-2, v2s s0-1)' = 0.2323 and row 2 'pillars, every local seed' = 0.2322 differ ONLY in a3d_hemi seeds — V2_SHIPPED pins r3d_hemi48_f?_s[012] while PILLARS globs s? (scripts/build_master_v3.py:18-19). Going 3 seeds -> 5 seeds of the most heavily weighted arm bought 0.0001 nats for 10 checkpoints = 1.0e-05 nats/ckpt. The seed curve on a3d_hemi is flat.
- Gain per checkpoint of every option the run-8 gate table actually measured, against the 0.2322 all-seeds-pillars baseline (RESULTS_LOG.md:339-344): a3d_crophistsoft 1 seed +0.0019/5 ckpts = 3.8e-04; a3d_hemimw 1 seed +0.0009/5 = 1.8e-04; a3d_hemisoft 3 seeds +0.0023/15 = 1.53e-04; both gated arms together +0.0033/20 = 1.65e-04; a3d_hemi extra seeds +0.0001/10 = 1.0e-05. New arms beat extra seeds of a saturated arm by 15-38x, and the arm at ONE seed (a3d_crophistsoft) is the best per checkpoint on record.
- Left-right flip TTA is ALREADY ON at inference and adds ZERO checkpoints: InputSpec.variants (datsc/train_v3.py:73-80) yields both inp and self.flip(inp) unconditionally for every box and every z-shift, in both the 2D and 3D branches, and datsc/infer_v3.py:157 averages exactly spec.variants(x). The 2D z-shift set {-1,0,+1} is likewise already in the shipped specs and was matched at re-OOF time (gen_run8.py:127). There is nothing to buy here.
- Affine TTA is the only TTA lever left and it was measured and rejected: 'Affine TTA (3 reproducible small affines x flip) on the striatal-crop 3D fold-0 checkpoint: 0.2257 -> 0.2269 — no gain, off by default' (RESULTS_LOG.md:164). It is off by default (tta_affine=0, datsc/train_v3.py:43,49) and would multiply inference time by 4 against a 150-minute ladder.
- MODEL SOUP — would it load with no code change? YES. datsc/infer_v3.py:150-152 builds the architecture from arm_cfg['model_kw'] and calls model.load_state_dict(torch.load(...)); a weight-average of same-architecture fp16 state_dicts has identical keys, shapes and dtype, and the manifest simply lists one checkpoint path. No datsc/ edit is required, so no untested code would ship (scripts/build_master_v3.py:126-128 packs the working tree's main_v3.py, tests_synth.py and datsc/*.py). It is still REJECTED for three reasons below.
- Soup blocker 1 — no shared basin: every seed is independently initialised (datsc/train_v3.py:93-96 _seed_all is called per job, and 2D backbones load ImageNet weights only as a starting point), so seeds sit in permutation-distinct minima and a naive average is not a valid model. Only the a3d_ssl family warm-starts from a shared encoder (datsc/train_v3.py:186-193) and even there the heads are fresh.
- Soup blocker 2 — it invalidates the gate: the nested estimate is fitted on the per-fold oof_*.csv files (scripts/build_master_v3.py:25-30, 105), so a souped checkpoint would ship an ensemble the gate table never measured — precisely the silent-degradation path the run-9 build guards were added to close (aidlc-docs/efforts/008-run9-endgame/effort-state.md:25; docs/superpowers/specs/2026-09-13-run9-endgame-spec.md:42-44). Re-validating it needs a full Kaggle re-OOF (scripts/reoof_v3.py), i.e. spending the GPU budget on validation instead of on seeds.
- Soup blocker 3 — unnecessary: soup trades size for seeds, and at 3.83 MB the 3D checkpoints are not what consumes the envelope. The only checkpoints worth souping are the 2D ones (37-51 MB) and those are exactly the arms with a single shared-ImageNet-init lineage per seed, i.e. the hardest to validate in the time left.
- THE REAL SECOND CONSTRAINT — the 150-minute ladder. datsc/infer_v3.py:24 sets LADDER_STOP_S = 150*60; main_v3.py:67-69 iterates man['arms'] IN MANIFEST ORDER and breaks past that clock, and datsc/infer_v3.py:144 stops starting checkpoints inside an arm. build_zip writes arms in greedy order (scripts/build_master_v3.py:102, names = pillars + additions), so the shipped order is a3d_hemi(25), a3d_ssl(10), featc(1), a2d_cnx_cor(5), a2d_v2s(10), a3d_hemisoft(15), a3d_crophistsoft(5) — the two arms that justified Master v3 run LAST and are the first thing the ladder drops. effort-state.md:23 already names the ladder as the leading explanation for the arithmetically impossible '+0.0084 from lambda'. Checkpoint compute is roughly arm-independent (3D hemi ~15 GFLOP/scan/ckpt over 2 flip variants vs EfficientNet-V2-S ~17 GFLOP over 6 variants), so +20 checkpoints is ~+28% inference work regardless of which arm they go to.
- Test-set scale for the ladder budget: 'extrapolated ~12 min for 2,000 test volumes on the 24-vCPU/A100 box (limit 180 min)' for preprocessing (RESULTS_LOG.md:42); the Master v3 local container check took 566 s for 71 checkpoints on 20 scans (RESULTS_LOG.md:360) against 317 s for 30 checkpoints (RESULTS_LOG.md:156).
- A >1 GB zip has already been accepted and scored by the platform: submission_run5.zip, 39 ckpts, 1421 MB, smoke id-321496 = 0.2388 (RESULTS_LOG.md:212). So a 924 MB zip carries no upload risk; the 844 MB dialog failure at RESULTS_LOG.md:159-160 was a Playwright automation fault, and 847 MB has since uploaded twice by hand.
- The landmine: scripts/build_master_v3.py:159 defaults --max-mb to 850 and :136-146 silently applies DROP_ORDER (:22), whose first action deletes a2d_v2s _s1 — a pruning measured at ~0.01 nats (aidlc-docs/HANDOFF.md:121). Any build that forgets --max-mb 1050 destroys ten times the gain this whole session can produce. Since the run-9 guard landed (:147-149) it at least exits non-zero once DROP_ORDER is exhausted, but the first drops are still silent.
- No edit to scripts/ is needed for this plan: NEW already globs every seed (scripts/build_master_v3.py:20, r3d_hemisoft_f?_s? and r3d_crophistsoft_f?_s?), and TEMPLATE_ARM (:21) already maps both new arms onto shippable manifest templates.
- Arithmetic in this item is re-runnable with `py` alone (no torch): params = sum over the widths (24,48,96,192) Block3D/stem/head formulas; MB = params*2/1e6*0.92; the three headroom lines are (ceiling-847)/per_ckpt_MB.

### Risk if wrong

Three distinct failure modes, priced. (1) If the ladder is ALREADY firing at 71 checkpoints, adding 20 more pushes a3d_hemisoft and a3d_crophistsoft — which run last in manifest order — past the 150-minute stop, and combine() renormalises over the survivors, so the shipped ensemble silently becomes the five pillars. That costs the full 0.0033 the run-8 arms bought, i.e. ~3x the gain this session can plausibly return, on the ONLY slot left. This is why the View-log check is a hard precondition, not a nicety. (2) If --max-mb is left at its 850 default, DROP_ORDER deletes a2d_v2s seed 1 for -0.01 nats, four to twenty times the session's gain. (3) If the seed curve on a3d_crophistsoft is as flat as the measured a3d_hemi curve (1.0e-05 nats/ckpt), the session returns ~0.0002 and 4 GPU-hours are wasted — recoverable, since the banked 847 MB / nested 0.2266 zip stays shippable and the decision rule refuses to swap it in below a 0.0010 improvement. Getting the SIZE model wrong in the other direction (believing 11.9 MB/ckpt) costs nothing in log loss but silently caps the session at 17 checkpoints when 52 were affordable, i.e. it forfeits roughly two thirds of the available spend.

### Action

- STEP 0 — PRECONDITION, free, ~30 min, do this before any push. On drivendata.org open View log for BOTH real submissions (the 0.2557 and the 0.2473 runs). Count the lines 'stage written after arm <name>' — there must be exactly 7, ending with a3d_crophistsoft — and the per-arm 'arm <name>: checkpoints used N/N' lines, and search for the string 'ladder:'. If 'ladder:' appears, or any torch arm shows used < N, or the final 'done in Ns' is within ~15 min of 9000 s, then CANCEL all training: instead rebuild the SAME 71-checkpoint ensemble with the cheap arms first and re-verify. Only if the logs are clean with visible margin, continue to Step 1.
- STEP 1 — generate the session-A notebook (laptop, ~10 s). Writes a NEW folder, so kaggle/run8_distill/ is left untouched and run 8's output stays mounted as the resume source: py kaggle/run8_distill/gen_run8.py --out kaggle/run9_seeds --slug dat-run9-seeds --kernel-source skyfall23/dat-run8-distill --hemisoft-seeds 0 1 2 3 4 --crophistsoft-seeds 0 1 2 --skip r3d_hemisoft_site r3d_hemi_more r3d_hemimw r3d_hemisoft_full
- STEP 2 — verify the generator's own printout before spending GPU. It must say '40 training jobs' (25 hemisoft + 15 crophistsoft) and 'kernel sources [\'skyfall23/dat-run8-distill\']'. 20 of those 40 are already in run 8's output, so the runner cell will print '20 jobs to run, 20 already done'. If it prints any other job count, the --skip list is wrong — do not push.
- STEP 3 — push session A (target: Sept 14, morning UTC). kaggle kernels push -p kaggle/run9_seeds  — then on the notebook page set Session options -> GPU T4 x2, Internet On (aidlc-docs/HANDOFF.md:117) and push again if it landed on a single GPU. Expect ~2.1 h of training wall plus the re-OOF/super-stack/gate cells, ~3 h total, 4.1 GPU-h of the 30 available.
- STEP 4 — poll, never interactively. kaggle kernels status skyfall23/dat-run9-seeds
- STEP 5 — fetch. py scripts/watch_kaggle_output.py --kernel skyfall23/dat-run9-seeds --out data/run9a
- STEP 6 — re-gate WITH THE RAISED CEILING AND THE SHIPPED LAMBDA. Omitting --max-mb 1050 here silently deletes a2d_v2s seed 1 for -0.01 nats: py scripts/build_master_v3.py --src data/run9a/artifacts_v3 --src data/run8/artifacts_v3 --src data/master_pool/artifacts_v3 --max-mb 1050 --lam 0.01 --smoke data/smoke --zip data/submission_master_v3c.zip
- STEP 7 — read the output against three hard checks, not against the printed GATE PASS/FAIL (whose 0.2323 baseline is stale). (a) the 'greedy best subset' equal column must be <= 0.2256, i.e. at least 0.0010 better than the banked 0.2266; (b) the final line must read '91 checkpoint files across 7 arms, 924 MB' with NO 'zip ... > 1050: dropped ...' line above it; (c) the container check must print FORMAT OK, every arm N/N, and no 'ladder:'. If (a) fails, the seed curve is flat — keep the banked zip and go to Step 10. If (b) or (c) fails, keep the banked zip and stop.
- STEP 8 — decide session B (Sept 15 morning UTC, ONLY if Step 7(a) passed with >= 0.0010). Same command with two more seeds each, +20 checkpoints -> 111 ckpts / ~1000 MB / 4.1 more GPU-h: py kaggle/run8_distill/gen_run8.py --out kaggle/run9_seeds --slug dat-run9-seeds2 --kernel-source skyfall23/dat-run9-seeds --hemisoft-seeds 0 1 2 3 4 5 6 --crophistsoft-seeds 0 1 2 3 4 --skip r3d_hemisoft_site r3d_hemi_more r3d_hemimw r3d_hemisoft_full   then: kaggle kernels push -p kaggle/run9_seeds
- STEP 9 — HARD STOP: Sept 15 18:00 UTC. No 'kaggle kernels push' after this instant, whatever the gate table says. The budget behind it: a session is ~3 h and can be cut and need one full retry (~6 h), the paged fetch of a ~1 GB checkpoint dataset plus rebuild plus local_e2e is ~1 h, a DrivenData smoke test is a 12 min upload plus ~1 h for the score and is capped at 3/day (aidlc-docs/HANDOFF.md:120,125), the freeze is Sept 16 12:00 and the upload ~16:00 leaving ~8 h of margin to the 23:59 deadline. Sept 15 18:00 leaves 18 h of slack; anything later eats the retry window.
- STEP 10 — ship. Smoke-test the EXACT frozen zip on DrivenData, confirm FORMAT OK / all arms N/N / no 'ladder:' in the platform log, then upload as the single remaining real submission at ~Sept 16 16:00 UTC. If no session beat 0.2256, ship the banked data/submission_master_v3_lam01.zip unchanged — do NOT spend the slot at Sept 14 03:38 out of habit.

### Rejected alternatives

- More plain a3d_hemi seeds (--hemi-more-seeds 5 6, the 'safe gain' line at aidlc-docs/HANDOFF.md:93). REJECTED on direct measurement: seeds 3->5 of this exact arm moved the nested estimate from 0.2323 to 0.2322 (RESULTS_LOG.md:338-339), 0.0001 nats for 10 checkpoints = 1.0e-05 nats/ckpt, the worst ratio in the table and 38x below a3d_crophistsoft. a3d_hemi is at 5 seeds and saturated; more seeds do not raise its 1/7 equal weight, they only shave OOF noise that is already averaged over 25 checkpoints.
- More seeds of a3d_hemimw (--hemimw-seeds 0 1 2, the HANDOFF step-B option (ii)). REJECTED: the arm was measured at 0.2313 against the 0.2322 baseline and the greedy loop (scripts/build_master_v3.py:79-84) declined it. Its logits are also the most collinear with a3d_hemisoft (same hemi architecture, same soft-target table, only the sample weights differ), so extra seeds buy variance reduction on an arm the stack does not want.
- a2d_v2s seed 2, which already exists in the run-5 pool (RESULTS_LOG.md:195) and would cost ZERO GPU-hours — just widen the glob at scripts/build_master_v3.py:18 from _s[01] to _s?. REJECTED on size: 5 checkpoints x 37.1 MB = 186 MB, which is 92% of the entire 1050 MB headroom for one extra seed of an arm that already has two, and it cannot be combined with the 3D plan (924 + 186 = 1110 MB). Per MB it is 2.7e-06 nats vs the 3D arms' ~1e-04. Reconsider only if a dedicated size test establishes that a ~1120 MB zip runs inside the 150-minute ladder.
- A second a2d_cnx_cor seed. REJECTED twice over: 5 x 51.2 = 256 MB (more than the whole headroom), and gen_run8.py has no job family that trains it — it would need a different generator and an untested notebook.
- More a3d_ssl seeds. REJECTED: cheap in MB (3.8 MB/ckpt) but gen_run8.py's JOBS8 (gen_run8.py:60-67) contains no r3d_ssl family, so it needs the run-6c generator and a new, unvalidated notebook 48 h from the deadline. Not worth the packaging risk for an arm whose per-seed gain is unmeasured.
- Spending the full 52-checkpoint envelope in one go (10 seeds, ~1038 MB, 10.3 GPU-h). REJECTED on the ladder: +52 checkpoints is ~+73% inference work against a 150-minute stop whose current margin is unknown, and the arms it would drop are exactly the ones being added. The plan deliberately buys 20 (+28%) and keeps ~130 MB and the whole ladder question in reserve.
- Model soup / weight-averaging several seeds into one checkpoint. REJECTED although the shipped inference path WOULD load it unmodified (datsc/infer_v3.py:150-152 is a plain build_model + load_state_dict, and a soup of same-architecture fp16 state_dicts is key-, shape- and dtype-identical). Three reasons: seeds are independently initialised (datsc/train_v3.py:93-96) so there is no shared loss basin and a naive average is not a valid model; the gate table is fitted on per-fold OOF csvs (scripts/build_master_v3.py:25-30) so a souped checkpoint ships an ensemble that was never measured — the exact silent-degradation path the run-9 guards closed; and it is solving the wrong problem, since 3D checkpoints cost 3.8 MB and are not what fills the zip.
- Left-right flip TTA as a new lever. NOT REJECTED — IT IS ALREADY ON AND FREE. InputSpec.variants (datsc/train_v3.py:73-80) yields self.flip(inp) for every variant unconditionally in both arms, and datsc/infer_v3.py:157 averages exactly that set. It adds zero checkpoints and zero MB, and the OOF csvs were produced with the same flip (datsc/train_v3.py:297 tta_flip=True) so the gate already prices it in. There is nothing left to buy.
- Affine TTA (tta_affine > 0). REJECTED on measurement: 0.2257 -> 0.2269 on the striatal-crop fold-0 checkpoint, i.e. it hurt (RESULTS_LOG.md:164), and it would multiply per-checkpoint inference by 4 against the 150-minute ladder.
- Wider 2D z-shift TTA. REJECTED: {-1,0,+1} is already the shipped set and the re-OOF was explicitly run with --tta-z -1 0 1 so the stack is fitted on what main.py produces (gen_run8.py:127). Widening it would desynchronise the gate from the submission and triple 2D inference cost.
- Running gen_run8.py with its defaults or with --kernel-source alone. REJECTED: the defaults include r3d_hemisoft_site (5 jobs) and r3d_hemisoft_full (2 jobs) whose checkpoints load_arm discards via its '_site'/'_fall' filter (scripts/build_master_v3.py:26) — 7 jobs, ~1.5 GPU-h, producing nothing that can be shipped or gated. Both must be in --skip.
- Pushing into the existing kaggle/run8_distill folder with the default slug (the pattern at aidlc-docs/HANDOFF.md:55). REJECTED in favour of --out kaggle/run9_seeds --slug dat-run9-seeds: it avoids a kernel sourcing its own output, leaves run 8's notebook and its output dataset intact as a fallback, and overwrites nothing (--soft-table still resolves against the script's own directory, gen_run8.py:34).
- Submitting at Sept 14 03:38 when the slot opens. REJECTED: the slot opens and stays open to the deadline (aidlc-docs/efforts/008-run9-endgame/effort-state.md:21), and the standing plan would spend it on master_v2 at nested 0.2323 — worse than the banked 0.2266.

### Open questions

- Is the 150-minute ladder already firing at 71 checkpoints? This is the single question that decides whether ANY checkpoint may be added. Resolved for free in 30 minutes by Step 0: the View log of either real submission must show 7 'stage written after arm' lines, 'checkpoints used N/N' for all 6 torch arms, and no 'ladder:'. Until it is answered, treat the +28% inference increase as unpriced.
- What is the actual per-checkpoint GPU inference cost on the A100 for the 3D vs 2D arms? The FLOP estimate says they are within ~25% of each other (3D hemi runs 2 flip variants of a 2.1M-param net on a 48x48x32 crop; EfficientNet-V2-S runs 6 variants of a 21M-param net on 192x192 slabs), which is why the plan prices extra checkpoints as uniformly expensive in time even though they are 10x cheaper in MB. Resolved by the per-arm timestamps in the same View log ('stage written after arm X at Ns').
- Does the a3d_crophistsoft seed curve behave like the 1->2 step (steep) or like a3d_hemi's 3->5 step (flat)? Session A is itself the experiment; the Step 7(a) threshold of 0.0010 is the decision rule and Step 8 is conditioned on it.
- Does Kaggle accept a kernel whose --kernel-source is a notebook owned by the same account (dat-run9-seeds2 sourcing dat-run9-seeds)? The repo documents the resume pattern (aidlc-docs/HANDOFF.md:116) but only ever self-sourced dat-run8-distill. If session B's runner prints '40 jobs to run, 0 already done' the mount failed — kill the session immediately rather than retrain run 8 from scratch.
- Is there a DrivenData hard cap between 1050 MB and 1421 MB? A 1421 MB zip scored on the platform (RESULTS_LOG.md:212), so 924 MB is certainly safe, but nothing in the repo establishes the actual limit. Only matters if the a2d_v2s third seed is revisited.
- Would reordering the manifest so a3d_hemisoft and a3d_crophistsoft run BEFORE the two 2D arms be a strictly free insurance policy? It is a scripts/build_master_v3.py:102 change only, so no untested code ships (the zip packs main_v3.py, tests_synth.py and datsc/*.py, :126-128), and it moves the gate-justifying arms out of the ladder's firing line. Out of scope for this item but it should be settled before the freeze.

> **Needs checking** — could not be established from the repo, the rules or arithmetic: numbers.


---

## Inference ladder and packaging risk

**Question.** Can the 150-minute ladder silently drop arms on the hidden test set, and what makes the final zip verifiably the ensemble that was gated?

### Verdict

Leave LADDER_STOP_S, the arm order and every file in datsc/ exactly as they are - the ladder has roughly 6x headroom even at the 1050 MB / 88-checkpoint ceiling - and spend the remaining packaging effort on three verification steps instead: transcribe 'done in Xs' and the seven 'stage written after arm' lines from the two real-submission View logs, build the final zip at an explicit --max-mb that emits NO 'dropped' line, and platform-smoke the byte-identical (sha256-recorded) frozen zip before uploading it.

### Confidence

medium

### Evidence

- datsc/infer_v3.py:24 - LADDER_STOP_S = 150 * 60 = 9000 s, a single flat deadline; rules cap is 10800 s (DAT_PARKINSONS_CHALLENGE.md:71).
- main_v3.py:68-69 checks the ladder before each arm; datsc/infer_v3.py:144-145 before each checkpoint. Both only print ('ladder: skipping remaining arms' / 'checkpoints'); neither raises, so truncation is silent to the exit code.
- datsc/infer_v3.py:176-181 - combine() takes names = arms that produced probs and renormalises w over them, so a truncated ensemble still writes a well-formed, confidently-wrong submission.
- main_v3.py:55 passes deadline_s = infer.LADDER_STOP_S to preprocess_all, i.e. preprocessing alone is allowed to consume the entire 150 min before a single arm runs.
- DAT_PARKINSONS_CHALLENGE.md:122 - platform log of the prior-only smoke id-321144: 'preprocessed 20/20 volumes in 6s' with 20 workers => ~6 CPU-s/volume. Arithmetic: 2000 x 6 / 20 = 600 s = 10.0 min. This is the only platform-measured timing anchor in the repo.
- RESULTS_LOG.md:42 - independent Kaggle measurement, 150 volumes in 204 s on 4 CPUs = 5.4 CPU-s/volume, agreeing with the platform's 6 CPU-s/volume within 10%.
- RESULTS_LOG.md:62 - SimpleITK registration is 1.6-1.8 s/volume and is the bulk of that 6 CPU-s.
- main_v3.py:52 - n_workers = max(1, min(20, cpu_count-2)) = 20 on the 24-vCPU box, matching the measured configuration.
- datsc/volume.py:17 - VOL_SHAPE = (96, 96, 72) => 1.327 MB/volume float16 (2.65 GB at N=2000); datsc/infer_v3.py:140-141 builds a float32 copy per arm = 5.31 GB at N=2000.
- datsc/infer_v3.py:155-157 - batches of 16 are moved to the device inside the per-checkpoint loop, and spec.variants()/cropping happen AFTER the transfer, so the full 96x96x72 box (5.31 GB at N=2000) crosses PCIe once per checkpoint. This, not the convolutions, dominates per-checkpoint cost.
- datsc/infer_v3.py:150 - build_model(...) is reconstructed from scratch for every checkpoint (torchvision backbone constructed and randomly initialised, then .to(device)); datsc/models_v3.py:115-121 confirms weights=None so there is no network access. This per-checkpoint fixed cost (~0.3-1.0 s) is the one that is amortised over only 20 scans in the smoke run.
- Budget arithmetic at N=2000, 71 ckpts: 600 s preprocess + 71 x 5 s checkpoints + ~400 s featc + ~60 s startup/import/CUDA + ~42 s per-arm stacking = ~1460 s = 24 min, i.e. 16% of the 9000 s ladder. At 88 ckpts: ~1540 s = 26 min, 17%. Re-runnable: total(K, N) = 0.3*N + K*5*(N/2000) + 0.4*N + 60.
- Solving total(K, 2000) = 9000 gives K ~= 1580 checkpoints; solving total(71, N) = 9000 gives N ~= 10200 volumes (18700 if featc is free). The ladder is ~20x away from biting on checkpoint count and ~5x away on test-set size.
- RESULTS_LOG.md:349-360 - the shipped zip is 847 MB / 71 checkpoints = 11.93 MB/ckpt; (1050-847)/11.93 = 17.0 more checkpoints, total 88.
- RESULTS_LOG.md:361-362 - DrivenData smoke id-321850 ran the 847 MB / 71-ckpt zip and returned 0.2374, proving a 71-checkpoint zip completes inside the 360 s smoke cap on the platform. The wall-clock itself was not transcribed.
- RESULTS_LOG.md:212 - smoke id-321496 ran a 1421 MB / 39-ckpt zip and returned 0.2388. So 1421 MB both uploads and fits the 6-minute cap; --max-mb 1050 is strictly inside a demonstrated envelope.
- RESULTS_LOG.md:226-228 and :243-245 - 1815 MB and 2275 MB zips had to be repacked 'to satisfy the platform upload size limit', bracketing that limit in [1421, 1815) MB.
- RESULTS_LOG.md:39 - smoke id-321152, a 50.7 MB / 15-checkpoint zip, 'Completed (8 s, 15/15 checkpoints, exit 0)'. The only other platform wall-clock on record, and it is consistent with checkpoint loading, not compute, being the fixed cost.
- RESULTS_LOG.md:360 - the local container check of the same 71-ckpt zip took 566 s on a laptop CPU, which is ABOVE the 360 s smoke cap. The zip passed the platform smoke anyway (A100 + 24 vCPU), so the local wall-clock carries no information about smoke margin.
- scripts/local_e2e_v3.py:35-37 asserts that no 'ladder:' line appears, but the local run has 566 s against a 9000 s ladder (15.9x margin), so that assertion can never fire locally and provides no protection for the real run.
- scripts/local_e2e_v3.py:23 launches sys.executable on the host CPU; the platform is Python 3.12 + PyTorch on CUDA 12.9 (DAT_PARKINSONS_CHALLENGE.md:62). datsc/infer_v3.py:154 enables autocast only when device.startswith('cuda'), so fp16 numerics are never exercised by the local check - only a platform smoke exercises them.
- scripts/build_master_v3.py:126-128 - the zip is packed from the WORKING TREE: ROOT/main_v3.py as main.py, ROOT/tests_synth.py, and every datsc/*.py at build time. Any uncommitted edit under datsc/ ships; anything in datsc/ that is not a .py file does not ship.
- scripts/build_master_v3.py:102-107 - manifest arm order is just the iteration order of best_row[1] (pillars first, run-8 arms last). Reordering is a change to the build script or to the manifest JSON, NOT to datsc/, so it ships no untested inference code; combine() (datsc/infer_v3.py:176-181) is order-independent for a fixed arm set.
- scripts/build_master_v3.py:136-146 - DROP_ORDER trims checkpoints and can delete whole arms AFTER the gate table has been printed, renormalising man['weights'] at :144 without re-gating. The new size assert at :147-149 fires only when DROP_ORDER is EXHAUSTED - a SUCCESSFUL trim is still a silent divergence between the gated ensemble and the shipped one, announced only by the 'zip N MB > max: dropped ...' print at :145.
- datsc/infer_v3.py:52-60 - futures are resolved in submission order and every exception is swallowed into ok[i] = False. A BrokenProcessPool from a single crashed worker (malformed NIfTI segfaulting SimpleITK, OOM) is an Exception, so it is caught per-future and every REMAINING volume is also marked failed. The only trace is the aggregate 'preprocessed K/N volumes' line at :69.
- main_v3.py:76 - volumes with ok=False are written as the prior pi (0.548), which costs ~0.69 nats each; main_v3.py:87 - os._exit(0) forces exit code 0, so a run degraded this way still reports success to the platform.
- DAT_PARKINSONS_CHALLENGE.md:74 (300 lines x 300 chars) vs the emitted log: 3 header lines + 2 per arm x 7 + 1 'done in' = 18 lines nominally, 89 even if all 71 checkpoints print a 'checkpoint failed' line (datsc/infer_v3.py:161, truncated to 160 chars). The log cap is not binding and 'ladder:' lines will always be visible.
- scripts/build_master_v3.py:111 hard-sets typ to None and scripts/local_e2e_v3.py:38 asserts it; main_v3.py:60-64 would otherwise print median typicality and a count of the hidden test set, which DAT_PARKINSONS_CHALLENGE.md:86 makes grounds for disqualification.
- datsc/config.py:70-75 - find_nifti_path only tries niftis/<uid>.nii.gz and niftis/<uid>.nii, flat. This layout assumption is retired as a risk: the two real runs scored 0.2557 and 0.2473 (RESULTS_LOG.md:379, :397), which is impossible if the files were not found.
- Analytic, against the council's hypothesis (COUNCIL_VERDICT_ENDGAME.md:39-41): slot 1 and slot 2 ran the SAME 71-checkpoint 847 MB zip differing only in lam (RESULTS_LOG.md:376, :394). Their runtime cost is identical, so ladder truncation cannot produce a 0.0084 difference between them unless the ladder is marginally binding - which the budget above says it is not, by ~6x. The ladder is therefore not a plausible explanation for the over-ceiling lambda 'gain'.

### Risk if wrong

If the headroom estimate is wrong in the direction that matters - the hidden test set is ~5x larger than assumed, or featc/hist normalisation costs an order of magnitude more per volume than estimated - the ladder truncates the arms that run last (a3d_hemisoft, a3d_crophistsoft) and combine() renormalises the weights over the survivors, so the final submission silently becomes roughly the five-pillar ensemble: about +0.003 nested, ~0.003-0.005 public log loss, worth a few leaderboard places from rank ~16. That is the benign failure. The malignant one is not the ladder at all: a single crashed preprocessing worker marks every subsequent volume failed and fills those rows with the prior, which at 50% loss of coverage costs roughly +0.2 log loss and exits 0 while doing it - it burns the last slot with no crash to show for it. Conversely, if this verdict is wrong in the other direction - i.e. reordering arms or raising --max-mb to 1050 really was unsafe - the cost is a zip that differs from the one that was gated, which is exactly the failure the endgame is trying to avoid.

### Action

- 1. FIRST, free and decisive (15 min, no GPU): open the DrivenData submissions page, click View log on BOTH real submissions (the 71-ckpt master_v3 and master_v3_lam01 runs) and copy into RESULTS_LOG.md: the 'preprocessed N/N volumes in Ys' line, all seven 'stage written after arm <name> at Xs' lines, every 'arm <name>: checkpoints used k/k' line, the 'arm featc: featc model (...) on N volumes' line, the final 'done in Xs; failed volumes: M' line, and any 'ladder:' / 'checkpoint failed (' / 'arm ... failed:' / 'Traceback'. The per-arm timestamps give the exact real-set cost of one arm and one checkpoint, which turns every estimate below into a measurement.
- 2. From that log compute headroom directly: per_ckpt_s = (t_arm_end - t_arm_start) / n_ckpts for the largest arm, then projected_total(K) = t_preprocess + K * per_ckpt_s + featc_s + 60. Only if projected_total(88) > 0.6 * 9000 should anything in this item be revisited.
- 3. Do NOT edit datsc/. No change to LADDER_STOP_S, no change to arm order, no BrokenProcessPool retry, unless step 2 shows the projection above 0.6 * 9000 - the working tree ships verbatim (scripts/build_master_v3.py:126-128) and any datsc/ edit converts a six-times-proven inference path into one validated only by a CPU-only local run plus one smoke test.
- 4. Build the final zip at an explicit ceiling and require a clean build: `git status --porcelain` must be EMPTY before building, then `python scripts/build_master_v3.py --src data/run8/artifacts_v3 --src data/master_pool/artifacts_v3 --lam 0.01 --T 0.9 --max-mb 1050 --zip data/submission_v9.zip --smoke <smoke_dir>`.
- 5. Read the build tail and REJECT the zip if any line matching 'zip .* MB > .*: dropped' appears. A successful DROP_ORDER trim silently changes the ensemble after the gate table was printed (scripts/build_master_v3.py:136-146) and renormalises the weights without re-gating. If a drop line appears, raise --max-mb (up to 1400, inside the demonstrated 1421 MB acceptance) and rebuild rather than accepting the trim.
- 6. Confirm the built manifest matches the gate: `python -c "import json,zipfile; z=zipfile.ZipFile('data/submission_v9.zip'); m=json.loads(z.read('artifacts/manifest.json')); print(len(m['arms']), sum(len(a['checkpoints']) for a in m['arms']), m['weights'], m['T'], m['lam'], m['typ'], m['note'])"` - arm names and per-arm checkpoint counts must equal the gate table's best row, weights must all be 1/len(arms), and typ must be null.
- 7. Record provenance in RESULTS_LOG.md before uploading: `python -c "import hashlib,pathlib; p=pathlib.Path('data/submission_v9.zip'); print(p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest())"` plus `git rev-parse HEAD`. This sha256 is what makes the zip verifiably the one that was gated.
- 8. Run the container check on that exact file: `python scripts/local_e2e_v3.py data/submission_v9.zip <smoke_dir>` - require FORMAT OK, every arm N/N, an 'arm featc' line, and note that its 'no ladder:' assert (scripts/local_e2e_v3.py:35-37) proves nothing about the real run.
- 9. If Docker is available, repeat step 8 inside the official runtime image (competition-sfmn-parkinsons-runtime, Python 3.12 / CUDA 12.9, DAT_PARKINSONS_CHALLENGE.md:62). This is the only local way to catch a torch-version or weights_only=True incompatibility before spending the slot.
- 10. Platform-smoke the byte-identical file (re-check the sha256 after upload prep; smokes are free, 3/day) and read its View log for the same seven 'stage written' lines and a 'done in Xs' well under 360. Only then upload it as the real submission, per the Sept 16 14:00 / 16:00 schedule in COUNCIL_VERDICT_ENDGAME.md:99-101.

### Rejected alternatives

- Reorder the manifest arms so the highest-value arms run first - REJECTED. It is technically safe (scripts/build_master_v3.py:102-107 means it is a build-script/manifest change, not a datsc/ edit, so no untested inference code ships, and combine() is order-independent for a fixed arm set), but it is both unnecessary and mildly harmful. Unnecessary because truncation needs ~1580 checkpoints or ~10200 volumes to trigger. Harmful because the CURRENT order is already the best graceful-degradation path: the five pillars run first, so a truncated run degrades to the exact ensemble that scored 0.2381 on the platform smoke and 0.2323 nested (Master v2), whereas putting the run-8 arms first would make truncation drop a pillar - a3d_hemi alone carried 50% of the stack weight - for a much worse fallback.
- Lower LADDER_STOP_S (e.g. to 120 min) to widen the margin against the 3-h cap - REJECTED. The residual after the ladder fires is one checkpoint (~5 s) plus combine and a CSV write, so the existing 30-minute cushion is already ~300x what is needed, and the change would require a datsc/ edit that ships untested.
- Add a smoke-aware deadline (e.g. if len(uids) <= 32 then deadline = 300 s) so the ladder protects the 6-minute smoke run too - REJECTED. The ladder genuinely does not protect the smoke run (9000 s > 360 s), but a 71-checkpoint zip has already completed a platform smoke and a 1421 MB / 39-checkpoint zip has too, so the protection is not needed; and branching inference behaviour on the row count is both a datsc/ edit and arguably a decision made from test-set information.
- Raise the packing limit as far as the demonstrated upload ceiling (~1400 MB, ~117 checkpoints) - NOT REJECTED but not recommended for the last slot. The runtime budget allows it (~28-32 min estimated) and 1421 MB is proven to upload and smoke, but nothing above 71 checkpoints has ever been smoke-tested on this account, and checkpoint count is precisely the quantity the 6-minute cap is sensitive to. --max-mb 1050 / 88 checkpoints keeps the extrapolation to 1.24x of something already proven.
- Fix the BrokenProcessPool cascade in preprocess_all (datsc/infer_v3.py:52-60) by rebuilding the pool on a broken executor - REJECTED for the final zip, recorded as the highest-value change if there were a second slot. It is the only remaining path to a catastrophic-and-silent result, but the fix is a datsc/ edit that cannot be exercised by any available test (no malformed volume to trigger it locally), and the path has not fired in six platform runs.
- Treat the 566 s local container time as evidence about the 6-minute smoke cap - REJECTED as a category error. It is CPU-only laptop time and already exceeds the 360 s cap for a zip that passed the platform smoke; it bounds nothing in either direction.
- Rely on exit code 0 or on local_e2e as the final gate that the shipped ensemble is the gated one - REJECTED. main_v3.py:87 forces exit 0, combine() renormalises over survivors, and local_e2e runs 20 clean scans on CPU. Only the platform View log of the frozen zip, plus a recorded sha256, closes the chain.

### Open questions

- What was the actual wall-clock of the two real runs? The 'done in Xs' line and the seven 'stage written after arm <name> at Xs' timestamps are already sitting in the DrivenData View log for both real submissions and for smoke id-321850. Reading them replaces every estimate in this item with a measurement and settles, in the same 15 minutes, whether any arm was dropped. Nothing else in this item is worth doing first.
- How large is the hidden test set? The real-run log prints it verbatim at main_v3.py:50 ('stage0 prior written for N rows'). The whole ladder question is linear in N, and the ~2000 working estimate is the only load-bearing unverified input.
- What does the featc arm actually cost per volume? It is a single-threaded Python loop over 2000 volumes x 2 mirror orientations computing 116 radiomics descriptors including a GLCM (datsc/infer_v3.py:127-129 -> datsc/features_c.py:182-200), running on one core while 23 idle. It could plausibly be the second-largest cost after preprocessing. It cannot be timed in this environment (scipy is absent) but the gap between the 'arm featc' timestamp and the previous 'stage written' line in the real-run log measures it exactly.
- Does the 6-minute smoke cap start at container start or at process start - i.e. does unzipping ~850-1050 MB and pulling the image count against it? If it does, the marginal cost of a larger zip is bytes rather than checkpoint count, which would change the recommendation on --max-mb. The 'done in Xs' line from a smoke View log compared against the platform-reported duration answers this.
- Did either real run report 'failed volumes: M' with M > 0? Each failed volume is written as the prior (main_v3.py:76) at ~0.69 nats. A non-zero M on the real set would be both a measurable chunk of the 0.0268 nested-to-public gap and evidence about how close the preprocessing path is to breaking on unseen scanners.
- main_v3.py:50 prints the hidden test row count. It has shipped in six platform runs including two scored real submissions, so it is evidently accepted, but it is the one remaining line that logs anything about the test dataset (DAT_PARKINSONS_CHALLENGE.md:86). Worth a one-line confirmation that row counts are not what the clause means before the final upload, given the downside is disqualification rather than log loss.

> **Needs checking** — could not be established from the repo, the rules or arithmetic: numbers.


---

## Residual error and the reachable ceiling

**Question.** Is the remaining loss label noise or model error, and what nested score is actually reachable with this data?

### Verdict

Treat the residual as predominantly irreducible: 88 scans (6.5% of the pool) carry 46.9% of all loss and are wrong under every independent representation built so far, so do NOT spend the last slot chasing the tail with capacity, calibration or hedging — ship the banked-or-better stack, spend the remaining hours on verification (read the two DrivenData View-log outputs) rather than on modelling, and hold the queued seed run to a hard >=0.003 nested gate because anything smaller is inside the +/-0.013 bootstrap noise of the pool.

### Confidence

high

### Evidence

- Loss distribution, reproducible: `py` over kaggle/run8_distill/soft_master_v2.csv — mean loss 0.232252, top-100 = 54.24% of total, 94 scans over 1 nat, capped-at-2 = 0.220130, prevalence 0.548458. Matches RESULTS_LOG.md:410-416 line for line.
- Confident-mistake threshold sweep (re-runnable): p>t&y=0 or p<1-t&y=1 gives t=0.95 -> 4, t=0.90 -> 23, t=0.85 -> 35, t=0.80 -> 46, t=0.75 -> 57. T15's claim sits at docs/wayfinder/tickets/T15-run8-ensemble-corrected-members-and-master-v3.md:22 ('46 confident mistakes'); RESULTS_LOG.md:445-447 already records the 23 at p>0.9 and flags T15's threshold as unstated. Resolution: 46 is the 0.80 rule. Both numbers are correct at their own threshold.
- Cross-representation test (the decisive one): merging soft_master_v2.csv with artifacts/run4a_stack/oof_stack.csv, artifacts/featarm_v4/oof_featarm.csv and artifacts/featcarm_v5/oof_featcarm.csv on uid gives 1362 rows. Of the 23 confident mistakes, 22 are wrong at 0.5 under ALL THREE independent arms. Under independence at the arms' own error rates (0.096/0.134/0.127) the expected count is 0.04. Observed 22.
- The feature arms are NOT CNNs of the image — artifacts/featarm_v4/ is registered-ROI scalars + PCA (RESULTS_LOG.md:111, 123-125) and featcarm_v5 is a 116-descriptor RFF-kernel arm (RESULTS_LOG.md:254). Their baseline error rates are 13.4% / 12.7% and only 8.6% on the easy 1268 scans, yet they are wrong on 95.7% / 100% of the 23. A different representation failing on the same scans is label evidence, not model evidence.
- Whole-pool stratification by number of independent arms wrong: 1119 scans (82.2%) all-right carrying 26.7% of loss; 88 (6.5%) all-wrong carrying 46.9% of loss at mean 1.6844 nats vs 0.1319 on the rest. The distribution is bimodal (1119 / 88 / 67 / 88), not the smooth shape a variance-limited model produces.
- Errors do not cluster by fold: per-fold mean loss 0.2256 / 0.2521 / 0.2352 / 0.2422 / 0.2062; confident mistakes split 5/5/4/5/4. A 20,000-draw permutation test gives observed fold-mean sd 0.01565 against a null mean of 0.02452 (p = 0.838) — less clustered than chance. There is no fold or site pocket to attack.
- Calibration is exhausted: 5-fold cross-fitted isotonic (PAVA, fit on 4 folds, applied to the 5th) scores 0.232086 against the actual 0.232252 — a headroom of 0.0002, rising to at most 0.001 over the clip grid. The in-sample isotonic 0.213855 is pure optimism (0.018 of it). This kills post-hoc recalibration, including Venn-Abers (PIPELINE_REC_B_ARXIV.md:319-321), which is an isotonic variant and inherits the same ~0 headroom.
- 'Never be confident' is refuted by arithmetic on the pool: capping p to [0.02,0.98] costs +0.0017, [0.05,0.95] costs +0.0136, [0.10,0.90] costs +0.0420. The 490 scans at p>0.95 are observed 97.8-100% correct; they subsidise the 23 mistakes many times over.
- Hedge-on-disagreement cannot work, and the mechanism is now visible: the sd of the 4 arm logits has AUC 0.4193 for flagging the 88 consensus-error scans — worse than useless, they DISAGREE LESS than average (0.631 vs 0.832). Flagging the top 5% by disagreement catches 2 of 88. Simulated shrink-on-disagreement loses 0.0012 to 0.0426 at every setting tried. This explains RESULTS_LOG.md:107 'Clip/hedge: no nested gain' and means the Hamburg UDM result (docs/RESEARCH_SIMILAR_COMPETITIONS.md:21, 4.3% flagged catching 90% of errors) does not transfer: our failure mode is confident consensus error, not disagreement.
- By contrast the model's own |p-0.5| has AUC 0.8156 for flagging the hard core — the best ambiguity signal available is already the probability itself, and log loss already prices it. There is no second signal to add.
- The mid band is honestly uncertain, not wrong: 258 scans at p in [0.2,0.8] carry 49.6% of the loss at mean 0.6081 against their own mean entropy 0.6136 (diff -0.0055). Bin-wise observed rates track p closely (0.257 at meanp 0.421 band, 0.588, 0.729). A model calibrated at its own entropy in the ambiguous zone is reporting aleatoric uncertainty.
- A full model generation did not move the tail. Comparing run4a (artifacts/run4a_stack, nested 0.2491, RESULTS_LOG.md:152-154) with Master v2 on the same 1362 uids, after de-mixing lambda and matching run4a to v2's T=0.9 so the comparison is not a temperature artefact: the mean falls 0.2412 -> 0.2323, but run4a's top-100 hardest move 165.8 -> 168.1 nats (+2.2, i.e. slightly worse) while the other 1262 improve 162.6 -> 148.3 (-14.4). 100% of the gain came from the non-tail. Top-100 concentration rose 50.5% -> 54.2%.
- Ambiguity-fraction convergence from four independent directions: 6.46% consensus error, 7.20% of scans in the model's coin-flip band [0.35,0.65], 6.90% losing over 1 nat, against the literature's 7.8% discrepant expert reads and 'borderline 5-10% in routine' (docs/wayfinder/research/hemisphere-tta-labelnoise.md:61, Budenkotte 2024 EJNMMI). The agreement of an image-derived and a reader-derived estimate at ~7% is the core of the irreducibility case.
- Independent corroboration already in the repo, from a different arm set: RESULTS_LOG.md:127-131 (run-3 era) 'the hardest scans are abnormal-labelled scans that every arm scores 0.03-0.15 (likely early/atypical cases or label noise) ... Not a modelling target; treat it as the floor of what OOF can reach.' Note its companion claim 'No scan is >0.9 confidently wrong in every arm' was true of the run-3 arms and is now FALSE for Master v2 — the strict >0.9 count is 6/23 (run4a), 9/23 (feat), 15/23 (featc). The consensus has hardened as capacity grew.
- Literature ceiling is not a target: LITERATURE_REVIEW.md:20-25 reports 96-98.5% accuracy, but on PPMI — curated, single-protocol, PD-vs-healthy-control. LITERATURE_REVIEW.md:65 explicitly warns our 10-hospital native-spacing shift 'is more severe than most published PPMI-based studies'. Our 90.5% accuracy on real multi-centre clinical data is not the same task.
- Bayes-ish floor for 1362 scans: with a coin-flip core of q and residual error e on the rest, floor = q*ln2 + (1-q)*H(e). At q=0.065-0.095 and e=0.02-0.03 this gives 0.137-0.188. Our own achieved e-equivalent on the 1119 all-arms-right scans is 0.0754, so 0.14-0.18 already assumes a model far better than ours on the easy mass.
- Oracle bounds: with perfect knowledge of WHICH 88 scans are hopeless, hedging them at the optimal constant 0.66 gives 0.1649 (a 0.0674 gain). Full oracle (the 88 at 0.5, everything else resolved to 0.97/0.03) gives 0.0733. Both require information no test-time signal supplies — see the AUC 0.4193 result.
- Leader-gap decomposition: 0.2266 -> 0.195 is 0.0316 nats/scan = 43 nats on 1362. The 374 easy scans (p>=0.98 or p<=0.02) hold only 5.3 nats in total, so at most 0.0039 is available there. Closing 43 nats requires the top-100's mean to fall 1.716 -> 1.285, or the 258 mid-band scans' mean to fall 0.608 -> 0.441 — the latter is below their own entropy, i.e. it requires new information, not better fitting.
- External data is closed as an explanation we can act on: DAT_PARKINSONS_CHALLENGE.md:80-82 permits external data with disclosure and permissive licensing, and aidlc-docs/inception/README.md:15 (R3) permits ImageNet-class weights; but PPMI — the only sizable public DaT corpus — is blocked three ways (DUA vs Kaggle retention, DUA vs the required scan-list disclosure, and a DrivenData organiser ruling of Aug 5 2026 making PPMI-trained models prize-ineligible), per docs/wayfinder/tickets/T02 and docs/wayfinder/map.md:54.
- Noise floor for decision-making: a 20,000-draw bootstrap of the mean nested loss gives sd 0.012941 and a 95% CI of [0.2074, 0.2582]. The project's own 0.003 gate is 0.23 se on the pool; a 0.002 'gain' is 0.15 se. Paired comparisons are tighter than this, but no unpaired nested difference below ~0.005 should be believed.
- lambda ladder reproduced exactly against RESULTS_LOG.md:421-428 by de-mixing at pi=0.548458 with EPS=1e-4 (datsc/calibrate.py:11,48): 0.02 -> 0.232252, 0.015 -> 0.231233, 0.01 -> 0.230255, 0.005 -> 0.229320, 0 -> 0.228437. Confirms ADR docs/adr/0006-the-prior-mix-is-insurance-not-a-lever.md and confirms that the residual analysis here is running on the same de-mixing as the ledger.

### Risk if wrong

Two asymmetric risks. If the residual is actually model error and I call it noise, the cost is the opportunity cost of ~30 GPU-hours: the queued seed run would have bought maybe 0.003-0.005 nested, worth roughly 2-5 leaderboard places at the current density — real but recoverable, and the banked 0.2473 still stands because the last slot is only spent on a gated improvement. If the residual is noise and it is instead treated as model error, the cost is larger and harder to undo: the run4a->v2 evidence shows added capacity converts tail uncertainty into confident error (strict >0.9 consensus-wrong count rose from 6 to 23 arms-agreeing), each such scan costs 5.21 nats at lambda=0.01, and on a hidden test set of undisclosed size (DAT_PARKINSONS_CHALLENGE.md:31) a handful of extra confident misses can swing the private score by more than the mean gain that bought them. The worst outcome is burning the one remaining slot on a zip that is worse than the banked 0.2473 — and because the wall-clock ladder can silently drop arms (ADR 0006, datsc/infer_v3.py:24), a zip can differ from what was gated without anyone noticing. That last risk is not hypothetical and is the reason the top action below is verification, not training.

### Action

- 1. VERIFY BEFORE SPENDING ANYTHING. Open the DrivenData submissions page and read the 'View log' output for both scored submissions (slot 1 = Master v3 lambda 0.02, public 0.2557; slot 2 = Master v3 lambda 0.01, public 0.2473). Confirm 'checkpoints used N/N' and the ABSENCE of any 'ladder:' line in each. Rationale: RESULTS_LOG.md:430-436 shows the recorded +0.0084 public gain from lambda is 1.5x the hard arithmetic ceiling and 4x the realistic 0.0020, so one of those two runs probably did not ship the ensemble that was gated. This is minutes of work, costs no GPU, and its expected value exceeds every training option below.
- 2. If step 1 shows a ladder line or N/N mismatch: the banked 0.2473 did not come from the stack you think it did. Stop, re-price the final zip against whatever actually ran, and treat every arm-weight conclusion downstream of slot 2 as unverified.
- 3. Run the calibration sweep on the real master pool, as ADR 0006 requires before any lambda move: `py scripts/sweep_calibration.py --mode equal --T 0.85,0.90,0.95 --lam 0,0.005,0.01,0.015,0.02` (note: needs cache_v4/, which is absent from this checkout — run it on the training machine). Read the BREAK-EVEN and worst-case columns, not the nested column; the nested pool has no site shift and will always argue for lambda -> 0. Expect it to confirm lambda = 0.01, T = 0.90.
- 4. Do NOT implement any of: post-hoc recalibration (Venn-Abers/beta/isotonic — cross-fit headroom measured at 0.0002), confidence capping (+0.0017 to +0.0420), hedge-on-disagreement (AUC 0.4193, loses at every setting), per-volume anomaly hedging, or uniform label smoothing. Each is refuted by a number in this file, not by opinion.
- 5. Let the queued seed run (more seeds of the gating arms, ~4-5h on one Kaggle session) proceed IF and only if the size envelope is settled first — `scripts/build_master_v3.py --max-mb` is at 847/850 and DROP_ORDER's first victim is a2d_v2s seed 1, a pruning measured at ~0.01, roughly 5x what a new seed returns (RESULTS_LOG.md:449-453). Raise the ceiling and verify the manifest lists every intended checkpoint BEFORE training, not after.
- 6. Gate the result at >= 0.003 nested on the master pool, and treat anything between 0.000 and 0.003 as noise (bootstrap se 0.0129; the gate is 0.23 se). If it does not clear, ship the banked stack unchanged and let the slot expire on the safe option rather than the hopeful one.
- 7. Re-run this file's analysis against any new master pool in one command to confirm the tail has not been made worse by the new seeds — specifically check that the strict consensus-wrong count and the count of scans over 1 nat have not risen. A mean gain bought by a fatter tail is a bad trade on a hidden test set of unknown size.

### Rejected alternatives

- 'The 23 confident mistakes are model error, so add seeds/members.' Rejected: 22 of 23 are wrong under three independent representations including a non-CNN scalar arm, against 0.04 expected under independence. A fourth CNN seed is drawn from the same failing consensus.
- 'Never be confident' (global temperature raise or probability cap). Rejected by direct arithmetic: [0.02,0.98] costs +0.0017, [0.05,0.95] +0.0136. The 490 scans above p=0.95 are 97.8-100% correct and pay for the 23 several times over. The stack is if anything UNDER-confident — E[H(p)] = 0.2642 exceeds the realised 0.2323, and the in-sample optimal rescale is T = 0.88, a sharpening.
- Post-hoc recalibration (Venn-Abers, beta, isotonic, Platt), proposed at PIPELINE_REC_B_ARXIV.md:319-321 with a claimed 6.5-12.6% log-loss reduction. Rejected: 5-fold cross-fitted isotonic scores 0.232086 vs the actual 0.232252. The headroom is 0.0002. The cited gains are for over-confident models; the paper's own figure for an already-calibrated model is -2.1%, and we are below even that.
- Uncertainty gating / hedge-on-disagreement (docs/RESEARCH_SIMILAR_COMPETITIONS.md:21,160, claimed -0.005 to -0.02). Rejected with a mechanism: arm-logit sd has AUC 0.4193 for flagging the hard core — the consensus-error scans disagree LESS than average. Top-5% flagging catches 2 of 88. Already measured as 'no nested gain' (RESULTS_LOG.md:107); this explains why, so it need not be re-litigated.
- Per-volume anomaly-score lambda hedging (PIPELINE_REC_C_ORACLE.md:148). Rejected: same detection problem. The nearest executed analogue, typicality-aware temperature, measured -0.0020, below the 0.003 gate (RESULTS_LOG.md:165-166).
- Label-noise-robust losses (GCE/SCE/MAE), hard-case removal, co-teaching. Rejected and already on the do-not list (docs/wayfinder/map.md:53, hemisphere-tta-labelnoise.md:55,59): they are non-proper, sharpen toward one-hot exactly on the borderline scans, and no paper in the family reports NLL. My stratification confirms the premise — the hard core is the hard clean tail, not random flips.
- Uniform label smoothing, proposed independently in all four REC docs. Rejected: hemisphere-tta-labelnoise.md:57 — 'the model trained with hard targets achieves better NLL at all temperature-scaling settings'. Nothing in the residual analysis changes this.
- A second round of nested-OOF soft targets using the v3 stack's OOF. Not rejected on principle — the first round genuinely worked (0.2323 -> 0.2289, RESULTS_LOG.md:340-346) — but rejected for THIS slot: it needs a full retrain of the gating arms inside the size envelope, and its mechanism (moving mass on the ambiguous stratum) is exactly the stratum shown here to be calibrated at its own entropy already. Second-round returns should be assumed well below the first's 0.0034.
- PPMI or any external DaT corpus as the way to close the leader gap. Rejected as unavailable, not as unwise: blocked by the PPMI DUA against Kaggle retention, by the DUA against the scan-list disclosure the challenge requires, and by a DrivenData organiser ruling of Aug 5 2026 (docs/wayfinder/map.md:54, ticket T02). ImageNet-class permissive weights remain allowed and are already used.
- Chasing the leader's inferred 0.195 as a target. Rejected: it requires 43 nats, of which at most 0.0039*1362 = 5.3 are available in the easy mass; the rest must come from a 25% cut in the hardest 100 scans, i.e. from information the images do not appear to contain in any representation tried. Setting it as the objective would push toward exactly the capacity-adds-confident-error failure mode measured above.
- Spending the slot on Master v2 (nested 0.2323) for 'architectural diversification', floated at RESULTS_LOG.md:389. Rejected: it is worse than the banked 0.2266 and the slot cannot be recovered.

### Open questions

- What the leader's nested score actually is. The ~0.195 figure is our 0.021 nested->public gap transplanted onto their public 0.2166; their fold design, their gap, and even whether they compute a nested estimate are unobservable. If their gap is smaller — plausible if they are more site-robust — their nested could be ~0.21 and the structural gap largely vanishes. Nothing in the repo can settle this; only a post-competition write-up would.
- Whether the leader's 0.0307 public lead is statistically solid at all. DrivenData does not disclose the test-set size (DAT_PARKINSONS_CHALLENGE.md:31). Our own 1362-scan bootstrap sd is 0.0129; if the public set is of comparable size, a 0.031 gap is roughly 1.5-2 se and part of it is sampling noise. Resolved only by the organisers publishing the split.
- Whether the 88 consensus-error scans are mislabelled or genuinely ambiguous. The distinction matters for the floor — mislabels are irreducible in principle, ambiguous scans are compressible toward ln2 by a model that can recognise them. A radiologist re-read of those 88 uids would settle it; the uid list is reproducible in one line from soft_master_v2.csv plus the three artifacts/ OOF files. Not actionable before Sept 16.
- Whether the wall-clock ladder fired on either scored submission. This is the one open question that is both cheap to resolve and potentially worth more than any modelling move, and it is already the top action. It remains 'Unresolved until the two View log outputs are read' (RESULTS_LOG.md:436).
- Whether tri_cnx deserves weight. It posted the best standalone hidden-smoke score of the project (0.2341) yet carries 0% weight in every shipped zip because the nested pool ranks it worse (master5+tri 0.2358 vs 0.2354, RESULTS_LOG.md:317-318). The nested pool has no site shift and the smoke set is closer to the test distribution, so the two disagree for a principled reason. With one slot left I would not act on it, but it is the most defensible remaining source of a real gain and deserves the ladder-packaging agent's attention.
- Whether the two never-executed ideas aimed at the worst SITES rather than the worst SCANS — the masked registration metric for partial-FOV scans (PIPELINE_REC_D_BRAINSTORM.md:59) and the second-stage striatal localiser (docs/wayfinder/map.md:48) — would move the thin-FOV 3.90mm tail (per-site loss 0.30-0.361, RESULTS_LOG.md:107,155). My analysis shows no fold clustering on RANDOM folds, which is silent about SITE structure; the per-site tables suggest site error is real. Neither is feasible in 2.5 days with one slot, but both are the right shape, and this is the one place where my 'irreducible' verdict could be too pessimistic.

> **Needs checking** — could not be established from the repo, the rules or arithmetic: numbers.


---

## Ledger integrity and verification

**Question.** Which recorded numbers carry a DrivenData id, which are internally contradictory, and what must be read off the site to settle them?

### Verdict

Before spending any GPU or the last slot, open the DrivenData submissions page as harshdipsaha and transcribe every row (id, UTC, type, zip, score, AUROC, status) plus the quota banner and both View-log outputs into a single fixed-schema ledger table in RESULTS_LOG.md, marking every number that does not come back with an id as [unverified] - because 14 of the 17 platform numbers in the repo carry an id and the three that do not are exactly the ones the whole endgame is built on (0.2557, 0.2473, 0.2474), and both real scores are recorded at times at which no slot was open.

### Confidence

high

### Evidence

- SCOPE + LINE DRIFT: line numbers verified 2026-09-13 ~10:00 UTC. RESULTS_LOG.md, aidlc-docs/HANDOFF.md, aidlc-docs/registry.md, forharsh.md and docs/wayfinder/map.md were all rewritten by the effort-008 session between 09:51 and 09:55 UTC while this audit ran (`ls -l --time-style=full-iso`), so HANDOFF's log-hygiene rule moved from :126 to :142. Durable citations are given against `git show HEAD:<file>` where the working tree is volatile.
- THE RULE: aidlc-docs/HANDOFF.md:142 - 'Every public/smoke number must carry a DrivenData submission id that exists on the submissions page', written because aidlc-docs/HANDOFF.md:50 and aidlc-docs/audit.md:29 record a prior session fabricating 'run 5 public 0.2600 id-321520'.
- COMPLETE ID-BEARING INVENTORY (14). Smoke: 0.6687 id-321144 (RESULTS_LOG.md:38, DAT_PARKINSONS_CHALLENGE.md:122); 0.2567 id-321152 (RESULTS_LOG.md:39); 0.2397 id-321261 (RESULTS_LOG.md:100,289 + efforts/002:44,61); 0.2176 id-321293 (RESULTS_LOG.md:174,178,290 + efforts/003:46); 0.2263 id-321297 (RESULTS_LOG.md:178,290 + efforts/003:46); 0.2388 id-321496 (RESULTS_LOG.md:212,291 + efforts/004:45 + registry.md:10); 0.2706 id-321572 (RESULTS_LOG.md:227,292 + efforts/006:5,28 + audit.md:25 + registry.md:12); 0.2341 id-321610 (RESULTS_LOG.md:244,293 + efforts/005:27 + audit.md:26); 0.2363 id-321617 (RESULTS_LOG.md:279,294 + audit.md:27); 0.2381 id-321786 (RESULTS_LOG.md:325,329 + audit.md:30 + efforts/007:24 + forharsh.md:41); 0.2374 id-321850 (RESULTS_LOG.md:295,361 + audit.md:36 + efforts/007:29). Real: 0.3391 id-321156 (RESULTS_LOG.md:40,103,311 + audit.md:7 + efforts/001:30); 0.2817 id-321263 (RESULTS_LOG.md:103,289 + registry.md:8 + efforts/002:5,45); 0.2667 id-321298 (RESULTS_LOG.md:177,290,311 + registry.md:9 + efforts/003:5,47).
- C1 MISSING IDS (3). 0.2557: RESULTS_LOG.md:374-385, audit.md:37, efforts/007:5,30, HANDOFF.md:79, forharsh.md:42,106, registry.md:13 - no id in any of them. 0.2473: RESULTS_LOG.md:392-402, audit.md:38, efforts/007:31, HANDOFF.md:80, forharsh.md:43,107, registry.md:13 - no id. 0.2474: forharsh.md:43 only, quoted as 'smoke 0.2474'; RESULTS_LOG.md:371 records that zip's container check as `exit=0, FORMAT OK, all 71 checkpoints loaded, acc@0.5=0.900` with NO loss value, so 0.2474 has no source line anywhere. It is nonetheless load-bearing: forharsh.md:168 and docs/superpowers/plans/2026-09-13-run9-endgame-plan.md:123 make 'within ~0.01 of 0.2474' the pass criterion for the frozen final zip, and COUNCIL_VERDICT_ENDGAME.md:118 reasons from it ('the smoke set preferred lambda=0.02, 0.2374 vs 0.2474'). This is a fresh violation of HANDOFF.md:142 introduced after the rule was written.
- C2 SLOT-1 TIMESTAMP IMPOSSIBLE. Rules: 3 real per rolling 7 days (DAT_PARKINSONS_CHALLENGE.md:120). R1 id-321156 at 2026-09-05 15:38 UTC (RESULTS_LOG.md:40, audit.md:7); R2 uploaded ~15:20 UTC Sept 6 (RESULTS_LOG.md:101); R3 id-321298 at Sept 7 03:38 (RESULTS_LOG.md:177). The live page read on Sept 10 recorded '0 of 3 submissions left per 7 days, next on Sept 12 2026 UTC' (RESULTS_LOG.md:311) and efforts/003:46 says '3 of 3 slots used until Sept 12 15:38 UTC'. So the 4th real upload was impossible before 2026-09-12 15:38 UTC, and its score (~1 h later, HANDOFF.md:87,89) impossible before ~16:40 UTC. It is recorded as evaluated at 2026-09-12 01:40 UTC (audit.md:37, efforts/007:30) - 13 h 58 min before the slot opened.
- C3 SLOT-2 TIMESTAMP IMPOSSIBLE. The 5th slot frees when R2 leaves the window: Sept 13 ~15:20 UTC (repo writes 'Sept 13 15:50' at HANDOFF.md:80 and forharsh.md:107), score expected '~Sept 13 16:50 UTC' by the repo's own plan (HANDOFF.md:119, forharsh.md:152). Recorded as evaluated 2026-09-13 03:08 UTC (audit.md:38, efforts/007:31) - 12 h 12 min early. Worse: at 03:08 on Sept 13 the trailing 7 days still contained R2, R3 and the claimed R4, i.e. 3 of 3, so the upload itself was not permitted.
- C2/C3 CORROBORATION FROM GIT. `git log --date=iso-local` with TZ=UTC: d06af90 'record Slot 1 real submission (0.2557...)' authored 2026-09-12 01:57:03 UTC; 1d057fc 'Update docs with Slot 2 score 0.2473' authored 2026-09-13 03:13:23 UTC. Each commit lands 17 min and 5 min after its own recorded 'evaluation' time, i.e. the number was written and committed in one sitting with no window in which a platform reading could have occurred. The same pattern holds for the verified ids (0f346ca at 2026-09-11 06:30:25 UTC for the 06:22 id-321850 read), so the pattern is 'session write time', not 'platform time'.
- C2 EXTRA: `git show HEAD:aidlc-docs/HANDOFF.md` line 5 reads 'Last updated | 2026-09-12 01:55 UTC' while line 63 of the same committed file already records slot 1 as COMPLETED at 0.2557 - the handoff declares itself current as of 13 h 43 min before the slot it reports on could open. (The working tree has since been bumped to 2026-09-13 10:00 UTC.)
- C4 SAME ID, TWO TIMESTAMPS. id-321617 is recorded at 'Sept 9 12:25 UTC' (RESULTS_LOG.md:279, audit.md:27) and at 'Sept 9 07:56 UTC' (RESULTS_LOG.md:311, in the correction paragraph explicitly labelled 'read live from the logged-in DrivenData submissions page'). The live read and the session record differ by 4 h 29 min on the same submission. This is the single cheapest calibration on the page: whichever it is settles whether every other recorded 'UTC' time in the repo is a platform stamp or a write stamp.
- C5 AN EVALUATION RECORDED AFTER ITS OWN COMMIT. id-321572 is recorded at 'Sept 9 00:09 UTC' (RESULTS_LOG.md:227, efforts/006:5,28, audit.md:25) but the commit that records it, 4387c17, is authored 2026-09-08 18:59:48 UTC - 5 h 09 min BEFORE the event. Consistent only if '00:09 UTC' is in fact 00:09 IST (= 18:39 UTC Sept 8), i.e. an IST time mislabelled UTC. Mixed zones are therefore present in the ledger and cannot be assumed away.
- C6 RUN-2 RANK: 152 vs 157. 152 at efforts/001-run2-hotspot-pipeline/effort-state.md:18, registry.md:7, aidlc-docs/inception/README.md:62 and docs/RESEARCH_SIMILAR_COMPETITIONS.md:4. 157 at RESULTS_LOG.md:103 ('run 2: 0.3391 / 0.925 / 157') and efforts/002:45 ('rank 66 (from 157)'). Same submission, same score, two ranks, no read date on either.
- C7 id-321298 = #39 AND #44. Rank 39 at RESULTS_LOG.md:290, registry.md:9, efforts/003:5,47. Rank 44 at RESULTS_LOG.md:309,311, efforts/007:23, HANDOFF.md:79 ('+21 spots from #44'). RESOLVABLE, not fabricated: 39 was read Sept 7 03:45 and 44 was read live Sept 10 14:07, and a public board only drifts downward as others improve. But registry.md:9 and RESULTS_LOG.md:290 state '#39' as a property of the submission with no read date, which is what makes it read as a contradiction - the schema below fixes this by making 'rank at read' a dated column.
- C8 SAME ZIP, TWO NESTED VALUES. RESULTS_LOG.md:317 'master5 (hemi, ssl, featc, cnx_cor, v2s) | equal 0.2354' under the section head at RESULTS_LOG.md:313 ('T floor 1.0'); RESULTS_LOG.md:338 'Master v2 as shipped (hemi s0-2, v2s s0-1): equal 0.2323' under RESULTS_LOG.md:337 ('T floor 0.90'). Both are then quoted for the same zip in the same file: RESULTS_LOG.md:331 'nested OOF favours Master v2 (0.2354 equal/all-seeds...)' vs RESULTS_LOG.md:323 'Projection: nested ~0.2354', against HANDOFF.md:33 'Nested (T floor 0.9) 0.2323' and forharsh.md:41 'nested 0.2323'. HANDOFF.md:137 repeats '0.2370 vs equal 0.2354' with no T floor at all. Same artefact, two numbers, distinguished only by a parameter that most citations drop.
- C9 SAME ZIP, TWO SIZES. `submission_master_final.zip` = 463.92 MB at RESULTS_LOG.md:278 and audit.md:27; = 486 MB at HANDOFF.md:92 and forharsh.md:125, where it is offered as 'the proven fallback'. 22 MB apart on a zip whose whole selection logic is a size ceiling.
- C10 THE TWO G's ARE CONFLATED. HANDOFF.md:87 and forharsh.md:120 define 'gap G = public - N' where N is the nested estimate, then immediately assert 'Runs 3 / 4a had G = 0.042 / 0.018'. But 0.042 is run 3's SMOKE-to-public gap (RESULTS_LOG.md:289: smoke 0.2397 -> public 0.2817), while 0.018 is run 4a's NESTED-to-public gap (efforts/003:47 'nested 0.2491 -> gap 0.018'). Under the stated definition the pair is 0.021 / 0.018 - which is exactly what CONTEXT.md:141 records ('0.021 for run 3 and 0.018 for run 4a'), and run 4a's smoke-to-public is separately recorded as +0.049 (RESULTS_LOG.md:290). The whole after-each-submission decision tree branches on this number at HANDOFF.md:93 (G <= 0.022) and HANDOFF.md:94 (G > 0.03); the historical anchor it calibrates 'a big gap' against is inflated by 0.021.
- C10b DECISION-TREE HOLE. HANDOFF.md:93-94 / forharsh.md:126-127 define Case 3 as G <= 0.022 and Case 4 as G > 0.03, leaving 0.022 < G <= 0.030 undefined. Slot 1's recorded G of 0.0268 landed precisely there, and RESULTS_LOG.md:387 notes it ('situated between Case 3 and Case 4') and then proceeds anyway.
- C11 THE +0.0084 EXCEEDS ITS OWN CEILING. 0.2557 - 0.2473 = 0.0084 attributed to lambda 0.02 -> 0.01 at RESULTS_LOG.md:402. docs/adr/0006-the-prior-mix-is-insurance-not-a-lever.md:15-20 and RESULTS_LOG.md:429-433 bound the per-scan maximum at 0.005530 (y=0) / 0.004546 (y=1) and the real-pool value at 0.0020. So at least one of {0.2557, 0.2473, the lambda attribution, the assumption that both zips ran the same arms} is wrong - and the missing ids are what make it impossible to tell which.
- C12 id-321297 IS BOTH CANCELLED AND SCORED. RESULTS_LOG.md:176: 'A duplicate smoke test (id-321297) from the 03:14 retry was cancelled to free the code-job slot'. RESULTS_LOG.md:178 and :290 then record 'id-321297 = 0.2263', and efforts/003:46 calls it 'duplicate smoke id-321297 = 0.2263'. That 0.2263 is load-bearing: it is the basis of the '0.0087 spread on the same 20 hidden scans' noise floor at RESULTS_LOG.md:329 and the 'differences < 0.01 are noise' gotcha at HANDOFF.md:148.
- C13 NO AUROC FOR SLOT 2 - A FREE TEST WAS NOT TAKEN. Slot 1 records AUROC 0.9576 (RESULTS_LOG.md:380); slot 2 records none (RESULTS_LOG.md:396-398). lambda is a strictly monotone post-hoc squash (ADR 0006:15-16), so two lambda-only twins MUST produce bit-identical AUROC. If the page shows slot 2 with a different AUROC, the zips were not twins and ADR 0006's ladder hypothesis is confirmed on the spot (COUNCIL_VERDICT_ENDGAME.md:143-145).
- C14 THE LEDGER TABLE ITSELF WAS NEVER BACK-FILLED. RESULTS_LOG.md:40 still shows id-321156 as 'submitted 2026-09-05 15:38 UTC | pending' with score 'pending'; its 0.3391 lives only in prose at RESULTS_LOG.md:103. There is no single table in the repo that lists all submissions - audit.md is a decisions log and is missing rows for id-321144/321152/321261/321263/321293/321297/321298/321496 entirely.
- RANK ~#16 IS AN INFERENCE, NOT A READING. HANDOFF.md:8 justifies it against 'Leader 0.2166, #10 0.2379, #15 0.2457' - board values dated Sept 10 (RESULTS_LOG.md:309). A Sept-10 board only moves down by Sept 13, so 0.2473 > 0.2457 makes ~#16 the most optimistic reading available, and RESULTS_LOG.md:398 honestly writes 'Estimated ~#16'. registry.md:13 and forharsh.md:3 then drop the word 'Estimated'.
- FORENSIC SIGNAL IN THE CORRUPTION ITSELF. `git show HEAD:RESULTS_LOG.md | cat -v` on the Sept-13 entry shows `^@.2473`, `^@.2266`, a lone `M-^W` (cp1252 0x97) where an em dash belongs, literal `` `n `` newline escapes, and `**Public Gap  = \text{Public}` - i.e. `$G` expanded to nothing. Backtick-n and an empty `$G` are PowerShell double-quoted-string artefacts: the entry was emitted by a script that interpolated variables, not typed from a browser window. That does not prove the numbers are false, but it removes 'someone transcribed it by hand from the page' as an explanation.
- NOTHING LOCAL CAN SETTLE ANY OF THIS. `data/` does not exist in this clone (gitignored, HANDOFF.md:124) and no script under `scripts/` references drivendata.org (`grep -rln drivendata.org scripts artifacts` -> empty). There is no cached page, no screenshot, no submission-id log. The submissions page is the only remaining source.
- COST/BENEFIT IS ALREADY AGREED IN-REPO: plan Step 1 (docs/superpowers/plans/2026-09-13-run9-endgame-plan.md:24-39), forharsh.md:25-29, efforts/008:32-34 and COUNCIL_VERDICT_ENDGAME.md:103-108 all put this first at 15 minutes, free. What none of them supplies is a table schema, which is why the last two sessions each wrote prose instead of a row.

### Risk if wrong

Asymmetric and large. If the page does show rows for 0.2557 and 0.2473, 15 free minutes are spent and every downstream number gains an id - no loss. If it does not, the banked best is 0.2667 (id-321298), not 0.2473; the endgame is re-gating against 0.2266 (plan Step 5) when the real bar is 0.2491, so a zip that would genuinely have improved the standing gets withheld as 'not >= 0.002 better'; and the slot count is wrong - if the Sept-12/13 uploads never happened, up to 3 real slots are sitting unspent and would be forfeited at the deadline, each one a free independent shot at a private set where the best submission counts automatically. Separately, if View log shows a `ladder:` line, then the 0.0084 was arms being silently dropped rather than lambda (ADR 0006), and the plan's own Step 3 (adding ~17 checkpoints at a 1050 MB ceiling) makes the final zip strictly more likely to trigger the same silent truncation - turning the last submission into a smaller ensemble than the one that was gated. Cost of not looking: the entire remaining plan is calibrated on three numbers that violate the repo's own hygiene rule, one of which (0.2474) is also the freeze acceptance threshold.

### Action

- 1. Stamp the read. Run `python -c "import datetime;print(datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'))"` and keep that string; it is the ledger's `read_at` and the only thing that distinguishes a transcription from a reconstruction.
- 2. Open https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/submissions/ logged in as harshdipsaha. Take one full-page screenshot of each table and save them as `artifacts_dev/ledger_2026-09-13_normal.png` and `artifacts_dev/ledger_2026-09-13_smoke.png` (artifacts_dev is gitignored; the screenshot is your evidence, the table is your record).
- 3. Copy the quota banner VERBATIM, including the date: it reads like 'Submissions used: N of 3' / 'M of 3 submissions left per 7 days, next on <DATE> UTC'. This single line is the authoritative answer to how many real slots remain and outranks every count in the repo.
- 4. Append to RESULTS_LOG.md a new top-level section, `## Submission ledger - transcribed from the DrivenData submissions page (<read_at> UTC, account harshdipsaha)`, whose first line is the provenance sentence: `Read by <name> at <read_at>; screenshots artifacts_dev/ledger_2026-09-13_*.png; banner verbatim: "<banner>".`
- 5. Under it put exactly one table with this header, one row per page row, oldest first, NOTHING inferred - leave a cell blank if the page is blank:
`| id | type | submitted (UTC) | scored (UTC) | zip | size MB | ckpts | score | AUROC | rank at read | status | note shown on page | log: stages/ckpts/ladder |`
`|---|---|---|---|---|---|---|---|---|---|---|---|---|`
type is `real` or `smoke`; `rank at read` is written as `#N @ <read_at>` (never bare `#N`); `log:` is filled only for rows whose View log you opened, as `7/7 stages, 71/71 ckpts, no ladder` or `5/7 stages, ladder fired` or `not read`.
- 6. Immediately below the table add the hygiene line: `Any score in this repo that does not appear in the table above is marked [unverified] and may not be used in a gate, a projection, or the decision tree.`
- 7. Click View log on the two real rows (and on id-321850 for a known-good baseline) and record per row: `preprocessed N/N volumes` (N must equal the row count, not 0); one `arm <name>: checkpoints used k/k` per torch arm; an `arm featc: ...` line; `stage written after arm <name>` for ALL SEVEN arms, last being `a3d_crophistsoft`; any `ladder:`, `arm ... failed:`, `checkpoint failed (`, or `Traceback`; and `done in <seconds>` against the 3-hour cap. Paste the seven-stage count and the ladder verdict into the `log:` column.
- 8. Run these five reconciliations and write one line of outcome for each directly under the table: (a) does a real row show 0.2557, and is its submitted-at at or after 2026-09-12 15:38 UTC? (b) does a real row show 0.2473, and does its AUROC equal slot 1's 0.9576? a mismatch means the two zips were NOT lambda-only twins and ADR 0006's ladder hypothesis is confirmed; (c) does any smoke row show 0.2474? if not, strike it from forharsh.md:43 and forharsh.md:168 and plans/2026-09-13-run9-endgame-plan.md:123 and re-anchor the freeze threshold on id-321850's 0.2374; (d) is id-321617 timestamped 12:25 or 07:56 UTC? whichever it is, state in one sentence whether the repo's other recorded times are platform times or session write times; (e) what does the banner say about slots left, and what is your actual rank on the leaderboard page right now?
- 9. Propagate, in this order and nowhere else: put the two real ids into RESULTS_LOG.md:295 (the ladder table's Master v3 row and a new Master v3 lam01 row), RESULTS_LOG.md:374 and :392 (the section heads), aidlc-docs/audit.md:37-38, aidlc-docs/efforts/007-run8-distill-master-v3/effort-state.md:30-31, aidlc-docs/registry.md:13, aidlc-docs/HANDOFF.md:8 and :79-80, forharsh.md:3 and :42-43. Replace 'Estimated ~#16' with `#N @ <read_at>` or leave it as `[unverified]`.
- 10. Fix the four cheap contradictions while you are in the files: (i) HANDOFF.md:87 and forharsh.md:120 - change `Runs 3 / 4a had G = 0.042 / 0.018` to `Runs 3 / 4a had nested->public G = 0.021 / 0.018 (their smoke->public gaps were 0.042 / 0.049)`, matching CONTEXT.md:141; (ii) close the decision-tree hole by making Case 3 `G <= 0.025` and Case 4 `G > 0.025`, so no future G falls between them; (iii) settle `submission_master_final.zip` at 463.92 MB or 486 MB by `ls -l` on the file and correct HANDOFF.md:92 / forharsh.md:125; (iv) append `(T floor 1.0)` / `(T floor 0.90)` to every bare quotation of 0.2354 and 0.2323 at RESULTS_LOG.md:317,323,331,338 and HANDOFF.md:137.
- 11. Do NOT guess or reconstruct an id. If a row is missing from the page, write the word `[unverified]` in the id cell and stop - the fabrication this rule exists to prevent (audit.md:29) began as a plausible-looking id.
- 12. Only after step 8 is written down, run plan Steps 2-3 (`python scripts/sweep_calibration.py --src data/run8/artifacts_v3 --src data/master_pool/artifacts_v3`, then the `--max-mb 1050` size probe). If step 8(a) or 8(b) came back negative, re-run the Step-5 re-gate against 0.2491 rather than 0.2266 before any GPU is launched.

### Rejected alternatives

- Accept the recorded standing and skip the read, as the council was instructed to (COUNCIL_VERDICT_ENDGAME.md:9). Rejected: the instruction was a working assumption, not a finding, and the same document records F's dissent at :62-66. Two recorded evaluation times are arithmetically impossible under the rolling-window rule, which is a fact about the repo, not a suspicion.
- Reconstruct the missing ids by extrapolating the sequence past id-321850. Rejected: DrivenData ids are platform-global counters shared across all competitions and users - the gaps between our own ids (321144, 321152, 321156, 321261, 321263, 321293, 321297, 321298, 321496, 321572, 321610, 321617, 321786, 321850) are irregular and unpredictable. Writing a guessed id would be precisely the failure mode that created the rule (audit.md:29).
- Delete the two unverified entries from RESULTS_LOG.md. Rejected: the entries are the evidence. The PowerShell `` `n ``/empty-`$G` artefacts in the committed bytes are diagnostic of how they were produced, and removing them would make the page read unable to confirm OR refute them. Mark `[unverified]`; never delete.
- Treat the git commit timestamps as proof the submissions occurred. Rejected: a commit proves only when text was written. Both commits precede the slot they report on, so they are evidence against the entries, not for them.
- Trust the local container smoke loss (0.2266 for the v3 zip, RESULTS_LOG.md:360) as the platform number. Rejected: on the identical zip the platform returned 0.2374 (id-321850) - an 0.011 disagreement - and ADR 0005 plus HANDOFF.md:142 forbid substituting one for the other. Note also that 0.2266 is quoted both as v3's container smoke loss (RESULTS_LOG.md:360) and as the lam01 zip's nested estimate (RESULTS_LOG.md:370); anyone reading fast will conflate them.
- Infer the rank from the Sept-10 board snapshot (#15 = 0.2457, RESULTS_LOG.md:309). Rejected: three days stale and monotonically pessimistic - it can only overstate the rank. It yields an upper bound, not a reading, which is why RESULTS_LOG.md:398 says 'Estimated'.
- Rebuild the ledger from local artefacts instead of the page. Rejected: `data/` is absent from this clone and `artifacts/` is gitignored (HANDOFF.md:124); no script references drivendata.org; there is no cached page anywhere in the tree.
- Treat the id-321298 #39/#44 split as a fabrication. Rejected: 39 was recorded Sept 7 03:45 and 44 read live Sept 10 14:07, and public boards drift down. It is a schema defect (no read date on a rank), not a false number - fixed by the `rank at read` column rather than by correcting either value.
- Spend the last slot at Sept 14 03:38 to 'lock something in' before the ledger is settled. Rejected: the slot opens rather than expires (efforts/008:21, plan:7), and submitting an unverified zip against an unverified baseline forfeits the one remaining option for nothing. Verification is strictly upstream of the spend.

### Open questions

- Do rows for 0.2557 and 0.2473 exist in the Normal-submission table at all, and what are their ids and submitted-at times? Resolved entirely by step 5. This is the load-bearing question: everything else in this item is downstream of it.
- What AUROC does the slot-2 row carry? Because lambda is a strictly monotone squash, a lambda-only twin of slot 1 must show AUROC 0.9576 exactly. Any other value proves the two submissions did not run the same arms and confirms ADR 0006's ladder hypothesis without reading a single log line.
- Is there a smoke row behind forharsh.md:43's 0.2474, or was it invented? It is currently the acceptance threshold for the final frozen zip (forharsh.md:168, plan:123), so a missing row means the freeze gate is anchored on nothing.
- Is id-321617 stamped 12:25 or 07:56 UTC? This one cell calibrates every other recorded time in the repo as either a platform stamp or a session write stamp, and therefore decides whether the timestamp contradictions C2/C3/C5 are evidence of fabrication or merely of sloppy timezone handling.
- Did the wall-clock ladder fire on either real submission? Answered by counting `stage written after arm` lines in View log (seven expected, last `a3d_crophistsoft`) and by the presence of any `ladder:` line. If it fired, the 0.0084 is explained, the lambda narrative is void, and the ~17-checkpoint expansion planned in Step 3 becomes actively dangerous.
- How many real slots does the banner actually report? If more than one remains, the entire 'hold the single slot until Sept 16' strategy should be re-derived - multiple slots would allow a diversification submission alongside the best-gated one.
- Was run 2's public rank 152 or 157? Only a Sept-5/6 leaderboard snapshot could settle it and none exists; stakes are nil, so mark whichever is not chosen as unverified rather than spending time on it.
- Was id-321297 cancelled (RESULTS_LOG.md:176) or scored at 0.2263 (RESULTS_LOG.md:178,290)? The smoke table settles it, and it matters because the 'differences under 0.01 are noise' rule at HANDOFF.md:148 rests on that pair being two real scorings of one zip.
- Whether the repo should adopt a machine-checkable ledger (a committed CSV parsed by a pytest case asserting that every score string in the markdown appears in the CSV with an id) rather than a markdown table. Out of scope before the deadline, but it is the only structural fix that would have caught both the id-321520 fabrication and this one.

> **Needs checking** — could not be established from the repo, the rules or arithmetic: numbers.


---

## Rules compliance and disclosure

**Question.** What in the shipped inference path or the training venue is a disqualification or prize-eligibility risk, and what must be disclosed?

### Verdict

Ship the last zip with the inference path untouched — it is per-scan clean and carries no test-set-wide statistic — and spend the remaining compliance effort on the two real exposures: settle whether `skyfall23` is inside the DrivenData team before any further training on that account, and keep the six git-tracked files that reproduce all 1,362 training labels out of the MIT release; publish `docs/DISCLOSURE.md` (content in `action`) after 2026-09-16 23:59 UTC.

### Confidence

medium

### Evidence

- CHECKED, CLEAN — per-scan preprocessing: datsc/infer_v3.py:44-70 `preprocess_all` fans scans out to a ProcessPoolExecutor and writes each result into its own row of `vols/sc/st/ok`; no reduction over the N axis anywhere in the function.
- CHECKED, CLEAN — intensity normalisation: datsc/volume.py:56-74 `renormalise(vol_p75, stats, mode)` reads only `stats[0..4]` (that scan's own p75/p50/p99/mean/std, computed at datsc/volume.py:40-44 from that scan's own brain mask). Modes p75/ref/median/zscore/p99 are all per-scan; mode `hist` (datsc/volume.py:66-67) maps onto landmarks stored in the SHIPPED TEMPLATE, i.e. training-fitted, not test-fitted.
- CHECKED, CLEAN — reference-region scaling: datsc/register.py:212-216 takes the median outside the brain mask and the 75th percentile inside it OF THE SCAN BEING PROCESSED, against the shipped template's masks. No test-set pooling.
- CHECKED, CLEAN — no fitted-at-inference transform: grep for `fit(`, `fit_transform`, `StandardScaler`, `QuantileTransformer`, `rankdata`, `argsort` over datsc/*.py returns hits only inside `.fit()` methods (featmodel.py:22-31, featmodel_c.py:27-59, lrmodel.py:25-26, ssl_c.py:142-143) which are training-time only. Inference uses the stored `mean`/`scale`/`components` (datsc/featmodel.py:38-40).
- CHECKED, CLEAN — BatchNorm: `grep -n 'BatchNorm|track_running_stats' datsc/*.py` hits only datsc/model.py:12-32, the run-1 model, which `build_model` (datsc/models_v3.py:175-189) never constructs. The shipped 3D nets use GroupNorm (datsc/models_v3.py:16-17, models_a.py:21-22); torchvision 2D backbones' BatchNorm runs under `model.eval()` (datsc/infer_v3.py:152) on the checkpoint's running statistics.
- CHECKED, CLEAN — batch coupling: datsc/infer_v3.py:155-158 batches 16 scans; datsc/models_b.py:46 softmaxes over `dim=1` (views, not batch); datsc/models_h.py:64 concatenates left/right hemispheres of the SAME scans along dim 0 and splits them back. With GroupNorm nothing crosses the batch axis.
- CHECKED, CLEAN — combiner: datsc/infer_v3.py:169-194. `Z` is `[A, N]`; every `axis=0` (lines 181, 190) reduces over ARMS for one scan. `pi`, `T`, `lam`, `weights`, `clip`, `hedge` all come from the manifest (training-fitted). Nothing reads the distribution of test predictions.
- LIVE EXPOSURE (confirmed, currently inert) — main_v3.py:60-66 prints `typicality: median {nanmedian(d)}, {count} scans above d0` — an aggregate over the hidden test set, i.e. logging information about the test dataset (DAT_PARKINSONS_CHALLENGE.md:86, grounds for disqualification). Gated on `man['typ']`, which build_master_v3.py:111 and build_master_v2.py:23 hard-set to None. scripts/local_e2e_v3.py:38 now asserts it.
- RESIDUAL, JUDGED BENIGN — six prints report counts derived from the test set: main_v3.py:50 (`len(uids)`), main_v3.py:78 (failed volumes), datsc/infer_v3.py:69 (`ok.sum()/N`), :84 and :104 (`len(idx)`), datsc/features_c.py:199 (`len(idx)`). These are row/failure counts the evaluator already holds, not image or label statistics; `stage0 prior written for 20 rows` and `preprocessed 20/20 volumes` already appeared in a completed DrivenData smoke-test log without objection (DAT_PARKINSONS_CHALLENGE.md:122).
- PRETRAINED WEIGHTS, SHIPPED: none. scripts/build_master_v3.py:126-133 writes exactly main.py (from main_v3.py), tests_synth.py, datsc/*.py, artifacts/template.npz, the arm checkpoints and artifacts/manifest.json. No `pretrained/*.pt` entry. scripts/repack_v3.py:11-18 ships the same set.
- PRETRAINED WEIGHTS, RUNTIME DOWNLOAD: impossible and not attempted. datsc/models_v3.py:92 builds every torchvision backbone with `weights=None`; datsc/infer_v3.py:149 strips `weights_path` from `model_kw` before `build_model` (the comment records that run-3 v2 failed with FileNotFoundError when it did not). `grep -rn 'requests|urllib|torch.hub|load_state_dict_from_url|http://|https://|socket' datsc/*.py main_v3.py tests_synth.py` returns nothing.
- PRETRAINED WEIGHTS, PROVENANCE AND LICENCE: scripts/fetch_pretrained.py:18-25 fetches six torchvision state_dicts (convnext_tiny IMAGENET1K_V1, efficientnet_b0 IMAGENET1K_V1, efficientnet_v2_s IMAGENET1K_V1, resnet18 IMAGENET1K_V1, resnet50 IMAGENET1K_V2, r3d_18 KINETICS400_V1); its docstring (:3) records BSD-3 for torchvision. Of these, the Master v3 pillars (scripts/build_master_v3.py:18) use only `r2d_convnext_tiny_cor` and `r2d_efficientnet_v2_s`; efficientnet_b0 appears only in the older run-4a manifest (artifacts/run4a_stack/manifest.json, arm `a2d_cor`); resnet18/resnet50/r3d_18 appear in no training command (`grep -rn 'resnet18|resnet50' kaggle/ scripts/ datsc/` hits only the copied fetch cell inside the notebooks).
- EXTERNAL DATA: none. docs/wayfinder/tickets/T02 and T04 (both `status: closed`) record PPMI ruled out on 2026-09-07 — DUA bars retention-granting platforms and scan-list sharing, and a DrivenData organiser ruled PPMI-trained models prize-ineligible. docs/wayfinder/map.md:39 and :54 confirm 'External data leaves the route'. aidlc-docs/audit.md (2026-09-10 17:00) records the re-check: no open DaT-SPECT dataset outside PPMI; MedicalNet not pursued.
- FINDING — competition labels are tracked in git: `git ls-files` shows artifacts/featarm_v4/oof_featarm.csv, oof_featarm_site.csv, artifacts/featcarm_v5/oof_featcarm.csv, oof_featcarm_site.csv, artifacts/run4a_stack/oof_stack.csv and kaggle/run8_distill/soft_master_v2.csv. Re-runnable check: each has 1,362 unique `uid` rows with a `y` column summing to 747 positives — i.e. a verbatim reconstruction of `train_labels.csv`, matching config.py:35 `DEFAULT_PRIOR = 747/1362`. Rules: 'Never share, copy, or publish the data' (DAT_PARKINSONS_CHALLENGE.md:84-85) and the MIT open-source release requirement (:88) collide here.
- FINDING — the raw competition data was shared with a second Kaggle account: docs/how-to-run-on-a-second-kaggle-account.md:13-16 instructs adding the teammate as a collaborator on `dat-parkinsons-challenge-data` (raw competition data, 'Can view'), `dat-parkinsons-cache-v4` ('Can view') and `dat-run3-checkpoints` ('Can edit'); aidlc-docs/HANDOFF.md:42 confirms the three datasets are 'private, owned by harshdipsaha, shared with skyfall23'. Rules clause at risk: 'One account per participant; no privately sharing code/data outside your team' (DAT_PARKINSONS_CHALLENGE.md:87) plus the staff test 'must not be accessible to anyone who hasn't agreed to the rules' (:95).
- MITIGATING — the repo treats skyfall23 as inside the team and as compute only: docs/how-to-run-on-a-second-kaggle-account.md:52 'Submissions go only through our DrivenData account (one team account)' and :58 'the teammate's account is a compute venue, not a second competition entry'; :57 'The datasets stay private and are deleted after the competition; no raw scan is written to notebook outputs.' aidlc-docs/audit.md (2026-09-08 08:00) records the user instruction that introduced the account.
- KAGGLE PRIZE-ELIGIBILITY EXPOSURE: unresolved by design. DAT_PARKINSONS_CHALLENGE.md:95-97 records the staff test (no third-party retention, fully deletable, no access by non-agreeing parties), the unanswered Kaggle question, and the safer alternative; docs/adr/0002-train-on-kaggle-despite-prize-risk.md accepts the risk explicitly and pre-commits to 'disclose the training venue if a prize is at stake' and to a one-day re-run on a rented instance if eligibility becomes a hard requirement.
- PSEUDO-LABELLING: none. scripts/make_soft_targets.py:1-2 and :19-33 build the soft-target table from the NESTED out-of-fold predictions of the five pillars on TRAINING scans, keyed by `cache_v4/folds.csv`; the output kaggle/run8_distill/soft_master_v2.csv has 1,362 rows (uid,y,p,loss,fold) and carries the true label in every row. No test scan, smoke or hidden, enters any training or fitting step.
- SMOKE-SET POLICY: docs/adr/0005-smoke-set-is-validation-only.md:15 'Nothing — no weight, temperature, prior, member gate or training example — is ever fitted on them'; :19 anchors selection to the nested out-of-fold estimate over 1,362 scans. scripts/build_master_v3.py:8 repeats 'The smoke set is never fitted on (ADR 0005)'.
- DELETION OBLIGATION: DAT_PARKINSONS_CHALLENGE.md:84 'delete all local copies after the challenge ends'. Locally `data/` and `cache_v4/` are already absent and both are gitignored (.gitignore lines `data/`, `cache_v4/`, `pretrained/`, `artifacts_v3/`), so the laptop side is nearly done; the open side is Kaggle — three private datasets plus the notebooks and their saved output versions (aidlc-docs/HANDOFF.md:41-42), which the Kaggle CLI cannot delete (no `datasets delete` / `kernels delete` subcommand — web UI only).

### Risk if wrong

This item is not priced in log loss — it is priced in the whole prize. If the inference-path audit is wrong (a hidden cross-scan statistic, or a manifest that ships with `typ` set) the submission is disqualified under DAT_PARKINSONS_CHALLENGE.md:86 and all five banked submissions including the 0.2473 go with it; the same clause makes a single stray aggregate print fatal at zero score benefit. If the Kaggle/second-account read is wrong, the downside is narrower: the score stands on the leaderboard but the €12,500/€7,500/€5,000 is withheld, and, in the worst reading, the raw-data share with a non-team account is a data-sharing breach rather than a venue question. If the label-file finding is wrong (i.e. the release is published with them), the breach happens AFTER the prize decision and damages the BOAS publication and the sponsor relationship rather than the ranking. Being wrong in the cautious direction is cheap: leaving `typ` null costs about 0.000-0.002 nested, revoking the Kaggle share costs only the teammate's remaining GPU hours, and stripping label columns from the public repo costs nothing at all.

### Action

- 1. BEFORE UPLOADING THE LAST ZIP — verify the manifest is inert (5 seconds, no side effects):
   py -c "import zipfile,json;z=zipfile.ZipFile(r'data/submission_master_v3_lam01.zip');m=json.loads(z.read('artifacts/manifest.json'));print('typ=',m.get('typ'),'| clip=',m.get('clip'),'| hedge=',m.get('hedge'),'| T=',m['T'],'lam=',m['lam'],'| arms=',[a['name'] for a in m['arms']])"
   REQUIRED: typ= None. If it is anything else, do NOT upload — rebuild with scripts/build_master_v3.py (which hard-sets typ to None at :111) instead of scripts/fit_stack_v3.py --typ (which sets it at :199). Note that scripts/repack_v3.py copies a manifest through verbatim, so repacking a --typ zip carries the flag forward.
- 2. DO NOT edit main_v3.py:64 before the final submission. Deleting the aggregate print is the correct permanent fix but it changes the shipped entrypoint with one real submission left; the gate (scripts/local_e2e_v3.py:38) plus step 1 already make the line unreachable. Schedule the deletion for the post-deadline MIT release instead (step 8).
- 3. BEFORE ANY FURTHER TRAINING ON `skyfall23` — settle the team question, because more training deepens the exposure. Decision tree:
   (a) If skyfall23 is a SECOND KAGGLE ACCOUNT OF THE SAME PERSON: there is no sharing-outside-the-team issue at all (DrivenData's one-account rule is about DrivenData accounts, and only one, harshdipsaha, was ever used to submit). Proceed; disclose it in section 4 of DISCLOSURE.md. Note only that running two Kaggle accounts may breach KAGGLE's terms, which is not a DrivenData matter.
   (b) If skyfall23 is ANOTHER PERSON who IS a registered member of the DrivenData team: compliant as-is. Record the team name and the registration date for section 4.
   (c) If skyfall23 is ANOTHER PERSON who is NOT registered on DrivenData: either form/join the team on the DrivenData site now (competition page -> Team) so that person has accepted the rules, or stop using the account and revoke the collaborator grants on `dat-parkinsons-challenge-data` and `dat-parkinsons-cache-v4` immediately (Kaggle -> dataset -> Settings -> Collaborators -> remove). Do NOT leave the raw-scan dataset shared with a non-participant; that is the one live clause that is independent of the Kaggle-venue gray area.
- 4. AFTER THE DEADLINE (2026-09-16 23:59 UTC), NOT BEFORE — inventory what must be deleted:
   kaggle datasets list --mine
   kaggle kernels list --mine
   Expected: datasets harshdipsaha/dat-parkinsons-challenge-data, dat-parkinsons-cache-v4, dat-run3-checkpoints; notebooks harshdipsaha/dat-parkinsons-train and skyfall23/dat-run7-hemisphere, dat-run6b-triplanar, dat-run6c-features-ssl, dat-run8-distill (check for a dat-run6a-crop3d too). Run the same two commands with KAGGLE_CONFIG_DIR pointed at the second account.
- 5. Delete them through the Kaggle WEB UI — the CLI has no delete subcommand for either resource. For each dataset: Settings -> remove every collaborator -> Delete dataset. For each notebook: open it -> Settings/File -> Delete notebook (this removes its saved output versions, which hold checkpoints and OOF tables). Screenshot each confirmation; the dates go into section 4 of DISCLOSURE.md.
- 6. Delete every local copy of the competition data and confirm:
   Remove-Item -Recurse -Force data, cache_v4, cache_v3, pretrained -ErrorAction SilentlyContinue
   py -c "import os;print([p for p in ('data','cache_v4','cache_v3','pretrained') if os.path.exists(p)] or 'all gone')"
   Keep the submitted zip and artifacts/ only until the results are final, then decide with the organisers (they are fitted parameters, not scans).
- 7. Purge the training labels from anything that will be published. Confirm the exposure first:
   py -c "import pandas as pd,glob;fs=glob.glob('artifacts/**/oof_*.csv',recursive=True)+['kaggle/run8_distill/soft_master_v2.csv'];[print(f,len(d:=pd.read_csv(f,dtype={'uid':str})),list(d.columns),int((d.y==1).sum())) for f in fs]"
   Then, in the RELEASE repository only (leave the private working repo alone until the prize decision), either delete those six files or strip the identifying columns:
   py -c "import pandas as pd,glob;fs=glob.glob('artifacts/**/oof_*.csv',recursive=True)+['kaggle/run8_distill/soft_master_v2.csv'];[pd.read_csv(f,dtype={'uid':str}).drop(columns=[c for c in ('uid','y') if c in pd.read_csv(f,nrows=0).columns]).to_csv(f,index=False) for f in fs]"
   The release must not carry `uid,is_pathologic` pairs for the 1,362 training scans in any form.
- 8. In the release repo, delete the aggregate print at main_v3.py:64 (keep the per-scan typicality machinery; only the log line is a rules problem), then re-run the gate: py scripts/local_e2e_v3.py <zip> <smoke_dir>
- 9. WRITE `docs/DISCLOSURE.md` WITH EXACTLY THE CONTENT BELOW (fill every <...> placeholder, resolve the <CONFIRM: ...> in section 4 from step 3, and paste the final submission's score into section 10). Send it to the organisers with the solution, unprompted, whether or not the solution places:

# Disclosure — DaT Parkinson's Challenge (DrivenData #311)

DrivenData account: `harshdipsaha`. Submitted solution: `submission_master_v3_lam01.zip`
(7 arms, 71 checkpoints, equal weights 1/7, T = 0.90, lambda = 0.01) and the earlier
submissions listed in section 10. Written <DATE, after 2026-09-16 23:59 UTC>.

This document is provided under the challenge rule that all external data and pretrained
models be disclosed to the organisers for independent verification, regardless of prize
eligibility. It also records, unprompted, the training venue, how the competition data was
handled and deleted, how the submitted code treats the hidden test set, and our position on
pseudo-labelling.

---

## 1. External data

**None.** No dataset other than the competition's own `train_labels.csv` and `niftis/` was
used at any stage — not for training, not for pretraining, not for calibration, not for
intensity normalisation, not for the anatomical template.

The Parkinson's Progression Markers Initiative (PPMI) DaT-SPECT collection was researched
and formally ruled out on 2026-09-07 (`docs/wayfinder/tickets/T02-*.md`,
`docs/wayfinder/research/ppmi-dat-spect.md`): its Data Use Agreement bars processing on
third-party platforms that retain uploaded content and prohibits sharing participant-level
data or scan lists, and a DrivenData organiser had already ruled PPMI-trained models
prize-ineligible. **No PPMI data was ever requested with intent to use it here, downloaded,
or processed.** No other open DaT-SPECT corpus was identified.

The anatomical template shipped inside the submission (`artifacts/template.npz`) is *not*
external data: it is a label-blind iterative mean of the competition's own training volumes
(`scripts/build_template.py`), i.e. a model parameter derived from the challenge data.

## 2. External pretrained models

Two ImageNet-1k backbones distributed by torchvision are ancestors of weights inside the
submitted ensemble. All were obtained once by `scripts/fetch_pretrained.py`.

| Asset | Source | Licence | Free & public to all participants | In the submitted model |
|---|---|---|---|---|
| ConvNeXt-Tiny `ConvNeXt_Tiny_Weights.IMAGENET1K_V1` | torchvision | BSD-3-Clause | yes (`pip install torchvision`) | **yes** — arm `a2d_cnx_cor` |
| EfficientNet-V2-S `EfficientNet_V2_S_Weights.IMAGENET1K_V1` | torchvision | BSD-3-Clause | yes | **yes** — arm `a2d_v2s` |
| EfficientNet-B0 `EfficientNet_B0_Weights.IMAGENET1K_V1` | torchvision | BSD-3-Clause | yes | no — used in earlier submissions (arm `a2d_cor`, runs 3-5) only |
| ResNet-18 `IMAGENET1K_V1`, ResNet-50 `IMAGENET1K_V2` | torchvision | BSD-3-Clause | yes | no — downloaded, never trained with |
| R3D-18 `R3D_18_Weights.KINETICS400_V1` | torchvision | BSD-3-Clause | yes | no — downloaded, never trained with |

**Licence position.** torchvision, including the weights it distributes, is BSD-3-Clause:
permissive, no non-commercial restriction, and compatible with releasing the resulting model
for broad (including commercial) use. The assets are freely and publicly available to every
participant from PyPI without registration or approval. We note for completeness that the
underlying ImageNet-1k *images* are distributed by their owners under research terms; what we
used and redistribute are the torchvision-published *weights* under BSD-3-Clause, which is the
standard practice for ImageNet transfer learning in this and prior DrivenData challenges. We
raise it here rather than leave it implicit.

**How they enter the submission.** A pretrained `state_dict` is used only as the
initialisation of a fine-tuning run on the competition's training scans. **The submitted zip
contains no pretrained weight file.** At inference the backbone is constructed with
`weights=None` (`datsc/models_v3.py:92`), every parameter comes from our fine-tuned
checkpoint, and the training-time `weights_path` key is stripped from the manifest before the
model is built (`datsc/infer_v3.py:149`). The evaluation container has no network access and
the shipped code contains no network call of any kind (no `requests`, `urllib`, `torch.hub`
or URL literal in `main.py`, `datsc/*.py`, `tests_synth.py`).

**Self-supervised pretraining.** The `a3d_ssl` arm was pretrained with VICReg on the
competition's **own training scans only** (`datsc/ssl_c.py`); no external corpus was involved.

## 3. Software

All permissive and all already present in the official runtime image; no additional package
was requested: nibabel (MIT), numpy, scipy, pandas, torch, torchvision (all BSD-3-Clause).

## 4. Training venue, what was uploaded, and deletion

Training ran in **private Kaggle notebooks** (2x T4) on two Kaggle accounts:
`harshdipsaha` (runs 3-5) and `skyfall23` (runs 6b, 6c, 7, 8). `skyfall23` is
<CONFIRM: "a second Kaggle account of the same participant" OR "the Kaggle account of
<name>, a registered member of our DrivenData team who has read and accepted the competition
rules">. Submissions were made from a single DrivenData account, `harshdipsaha`.

The competition scans and every derived artefact lived in three **private** Kaggle datasets
owned by `harshdipsaha` and shared, privately, only with `skyfall23`:

| Dataset | Contents |
|---|---|
| `dat-parkinsons-challenge-data` | the raw competition NIfTIs and `train_labels.csv` |
| `dat-parkinsons-cache-v4` | the registered 2 mm cache derived from them, `template.npz`, the fold assignment, and the torchvision weight files |
| `dat-run3-checkpoints` | model checkpoints, out-of-fold prediction tables, the feature-arm models |

No dataset and no notebook was ever public. No raw scan or derived volume was written to a
notebook output. The data was never used for anything but this challenge, and never during
any period outside the challenge.

**Kaggle's retention terms, disclosed voluntarily.** Kaggle's Terms of Use grant Kaggle a
perpetual, irrevocable, worldwide licence over uploaded content, and Kaggle's Privacy Policy
describes deletion as taking roughly two months (up to six for encrypted backups). The forum
thread on AI assistants and cloud GPU compute
(community.drivendata.org/t/clarification-on-ai-coding-assistants-and-cloud-gpu-compute-with-competition-data/11479)
established the test — data not retained by a third party, not accessible to anyone who has
not agreed to the rules, every copy fully deletable — but the Kaggle-specific question posed
in that thread was still unanswered when we had to choose a venue. We chose Kaggle knowingly,
for cost and GPU quota, and recorded the decision and its risk at the time
(`docs/adr/0002-train-on-kaggle-despite-prize-risk.md`). We disclose it here so the
organisers can rule on prize eligibility rather than discover the venue. If the venue is
judged disqualifying for prize purposes, we ask to be told: every checkpoint is reproducible
from the shipped scripts on a self-controlled, wipeable instance in roughly one day.

**Deletion.** On <DATE> the three private Kaggle datasets, all Kaggle notebooks and all their
saved output versions were deleted through the Kaggle web interface; the collaborator grants
were revoked first. All local copies were deleted: `data/` (raw scans and labels),
`cache_v4/`, every checkpoint directory and every built `submission_*.zip`. We hold no
competition scan, no derived volume cache and no copy of `train_labels.csv`. We will
re-confirm with Kaggle after their stated retention window. The only competition-derived
material we retain is the fitted model parameters inside the submitted solution (checkpoints,
`template.npz`, the feature-model `.npz` files), which contain no recoverable scan; if the
organisers prefer these destroyed too, we will destroy them on request.

## 5. What the submitted code does with the hidden test set

- **Every test scan is processed independently.** Loading and resampling
  (`datsc/io.py`), brain masking and p75 reference normalisation
  (`datsc/preprocess.py:50-54`), the crop-at-centre-of-mass box and the per-scan statistics
  (`datsc/volume.py:23-53`), all alternative intensity normalisations
  (`datsc/volume.py:56-74`, which read only that scan's own stored statistics), registration
  to the shipped template and reference-region scaling (`datsc/register.py:205-220`), and all
  feature extraction (`datsc/features_v3.py`, `features_v4.py`, `features_c.py:182-199`) use
  only that scan's voxels plus parameters fitted on the training set and frozen inside the
  shipped artifacts.
- **No statistic is computed across the test set.** There is no test-set-wide normalisation,
  no quantile or rank transform over the batch, no clustering of test scans, no transductive
  step. The standardisers and PCA bases in the feature arms are the training-time values
  loaded from disk (`datsc/featmodel.py:38-40`, `datsc/lrmodel.py`).
- **Networks run in `eval()` mode** (`datsc/infer_v3.py:152`). The 3D models use GroupNorm
  (`datsc/models_v3.py:16-17`), which is per-sample; the torchvision 2D backbones' BatchNorm
  layers use the running statistics stored in the checkpoint. Batching (16 scans,
  `datsc/infer_v3.py:155`) is a throughput device and does not couple scans.
- **The per-scan "typicality" score** is a Mahalanobis distance to *training-set* statistics
  stored in the model file (`datsc/featmodel.py:45-52`), not to the test set. It is disabled
  in the submitted manifest (`"typ": null`).
- **Nothing about the test data is printed or logged.** The container log contains only the
  number of rows in the supplied `submission_format.csv`, the number of volumes that
  preprocessed successfully, per-arm checkpoint counts and timings — facts the evaluator
  already holds — and no image statistic, no prediction, no per-scan value.

## 6. Pseudo-labelling — explicit statement

**No pseudo-labelling of any kind was performed.** No prediction on a test scan — smoke
test, public leaderboard or private leaderboard — was ever used as a training target, as a
calibration target, or as a selection signal for ensemble weights, temperature, prior mix,
clipping or member gating.

Because one of our runs is named "distillation", we describe it precisely so it is not
mistaken for pseudo-labelling. In run 8 the soft target of a **training** scan is a mixture
of that scan's **known hard label** with the *nested out-of-fold* probability the existing
ensemble assigns to that **same training scan** (`scripts/make_soft_targets.py`; the table
`kaggle/run8_distill/soft_master_v2.csv` has exactly 1,362 rows — one per training scan — and
carries the true label `y` in every row). This is self-distillation on labelled training
data, computed entirely before any test scan is seen. No test scan, and no prediction on a
test scan, participates in it.

## 7. Smoke-test policy

DrivenData's smoke test scores a submission on twenty scans drawn from the training set. Our
policy, fixed in `docs/adr/0005-smoke-set-is-validation-only.md` on 2026-09-07 and never
departed from: those twenty scans are an out-of-distribution sanity check and, at most, a
coarse tie-breaker between two otherwise-equal candidates. **No weight, temperature, prior,
clip threshold, ensemble member or training example was ever fitted on them or selected by
them.** Candidate selection was anchored throughout to a nested out-of-fold estimate over all
1,362 training scans; twenty scans carry a log-loss standard error of about 0.05 and would
have been fitted noise. The smoke-to-public track therefore remains an honest read-out of
transfer, because the smoke score was never optimised.

## 8. Model provenance summary (the submitted ensemble)

| Arm | Architecture | Initialisation | External asset |
|---|---|---|---|
| `a3d_hemi` | hemisphere-level shared 3D encoder, GroupNorm | random | none |
| `a3d_hemisoft` | as above, trained on run-8 soft targets | random | none |
| `a3d_crophistsoft` | 3D striatal crop, histogram-standardised, soft targets | random | none |
| `a3d_ssl` | 3D ResNet, GroupNorm | VICReg self-supervised on the competition training scans | none |
| `a2d_cnx_cor` | ConvNeXt-Tiny on coronal 2.5D slabs | ImageNet-1k | **ConvNeXt-Tiny BSD-3** |
| `a2d_v2s` | EfficientNet-V2-S on axial 2.5D slabs | ImageNet-1k | **EfficientNet-V2-S BSD-3** |
| `featc` | 116 hand-crafted radiomic/shape descriptors + striatal-patch PCA -> logistic regression | fitted on training scans | none |

Combination: equal weights, temperature 0.90, prior mix lambda = 0.01 toward the training
prevalence 747/1362 = 0.5485. All constants are fitted on training data only.

## 9. Open-source release

If the solution places, it will be released under the MIT License as required, and indexed
for BOAS. The public release will contain the inference and training code, the model weights
and this disclosure, and will contain **no competition data in any form**. Specifically, the
out-of-fold prediction tables and the soft-target table present in our working repository
(`artifacts/featarm_v4/oof_featarm*.csv`, `artifacts/featcarm_v5/oof_featcarm*.csv`,
`artifacts/run4a_stack/oof_stack.csv`, `kaggle/run8_distill/soft_master_v2.csv`) each carry
the `uid,y` pair for all 1,362 training scans and are removed, or stripped of the `uid` and
`y` columns, before publication. `template.npz` and the feature-model `.npz` files are
aggregate fitted parameters from which no individual scan can be recovered.

## 10. Submission ledger

| Date (UTC) | Package | Public log loss |
|---|---|---|
| 2026-09-05 | run 3 | 0.3391 |
| 2026-09-06 | run 4 | 0.2817 |
| 2026-09-07 | run 4a | 0.2667 |
| 2026-09-12 | Master v3 (lambda 0.02) | 0.2557 |
| 2026-09-13 | Master v3 (lambda 0.01) | 0.2473 |
| <DATE> | <final> | <score> |

## 11. Attestation

The statements above are accurate to the best of our knowledge. We are available for any
verification the organisers wish to run, including re-execution of training from the released
scripts.

<Name> — DrivenData `harshdipsaha` — <date>

- 10. Alongside it, create the release README skeleton required by T12 (docs/wayfinder/tickets/T12): title and one-paragraph method summary; MIT LICENSE file; `Data` section pointing to DrivenData and stating no competition data is redistributed; `External assets` section reproducing the table from DISCLOSURE.md section 2; `Reproduce` section = fetch_pretrained.py -> cache_volumes.py -> register_cache.py -> build_template.py -> train_v3.py -> fit_stack_v3.py -> build_master_v3.py -> local_e2e_v3.py; `Inference` section = unzip, `python main.py` against /code_execution/data; hardware and runtime (A100 80GB, ~3 h cap, measured wall clock); and a link to docs/DISCLOSURE.md.
- 11. Mark T12 resolved in docs/wayfinder/tickets/T12-disclosure-and-open-source-release-draft.md once the user has approved the text (the ticket is `mode: HITL`, resolved by user approval, not by the file existing).

### Rejected alternatives

- Delete the `typicality:` print from main_v3.py:64 before the last submission — rejected: it is a print-only change, but it edits the shipped entrypoint with one real submission and no spare slot, and the line is already unreachable (build_master_v3.py:111 sets typ=None; scripts/local_e2e_v3.py:38 asserts it). The risk of an unverified entrypoint edit exceeds the risk of a dead branch. Scheduled for the release instead.
- Remove the six operational count prints (main_v3.py:50/:78, infer_v3.py:69/:84/:104, features_c.py:199) to be maximally safe — rejected: they report row and failure counts the evaluator already holds, DrivenData's own smoke run logged `stage0 prior written for 20 rows` and `preprocessed 20/20 volumes` without objection (DAT_PARKINSONS_CHALLENGE.md:122), and a silent run would remove the only evidence that all arms actually fired, which the packaging gate depends on.
- Retrain everything on a rented, self-wipeable instance before the deadline to cure the Kaggle exposure — rejected: ADR 0002 estimates a one-day re-run and there is not a day of slack with one submission slot left; more importantly a re-run does not undo the upload that already happened, so it buys nothing for eligibility now. It stays the remedy if the organisers rule the venue out.
- Delete the Kaggle datasets now, to be clean early — rejected: the rules require deletion after the challenge ends (DAT_PARKINSONS_CHALLENGE.md:84), and deleting now destroys the ability to run the remaining GPU hours or to rebuild a zip if the last one fails verification. Deletion is step 4-5, after 2026-09-16 23:59 UTC.
- Rewrite git history in the private working repo to expunge the label-carrying CSVs — rejected as premature: the repo is private (docs/how-to-run-on-a-second-kaggle-account.md:9 refers to it as a private repo) and the files are legitimately needed to rebuild and gate the final zip. The obligation bites at publication, so the strip happens in the release repo (step 7).
- Stay silent about the Kaggle venue and hope it is not asked — rejected: disclosure of the venue is what ADR 0002 pre-committed to, the staff guidance in the forum thread is explicit about the retention test, and a venue discovered by the organisers after a prize award is far worse than one declared with a stated remedy.
- Claim the run-8 soft targets are simply 'label smoothing' and say nothing further — rejected: the run is named distillation and the word invites the pseudo-labelling question. Describing the construction precisely (nested OOF probability of the same TRAINING scan, mixed with its true label) is both accurate and the strongest possible answer.
- Treat the shipped `template.npz` and the feature-model `.npz` files as competition data that must be destroyed — rejected: they are aggregate fitted parameters with no recoverable scan, in the same class as the model checkpoints, and destroying them would make the MIT release non-functional. Flagged in DISCLOSURE.md section 4 with an offer to destroy on request.

### Open questions

- Is `skyfall23` a second Kaggle account of the same participant, or another person — and if another person, are they a registered member of the DrivenData team? Nothing in the repo records a DrivenData team registration; aidlc-docs/HANDOFF.md:39-41 shows one DrivenData login (harshdipsaha) and two Kaggle accounts. Resolved by looking at the competition page's Team section while logged in as harshdipsaha, and by asking the user directly. This single fact decides whether sharing `dat-parkinsons-challenge-data` was a rules breach or a non-event.
- Does DrivenData still allow team formation three days before the deadline? Most DrivenData competitions close team changes before the final deadline. Resolved by the Team tab on the competition page; only matters under branch (c) of action step 3.
- Will DrivenData accept a Kaggle-trained solution for prizes? Still unanswered on the forum as of the repo's last research (DAT_PARKINSONS_CHALLENGE.md:97). Resolvable now at near-zero cost by posting the question again in the existing thread, or by emailing info@drivendata.org with the disclosure text — an answer before the deadline would let the last slot be spent knowing whether the prize is live.
- Does the organisers' reading of 'no NC licence' reach past torchvision's BSD-3 weights to ImageNet's own non-commercial research terms for the underlying images? Every ImageNet transfer-learning solution in DrivenData history has the same structure, so the expected answer is no, but section 2 of the disclosure raises it explicitly so the organisers rule rather than assume.
- Was `kaggle/run6a_crop3d` ever executed on Kaggle, and under which account? aidlc-docs/efforts/005-*/effort-state.md records unit A as 'merged' with no Kaggle execution row, unlike units B and C. Resolved by `kaggle kernels list --mine` on both accounts (action step 4) — it only changes the deletion inventory.
- Should the final submitted zip itself be deleted after the results are published, or retained as the artefact backing the MIT release? The rules require deleting competition DATA, and checkpoints are fitted parameters, not data — but the organisers may have a house position. Resolved by asking in the disclosure email.

> **Needs checking** — could not be established from the repo, the rules or arithmetic: numbers.
