# Plan: Run 9 — the last submission (effort 008, ticket T16)

Spec: [2026-09-13-run9-endgame-spec.md](../specs/2026-09-13-run9-endgame-spec.md). ADR: [0006](../../adr/0006-the-prior-mix-is-insurance-not-a-lever.md).

## Timeline

Now is **Sept 13 09:05 UTC**. The slot opens **Sept 14 03:38** and stays open. Deadline **Sept 16 23:59**.

| When (UTC) | What | Cost |
|---|---|---|
| Sept 13 09:30 | Step 1 — verify the ledger and the two submission logs | 15 min, free |
| Sept 13 10:00 | Step 2 — calibration sweep on the real master pool | 10 min, no GPU |
| Sept 13 10:30 | Step 3 — size envelope: re-gate at a raised `--max-mb`, measure real headroom | 30 min, no GPU |
| Sept 13 12:00 | Step 4 — GPU session A (seeds for the best-subset arms) | ~4.5 h wall, ~9 GPU-h |
| Sept 14 03:38 | **Slot opens. Do not submit.** Fallback zip stays on disk. | — |
| Sept 14 08:00 | Step 5 — fetch, re-gate, decide whether session B is worth it | 1 h |
| Sept 14 10:00 | Step 6 — GPU session B, only if session A gated in | ~4.5 h wall, ~9 GPU-h |
| Sept 15 | Buffer. Re-gate, rebuild, container-check. No new training after 18:00. | — |
| Sept 16 12:00 | **Freeze.** Final zip built, full suite green, `FORMAT OK`, all arms N/N | hard gate |
| Sept 16 13:00 | Smoke-test the exact frozen zip on DrivenData (done ~13:38) | free, ~38 min |
| Sept 16 16:00 | **Real upload** (~12 min upload, ~20 min queue, ~1 h scoring → ~17:32; margin 6 h 27 min) | the last slot |
| Sept 16 20:27 | Latest start if the run uses the full 3-h cap. 22:27 for a typical ~1 h score; 23:47 absolute. | — |

The smoke test is at 13:00, not 14:00: only one code job runs at a time, and a 14:00 smoke would finish 14:38
and leave 1 h 22 min before a 16:00 upload — under the ≥ 2 h clearance this plan requires in Step 7.

## Step 0 — the conclusion this plan reached: verify, don't train

The deep-research pass (`docs/research/dat-endgame/report.md`, 7 items, all validated) changed the shape of
this plan. The remaining error is **predominantly irreducible**, and the evidence is cross-representational:

- **88 scans (6.46 % of the pool) carry 46.9 % of all loss and are wrong under every independent
  representation built so far** — including the non-CNN registered-ROI scalar arm — against 0.04 scans
  expected if the three representations failed independently.
- Errors do **not** cluster by fold (permutation p = 0.838 — *less* clustered than chance), and the mid band is
  calibrated at its own entropy (0.6081 against 0.6136).
- The image-derived ambiguity rate (6.5–7.2 %) converges with the literature's 7.8 % discrepant-expert-read
  rate for DaT-SPECT.

Both escape routes are closed. **Calibration is exhausted**: a cross-fitted isotonic fit scores 0.232086
against the actual 0.232252 — 0.0002 of headroom, which also rules out Venn-Abers. **"Never be confident"
strictly loses**: +0.0017 at a 0.98 cap, +0.0136 at 0.95. And **hedging on arm disagreement cannot work**,
because the hard core has AUC 0.4193 — those scans disagree *less* than average, which at last explains the
repo's long-standing unexplained result that clip and hedge showed no nested gain.

So the last slot is not won by modelling. **The highest-value remaining action costs zero GPU: read the two
DrivenData `View log` outputs.** The recorded "+0.0084 from λ" is 1.5× the arithmetic ceiling, so one of the
two scored zips probably did not ship the ensemble its gate table measured — and that question outvalues every
training option on the table. The seed run in Step 4 is worth doing but must clear a hard **≥ 0.003** nested
gate, because anything smaller sits inside the ±0.013 bootstrap noise of the pool.

**Before spending GPU on `skyfall23`, settle one unrecorded fact** (Step 8): the raw-scan Kaggle dataset was
shared with that account, and whether that is inside "no privately sharing code/data outside your team"
depends on whether skyfall23 is a second account of the same person, a registered DrivenData teammate, or
neither. Nothing in the repo says which.

## Step 1 — verify what actually shipped

Log in to the submissions page. Transcribe **every** row into `RESULTS_LOG.md`: id, UTC timestamp, type
(smoke/real), score, AUROC. Read the banner verbatim ("Submissions used: N of 3 … next on …").

Then click **View log** on the two real submissions and record, for each:

- `preprocessed N/N volumes` — N must equal the row count, not 0
- one `arm <name>: checkpoints used k/k` line per torch arm, and an `arm featc: feat model on N volumes` line
- `stage written after arm <name>` for **all seven** arms, last being `a3d_crophistsoft`
- any `ladder:`, `arm ... failed:`, `checkpoint failed (`, or `Traceback`
- `done in <seconds>` against the 3-hour cap

**The AUROC test — free, and decisive.** AUROC is on the leaderboard, and λ is a strictly monotone transform,
so it cannot change any ranking. If slots 1 and 2 were genuinely λ-only twins, their AUROC must be **identical
(0.9576)**. Slot 2's AUROC was never recorded. Read it. If it differs at all, the two runs did not ship the
same predictions and the whole "λ bought 0.0084" story is void.

Note the ladder is *not* the explanation, despite being the obvious suspect: slots 1 and 2 uploaded the same
847 MB / 71-checkpoint zip, so their runtime is identical, and a 71-checkpoint run costs roughly 24 minutes
against the 9000 s ladder — truncation would need on the order of 1,580 checkpoints. Read `done in Xs` and the
seven `stage written after arm` lines to confirm, but expect them to be clean. That leaves a recorded number
being wrong as the live hypothesis, which is what the AUROC reading and the missing ids settle.

## Step 2 — calibration sweep (no GPU)

```
python scripts/sweep_calibration.py --src data/run8/artifacts_v3 --src data/master_pool/artifacts_v3
```

Read the TAIL RISK table, not just the nested column. The nested pool has no shift and will always point at
λ → 0; the BREAK-EVEN column says how many confidently-wrong scans erase that gain. **Keep λ = 0.01 and
T = 0.9 unless break-even exceeds a number you would confidently bet against.** Record the table.

This must run **on the training machine**: it needs `cache_v4/` and `data/master_pool/artifacts_v3`, both of
which are gitignored and absent from a fresh clone. Expect it to confirm rather than change the decision —
ADR 0006 already fixes T = 0.90, λ = 0.01, `clip = null` on measured grounds.

## Step 3 — the size envelope

The shipped zip is 847 MB against `--max-mb 850`. Establish the real headroom before training anything:

```
python scripts/build_master_v3.py --src data/run8/artifacts_v3 --src data/master_pool/artifacts_v3 \
    --lam 0.01 --T 0.9 --max-mb 1050 --zip data/submission_v9_probe.zip --no-e2e
```

Read the tail: per-arm checkpoint counts, total size, and — critically — **no** `zip ... > 1050: dropped ...`
line. If a drop line appears at 1050, the trim is active and every later decision is being made about a
different ensemble than the gate table scored. The build now exits non-zero rather than writing an oversized
zip, so a `FATAL:` line means raise the ceiling or ship fewer seeds; it does not mean edit `DROP_ORDER`.

**Checkpoint size is per-arm, not an average.** 847/71 ≈ 11.9 MB is a zip average and is the wrong unit:
checkpoints are fp16 (`datsc/train_v3.py:300-302`) and a 3D hemisphere checkpoint is **≈3.83 MB** — confirmed
empirically as (847−732)/30 across the Master v2→v3 zips (`RESULTS_LOG.md:323,349-357`) and from the
HemiNet3D parameter count (`datsc/models_h.py:24-49`) — against **≈37.1 MB** for `a2d_v2s` and **≈51.2 MB**
for `a2d_cnx_cor`.

So 1,050 MB buys roughly **52** more 3D checkpoints, but only ~5 v2s or ~3 cnx_cor ones.

**In fact neither megabytes nor the ladder binds.** The upload limit is at least 1421 MB — a 1421 MB /
39-checkpoint zip already smoke-scored (`id-321496`) — and a 71-checkpoint run costs ~24 min against the
9000 s ladder, so truncation would need on the order of 1,580 checkpoints. **What caps the spend is
diminishing returns**: `a3d_hemi` seeds 3→5 bought 0.0001 nats for 10 checkpoints. Add ~20 checkpoints where
they measurably pay, and stop — not because you must, but because nothing beyond that has been shown to help.
Still keep the ceiling honest: overflowing deletes `a2d_v2s` seed 1 first, worth ≈0.01, and a *successful*
trim is now fatal by default because the gate table scored the untrimmed set.

## Step 4 — GPU session A

Spend seeds only on arms already in the greedy best subset from Step 3's table, and only where a seed has
actually been shown to pay. **The measurement that decides this:** `a3d_hemi` seeds 3→5 bought **0.0001 nats
for 10 checkpoints** — the two gate rows at `RESULTS_LOG.md:338-339` differ only by the `_s[012]` vs `_s?`
glob. That is ~38× worse per checkpoint than `a3d_crophistsoft`. **More plain hemisphere seeds are rejected.**

One session, 20 jobs, ≈4.1 GPU-h, +77 MB → ≈924 MB / 91 checkpoints:

```
python kaggle/run8_distill/gen_run8.py --out kaggle/run9_seeds --slug dat-run9-seeds \
    --kernel-source skyfall23/dat-run8-distill \
    --hemisoft-seeds 0 1 2 3 4 --crophistsoft-seeds 0 1 2 \
    --skip r3d_hemisoft_site r3d_hemi_more r3d_hemimw r3d_hemisoft_full
kaggle kernels push -p kaggle/run9_seeds
```

Every flag above is verified against `gen_run8.py:26-40`, and all four `--skip` names against the `JOBS8`
table at `gen_run8.py:60-67`. The skipped jobs are skipped for a reason: `r3d_hemisoft_site` and
`r3d_hemisoft_full` write OOF files tagged `_site` and `_fall`, which `load_arm` filters out
(`build_master_v3.py:26`), so **their checkpoints can never enter this zip** — running them is pure waste.
(Those site-fold OOFs matter for site-pool temperature fitting elsewhere; the `build_master_v3` path fits T
per random fold against a floor, so nothing here needs them.)

`--kernel-source` mounts the previous version's output so finished jobs print "already done" and only new
seeds run. Before pushing: Edit → Session options → **GPU T4 x2**, Internet On. Set the `KAGGLE_USERNAME` /
`KAGGLE_KEY` secrets once (skyfall23's own) — without them a cut session loses everything.

Watch with `kaggle kernels status skyfall23/dat-run8-distill`; fetch with
`python scripts/watch_kaggle_output.py --kernel skyfall23/dat-run8-distill --out data/run9a`.

## Step 5 — re-gate

```
python scripts/build_master_v3.py --src data/run9a/artifacts_v3 --src data/run8/artifacts_v3 \
    --src data/master_pool/artifacts_v3 --lam 0.01 --T 0.9 --max-mb 1050 \
    --smoke data/smoke --zip data/submission_v9.zip
```

Compare the gate table against **0.2266** (what is banked), not against 0.2323. Keep the new zip only if it is
≥ 0.002 nested better *and* fits without a drop line. If the greedy best subset now exceeds the checkpoint
budget from Step 3, drop the weakest arm rather than letting `DROP_ORDER` choose.

## Step 6 — session B

Only if session A produced a gate gain. Same shape, more seeds of whatever gated in. If session A gained
nothing, stop training: the banked 0.2266 zip is the thing to beat and more seeds of a non-gating arm will not
beat it. **No training starts after Sept 15 18:00 UTC.**

## Step 7 — freeze and ship

```
git status --short                       # must be empty; the zip packs the working tree
python -m pytest -q                      # 99 passed
python scripts/local_e2e_v3.py data/submission_v9.zip data/smoke
```

`local_e2e_v3.py` must print `FORMAT OK`, `failed volumes: 0`, and every arm N/N — it now also fails if any
manifest arm was silent, if a `ladder:` or failure line appeared, or if `typ` is set. Then confirm the manifest
matches what the gate scored:

```
python -c "import zipfile,json;z=zipfile.ZipFile('data/submission_v9.zip');m=json.loads(z.read('artifacts/manifest.json'));print(m['T'],m['lam'],m['typ'],[(a['name'],len(a['checkpoints'])) for a in m['arms']])"
```

Smoke-test that exact file on DrivenData (free). **Pass criterion: within ~0.01 of 0.2374** — that is
`id-321850`, the Master v3 smoke, and it is the most recent smoke score in the repo that carries a real
submission id. (Do *not* use the "smoke 0.2474" figure quoted in `forharsh.md` for the λ=0.01 zip: it has no
id and no source line anywhere, so it cannot anchor a gate.) The smoke is a liveness check under a 6-minute
cap, not a ranking signal; checkpoint *count* is the fixed cost there, and a measured 1421 MB / 39-checkpoint
zip already smoke-scored successfully (`id-321496`), so size is not the hazard it looks like.

Then upload as the real submission, and write the id, score, AUROC and G into `RESULTS_LOG.md`.

## Step 8 — compliance, before the GPU and before any release

Two exposures, neither in the shipped inference path (which was audited clean: every normalisation is
per-scan, every reduction is over arms rather than scans, and no pretrained weight file ships at all).

1. **Settle skyfall23's status before spending the remaining GPU.** The raw-scan Kaggle dataset was shared
   with that account (`docs/how-to-run-on-a-second-kaggle-account.md:13-16`, `HANDOFF.md:42`). If skyfall23 is
   a second Kaggle account of the same person, this is fine. If it is a different person *registered on the
   DrivenData team*, fine. If it is a different person **not** on the team, that is sharing competition data
   outside the team, and the right move is to stop using the account and retrain on your own. Record the
   answer in `RESULTS_LOG.md` either way — it is currently nowhere in the repo.
2. **Six git-tracked files reproduce the entire training label set** — `artifacts/featarm_v4/oof_*.csv`,
   `artifacts/featcarm_v5/oof_*.csv`, `artifacts/run4a_stack/oof_stack.csv` and
   `kaggle/run8_distill/soft_master_v2.csv`, each carrying 1,362 uids with `y` summing to 747 positives, i.e. a
   verbatim `train_labels.csv`. This is harmless while the repo is private (confirmed private: an
   unauthenticated fetch of the GitHub page returns 404) but becomes a breach of "never share, copy, or
   publish the data" the moment the MIT release happens. **Purge them in the release commit**, not after.

Also note that `main_v3.py:64`'s test-set aggregate print is inert only because `build_master_v3.py:111` sets
`typ: None` — but `fit_stack_v3.py:199` can set it and `repack_v3.py:16-18` copies a manifest through
verbatim, so the new assertion in `local_e2e_v3.py` is the only thing standing between those two paths and a
disqualification. Do not remove it.

`docs/DISCLOSURE.md` (ticket T12) is written **after** the deadline; its full drafted content is in
`docs/research/dat-endgame/results/compliance.json` under `action`.

## Risks

| Risk | Why it bites | Mitigation |
|---|---|---|
| Overflowing 850 MB silently | `DROP_ORDER` deletes `a2d_v2s` seed 1 first (≈0.01), then whole gate-winning arms | `--max-mb 1050` set explicitly; build now exits non-zero instead of writing an oversized zip |
| The ladder drops arms on the hidden set | 150-min flat deadline; the run-8 arms run last and `combine()` renormalises silently | Step 1 reads the real logs; keep total checkpoints inside the Step-3 budget; `local_e2e_v3.py` fails on any `ladder:` line |
| Smoke test fails on a bigger zip | 6-minute cap, checkpoint loading is a fixed cost | Treat a smoke timeout as a size signal, not a model signal; the real run has 3 h |
| Chasing λ further | Ceiling 0.0055, realistic 0.0020, and λ=0 removes the 5.2-nat cap | ADR 0006; `tests/test_run9.py` pins it |
| Kaggle session cut | No upload secrets → everything lost | Set the secrets once; `--kernel-source` resumes |
| Submitting at 03:38 out of habit | Spends the last slot on a zip worse than what is banked | The slot does not expire; freeze is Sept 16 12:00 |
| One active code job | A smoke test near the deadline blocks the real upload | ≥ 2 h clear between the smoke test and the real upload |
| Editing `datsc/` late | `build_master_v3.py:126-128` packs the working tree → ships untested inference code | All changes confined to `scripts/`; `git status` clean before the build |
