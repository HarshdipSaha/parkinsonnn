# RUNBOOK — everything from here to the deadline, alone

Written 2026-09-13. **Deadline 2026-09-16 23:59 UTC. One real submission left.** Run everything from
`H:\augsepthacks\parkinsonnn`. Every step below says what to run, what you should see, and what to do when it
goes either way. If you only read one thing, read §1 and §7.

**Standing:** public 0.2473, nested 0.2266, rank ~#16 (recorded; the id was never written down — §1 fixes that).
**Already decided, do not re-open:** ship T = 0.90, λ = 0.01, `clip = null` (ADR 0006). Equal weights, all seeds.

---

## 0. The shape of the next three days

| When (UTC) | Do | Cost |
|---|---|---|
| Now | §1 verify the ledger, §2 the hard-core diagnostic | 45 min, free |
| Sept 13 afternoon | §3 calibration sweep, §4 size probe | 40 min, free |
| Sept 13 evening | §5 GPU session — **only if §2 or §4 says it is worth it** | ~4.5 h |
| Sept 14 03:38 | Slot opens. **Do not submit.** | — |
| Sept 14–15 | §6 re-gate; second GPU session only if the first gated in | — |
| Sept 15 18:00 | **No training may start after this.** | — |
| Sept 16 12:00 | §7 freeze | hard gate |
| Sept 16 13:00 | §7 smoke test | free |
| Sept 16 16:00 | §7 real upload | the last slot |
| Sept 17 | §9 disclosure and cleanup | — |

---

## 1. Verify the ledger (do this first, it is free and it gates everything)

Open <https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/submissions/> logged in as
harshdipsaha. Transcribe **every** row into `RESULTS_LOG.md` using this schema:

```
| id | type | submitted UTC | scored UTC | zip | MB | ckpts | score | AUROC | rank@read | status | note |
```

Read these five things and write them down verbatim:

1. The quota banner — `Submissions used: N of 3 … next on <DATE>`. Whether it carries a **clock time or only a
   date** settles whether the window is rolling-168h or calendar-day.
2. **The row count in the Normal-submission table.** It must be **5**. If it is 4, you have *two* slots and
   everything below relaxes. If it is 6, you have none.
3. The ids for the 0.2557 and 0.2473 rows. **Neither is recorded anywhere in this repo**, which breaks the
   house rule that every score carries an id — a rule written after a session once invented one.
4. **Slot 2's AUROC.** This is a free falsification test: λ is a strictly monotone transform and cannot change
   any ranking, so if slots 1 and 2 were genuinely λ-only twins their AUROC must be **identical (0.9576)**.
5. Whether any Errored/Cancelled row counts against the quota.

Then click **View log** on both real submissions and record: `preprocessed N/N volumes`, one
`arm <name>: checkpoints used k/k` per torch arm, an `arm featc: feat model on N volumes` line, seven
`stage written after arm ...` lines, `done in Xs`, and any `ladder:` / `Traceback`.

**If the AUROCs differ** → the two runs did not ship the same predictions. The recorded "λ gained 0.0084" is
void, and the real question is what differed. Read the two logs side by side for differing `checkpoints used`
counts. Treat the *better-scoring* configuration as your base.
**If the AUROCs are identical and both logs are clean** → the ensembles were identical, the +0.0084 is not
explainable by anything in the pipeline, and one of the two recorded public scores is simply wrong. Trust the
ids you just transcribed over anything previously written.
**If a row is missing** → you have more slots than you thought. Re-read §0; you can afford a bracket.

---

## 2. The hard-core diagnostic (the only live route to top 10)

```
python scripts/diagnose_hard_core.py
```

88 scans carry ~47 % of your loss and are wrong under every representation tried. That was read as irreducible
label noise — but all the arms share the *same* preprocessing, so "wrong everywhere" could instead be one
shared upstream defect. The published check only showed the errors do not cluster by **fold**, and folds are
random, so it could not have seen a site effect. This script tests site, intensity regime, voxel spacing, grid
shape and registration fallback, scoring enrichment with a permutation test on the max statistic.

**If it prints `VERDICT: CONCENTRATED`** — a group carries the hard core. This is the good outcome and the only
thing that could plausibly reach top 10.
1. `python scripts/qc_grid.py` on 10 hard-core scans from the enriched group and 10 easy ones from the same
   group. Look for a striatum in the wrong place, a clipped field of view, or a reference region landing on
   skull or background.
2. If registration is at fault, check whether those scans took the registration fallback and whether the
   fallback is worse than what it replaced for that group.
3. Fix it **only** if you can do so without editing `datsc/` — anything in `datsc/` ships untested inference
   code, because `build_master_v3.py:126-128` packs the working tree. Re-cache the affected group, retrain the
   affected arms, re-gate. Require **≥ 0.003** nested; anything less is inside the pool's ±0.013 noise.
4. If the defect is real but too big to fix safely by Sept 15 18:00, **write it up and stop.** It is the most
   valuable thing you have found and it is the first item of your next run.

**If it prints `VERDICT: UNIFORM`** — the label-noise reading holds. Stop looking for accuracy *and* for
calibration (measured at 0.0002 headroom). Skip §5 entirely, go to §4 then §7, and ship what you have.

**If it exits with `FATAL: ... sites.csv not found`** — you are not on the training machine, or `cache_v4/` has
moved. Point it at the right folder with `--cache <path>`.

---

## 3. Calibration sweep (confirmation, not decision)

```
python scripts/sweep_calibration.py --src data/run8/artifacts_v3 --src data/master_pool/artifacts_v3
```

Read the **BREAK-EVEN** column, not just the nested column. The nested pool has no site shift, so it is
monotone in λ and will always argue for λ → 0; that prices only the upside. λ's job is the downside.

**Expected:** it confirms T = 0.90, λ = 0.01. Record the table in `RESULTS_LOG.md` and move on.
**If break-even says otherwise** — i.e. it would take more confidently-wrong scans than you would bet against —
you may move to λ = 0.005, which was the minimax cell. Do **not** ship λ = 0: it buys ≤ 0.0018 and removes the
last cap on a confident miss (5.2 nats → 9.21).

---

## 4. Size probe (before any GPU)

```
python scripts/build_master_v3.py --src data/run8/artifacts_v3 --src data/master_pool/artifacts_v3 \
    --lam 0.01 --T 0.9 --max-mb 1050 --zip data/submission_v9_probe.zip --no-e2e
```

**What you want:** per-arm checkpoint counts, a total size, and **no line containing `dropped`**.

**If it prints `FATAL: ... and DROP_ORDER is exhausted`** → raise `--max-mb`. A 1421 MB zip has already been
accepted by the platform (`id-321496`), so 1050 is conservative. Do not edit `DROP_ORDER`.
**If it prints `FATAL: the zip fitted only after trimming ...`** → the gate table scored a different set from
the one that would ship, and dropping an arm also renormalises the weights. Raise `--max-mb` and rebuild. Only
use `--allow-trim` if you deliberately accept the trimmed set *and* re-read the gate table for it.
**If it succeeds** → note the checkpoint count. Checkpoints are fp16 and per-arm: 3D ≈ 3.83 MB, `a2d_v2s`
≈ 37.1 MB, `a2d_cnx_cor` ≈ 51.2 MB. Neither size nor the 150-minute ladder binds (a 71-checkpoint run uses
~24 min of it); what limits you is that extra seeds have stopped paying.

---

## 5. GPU session — only if §2 said CONCENTRATED, or you simply have the hours

Set the `KAGGLE_USERNAME` / `KAGGLE_KEY` secrets on the notebook once (Edit → Add-ons → Secrets, skyfall23's
own values), and set Session options → **GPU T4 x2**, Internet On. Without the secrets a cut session loses
everything.

```
python kaggle/run8_distill/gen_run8.py --out kaggle/run9_seeds --slug dat-run9-seeds \
    --kernel-source skyfall23/dat-run8-distill \
    --hemisoft-seeds 0 1 2 3 4 --crophistsoft-seeds 0 1 2 \
    --skip r3d_hemisoft_site r3d_hemi_more r3d_hemimw r3d_hemisoft_full
kaggle kernels push -p kaggle/run9_seeds
kaggle kernels status skyfall23/dat-run9-seeds
```

20 jobs, ≈ 4.1 GPU-h. `--kernel-source` mounts the previous output so finished jobs print "already done".
The four skipped jobs write OOF files tagged `_site`/`_fall`, which `load_arm` filters out
(`build_master_v3.py:26`) — **their checkpoints can never ship**, so running them is pure waste.
Do **not** add more plain `a3d_hemi` seeds: seeds 3→5 bought 0.0001 nats for 10 checkpoints.

**Before you push, settle one thing:** the raw-scan Kaggle dataset is shared with `skyfall23`, and the rules
forbid sharing competition data outside your team. If that account is a second account of yours, fine. If it is
another person registered on your DrivenData team, fine. If it is another person *not* on the team, stop using
it and train on your own account.

**If the session dies** → `kaggle kernels status`; push again. With the secrets set it resumes; without them it
restarts from zero (~4.5 h).
**If it lands on 1×T4 instead of 2** → it still works, roughly twice the wall-clock. Recheck the Sept 15 18:00
stop.

```
python scripts/watch_kaggle_output.py --kernel skyfall23/dat-run9-seeds --out data/run9a
```

---

## 6. Re-gate

```
python scripts/build_master_v3.py --src data/run9a/artifacts_v3 --src data/run8/artifacts_v3 \
    --src data/master_pool/artifacts_v3 --lam 0.01 --T 0.9 --max-mb 1050 \
    --smoke data/smoke --zip data/submission_v9.zip
```

Compare the GATE TABLE against **0.2266** — what is banked — not against 0.2323.

**If the best set beats 0.2266 by ≥ 0.003** → this is your submission. Go to §7.
**If it beats it by less than 0.003** → that is inside the pool's ±0.013 bootstrap noise. Ship the banked
configuration instead; rebuild it as a *new* zip (you may not submit the same zip twice) at λ = 0.01, T = 0.9.
**If it is worse** → something regressed. Rebuild without `--src data/run9a/artifacts_v3` and ship that.
**If it says "missing/incomplete OOF"** → an arm has fewer than 5 folds and was skipped automatically; that is
fine. If *everything* is missing, a `--src` points at the wrong folder (it needs the one containing `oof_*.csv`).

---

## 7. Freeze, smoke, ship

**Sept 16 12:00 — freeze.**

```
git status --short                 # must be EMPTY: the zip packs the working tree
python -m pytest -q                # expect 99 passed (94 existing + 5... now 6 in test_run9)
python scripts/local_e2e_v3.py data/submission_v9.zip data/smoke
python -c "import zipfile,json;z=zipfile.ZipFile('data/submission_v9.zip');m=json.loads(z.read('artifacts/manifest.json'));print(m['T'],m['lam'],m['typ'],[(a['name'],len(a['checkpoints'])) for a in m['arms']])"
certutil -hashfile data/submission_v9.zip SHA256
```

Record that SHA256. The file you smoke-test must be the byte-identical file you upload.

The container check must print `FORMAT OK`, `failed volumes: 0`, and every arm N/N. It now also fails if any
manifest arm was silent, if a `ladder:` or failure line appeared, or if `typ` is set. The manifest must show
`typ` as `None` — `main_v3.py:64` would otherwise print statistics about the hidden test set, which the rules
make grounds for disqualification.

**If `git status` is not empty** → commit or stash first. Anything uncommitted in `datsc/` or `main_v3.py`
ships untested inference code.
**If `local_e2e_v3.py` fails** → read which assertion. `ARMS IN MANIFEST BUT SILENT IN OUTPUT` means an arm
died without saying so; `DEGRADED RUN` means the ladder fired or a checkpoint failed. Do not upload. Fall back
to the last zip that passed.

**Sept 16 13:00 — smoke test.** Upload the frozen zip as a **Smoke test** (free, 3/day, does not touch your
quota). Pass = within ~0.01 of **0.2374** (`id-321850`), the most recent smoke score that carries a real id.
Do not gate on the "0.2474" figure floating in `forharsh.md`; it has no id and no source.

**If the smoke score is far worse** → the model is not the problem; check the log for a `Traceback` or a
missing arm. **If it times out** → too many checkpoints for the 6-minute cap; the real run has 3 hours, so
this is a size signal, not a quality signal. Drop the weakest arm's highest seed and re-freeze.

**Sept 16 16:00 — the real submission.** Only after the smoke job shows Completed (one code job runs at a
time). Choose **Normal submission**. Note = what it is + its nested number. Transfer ~12 min, queue ~20 min,
score ~1 h → back around 17:32, leaving ~6.5 h of margin.

**Hard stops:** start by **20:27** to be safe even against a 3-hour-capped run; by **22:27** to still see the
score; **23:47** is the last moment the transfer can begin. After 20:27, skip the smoke test rather than skip
the submission.

Immediately after: write the id, UTC time, public score, AUROC and G = public − nested into `RESULTS_LOG.md`.

---

## 8. If everything breaks

| Symptom | Do |
|---|---|
| Zip will not build at all | Ship `data/submission_master_v2.zip` (732 MB, nested 0.2323, smoke 0.2381 `id-321786`). Worse than banked, but valid. |
| Container check fails and you cannot fix it | Ship the last zip that passed. Never upload one that did not print `FORMAT OK`. |
| DrivenData "one active code job" | Wait. Do not cancel the real job to run a smoke test. |
| Upload stalls mid-transfer | Do not reload the page during `Upload in progress`. Let it finish or start over from §7. |
| You realise you have no slots left | Nothing to do. The best private submission counts automatically; 0.2473 is already banked. |
| Laptop sleeps mid-upload | Keep it plugged in and awake. Re-upload from §7 if it dropped. |

**The thing that cannot go wrong:** your banked score survives everything. A worse final submission, a crashed
one, or an unused slot all leave 0.2473 standing, because the best private-set submission counts
automatically. The only real loss is the option itself.

---

## 9. After the deadline (Sept 17) — Completed

1. **[COMPLETED]** Write `docs/DISCLOSURE.md` — [docs/DISCLOSURE.md](docs/DISCLOSURE.md) drafted, verified, and committed (fulfilling Ticket T12). Covers: zero external data, BSD-3 torchvision ImageNet weights only, Kaggle compute venue, no-pseudo-labelling, per-scan inference isolation, and the complete verified 6-submission ledger.
2. **Purge the six files that reproduce the training labels** before any public/MIT release:
   `artifacts/featarm_v4/oof_*.csv`, `artifacts/featcarm_v5/oof_*.csv`, `artifacts/run4a_stack/oof_stack.csv`,
   `kaggle/run8_distill/soft_master_v2.csv`. Each holds 1,362 uids with `y` — a verbatim `train_labels.csv`.
   Harmless while the repo is private; a breach of "never share, copy, or publish the data" the moment it is not.
3. Delete the private Kaggle datasets and notebook outputs on both accounts.
4. Final competition results verified: **Rank #50** on Private Leaderboard (best Private Log Loss: **0.3003**, AUROC: **0.9406** via Slot 5 `id-322091`). See [RESULTS_LOG.md](RESULTS_LOG.md) and [aidlc-docs/HANDOFF.md](aidlc-docs/HANDOFF.md).
5. Read `LESSONS.md` / `LESSONS.pdf`. Then read it again before your next competition.

