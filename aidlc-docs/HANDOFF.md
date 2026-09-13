# Handoff: resume here

| | |
|---|---|
| **Last updated** | 2026-09-13 10:00 UTC |
| **Deadline** | 2026-09-16 23:59 UTC; best submission on the **private** set counts automatically (organiser statement) |
| **Goal (user)** | as high as possible; honest reach: rank ≈ 15–20 (top 10 needs public 0.2379; rank 1 = 0.2166 is out of reach with this data) |
| **Current public score** | **0.2473** log loss, **Rank ~#16** (Master v3 lam=0.01, Slot 2 evaluated Sept 13; climbed +7 places). Leader 0.2166, #10 0.2379, #15 0.2457 |
| **Real submissions** | 5 used. **One slot left**: it opens Sept 14 03:38 UTC and stays open until the deadline — the next expiry after that is Sept 19 |
| **Best zip on hand** | `data/submission_master_v2.zip` (732 MB, nested **0.2323**) — note this is *worse* than the banked 0.2266; it is a diversification bet, not an upgrade |
| **Completed** | Master v3 bracket delivered and verified on public LB (0.2473, Rank ~#16) |
| **Now (effort 008)** | Endgame: **hold the last slot, ship late.** Freeze Sept 16 12:00, smoke 14:00, upload ~16:00. See `efforts/008-run9-endgame/effort-state.md` |
| **GPU** | **30+ GPU-hours remain** (the "≈10 GPU-h" figure below and in `forharsh.md` §3 is stale) |
| **Solo playbook** | `forharsh.md` / `forharsh.pdf` (repo root) — the short, ordered version of this file |

Reading order: **`RUNBOOK.md`** (the complete solo procedure, with every branch) → this file →
`docs/superpowers/plans/2026-09-13-run9-endgame-plan.md` → `efforts/008-run9-endgame/effort-state.md` →
`RESULTS_LOG.md`. After the deadline: `LESSONS.md` / `LESSONS.pdf`.

> **Three corrections to the plan below, from effort 008 (Sept 13).** The sections after this box were written
> assuming the last slot is spent on Sept 14 and that no training time remains. Both assumptions are wrong.
>
> 1. **The slot opens; it does not expire.** It is live until Sept 16 23:59. Submitting at 03:38 on Sept 14
>    throws away ~60 h of runway and spends the last option on a zip (nested 0.2323) worse than what is banked
>    (0.2266). Hold it. Freeze Sept 16 12:00.
> 2. **Do not chase λ any further.** λ 0.02→0.01 is capped at 0.0055 nats by arithmetic and is worth 0.0020 on
>    the real pool; the +0.0084 credited to it in `RESULTS_LOG.md` is impossible, so something else differed
>    between those two submissions — most likely the ladder dropping arms. λ 0.01→0 buys ≤0.0018 and removes
>    the last cap on a confident miss (9.21 nats vs 5.2). See `docs/adr/0006-the-prior-mix-is-insurance-not-a-lever.md`.
> 3. **Size is the binding constraint, not GPU.** 847 MB against `--max-mb 850` is ~3 MB of headroom. Any new
>    seed triggers `DROP_ORDER`, which deletes `a2d_v2s` seed 1 first — worth ≈0.01, five times what a seed
>    returns. Measure the envelope at `--max-mb 1050` *before* launching any training. Note checkpoint size is
>    **per-arm**, not an average: fp16 3D checkpoints are ≈3.83 MB each, but `a2d_v2s` is ≈37.1 MB and
>    `a2d_cnx_cor` ≈51.2 MB. So 1,050 MB buys ~52 more 3D checkpoints but only ~5 v2s ones. Neither size nor
>    the ladder actually binds (a 1421 MB zip already smoke-scored, `id-321496`; a 71-ckpt run uses ~24 min of
>    the 150-min ladder). What caps the spend is **diminishing returns** — `a3d_hemi` seeds 3→5 bought 0.0001
>    nats for 10 checkpoints — so add ~20 checkpoints where they still pay and stop.

---

## 1. Where things are

| Thing | Location |
|---|---|
| Repository (source of truth) | `H:\augsepthacks\parkinsonnn`, GitHub `HarshdipSaha/parkinsonnn`, branch `master` |
| Competition | https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/ — submissions page `.../submissions/` (log in as harshdipsaha in the Playwright/Chrome window) |
| Kaggle CLI on this laptop | `~/.kaggle/kaggle.json` is authenticated as **skyfall23** (the teammate); `kaggle kernels list --mine` lists their notebooks |
| Kaggle notebooks | `skyfall23/dat-run8-distill` (running); earlier: `skyfall23/dat-run7-hemisphere`, `dat-run6b-triplanar`, `dat-run6c-features-ssl`; `harshdipsaha/dat-parkinsons-train` (runs 3–5) |
| Kaggle datasets (private, owned by harshdipsaha, shared with skyfall23) | `dat-parkinsons-challenge-data` (raw), `dat-parkinsons-cache-v4` (registered 2 mm cache + template + torchvision weights), `dat-run3-checkpoints` (every checkpoint/OOF of runs 3–7) |
| Local data | `cache_v4/` (registered cache, folds.csv, folds_site.csv, sites.csv); `data/smoke/` (20 labelled smoke scans: validation only, never fitted on — ADR 0005); `data/master_pool/artifacts_v3` (all OOF csvs + manifest, no .pt); `data/ckpt_ds_v4`, `data/kaggle_output_v5`, `data/run6b`, `data/run6c` (checkpoints); zips `data/submission_*.zip` (run 5, run 7, master_final, master_v2) |
| Soft-target table (run 8) | `kaggle/run8_distill/soft_master_v2.csv` (uid, y, p_nested, loss, fold; 1362 rows; built by `scripts/make_soft_targets.py`) |
| Results and decisions | `RESULTS_LOG.md` (every number; Sept 10 sections at the end), `aidlc-docs/audit.md`, `docs/wayfinder/map.md` + tickets, `docs/adr/` |

## 2. What was done on Sept 10 (effort 007)

1. **Analysis of the master pool** (`artifacts_dev/stack_analysis.json`): equal weights beat fitted weights in nested CV; Master Final's seed-0-only pruning cost ≈ 0.011 on the V2-S arm; the leave-site-out pool prefers T ≈ 0.83 (the old floor 1.0 hurt by 0.005). → **Master v2** built with `scripts/build_master_v2.py` (equal weights, T 0.9, λ 0.02, all seeds). Local container check 0.2078 on the smoke set; nested (T floor 0.9) 0.2323.
2. **Ledger verified live**: exactly three real submissions exist; the "run 5 public 0.2600 id-321520" line another session wrote was fabricated and is corrected in `RESULTS_LOG.md`.
3. **Master v2 smoke test** id-321786 = **0.2381** (Master Final 0.2363: tied within the measured 0.009 same-zip noise). Master v2 is the slot-1 fallback.
4. **Run 8 designed and launched** (spec, plan, ticket T15): Master v2's nested loss is concentrated (top-100 scans = 54 %, 46 confident mistakes); run 7's self-bootstrap loss did nothing (0.2565 vs 0.2568), so the soft targets come from the ensemble: `y* = 0.6·y + 0.4·p_nested`, plus a mistake-weighted variant (`w = 1 + 2·loss_nested`, mean 1) and hemisphere seeds 3–4. Trainer knobs `--soft-csv/--soft-lam/--mistake-w`; validation always on hard labels; 94 tests green.
5. **External data re-checked on request**: no open DaT-SPECT dataset exists outside PPMI (ruled out: DUA + organiser); MedicalNet (MIT) is legal but CT/MRI-pretrained with a different architecture → not pursued. Prize verification is post-hoc and silent; a compliant, disclosed entry is what benefits from any reshuffle.

## 3. Runbook after run 8 (copy-paste)

```
kaggle kernels status skyfall23/dat-run8-distill
python scripts/watch_kaggle_output.py --kernel skyfall23/dat-run8-distill --out data/run8
python scripts/build_master_v3.py --src data/run8/artifacts_v3 --src data/master_pool/artifacts_v3 --smoke data/smoke --zip data/submission_master_v3.zip
```
`build_master_v3.py` prints the GATE TABLE (nested: Master v2 as shipped; pillars + each run-8 arm; + all; greedy best subset; equal/shrunk/fitted weights), decides PASS/FAIL (gain ≥ 0.003 over 0.2323), packs the best set with every seed, equal weights, T 0.9, trims below 850 MB, and runs `scripts/local_e2e_v3.py` (must print `FORMAT OK`, every arm N/N). Then the DrivenData smoke test by hand (section 5 of `forharsh.md`).

**Decision rule (plan §8, written before the run):**
- v3 replaces v2 in slot 1 **iff** nested(v3) ≤ 0.2293 **and** platform smoke(v3) ≤ 0.2481 (= 0.2381 + 0.010).
- Otherwise v2 goes first; v3 second if it is below 0.2323 at all; else the λ 0.01 bracket of v2.
- Slot 3: the λ 0.01 bracket of the slot-1 zip (`build_master_v2.py --lam 0.01` or `build_master_v3.py --lam 0.01`), unless a second GPU session produced a better-gated zip.

**Second GPU session** (≈ 10 GPU-h left; only if an arm gated in; resumes from run 8's own output, no secrets needed):
```
python kaggle/run8_distill/gen_run8.py --kernel-source skyfall23/dat-run8-distill --hemisoft-seeds 0 1 2 3 4 --skip r3d_hemimw r3d_crophistsoft
kaggle kernels push -p kaggle/run8_distill
```

## 4. The three submissions

| Slot (UTC) | Zip | Condition / Status |
|---|---|---|
| Sept 12 | `submission_master_v3.zip` | **COMPLETED**: Public **0.2557**, AUROC **0.9576**, **Rank #23** (climbed +21 spots from #44; G=+0.0268) |
| Sept 13 03:08 | `data/submission_master_v3_lam01.zip` | **COMPLETED**: Public **0.2473**, **Rank ~#16** (climbed +7 spots from #23; G=+0.0207). **id not recorded** — transcribe it |
| Sept 14 03:38 — **Sept 16 23:59** | the best zip that gates above **0.2266**; `data/submission_master_v2.zip` only as a last resort | **SUPERSEDED by effort 008**: the slot opens Sept 14 but is spent Sept 16 ~16:00 after the freeze, smoke test and container check. Master v2 (nested 0.2323) is *worse* than what is banked, so it is a diversification bet and not the default. See `docs/superpowers/plans/2026-09-13-run9-endgame-plan.md` |

Every upload: note = what it is + nested number; write the score and id into `RESULTS_LOG.md`.

## 4b. After each real submission: read the score, then run exactly this

The public score comes back ≈ 1 h after the upload. For the zip you submitted you know its nested estimate N (from the gate table; Master v2 = 0.2323). Compute the **gap G = public − N** and write both into the ladder table in `RESULTS_LOG.md`. Runs 3 / 4a had G = 0.042 / 0.018. Nothing else about the test set is visible, so G is the only feedback you get; use it as below. Start the GPU session of step B the same hour the slot-1 score arrives; do not wait for slot 2.

**After slot 1 (score ≈ Sept 12 16:40 UTC) — pick the first case that matches:**

1. **Public ≤ 0.2379 (top 10).** Change nothing. Slot 2 = the other zip (v2 or v3), slot 3 = the λ 0.01 bracket of the slot-1 zip (§4). Still launch step B so slot 3 can use more seeds if they arrive in time.
2. **Public worse than 0.2667 (our old best).** The zip broke on the platform, not the model. On the submissions page click *View log* for that id and look for `checkpoints used N/N` per arm and any `Traceback`. Slot 2 = the zip that did NOT go first (v2 if v3 broke); if v2 itself broke, slot 2 = `data\submission_master_final.zip` (486 MB, platform smoke 0.2363, id-321617 — the proven fallback; the run-4a zip is no longer on disk). Fix the packaging before slot 3 (`python scripts/local_e2e_v3.py <zip> data/smoke` must print `FORMAT OK`).
3. **G ≤ 0.022 (gap like run 4a or smaller).** The nested estimate is trustworthy: chase it. Run step B; slot 2 = the step-B zip if its gate table beats the slot-1 zip by ≥ 0.002 nested and its smoke test is within 0.01 of the slot-1 zip's smoke; otherwise slot 2 = the other of v2/v3. Slot 3 = the λ 0.01 bracket of the better of the two public scores.
4. **G > 0.03 (shift dominates).** Confident mistakes are being punished; hedge the calibration. Slot 2 = the **shift hedge** of the slot-1 zip (same models, T 1.0, λ 0.03 — less confident):
   ```
   python scripts/build_master_v2.py --T 1.0 --lam 0.03 --out data/submission_master_v2_hedge.zip
   python scripts/local_e2e_v3.py data/submission_master_v2_hedge.zip data/smoke
   ```
   (or the same with `python scripts/build_master_v3.py --src data/run8/artifacts_v3 --src data/master_pool/artifacts_v3 --T 1.0 --lam 0.03 --smoke data/smoke --zip data/submission_master_v3_hedge.zip` if v3 went first). Smoke-test it first (free). Still run step B: more seeds also reduce variance under shift, and feed slot 3.

**Step B — the second GPU session (≈ 2.5–3 h on skyfall23; the quota bar on the notebook page shows what is left).** Choose ONE line according to run 8's gate table, then push:

```
# (i) the soft-target hemisphere arm gated in (a3d_hemisoft in the best set):
python kaggle/run8_distill/gen_run8.py --kernel-source skyfall23/dat-run8-distill --hemisoft-seeds 0 1 2 3 4 --skip r3d_hemimw r3d_crophistsoft
# (ii) the mistake-weighted arm gated in instead (a3d_hemimw in the best set):
python kaggle/run8_distill/gen_run8.py --kernel-source skyfall23/dat-run8-distill --hemimw-seeds 0 1 2 --skip r3d_crophistsoft
# (iii) nothing from run 8 gated in: more plain hemisphere seeds (the safe gain):
python kaggle/run8_distill/gen_run8.py --kernel-source skyfall23/dat-run8-distill --hemi-more-seeds 3 4 5 6 --skip r3d_hemisoft r3d_hemisoft_site r3d_hemimw r3d_crophistsoft r3d_hemisoft_full
kaggle kernels push -p kaggle/run8_distill
```
`--kernel-source` mounts run 8's own output, so every job already trained prints "already done" and only the new seeds run (10 jobs ≈ 2.5 h). Watch with `kaggle kernels status skyfall23/dat-run8-distill`. When COMPLETE:
```
python scripts/watch_kaggle_output.py --kernel skyfall23/dat-run8-distill --out data/run8b
python scripts/build_master_v3.py --src data/run8b/artifacts_v3 --src data/run8/artifacts_v3 --src data/master_pool/artifacts_v3 --smoke data/smoke --zip data/submission_master_v3b.zip
```
Read its GATE TABLE against the zip you already submitted (not against 0.2323): use `submission_master_v3b.zip` only if it is ≥ 0.002 nested better than that zip, and only after a free smoke test.

**After slot 2 (score ≈ Sept 13 16:50 UTC), 10 h before slot 3 — no training fits any more, only assembly:**

1. Write the score and G into `RESULTS_LOG.md`. You now have two public scores; the zip with the better one is the **base**.
2. If the step-B zip exists, was smoke-tested, and beat the base's nested by ≥ 0.002 → slot 3 = that zip.
3. Otherwise slot 3 = the λ 0.01 bracket of the base (`--lam 0.01`, section 5), unless case 4 applied above and the hedge scored better than its parent — then slot 3 = the hedge of the other zip (`--T 1.0 --lam 0.03`).
4. Container check (`FORMAT OK`, all arms N/N) before every upload; never the same zip twice; the best of the three on the private set counts.

**After slot 3 (Sept 14):** nothing more can be submitted before the deadline. Write the final ladder into `RESULTS_LOG.md`, then `forharsh.md` §8.

## 5. Gotchas (do not rediscover)

| Area | Gotcha | Do this instead |
|---|---|---|
| Kaggle | A notebook without `KAGGLE_USERNAME`/`KAGGLE_KEY` secrets does not upload checkpoints; a cut session loses everything | Add the secrets once (skyfall23's own), or attach the previous version's output as a kernel source (`gen_run8.py --kernel-source ...`) so the next push resumes |
| Kaggle | `kaggle kernels push` only says "GPU"; the accelerator may land on one T4 or a P100 | Edit → Session options → GPU T4 x2, Internet On, then push again; the runner uses however many GPUs it gets |
| Kaggle | Interactive sessions die and take `/kaggle/working` with them | Only background commits (push); never run cells by hand |
| Kaggle | `kaggle datasets version` on Windows must run inside the folder; `--dir-mode skip` | see `scripts/watch_kaggle_output.py` for the paging download |
| DrivenData | Upload of 700–850 MB takes ≈ 12 min; the page shows `Upload in progress N%`; one active code job at a time; smoke tests 3/day | Use the site by hand; Playwright `browser_file_upload` works, `setInputFiles` and network listeners crashed the MCP server earlier |
| Stacking | Fitted simplex weights over-fit (nested 0.2370 vs equal 0.2354); seed-0 pruning costs ≈ 0.01; T floor 1.0 hurts under shift | Equal weights, all seeds, T 0.9 (`build_master_v2/v3.py`) |
| Smoke set | 20 scans; the same zip scored 0.2176 and 0.2263 | Differences < 0.01 are noise; decide by the nested estimate; never fit on it (ADR 0005) |
| Folds | Canonical folds: `cache_v4/folds.csv` (random) and `cache_v4/folds_site.csv` (leave-pseudo-site-out) | Soft targets come from random-fold OOF → site-fold read-outs of soft arms are slightly optimistic; gate on random folds only |
| Windows | Heredocs mangle backslashes; `artifacts/` and `data/` are gitignored | Small tables that must be committed live under `kaggle/<run>/` |
| Rules | 3 real submissions per rolling 7 days; smoke 3/day; no pseudo-labelling/transductive use; per-sample inference; ImageNet torchvision weights OK (disclose); PPMI / DINOv3 / non-commercial weights not | `docs/DISCLOSURE.md` after the deadline (T12) |
| Log hygiene | One session wrote a fabricated public score into `RESULTS_LOG.md` | Every public/smoke number must carry a DrivenData submission id that exists on the submissions page |

## 6. Skills and tools used

Research: deep-research, brainstorming, llm-council, EDA skills, web search (Sept 10: external data / disqualification policy). Engineering: to-spec, writing-plans, ai-dlc, wayfinder, pytest (`tests/` — 94 tests). Automation: Playwright MCP (Kaggle JupyterLab API, DrivenData uploads), Task Scheduler watcher, Kaggle CLI.
