# Handoff: resume here

| | |
|---|---|
| **Last updated** | 2026-09-10 17:45 UTC |
| **Deadline** | 2026-09-16 23:59 UTC; best submission on the **private** set counts automatically (organiser statement) |
| **Goal (user)** | as high as possible; honest reach: rank ≈ 12–20 (top 10 needs public 0.2379; rank 1 = 0.2166 is out of reach with this data) |
| **Current public score** | **0.2667** log loss, AUROC 0.9536, rank 44 (run 4a, id-321298, Sept 7). Leader 0.2166, #10 0.2379, #15 0.2457 |
| **Real submissions** | 3 of 3 used (Sept 5 15:38, Sept 6 15:50, Sept 7 03:38 UTC); they return **Sept 12 15:38, Sept 13 15:50, Sept 14 03:38 UTC** |
| **Best zip on hand** | `data/submission_master_v2.zip` (732 MB): 5 pillars, equal weights, T 0.9, all seeds; nested **0.2323**; DrivenData smoke **0.2381** (id-321786) |
| **In flight** | Run 8 on the teammate's account: `skyfall23/dat-run8-distill`, Version 1, started Sept 10 16:33 UTC, 42 jobs, ETA ≈ 21:00–21:30 UTC. **No upload secrets** on that notebook: nothing is saved until it completes. |
| **Solo playbook** | `forharsh.md` / `forharsh.pdf` (repo root) — the short, ordered version of this file |

Reading order: this file → `forharsh.md` → `efforts/007-run8-distill-master-v3/effort-state.md` → `docs/superpowers/plans/2026-09-10-run8-distill-plan.md` → `RESULTS_LOG.md`.

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

| Slot (UTC) | Zip | Condition |
|---|---|---|
| Sept 12 15:38 | `submission_master_v3.zip`, else `submission_master_v2.zip` | v3 only if gate PASS and smoke ≤ 0.2481 |
| Sept 13 15:50 | the other of v2/v3 (v3 if nested < 0.2323); else `submission_master_v2_lam01.zip` | never the same zip twice |
| Sept 14 03:38 | λ 0.01 bracket of the slot-1 zip, or the session-2 zip if it gated better | container check first |

Every upload: note = what it is + nested number; write the score and id into `RESULTS_LOG.md`.

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
