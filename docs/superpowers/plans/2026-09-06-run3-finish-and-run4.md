# Run 3 finish + Run 4 (toward rank 1) Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking. Read `aidlc-docs/HANDOFF.md` first.

**Goal:** Ship the run-3 registered 3D + 2.5D + ROI-LR ensemble tonight (slot 2 of 3), then close the gap to the leader (public 0.219) with measured, gated additions over Sept 7–15.

**Architecture:** unchanged from run 3 (`datsc/volume.py` → `datsc/register.py` → arms in `datsc/models_v3.py` trained by `datsc/train_v3.py`; combiner `scripts/fit_stack_v3.py` → `manifest.json`; inference `main_v3.py`). Run 4 adds ensemble members and calibration rigor, not new infrastructure.

**Tech Stack:** Python 3.12, PyTorch/torchvision, SimpleITK, scikit-learn, Kaggle T4 x2 via Playwright-driven JupyterLab, DrivenData code-execution runtime (A100, 3 h).

---

## Part A — finish run 3 (tonight)

### Task A1: Wait for queue 1 and record the preliminary stack

**Files:** `RESULTS_LOG.md`, `aidlc-docs/efforts/002-run3-registered-ensemble/effort-state.md`

- [ ] Step 1: Run `.playwright-mcp/poll_runner.js` (Playwright MCP, `filename` param) until the live runner tail ends with `all done in … min; failures: []`.
- [ ] Step 2: Read cells 52/53 outputs. Expected: a per-run table, then `weights={'a3d': …, 'a2d_b0': …, 'a2d_v2s': …, 'lr': …} T=… stack OOF: AUROC=… logloss=…`, `wrote submission_v3.zip: N checkpoints across 4 arms, ~650 MB`, and `FORMAT OK … smoke-test log loss = …`, `EXTRAPOLATED 2,000 volumes: prep ~… min + models ~… min`.
- [ ] Step 3: Append the numbers to `RESULTS_LOG.md` (per-arm mean fold logloss/AUROC, stack OOF, T, weights); commit.

### Task A2: Quick Save (insurance)

- [ ] Step 1: In the Kaggle editor click "Save Version" → "Quick Save" → Save (Playwright: `page.getByRole('button', {name: /Save Version/})`). Purpose: keep `artifacts_v3/` reachable if the session dies.

### Task A3: Stale re-run, queue 2, TTA re-OOF, final stack

- [ ] Step 1: If it is later than 14:30 UTC, edit cell 49 (queue 2) and remove the `r3d_full` entry (keeps ~25 min).
- [ ] Step 2: Select cell 47 → `notebook:run-all-below`. Watch with `poll_runner.js` (it reports the `[*]` runner).
- [ ] Step 3: When cell 52 finishes: expect `re-OOF` lines like `r2d_efficientnet_b0_f2_s0: logloss 0.2787 -> 0.27xx auroc … tta_z=(-1, 0, 1)` and a stack line with `+3 extra checkpoints` for a2d_b0 (and a3d if full 3D ran). The stack OOF should be ≤ the preliminary one. If TTA made an arm worse, re-run `reoof_v3.py --tta-z 0` for that arm's pattern (`--only "r2d_efficientnet_b0_*"`) and re-fit.
- [ ] Step 4: Cell 53 (e2e) must print `FORMAT OK` and an extrapolated total well under 180 min; cell 54 prints `md5 … size …` and the parts list.

### Task A4: Download, join, verify

- [ ] Step 1: Run `.playwright-mcp/download_parts.js`. Check `data/parts_v3/` and `.playwright-mcp/` for `submission_v3.zip.partNN` (each < 95 MB; move any from `.playwright-mcp/`).
- [ ] Step 2: `python scripts/join_parts.py data/parts_v3 <md5> --out submission_v3.zip` → expect `md5 … OK` and `testzip: None`.
- [ ] Step 3: Local sanity: unzip to a temp dir and run `python main.py --selftest` there (expects `selftest: template used` + `SELFTEST OK`).

### Task A5: Smoke test, then real submission

- [ ] Step 1: `python kaggle/gen_submit.py H:/augsepthacks/parkinsonnn/submission_v3.zip "run3: registered frame, 3D+B0+V2S+LR, TTA, T=…" smoke` → run `.playwright-mcp/dd_submit.js`. Poll the submissions page until `Completed`; open the log; verify every arm line and `exit code: 0`.
- [ ] Step 2: Same with `normal`. Record `id-…`, public log loss, AUROC, rank in `RESULTS_LOG.md`, effort 002, memory. Slot accounting: slots used 2 of 3; slot 1 returns Sept 12 15:38 UTC, slot 2 returns 7 days after this submission.

---

## Part B — run 4: measured additions toward 0.219 (Sept 7–15)

Budget: Kaggle ~17 GPU-h left this week (resets Sat), 30 h next week. Every addition is gated on the **nested** OOF log loss of the stack (fit weights/T on 4 folds, score the 5th, rotate) improving by ≥ 0.005; report per-pseudo-site numbers each time.

### Task B0: Dated schedule (added Sept 6 10:40 UTC)

| When | What |
|---|---|
| Sat Sept 6 night | Commit Version #2 → download → smoke → real submission 2 |
| Sun Sept 7 morning | B1 (below): nested estimate, per-site table, weak-site analysis (2.46 mm ultra-low, 43-slice), label-noise bound, EMA-lag guard in the report cell |
| Sun Sept 7 afternoon–night | B2 diversity members + B3 3D upgrades as chained commits (per-job uploads on) |
| Mon Sept 8 | B4 seeds + full-data; nested stack; pack; e2e |
| Tue Sept 9 | Real submission 3 (slot 3) |
| Fri Sept 12 15:38 UTC | Slot 1 returns → submission 4 (Sept 13) with B5 if gated in |
| Sept 15–16 | Final submissions; last one as a prevalence/temperature bracket of the best weights |

### Task B1: Honest nested estimate + diagnostics (CPU, local, 1 h)

**Files:** Create `scripts/nested_stack_v3.py`; Modify `scripts/fit_stack_v3.py` (import the shared combiner).

- [ ] Step 1: Download all `oof_*.csv` + `*.json` from Kaggle (small: base64 route or the `files/` trick on a tgz < 95 MB) into `artifacts/run3_YYYY-MM-DD/`.
- [ ] Step 2: Write `nested_stack_v3.py`: for k in folds: fit simplex weights + T on the other folds' OOF rows, apply to fold k; print mean nested logloss, and per-site (join `cache_v4/sites.csv`) logloss/AUROC/mean-p vs prevalence.
- [ ] Step 3: Compare nested vs plain OOF; if the gap > 0.01, reduce combiner freedom (equal weights) for the next submission.
- [ ] Step 4: Hard-fold analysis: fold 1 is worst for both arms — tabulate its site mix and the worst 30 scans (uid, site, p, y); view their slabs (`qc/slabs_v4.png` style) to check registration/QC.
- [ ] Step 5: Weak pseudo-sites (2.46 mm ultra-low: 61 scans, 89 % abnormal, ROI-LR AUROC 0.77; 3.90 mm 43-slice: 35 scans, 0.79): per-site registration QC (`ncc_reg`, `reg_scale`), mean-p vs prevalence, slab QC images. Candidate fixes to gate: stronger quantisation/blur for ultra-low, oversampling of weak sites, reference-region check on thin FOVs.
- [ ] Step 6: Label-noise bound: count scans where every seed of both arms is > 0.95 confident against the label; list uid/site; treat as the practical floor.
- [ ] Step 7: EMA-lag guard: in the report cell, flag any checkpoint whose `val_logloss` exceeds the last `val_raw_logloss` by > 0.02 (from `history`).

### Task B1b: Items from `docs/RESEARCH_SIMILAR_COMPETITIONS.md` (added Sept 6 11:30 UTC)

- [ ] Augmentation strength ablation: "unrealistic" Gaussian blur + noise (Buddenkotte & Buchert 2024 lifted cross-camera accuracy 0.953 → 0.975); ours caps blur at σ 4 mm / noise 15 % — try σ up to 8 mm and noise 30 % on fold 0/1 of both arms (2 × 2 runs, ~1 h). Keep if fold loss improves ≥ 0.01.
- [ ] Hamburg-recipe arm (Budd 2024): single 12 mm slab at 2 mm, 72×72 central crop, ~200k-param ResNet from scratch, 5 seeds × 5 folds (each run a few minutes on a T4). New arm `a2d_tiny`; needs `--n-slabs 1 --slab-mm 12 --hw 72` and a tiny 2D net in `datsc/models_v3.py`. Cheap, historically the strongest DaT recipe (98 % / 94 % / 98 % on three test sets).
- [ ] Uncertainty hedging in the combiner: shrink scans where the arms disagree strongly (e.g. spread of arm logits > threshold fitted on OOF) toward the prior; Hamburg's ensemble flags 4.3 % of scans and catches 90 % of its errors there. Gate on nested OOF (est. −0.005 to −0.02).
- [ ] Clip final probabilities to [0.02, 0.98] and evaluate on OOF (winners' log-loss playbook); ship the better of clipped/unclipped.
- [ ] TTA caution: BatchNorm backbones can lose accuracy under shifted inputs; the re-OOF step measures the z-shift TTA per arm — keep it only where it helps, otherwise `--tta-z 0`.
- [ ] Submission strategy: public LB is noise at this test size and the best *private* submission counts automatically → make each weekly slot a *diverse* variant (e.g. different T / hedging / member set), not a re-run.

### Task B2: Diversity members (GPU ~4 h)

Add jobs to `JOBS2` in `kaggle/gen_run3_inject.py`, regenerate, `python kaggle/push_cells.py`, run the queue-2 cell, then re-OOF → stack → pack → e2e → split.

- [ ] Step 1: 2.5D ConvNeXt-Tiny (weights in the dataset): `--backbone convnext_tiny --weights pretrained/convnext_tiny_in1k.pt --hw 224 --batch-size 24 --backbone-lr 1e-4 --ema 0.99`, 5 folds x 1 seed (~50 min). Arm name `a2d_cnx`.
- [ ] Step 2: 2.5D B0 with 5 slabs x 6 mm (`--n-slabs 5 --slab-mm 6`, stem adapted to 5 channels), 5 folds x 1 seed (~40 min). Tag collides with B0 — pass `--tag r2d_effb0_s5_f{f}_s{s}` (runner needs a `tag` field per job; add it).
- [ ] Step 3: Coronal V2-S, 5 folds x 1 seed (~50 min) if the coronal B0 arm got non-zero stack weight.
- [ ] Step 4: Gate each member: keep only if the nested stack improves ≥ 0.005 or its weight ≥ 0.1.

### Task B3: 3D arm upgrades (GPU ~4 h)

- [ ] Step 1: Full-resolution 3D (`--size 96 96 72`, batch 8; ~2.5x slower) fold 0 + fold 1 only, compare with the 64³ folds (0.274 / 0.334). Keep if ≥ 0.01 better on both.
- [ ] Step 2: 3D with 80 epochs and `--ema 0.997` fold 0/1 (checks under-training).
- [ ] Step 3: 3D multi-shift TTA: add `tta_shift` to `InputSpec.variants` for the 3D arm (`torch.roll` ±1 voxel in x and y after resize; 5 variants x flip). Re-OOF and gate.

### Task B4: More seeds + full-data models for the kept arms (GPU ~3 h)

- [ ] Step 1: Seeds 3–4 for B0 and 3D; full-data models (fold -1) 3 seeds for each kept arm (`--extra` in the stack cell picks them up).
- [ ] Step 2: Re-fit the stack; nested estimate; pack; e2e; split; download.

### Task B5: Registration refinement (CPU local, gated)

- [ ] Step 1: Build the template from all 1,362 scans (3 iterations) and re-register (`scripts/build_template.py --subset 1362 --n-iter 3`, `scripts/register_cache.py`); gate G1 (`scripts/gate_v3.py`) must beat 0.902 by ≥ 0.01 before re-uploading `cache_v5` and retraining anything.
- [ ] Step 2: Only if G1 improves: new Kaggle dataset version, retrain the two main arms (1 seed) and compare fold means.

### Task B6: Submission cadence

- [ ] Sept 9–10: submission 3 (best nested stack from B1–B4). Smoke first.
- [ ] Sept 12 15:38 UTC: slot 1 returns → submission 4 on Sept 13 with B5/B3 results if gated in.
- [ ] Sept 13–14: slots 2/3 return → final submissions Sept 15 (best model) and Sept 16 morning (same weights, alternative λ/T bracket, since the best private score counts automatically).
- [ ] After the competition: delete the Kaggle datasets and notebooks' data copies; write the MIT-release README + disclosure (torchvision ImageNet weights, Kaggle training venue).
