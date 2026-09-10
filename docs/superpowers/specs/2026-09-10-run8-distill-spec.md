# Spec: Run 8 — ensemble-corrected members and the final blend (Master v3) (effort 007, wayfinder ticket T15)

*Written Sept 10 2026 ≈ 20:30 UTC in the `to-spec` synthesis format. Companions: `docs/superpowers/plans/2026-09-10-run8-distill-plan.md` (implementation plan), `kaggle/run8_distill/APPROACH.md` (design and evidence), `kaggle/run8_distill/HOWTO.md` + `HOWTO.pdf` (one-command launch on the teammate's account). Supersedes ticket T13 (bootstrap loss → nested-OOF soft targets), which it implements.*

## Problem Statement

Our best submission scores 0.2667 public (rank 44). The best zip we hold, Master v2 (five pillars, equal weights, T 0.9), has a nested out-of-fold estimate of 0.2323 and a platform smoke score of 0.2381; it is booked for the Sept 12 15:38 UTC slot and projects to ≈ 0.25 public at the measured shift gap. Rank 10 needs 0.2379 public, rank 1 needs 0.2166. The ensemble's loss is concentrated: 100 of 1,362 training scans (7 %) carry 54 % of its nested loss, 94 scans lose more than 1.0 nat, and 46 scans are confidently wrong (probability more than 0.8 on the wrong side) although five diverse arms agree on them. Every member so far was trained on the hard labels of exactly these scans; the one label-noise treatment tried (self-bootstrapped targets, run 7) changed nothing (0.2565 vs 0.2568). The teammate's Kaggle account has ≈ 20 GPU-hours left and there is time for one training session before the Sept 12 slot and one more before the Sept 13 slots.

## Solution

Run 8 trains members that learn from the ensemble's own out-of-fold beliefs instead of the raw labels alone (nested-OOF soft targets; a variant that additionally up-weights the scans the ensemble gets wrong), adds the cheap safe gain of extra hemisphere seeds, and assembles a final blend, Master v3, from the Master v2 pillars plus whichever new arms pass the nested gate. Everything ships as one Kaggle notebook launched with one command from the teammate's account, resuming from and uploading to the shared checkpoint dataset, and one laptop command that assembles, gates and packs the zip and runs the container-style check. Honest expectation: nested ≈ 0.226–0.229 (from 0.2323), public ≈ 0.245–0.250, rank ≈ 15–20. Rank 1 is not reachable with these training scans and this shift gap; the spec says so rather than promising it.

## User Stories

1. As the competitor, I want members trained on soft targets that blend each training scan's label with the ensemble's nested out-of-fold probability, so that the 46 confidently mislabeled-or-ambiguous scans stop dragging every member toward memorising them.
2. As the competitor, I want the soft-target mix λ to be a command-line knob with a default of 0.4 (the range the literature supports), so that a second session can bracket it without code changes.
3. As the competitor, I want a variant that up-weights each training scan by the ensemble's nested loss on it (mean weight 1, so the learning rate stays comparable), so that the "correct the ensemble's mistakes" hypothesis is measured, not assumed.
4. As the competitor, I want both variants to fall back to the hard label and unit weight for any scan missing from the soft-target file, so that full-data (fold −1) jobs and future caches never crash on a missing uid.
5. As the competitor, I want validation metrics always computed against the hard labels, so that soft targets never make the out-of-fold numbers look better than they are.
6. As the competitor, I want the soft-target table (uid, nested probability, nested loss) embedded in the notebook itself, so that the notebook is one file and one command and needs no extra dataset.
7. As the competitor, I want the soft targets to come from Master v2's nested predictions (weights and temperature refit per fold, never on the scan being predicted), so that the targets carry no leakage of the scan's own label through the ensemble.
8. As the competitor, I want the hemisphere member (best single arm, OOF 0.2379) to be the primary carrier of the new losses, with three seeds on random folds and one seed on leave-pseudo-site-out folds, so that the site-pool temperature can be fitted for the new arm and its shift behaviour read next to the plain hemisphere member.
9. As the competitor, I want two more plain hemisphere seeds (seeds 3 and 4), so that the safe seed-averaging gain (≈ 0.003) lands even if the soft-target arms fail the gate.
10. As the competitor, I want one seed of the histogram-standardised crop-3D member trained on soft targets, so that the mechanism is tested on a second architecture, as run 7 did for DANN/DRO.
11. As the competitor, I want two full-data soft-target hemisphere models appended through the extras mechanism, so that the shipped arm uses every training scan.
12. As the competitor, I want the jobs ordered longest-first and resumable from the checkpoint dataset with an upload every four jobs, so that a 12-hour cap or a cancelled session loses at most four jobs.
13. As the competitor, I want the notebook to use however many GPUs Kaggle allocates (one or two T4s), so that a single-GPU allocation still completes.
14. As the competitor, I want the session budget to fit ≈ 5–6 hours wall on two T4s (≈ 9 GPU-hours), so that ≈ 10 GPU-hours remain for a second bracket session before the Sept 13 slots.
15. As the competitor, I want a stack cell in the notebook that scores every arm family present (runs 3–8), so that the printed per-arm table and nested estimate can be read from the Kaggle log before any download.
16. As the competitor, I want a laptop assembly command that computes nested estimates for Master v2, Master v2 + each new arm, and the best combination with equal and shrunk weights, so that the gate (≥ 0.003 nested gain over Master v2) is applied by numbers, not taste.
17. As the competitor, I want the assembly to build the Master v3 zip with all seeds of every member (no seed-0 pruning) and the temperature floor the site pool supports (0.9), so that the mistakes found on Sept 10 (pruning, fitted weights, floor 1.0) are not repeated.
18. As the competitor, I want the zip kept under 850 MB with the lowest-weight arm's checkpoints dropped first if needed, so that the DrivenData upload never fails on size.
19. As the competitor, I want the container-style check to run on the 20 smoke scans and print per-arm scores, all arms N/N, so that the smoke-to-public track can be read per member, never fitted on (ADR 0005).
20. As the competitor, I want a decision rule written down before the run (v3 replaces v2 in slot 1 only if the nested gain ≥ 0.003 and the platform smoke score is within 0.01 of v2's), so that the slot choice on Sept 12 is mechanical.
21. As the competitor, I want a one-page HOWTO PDF, so that the teammate can launch and fetch the run without the repo's history, using their own token in a separate Kaggle config directory (never pasted in chat, never committed).
22. As the competitor, I want the run to stay within the rules (no test-set use, per-sample inference, ImageNet weights only, no external data), so that the entry remains prize-eligible.

## Implementation Decisions

- Soft targets are a per-uid lookup applied inside the training loop after the batch is moved to the device and before mixup: target = (1 − λ)·label + λ·p_nested, label for uids not in the table. The dataset, the predict path, and the validation labels are untouched.
- Mistake weights are a per-uid lookup applied the same way: w = 1 + κ·loss_nested, then normalised to mean 1 over the training uids; with mixup the weights mix with the same λ as the inputs; the loss is the weighted mean of per-sample BCE. κ = 0 (default) reproduces today's loss exactly. The DANN/DRO branch already computes per-sample losses and applies the same weights.
- Two new trainer knobs, exposed by the standard runner and recorded in each checkpoint's metadata next to `loss`: the soft-target mix and the mistake-weight strength, plus the path of the table. Tags: `r3d_hemisoft_f<f>_s<s>[_site]`, `r3d_hemimw_f<f>_s<s>`, `r3d_crophistsoft_f<f>_s<s>`, `r3d_hemi48_f<f>_s<3|4>`, `r3d_hemisoft_fall_s<s>`.
- The soft-target table is generated on the laptop from the master pool by the same nested procedure as the Sept 10 analysis (equal weights, temperature refit per fold with floor 0.9, prior mix 0.02) and embedded in the notebook as a compressed string that a cell writes to `cache_v4/soft_master_v2.csv`; the runner passes that path to every soft job.
- The notebook generator reuses the run-7 generator's machinery (which reuses run 3's cells: package write-files, cache mount and resume, multi-GPU runner, fail-safe tail) by executing it in a namespace and replacing the job list and the arm list; the arm list gains the run-8 families.
- The stack cell keeps the run-7 super-stack (ten best arms by OOF, nested combiner) and additionally prints the nested estimate of "Master v2 pillars only" and "pillars + each new arm" so the gate is visible in the log.
- Assembly on the laptop is a new one-command script that: pools OOF files from every local artefact folder, computes the nested estimates (fitted / equal / shrunk) for the candidate arm sets, applies the gate, writes the manifest with all seeds, the chosen weights and T 0.9, packs the zip from local artefacts and the run-5/7 zips, and calls the existing container-style check.
- Inference is unchanged: every new checkpoint is a crop-3D hemisphere or crop-3D member under an existing arm type; the manifest lists them as additional checkpoints of `a3d_hemi`-style arms (new arm names `a3d_hemisoft`, `a3d_hemimw`, `a3d_crophistsoft`).
- The teammate's account: `KAGGLE_CONFIG_DIR` pointing at a directory holding their `kaggle.json`; the kernel id is `skyfall23/dat-run8-distill`; secrets `KAGGLE_USERNAME`/`KAGGLE_KEY` set once on the notebook by the account owner; the checkpoint dataset is already shared with edit rights (runs 6b/6c/7 ran there).

## Testing Decisions

- A good test drives the highest existing seam and checks observable outputs only: the standard runner CLI on the synthetic tiny cache (checkpoint + metadata written, metadata records the new knobs, finite loss), the generator (notebook validates with nbformat, contains the embedded table cell, the job tags and the device-count patch), the assembly script (manifest lists every seed, weights sum to one, T 0.9, zip under the size cap) and the container-style check (all arms N/N on the smoke set).
- Unit-level checks of the two lookups at the trainer's boundary: soft target equals the label when λ = 0 or the uid is missing; equals the blend otherwise; weights have mean 1 and are 1 when κ = 0; validation log loss in the metadata is computed against hard labels.
- Prior art: `tests/test_run7.py` (one-epoch training on the tiny cache with the hemisphere arch and bootstrap loss, container rebuild from metadata), `tests/test_run6a.py`–`test_run6c.py` (generator validation, HOWTO page count), `scripts/local_e2e_v3.py`.

## Out of Scope

- Residual (offset-boosted) members that consume the ensemble's logit at inference time — would change the inference pipeline days before the deadline.
- Any test-set or pseudo-label use, PPMI or other external data (ruled out), new registration, intensity TTA (T14), disclosure (T12).
- Choosing between slot-2/slot-3 variants beyond the written decision rule.

## Further Notes

- Evidence behind the design (Sept 10 analysis on the master pool, 1,362 scans): Master v2 nested 0.2323 (equal weights, T floor 0.9); top-100 scans carry 54 % of the loss; 46 confident mistakes; capping per-scan loss at 2.0 would give 0.2201 — that is the ceiling of "correcting mistakes", not the expectation. Self-bootstrap (run 7) 0.2565 vs plain 0.2568: no effect, so the soft target must come from the ensemble, not from the model itself.
- Budget: hemisphere fold ≈ 12.4 min on a T4 (run 7 metadata); 42 jobs ≈ 8.7 GPU-hours ≈ 4.5 h wall on two T4s plus ≈ 30 min cache and stack.
- Risk: soft targets built from random-fold OOF make the site-fold read-out of the new arm slightly optimistic (the ensemble that produced a scan's target saw scans from its site). The gate is therefore applied on random-fold nested estimates and the site-fold numbers are used only for the temperature.
- Honesty: the gap to rank 1 (0.2166) is ≈ 0.03 public; this run is worth ≈ 0.003–0.006 nested. It is the best remaining use of the GPU hours, not a path to first place.
