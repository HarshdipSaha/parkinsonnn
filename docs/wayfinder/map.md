---
label: wayfinder:map
title: The way to public 0.21–0.22 (DaT Parkinson's Challenge)
created: 2026-09-07 05:00 UTC
---

# Map: the way to public 0.21–0.22

## Destination

A verified, one-command Kaggle notebook (`kaggle/run7_hemisphere`, HOWTO included) implementing the route judged most likely to reach **public log loss 0.21–0.22 by Sept 16 2026 23:59 UTC**, with every decision that route depends on resolved and recorded here: which members enter the final super-stack, whether site-invariant training / histogram standardisation / PPMI pretraining are in, and how the three remaining real submissions (Sept 12 15:38, Sept 13 15:11 and ≈ 23:30 UTC) are spent.

## Notes

- Domain glossary: `CONTEXT.md`; ADRs `docs/adr/0001–0005`. Standing decisions that this map does NOT revisit: registered 2 mm frame (ADR 0001), Kaggle as venue with private datasets (ADR 0002), few-number combiner with temperature floor (ADR 0003), honest protocol / no site scalars (ADR 0004), the smoke set is validation-only (ADR 0005).
- Skills every session consults: `grilling` + `domain-modeling` for decisions, `research` for research tickets, `writing-plans` when a ticket graduates to a build, `verification-before-completion` before any claim of "verified".
- **Execution is carried into this map** (override of "plan, don't do"): the user asked for the notebook + HOWTO as the end product, so T07/T09/T11 are build tickets, resolved by delivered, tested artefacts.
- Where we stand (Sept 7 05:00 UTC): public 0.2667 / AUROC 0.9536 / rank 39 (run 4a, nested 0.2491); the **shift gap** (nested → public) has been 0.021 and 0.018; the leader is 0.2192 / 0.971. Reaching 0.215 needs roughly nested ≈ 0.20 with today's gap, or nested ≈ 0.22 with the gap closed — both a −0.03 to −0.05 step. Run 5 (site-invariant training, hist) is training; runs 6a/6b/6c are ready to launch; PPMI is applied for.
- Honest odds recorded at charting: ≤ 0.22 public by the deadline ≈ 10–15 % (needs PPMI or an unusually strong site-gap closure); ≤ 0.24 (top 10) ≈ 40 %.
- Rules that bound every ticket: no pseudo-labelling or transductive test use; per-sample inference within 3 h on 1×A100; public commercial-use weights only; external data only with legal right and disclosure.

## Charting round 1 (grilling; provisional answers adopted, the user overrides by editing the tickets)

❓ **Q1 — Destination**: public 0.21–0.22 by the deadline, private decides prizes (best private counts). ➡️ Adopt; bracket the last slot.
❓ **Q2 — Slot policy**: slot 1 → best gated super-stack (runs 3–6); slot 2 → run 7 (+ PPMI if landed); slot 3 → bracket (λ 0.01). ➡️ Adopt (T01, T10).
❓ **Q3 — Compute**: our account runs 5 then 7; the teammate's account runs 6a/6b/6c (6a not concurrently with 5). ➡️ Adopt (T06).
❓ **Q4 — PPMI**: proceed assuming no PPMI; PPMI is a conditional branch that becomes live when access lands (T04 → T11). ➡️ Adopt.
❓ **Q5 — Ensemble breadth vs inference budget**: ≤ 10 arms, ≤ ~60 checkpoints, zip ≤ 850 MB, projected container time < 60 min. ➡️ Adopt provisionally (T08 decides with numbers).
❓ **Q6 — Hemisphere-level member**: build it (the asymmetry descriptors rank top; unilateral onset is the disease's structure). ➡️ Build (T07).
❓ **Q7 — Registration refinement** (template from all scans): only if run 5 / T03 point at registration-sensitive errors. ➡️ Fog, not a ticket.
❓ **Q8 — Label-noise-robust loss**: try a bootstrapped-target loss on the crop-3D arm in run 7, gated. ➡️ Include as a run-7 variant (T07), after T03 reports.
❓ **Q9 — Calibration**: keep T floor 1.0, λ 0.02; bracket λ 0.01. ➡️ Adopt (T01).
❓ **Q10 — Obligations**: disclosure + release README drafted before the deadline. ➡️ T12.

## Decisions so far

<!-- one line per closed ticket: gist + link -->

## Not yet specified

- Registration refinement (study template from all 1,362 scans, 3 iterations) — only if run 5 / T03 show registration-sensitive errors.
- Single-scan test-time adaptation for the 3D arms — depends on T03's findings.
- Label-noise-robust objective beyond the run-7 trial — depends on T03 and the run-7 read-out.
- Knowledge distillation of the super-stack into one net if the inference budget (T08) turns out too tight.
- A second-stage striatal localiser to fix the residual registration errors of thin-FOV scans — only if the per-site tables after run 5/6 still show the 3.90 mm sites as the tail.

## Out of scope

- Pseudo-labelling, transductive normalisation, prevalence estimation from test predictions (competition rules).
- Non-commercial or non-public pretrained weights (DINOv3, RadImageNet).
- Anything after Sept 16 except the disclosure/release obligations (T12).
- A second DrivenData account or team (one team submits).
