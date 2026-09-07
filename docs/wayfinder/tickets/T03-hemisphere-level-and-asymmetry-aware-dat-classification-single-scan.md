---
id: T03
title: Hemisphere-level and asymmetry-aware DaT classification; single-scan test-time adaptation; label-noise-robust losses
label: wayfinder:research
type: research
mode: AFK
status: closed
assignee: claude
blocked_by: 
map: ../map.md
---

## Question

What published methods (2022–2026) classify DaT-SPECT or similar bilateral nuclear-imaging scans at the hemisphere level (Siamese / shared encoders on left and right striatum, max- or asymmetry-pooled heads), which single-scan (non-transductive) test-time adaptation methods have shown gains under scanner shift for 3D CNNs, and which label-noise-robust losses improve log loss when ~5 % of labels are ambiguous? Output: `docs/wayfinder/research/hemisphere-tta-labelnoise.md` with sources and a ranked recommendation of what to try in run 7.

## Resolution

Resolved Sept 7 05:15 UTC by `../research/hemisphere-tta-labelnoise.md` (≈ 95 sources). (1) No controlled per-hemisphere-vs-whole-volume DaT study exists; bilateral Siamese analogues elsewhere gain +0.01–0.07 AUC, so `HemiNet3D` is expected to be a diversity-type gain (−0.003 to −0.010 nested), and its sum + |difference| + max-side head matches the clinical picture (worst side first, but the other side already ≈ 38 % down at onset). (2) Single-scan test-time adaptation: only source-anchored BatchNorm-statistics blending has verified single-sample gains, on corruption benchmarks; entropy-minimisation (MEMO/Tent) sharpens confidence and hurts calibration; geometric TTA is expected flat — consistent with our flat z-shift and negative affine measurements. (3) Robust losses (GCE/SCE/MAE, co-teaching) are non-proper and never reported on log loss; uniform label smoothing worsens NLL after temperature scaling; the only family with log-loss evidence is **soft targets carrying uncertainty** — e.g. y* = (1−λ)·y + λ·p_OOF with λ 0.3–0.5. Ranked for run 7: HemiNet3D (+ optional per-side SBR auxiliary target); nested-OOF soft targets instead of `--loss bootstrap`; intensity-view TTA selected by nested log loss; BN-prior blend at inference for the B0/V2-S arms; worst-side canonical mirroring (lowest). Do-not list: robust losses, uniform label smoothing, case removal, MEMO/Tent, more geometric TTA, registration refinement (nothing found supports it).
