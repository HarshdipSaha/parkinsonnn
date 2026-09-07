---
id: T02
title: PPMI DaT-SPECT: what we would get and what the DUA allows
label: wayfinder:research
type: research
mode: AFK
status: closed
assignee: claude
blocked_by: 
map: ../map.md
---

## Question

For the Parkinson's Progression Markers Initiative DaT-SPECT data: which image products exist (raw reconstructed DICOM, spatially normalised NIfTI/Analyze, SBR tables), subject and scan counts per cohort (PD, healthy control, SWEDD, prodromal), voxel size / frame, label semantics for a normal/abnormal visual read, the Data Use Agreement's terms on cloud processing and on redistribution / sharing with challenge organisers, and the typical access lead time. Output: `docs/wayfinder/research/ppmi-dat-spect.md` with sources, and a recommended ingestion path into our registered frame (`scripts/cache_volumes.py` + `scripts/register_cache.py`).

## Resolution

Resolved Sept 7 05:05 UTC by the research note `../research/ppmi-dat-spect.md`. PPMI's DaT-SPECT is available (MNI 2 mm reconstructed DICOM, SBR + visual-read tables, ~1,100 PD / 300 HC / 80 SWEDD / 730 prodromal subjects), but it cannot enter this competition: (1) the DUA bars processing on third-party AI platforms that retain content — Kaggle's terms grant it a perpetual licence, so our training venue is excluded; (2) the DUA prohibits sharing participant-level data or scan lists, which the challenge rules require for external data; (3) a DrivenData organiser ruled (Aug 5 2026) that PPMI-trained models are not prize-eligible, and the rules require a licence needing no third-party approval. Lead time (≈ 1 week, prodromal reads 30 days) would also land ~Sept 14. **Decision: PPMI is out of scope for this map; it stays useful for post-competition research only.**
