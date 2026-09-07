---
id: T11
title: PPMI ingestion and pretraining notebook
label: wayfinder:task
type: task
mode: AFK
status: closed
assignee: claude
blocked_by: T02, T04
map: ../map.md
---

## Question

If PPMI access lands by Sept 10: `scripts/ppmi_ingest.py` (convert to NIfTI, register into the study frame, label map PD → 1, HC/SWEDD → 0, prodromal excluded, subject-level split), pretrain the crop-3D and hemisphere encoders on PPMI (fold −1), fine-tune on `cache_v4` with `--init`, gate on random + site folds; disclosure text for the organisers (scan list + PPMI acknowledgement).

## Resolution

Ruled out of scope Sept 7 05:05 UTC (see T02): no PPMI branch in this map.
