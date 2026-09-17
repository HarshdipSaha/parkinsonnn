# parkinsonnn — DaT Parkinson's Challenge (DrivenData #311)

Binary classification of 3D DaT-SPECT NIfTI volumes (normal vs abnormal), scored by log loss, delivered as a code-execution `submission.zip`. Private working repository.

| | |
|---|---|
| Status | **COMPLETED / CLOSED**. DrivenData Competition #311 concluded Sept 16, 2026. Official Private Leaderboard: **Rank #50** (Best Private Log Loss: **0.3003**, AUROC: **0.9406** via Slot 5 `id-322091`; Best Public: **0.2473**). All 6 official submissions and 20 smoke tests audited. |
| Resume work / Handoff | [aidlc-docs/HANDOFF.md](aidlc-docs/HANDOFF.md) |
| Results & Ledger | [RESULTS_LOG.md](RESULTS_LOG.md) |
| Compliance & Disclosure | [docs/DISCLOSURE.md](docs/DISCLOSURE.md) |
| Field Guide & Post-Mortem | [LESSONS.md](LESSONS.md) |

## Documentation

| Type | Document |
|---|---|
| How-to | [docs/how-to-train-and-submit.md](docs/how-to-train-and-submit.md): build the caches, train on Kaggle, pack, download, submit; [RUNBOOK.md](RUNBOOK.md) |
| Reference | [docs/ref-pipeline-v3.md](docs/ref-pipeline-v3.md): modules, cache layout, `manifest.json`, every script's flags; [docs/DISCLOSURE.md](docs/DISCLOSURE.md) |
| Explanation | [docs/concept-registered-frame.md](docs/concept-registered-frame.md): why the registered frame design was chosen; [LESSONS.md](LESSONS.md): post-mortem & 11 engineering rules |
| Rules and runtime | [DAT_PARKINSONS_CHALLENGE.md](DAT_PARKINSONS_CHALLENGE.md) |
| Research | [EDA_REPORT.md](EDA_REPORT.md), [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md), `PIPELINE_REC_*.md`, [COUNCIL_VERDICT.md](COUNCIL_VERDICT.md), [docs/RUN3_BRAINSTORM.md](docs/RUN3_BRAINSTORM.md), [docs/RESEARCH_SIMILAR_COMPETITIONS.md](docs/RESEARCH_SIMILAR_COMPETITIONS.md) |
| Domain language | [CONTEXT.md](CONTEXT.md) (glossary), [docs/adr/](docs/adr/) (decision records) |
| Plans and process | [docs/superpowers/plans/](docs/superpowers/plans/), [aidlc-docs/](aidlc-docs/README.md) (Efforts 001–009) |
| Jupyter Notebooks | [notebooks/](notebooks/) (01 EDA, 02 Scanner Shift, 03 Registration, 04 Radiomics, 05 Model Zoo, 06 Ensemble) |

## Layout

```
notebooks/        educational and reproducible Jupyter notebooks (01 EDA to 06 Master Ensemble)
datsc/            package shared by training and inference (run-3 modules end in _v3; run-2 modules kept for the submitted zip)
scripts/          cache building, template, registration, gate, training, stacking, packing
main_v3.py        DrivenData entrypoint (packed as main.py)
kaggle/           notebook generator / sync / submission helpers (runs 6-10)
tests/            pytest (run: python -m pytest tests -q)
artifacts/        template, run-2 artefacts (gitignored except .gitkeep)
```

Quick check: `python -m pytest tests/test_v3.py tests/test_infer_v3.py -q` and `DATSC_ARTIFACTS=artifacts python main_v3.py --selftest`.
