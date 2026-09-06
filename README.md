# parkinsonnn — DaT Parkinson's Challenge (DrivenData #311)

Binary classification of 3D DaT-SPECT NIfTI volumes (normal vs abnormal), scored by log loss, delivered as a code-execution `submission.zip`. Private working repository.

| | |
|---|---|
| Status | Run 2 submitted (public 0.3391, rank 152). Run 3 (registered frame, 3D + 2.5D + ROI-LR ensemble) in training. |
| Resume work | [aidlc-docs/HANDOFF.md](aidlc-docs/HANDOFF.md) |
| Results | [RESULTS_LOG.md](RESULTS_LOG.md) |

## Documentation

| Type | Document |
|---|---|
| How-to | [docs/how-to-train-and-submit.md](docs/how-to-train-and-submit.md): build the caches, train on Kaggle, pack, download, submit |
| Reference | [docs/ref-pipeline-v3.md](docs/ref-pipeline-v3.md): modules, cache layout, `manifest.json`, every script's flags |
| Explanation | [docs/concept-registered-frame.md](docs/concept-registered-frame.md): why the run-3 design is what it is |
| Rules and runtime | [DAT_PARKINSONS_CHALLENGE.md](DAT_PARKINSONS_CHALLENGE.md) |
| Research | [EDA_REPORT.md](EDA_REPORT.md), [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md), `PIPELINE_REC_*.md`, [COUNCIL_VERDICT.md](COUNCIL_VERDICT.md), [docs/RUN3_BRAINSTORM.md](docs/RUN3_BRAINSTORM.md), [docs/RESEARCH_SIMILAR_COMPETITIONS.md](docs/RESEARCH_SIMILAR_COMPETITIONS.md) |
| Domain language | [CONTEXT.md](CONTEXT.md) (glossary), [docs/adr/](docs/adr/) (decision records) |
| Plans and process | [docs/superpowers/plans/](docs/superpowers/plans/), [aidlc-docs/](aidlc-docs/README.md) |

## Layout

```
datsc/            package shared by training and inference (run-3 modules end in _v3; run-2 modules kept for the submitted zip)
scripts/          cache building, template, registration, gate, training, stacking, packing
main_v3.py        DrivenData entrypoint (packed as main.py)
kaggle/           notebook generator / sync / submission helpers
tests/            pytest (run: python -m pytest tests -q)
artifacts/        template, run-2 artefacts (gitignored except .gitkeep)
```

Quick check: `python -m pytest tests/test_v3.py tests/test_infer_v3.py -q` and `DATSC_ARTIFACTS=artifacts python main_v3.py --selftest`.
