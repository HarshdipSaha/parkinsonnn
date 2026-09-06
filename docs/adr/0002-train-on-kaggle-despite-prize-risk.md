---
status: accepted (2026-09-05, user decision)
---

# Train on Kaggle even though it may void prize eligibility

The rules require that competition data never be retained by a third party and that every copy be deletable; Kaggle's terms grant Kaggle a retention licence over uploaded content, and DrivenData staff had not cleared Kaggle notebooks when asked. The advisory council recommended a rented, wipeable GPU. The user chose Kaggle anyway ("Kaggle anyway, accept the risk") for cost and convenience, so the raw scans and the derived caches live in private Kaggle datasets.

**Consequences**: keep every dataset private and never write raw scans or derived boxes to notebook outputs; delete the datasets and notebooks after the competition; disclose the training venue if a prize is at stake; if eligibility becomes a hard requirement, the caches and scripts make a re-run on a rented instance a one-day job.
