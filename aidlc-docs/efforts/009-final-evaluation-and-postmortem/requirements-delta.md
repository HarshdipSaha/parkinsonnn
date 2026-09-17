# Effort 009 — requirements delta (vs `aidlc-docs/inception/`)

| Id | Kind | Requirement | Evidence / acceptance |
|---|---|---|---|
| R31 | NEW | Complete reconciliation and verification of all DrivenData submissions across the competition lifespan: all 6 scored normal submissions and all 20 smoke tests (14 completed, 6 canceled). | `RESULTS_LOG.md`, `aidlc-docs/HANDOFF.md`, `aidlc-docs/audit.md`, matching exact DrivenData IDs and scores. |
| R32 | NEW | Quantification of Private Leaderboard performance and domain shift. Compute gap $G_{\text{priv}} = \text{Private} - \text{Public}$ and $G_{\text{shift}} = \text{Private} - \text{Nested}$, confirming final ranking (#50, Log Loss 0.3003, AUROC 0.9406). | `aidlc-docs/efforts/009-final-evaluation-and-postmortem/effort-state.md`, `RESULTS_LOG.md`. |
| R33 | NEW | Structural post-mortem explaining the performance divergence between our pipeline (Rank #50, 0.3003) and top solutions (Rank #1 at 0.2532, Rank #6 at 0.2732, Rank #10 at 0.2760), addressing 3D vs 2D camera domain shift, random CV vs LOCO CV leakage, counterfactual lesion synthesis, scanner resolution augmentation, and calibration tail-risk. | Detailed analytical post-mortem in Effort 009 and `RESULTS_LOG.md`. |
| R34 | NEW | Formal closure of the competitive AI-DLC cycle, archiving models, scripts, and logs with full reproducibility guarantees. | Effort registry updated to complete; handoff updated to CLOSED. |
| R35 | NEW | Compliance disclosure draft (`docs/DISCLOSURE.md`) satisfying Ticket T12, documenting model licenses (BSD-3 torchvision only), zero external data, no pseudo-labeling, per-scan inference, and Kaggle compute provenance. | `docs/DISCLOSURE.md`; ticket T12 marked closed. |
