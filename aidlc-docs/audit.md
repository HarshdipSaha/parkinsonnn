# Audit: user decisions and approvals

| Date (UTC) | Decision | Source |
|---|---|---|
| 2026-09-05 | Train on Kaggle despite the unresolved prize-eligibility risk | User: "Kaggle anyway, accept the risk" |
| 2026-09-05 | Skip plan review; write scripts from the plan and run them on Kaggle | User instruction |
| 2026-09-05 15:38 | Real submission slot 1 used (id-321156, public 0.3391) | User: "download the zip and submit" |
| 2026-09-06 | Run deep-research and brainstorming, refine the approach, run it on Kaggle; goal is rank 1; approval gates waived by the instruction to run autonomously | User instruction |
| 2026-09-06 | AI-DLC initialised retroactively | User: "/ai-dlc initialise everything we did" |
| 2026-09-06 ~09:00 | Full handoff so a fresh session can resume from the docs alone (`HANDOFF.md`, plan, effort 003). Honest rank-1 estimate given: 10–15 % by the deadline | User instruction |
| 2026-09-06 ~09:20 | Documentation reformatted as clean Markdown | User: "in a better format" |
| 2026-09-06 ~10:30 | Interactive Kaggle session lost (websocket drop); training relaunched as a background commit with trimmed queues to keep tonight's submission feasible | Autonomous decision, consistent with the user's run-it instruction |
| 2026-09-06 ~10:10 | User provided a Kaggle API token (`.env`) and asked for per-job checkpoint persistence; user explicitly asked me to add the notebook Secrets myself without viewing the token (done via clipboard paste; `.env` gitignored) | User: "you can do it, just don't see it, just write it" |
| 2026-09-06 11:34 | Laptop-side watcher (Task Scheduler) added so the Kaggle output downloads without a session | Autonomous, after the user asked whether the zip downloads by itself |
| 2026-09-06 14:34 | Watcher allowed on battery (had silently stopped when the laptop was unplugged) | Autonomous fix |
| 2026-09-06 14:45 | Container bug fixed in code and the Kaggle zip repaired locally instead of re-running the 5 h commit | Autonomous; verified with a container-style local run |
| 2026-09-06 15:10 | Real submission uploaded before the smoke-test result was visible (slot 2 of 3) | User: "now submit" |
| 2026-09-06 15:20 | Keep the GPU closed after the run (no interactive sessions) | User: "close the gpu ok" — draft session confirmed off |
| 2026-09-06 15:50 | Public 0.2817 / rank 66. New target set by the user: 0.21 using all research + EDA findings | User: "IMPROVE THIS TO 0.21" |
| 2026-09-08 08:00 | Teammate Kaggle credentials (`skyfall23`) used due to GPU quota exhaustion on primary account | User instruction |
| 2026-09-08 11:10 | Run 7 (HemiNet3D bilateral Siamese) launched on Kaggle `skyfall23/dat-run7-hemisphere` | Autonomous execution per roadmap |
| 2026-09-08 14:00 | Run 6b (Tri-Planar 2.5D late fusion) launched on Kaggle `skyfall23/dat-run6b-triplanar` | Autonomous execution per roadmap |
| 2026-09-08 16:45 | Run 6c (116 radiomic descriptors + VICReg SSL) launched on Kaggle `skyfall23/dat-run6c-features-ssl` | User instruction ("start running: Run 6c") |
| 2026-09-08 23:25 | Large 1.8GB submission zip rejected by DrivenData upload; isolated slim package `data/submission_run7_hemi_only.zip` (63 MB) built and uploaded for pure HemiNet evaluation | Autonomous fix to bypass platform upload size limit |
| 2026-09-09 00:09 | DrivenData smoke test id-321572 evaluated: **0.2706 log loss** for single standalone HemiNet arm on hidden test set | Platform evaluation result |

