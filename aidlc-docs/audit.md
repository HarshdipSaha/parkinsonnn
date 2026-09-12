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
| 2026-09-09 06:35 | DrivenData smoke test id-321610 evaluated: **0.2341 log loss** for standalone Tri-Planar 2.5D package on hidden test set (beats full Run 5 ensemble 0.2388) | Platform evaluation result |
| 2026-09-09 12:25 | DrivenData smoke test id-321617 evaluated: **0.2363 log loss** for unified 5-pillar `submission_master_final.zip` (463.92 MB) on hidden test set | Platform evaluation result |
| 2026-09-10 13:30 | Sept 10 request: chances of top 10 and what can still improve → analysis run, Master v2 built with equal weights / all seeds / T 0.9 | User: "see the new results and tell what are chances of getting in top 10, anything u can do to become better?" |
| 2026-09-10 14:07 | RESULTS_LOG correction: the "run 5 public 0.2600 id-321520" entry was fabricated (another session); ledger re-read live: 3/3 real submissions, best 0.2667 | Autonomous correction, evidence from the logged-in submissions page |
| 2026-09-10 14:13 | DrivenData smoke test id-321786 for Master v2 (0.2381); the auto-mode classifier blocked the first click, the user approved explicitly | User: "yes do it" |
| 2026-09-10 15:00 | Run 8 spec/plan/ticket T15 (to-spec + writing-plans); honest expectation stated: nested 0.226–0.229, public ≈ 0.245–0.250, rank ≈ 15–20; rank 1 not reachable | User: "anything u think can lift our score to rank 1 ... /to-spec and then write and code" |
| 2026-09-10 16:33 | Run 8 pushed to `skyfall23/dat-run8-distill` with the default CLI config (already authenticated as skyfall23); no token handled in chat | User: "its already authenticated try it" |
| 2026-09-10 17:00 | External data / pretrained weights re-checked on request: no open DAT-SPECT dataset outside PPMI exists; MedicalNet (MIT, CT/MRI) is legal but modality- and architecture-mismatched → not pursued; disqualification-cascade policy confirmed as standard (no specific DrivenData incident found) | User questions; assessments only, no action |
| 2026-09-10 17:40 | Full handoff for solo execution (HANDOFF.md, forharsh.md/.pdf rewritten to the current state, effort 007, registry, audit) and push to GitHub | User: "/ai-dlc record and push everything ... assume this is our last message" |
| 2026-09-11 05:45 | Run 8 fetched from Kaggle `skyfall23/dat-run8-distill` (42 jobs COMPLETE); Gate passed with nested **0.2289** (+0.0034 gain over v2 0.2323); `submission_master_v3.zip` built (847 MB, 71 ckpts) | Autonomous gate execution per plan §8 |
| 2026-09-11 06:22 | DrivenData smoke test id-321850 evaluated: **0.2374 log loss** for `submission_master_v3.zip` on hidden smoke test; beats Master v2 (0.2381) and satisfies decision rule ($\le 0.2481$); Master v3 locked as official Slot 1 submission | Platform evaluation result |
| 2026-09-12 01:40 | Slot 1 official real submission (`submission_master_v3.zip`) evaluated: **0.2557 log loss**, **AUROC 0.9576**, **Rank #23** on Public Leaderboard (4th submission; +21 rank jump from #44); public gap $G = 0.2557 - 0.2289 = +0.0268$; smoke gap $0.2557 - 0.2374 = +0.0183$ | Platform evaluation result |
