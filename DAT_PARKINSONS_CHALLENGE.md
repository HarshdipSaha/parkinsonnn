# DaT Parkinson's Challenge — Research Notes

Research date: 2026-09-05
Source: [DrivenData competition #311](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/)

## Overview

- **Host:** DrivenData (NOT Kaggle), organized by the **French Society of Nuclear Medicine (SFMN)**, in partnership with **Health Data Hub** and **GaelO**. Part of France's "Health Data Challenges" call for projects, financially supported by the France 2030 plan.
- **Task:** Build a computer vision model that classifies 3D dopamine transporter (DaT) SPECT brain scans as **normal** or **abnormal**, to support early/accurate detection of parkinsonian syndromes.
- **Clinical motivation:** >20,000 DaT scans/year in France; ~1 in 5 cases is difficult to interpret (early-stage/atypical presentations). Winning models will be released open source and indexed in France's open health algorithm library (BOAS), with the dataset eventually published on data.gouv.fr.
- **Deadline:** **September 16, 2026, 11:59 PM UTC**.
- **Prizes:** €25,000 total — 1st €12,500 / 2nd €7,500 / 3rd €5,000 (paid in USD by DrivenData based on July 22, 2026 exchange rate).
- **Participation (as of research date):** 855 joined (757 solo, 45 teams).
- **Post-challenge:** Winners invited to an in-person ceremony in France; top teams may be invited to an optional research collaboration with SFMN on age/sex prediction from DaT scans.

## Dataset

- Format: 3D reconstructed volumes, compressed NIfTI (`.nii.gz`), 16-bit unsigned integer voxel intensities. One file per examination; filename (minus extension) = `uid`.
- **Not a fixed shape** — volume dimensions vary (e.g. `142×142×112`, `128×128×128`) and voxel spacing varies (e.g. 2.46 mm vs 3.895 mm isotropic) across the 10 contributing hospital centers. Pipelines must not assume fixed input shape/spacing.
- Files:
  ```
  data/
  ├── niftis/
  │   ├── xaji0y6d.nii.gz
  │   ├── pbhsahxt.nii.gz
  │   └── ...
  └── train_labels.csv   # uid, is_pathologic (0.0 = normal, 1.0 = abnormal)
  ```
- **Test set** is withheld entirely — not downloadable. It's only mounted inside the code-execution container at inference time.
- **Contributing hospitals:** CHU Toulouse, CHRU Nancy, CHRU Tours, Centre Henri Becquerel, CHRU Brest, CHU Grenoble Alpes, Hospices Civils de Lyon, Centre Jean Perrin, AP-HM Marseille, Institut Godinot.
- Exact training-set size (row count / GB) is **not publicly disclosed** without logging in and downloading. One forum participant (`solosolevd`) described it as **"very small for training neural nets."** Test-set size and its public/private split are also intentionally undisclosed by DrivenData staff.

## Performance metric

- **Log loss** (cross-entropy) is the ranking metric — lower is better. Rewards well-calibrated probabilities, penalizes confident-but-wrong predictions.
- **AUROC** is also shown on the leaderboard but **does not** affect ranking or prizes.
- Final prizes are awarded on the **private leaderboard**, using a private test set that may differ from the public-leaderboard test set. Best-scoring submission on the private set counts automatically — no manual "final submission" selection needed.

## Submission format — this is a code-execution challenge

- You do **not** submit predictions directly, and you do **not** push a Docker image to any registry or host a Space. You upload a **`submission.zip`** through DrivenData's web submissions page.
- `submission.zip` root **must** contain `main.py` (entrypoint). Model weights/extra modules can sit alongside it:
  ```
  submission.zip
  ├── main.py            # required entrypoint, executed by the evaluator
  ├── my_module/
  │   ├── preprocessing.py
  │   ├── model.py
  │   └── ...
  └── my_model_weights.bin
  ```
- Your code reads test volumes from a **read-only** mount and writes predictions:
  ```
  /code_execution/
  ├── data/
  │   ├── niftis/                 # test-set NIfTI files
  │   └── submission_format.csv
  ├── src/                        # your unzipped submission
  └── submission.csv              # you write predictions here
  ```
- Output `submission.csv` must have exactly two columns: `uid` (str), `is_pathologic` (float, 0.0–1.0 — a calibrated probability, not a hard 0/1 label).
- **Local testing uses Docker** to replicate the official runtime (`competition-sfmn-parkinsons-runtime`, hosted on Azure Container Registry, Python 3.12, PyTorch on CUDA 12.9, built with `uv`) — this is for your own debugging only, not the submission mechanism itself. Repo: https://github.com/drivendataorg/competition-sfmn-parkinsons-runtime
- Two-stage testing flow before a real submission: (1) test locally with Docker, (2) run a **smoke test** (small sample drawn from the training set, not scored for prizes) via the submissions page, (3) submit for real scoring on the full test set.

## Runtime environment & hard constraints

- Execution container hardware (provided by DrivenData, **not Kaggle**):
  - **1× NVIDIA A100 GPU, 80 GiB VRAM**
  - **24 vCPUs** (AMD EPYC 7V13)
  - **220 GB RAM**
- **Time limit: 3 hours max** for a full submission; **6 minutes max** for a smoke test.
- **No network/internet access** inside the container — all code, dependencies, and model weights must be bundled in the zip; no `pip install` at runtime.
- **No root access** inside the container.
- Log output capped at 300 lines × 300 characters/line.
- Requesting an additional Python package (not already in the runtime's `pyproject.toml`/`uv.lock`) is possible by following the request process in the runtime repo — must be available via PyPI and build cleanly.
- The execution machines are a **shared resource** across all competitors — DrivenData explicitly asks participants to test thoroughly locally/smoke-test first and cancel jobs expected to hit the time limit.

## Rules highlights

- **External data & pretrained models are allowed**, but:
  - All external data must be **disclosed to organizers** regardless of prize eligibility (for independent verification).
  - To be **prize-eligible**: external data/models must be freely & publicly available to all participants, and licensed to allow the resulting model to be released for broad (including commercial) use — **no NC / CC-NC / CC BY-NC licenses**.
- **Competition data restrictions:**
  - Use only for this challenge, only during the challenge period; delete all local copies after the challenge ends (unless a separate license allows continued use).
  - **Never share, copy, or publish the data.** Explicitly: you may not use tools like ChatGPT/Codex that store or retain uploaded data — though you may download model weights and run models locally.
- **No pseudo-labeling** / test samples must be processed independently; submissions must not print or log info about the test dataset (grounds for disqualification).
- One account per participant; no privately sharing code/data outside your team.
- Winning solutions require **MIT License** open-source release.
- Eligibility: 18+, legal resident of an eligible territory; excludes sponsor/DrivenData staff and Russian Federation residents (prize eligibility only).

## ⚠️ Open compliance question: training on Kaggle / cloud GPUs

A forum thread (["Clarification on AI coding assistants and cloud GPU compute with competition data"](https://community.drivendata.org/t/clarification-on-ai-coding-assistants-and-cloud-gpu-compute-with-competition-data/11479)) is directly relevant if training off-platform:

- DrivenData staff (`mdaley`) confirmed the compliance test is: data must not be retained by a third party, must not be accessible to anyone who hasn't agreed to the rules, and **every copy must be fully deletable** at the end — cloud notebooks/instances are fine **if** private to you and the provider's terms don't grant them retention rights.
- A later participant (`mukeshr06`) asked specifically about a Kaggle-Notebooks-only workflow (private notebook, no dataset upload, temp session storage only, deleted before session ends), noting that **Kaggle's Terms of Use grant Kaggle a perpetual, irrevocable, worldwide license over uploaded content**, and Kaggle's Privacy Policy states deletion can take ~2 months (up to 6 months for encrypted backups).
- **As of this research (2026-09-05), DrivenData staff had not yet answered that specific Kaggle question** — it remains an unresolved gray area for prize eligibility. Safer alternative per staff's already-confirmed guidance: a private, self-controlled, fully-deletable environment (your own machine, or a rented instance like RunPod/Lambda that you wipe yourself).

## Compute feasibility assessment (for training on Kaggle's free ~30 GPU-hrs/week)

- The **final scored inference run does not use Kaggle at all** — DrivenData provides the A100 80GB for that, capped at 3 hours. Kaggle would only matter for **training** beforehand.
- Given the dataset is single-country, ~10 hospitals, and described by a participant as "very small for training neural nets," it's realistically in the range of a few hundred to low thousands of 3D volumes — likely single-digit GB total.
- At that scale, training 3D CNNs (or 2.5D transfer-learning approaches) is cheap: full epochs run in minutes, full CV training runs likely take low single-digit hours each on a T4/P100.
- **Conclusion: 30 GPU-hrs/week is very likely more than sufficient** computationally. Over the remaining ~11 days there are two weekly quota resets (~60 hrs total available if needed). The realistic bottleneck for a competitive result is **cross-scanner generalization / preprocessing / calibration for log loss**, not raw compute — and (separately) the unresolved Kaggle data-compliance question above if prize eligibility matters.

## Key links

- [Competition home](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/)
- [Problem description](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/page/990/)
- [Code submission format](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/page/989/)
- [About the data/organizers](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/page/988/)
- [Official rules](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/rules/)
- [Leaderboard](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/leaderboard/)
- [Runtime repository (GitHub)](https://github.com/drivendataorg/competition-sfmn-parkinsons-runtime)
- [Community forum](https://community.drivendata.org/c/dat-parkinsons-challenge/110)
- [Kaggle/cloud-GPU compliance thread](https://community.drivendata.org/t/clarification-on-ai-coding-assistants-and-cloud-gpu-compute-with-competition-data/11479)
