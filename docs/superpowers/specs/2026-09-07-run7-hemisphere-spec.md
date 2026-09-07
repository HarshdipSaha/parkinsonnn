# Spec: Run 7 — hemisphere-level members and the super-stack (effort 006, wayfinder ticket T07)

*Written Sept 7 2026 05:30 UTC in the `to-spec` synthesis format. Companion documents: `docs/wayfinder/map.md` (the route to 0.21–0.22), `kaggle/run7_hemisphere/APPROACH.md` (design and evidence), `kaggle/run7_hemisphere/HOWTO.md` (one-command launch).*

## Problem Statement

Our best submission scores 0.2667 public log loss (rank 39) with a nested out-of-fold estimate of 0.2491; the leader is at 0.2192. Every arm so far reads the striatum as one volume or one plane, yet the strongest single descriptors in every run were left/right asymmetries, and the disease typically starts on one side. We have no member that models the two hemispheres separately, and no single notebook that stacks everything trained across runs 3–7.

## Solution

Run 7 adds a hemisphere-level member (one shared 3D encoder over the two mirrored striatal half-crops, a head that is symmetric in the hemispheres and pools the more abnormal side) trained on random and leave-site-out folds with site-invariant and label-noise-robust variants, and a super-stack cell that ranks every arm family present in the checkpoint dataset by out-of-fold log loss and fits the nested combiner over the ten best. Everything ships as one Kaggle notebook launched with `kaggle kernels push -p kaggle/run7_hemisphere`.

## User Stories

1. As the competitor, I want a member that sees each hemisphere separately, so that unilateral uptake loss is an explicit signal rather than something convolutions must discover across the midline.
2. As the competitor, I want the hemisphere head to be exactly mirror-symmetric, so that the flip augmentation and the flip test-time variant are label-preserving by construction.
3. As the competitor, I want the two half-crops embedded by one encoder in one pass, so that the member costs no more than the whole-crop net.
4. As the competitor, I want the member evaluated on leave-site-out folds, so that its shift behaviour is measured next to the crop-3D arm's 0.2818-style read-out.
5. As the competitor, I want a site-invariant variant (DANN + group-DRO) of the member, so that run 5's mechanism is tested on a second architecture.
6. As the competitor, I want a bootstrapped-target variant of the crop-3D arm, so that the label-noise hypothesis (100 hardest scans carry 46 % of the loss) gets a measured answer.
7. As the competitor, I want full-data hemisphere models appended through the extras mechanism, so that the shipped member uses every training scan.
8. As the competitor, I want the stack cell to discover every arm of runs 3–7 in the checkpoint dataset, so that outputs of notebooks run on either Kaggle account combine without editing code.
9. As the competitor, I want the stack limited to the ten best arms by out-of-fold loss, so that the compositions grid stays small and the combiner stays honest.
10. As the competitor, I want the notebook to resume from the checkpoint dataset and upload every four jobs, so that a timeout or cancel loses nothing.
11. As the competitor, I want the container-style check to print per-arm smoke-set scores, so that the smoke-to-public track can be read per member (never fitted on).
12. As the competitor, I want a one-page HOWTO PDF, so that a teammate can launch and fetch the run without this repo's history.
13. As the competitor, I want the runner to use however many GPUs Kaggle allocates, so that a single-GPU allocation still completes.
14. As the competitor, I want the assembly on the laptop to trim the zip below the accepted size if the super-stack grows, so that the DrivenData upload never fails on size.

## Implementation Decisions

- The hemisphere member is a third 3D architecture selectable through the existing model builder (`arch = "hemi"`), reusing the run-3 GroupNorm/SiLU residual blocks; the midline split is at half the crop width; the right half is mirrored along the left/right axis before encoding; the head input is the sum and the absolute difference of the two embeddings, plus the maximum of a shared per-side logit.
- Training reuses the run-3 trainer unchanged except two switches: a loss kind (plain or soft-bootstrapped targets with weight 0.9 on the label) and the architecture name; the site head for DANN uses the member's symmetric pooled features.
- Tags: `r3d_hemi48_f<f>_s<s>[_site]`, `r3d_hemiinv_*`, `r3d_cropboot_*`, `r3d_hemi48_fall_s<s>`; the standard runner (`scripts/train_v3.py`) trains them, so the resume/upload machinery applies unchanged.
- The super-stack cell lists every known arm family (runs 3–7, including the run-6 notebooks' families) as (name, OOF pattern, site pattern); arms without complete OOF coverage (< 1300 uids) are skipped; the ten best by OOF log loss go to the nested combiner with the temperature floor, prior mix and typicality gate of run 5.
- The generator reuses the run-3 cell machinery by executing it in a namespace and dropping its queue and tail cells, so package writefile cells, cache mount, resume and the multi-GPU runner stay identical across notebooks; the runner's GPU list is patched to the device count.
- The end-to-end cell is the run-6a variant that prints per-arm smoke-set scores.
- Inference budget: the hemisphere member costs one crop-3D forward per checkpoint; the container ladder and per-arm stage writes stay as in run 3.

## Testing Decisions

- Unit tests (`tests/test_run7.py`): forward shape and dispatch; exact mirror invariance of the logit (whole-scan flip equals hemisphere swap); one-epoch training on the synthetic tiny cache with the hemisphere architecture, bootstrapped loss and DANN/DRO together, followed by a container-style rebuild from the saved metadata and fp16 checkpoint.
- Real-data smoke: one epoch of the hemisphere member on the real cache on CPU (finite loss, checkpoint written).
- Notebook: generated file validated with nbformat; the generated runner asserted to contain the device-count patch; HOWTO.pdf asserted to be one page.
- Prior art: `tests/test_v3.py`, `tests/test_run6a.py`–`test_run6c.py`, `scripts/local_e2e_v3.py`.

## Out of Scope

- PPMI ingestion and pretraining (wayfinder ticket T11, conditional on access).
- Choosing the final submission members (ticket T08) and the slot plan (ticket T10).
- Registration refinement, single-scan test-time adaptation and distillation (map fog).

## Further Notes

- Honest expectation: hemisphere member 0.245–0.255 OOF; super-stack nested ≈ 0.235–0.242 with the run-5/6 members; public ≈ 0.25–0.255 at today's shift gap. The distance to 0.21–0.22 is the shift gap and external data, not this member alone.
- Run 7 must not run concurrently with another notebook that writes the same tag families into the checkpoint dataset on the same account.
