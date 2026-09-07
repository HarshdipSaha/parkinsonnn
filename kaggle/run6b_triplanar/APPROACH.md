# Run 6b — tri-planar late fusion with intensity standardisation (approach B)

## Where we stand (evidence this approach is built on)

| Submission | Nested OOF | Smoke (20 scans) | Public | Rank |
|---|---|---|---|---|
| run 2 (hotspot crops) | 0.2999 (plain) | 0.2567 | 0.3391 | 152 |
| run 3 (registered frame, 44 ckpts) | 0.2603 | 0.2397 | 0.2817 | 66 |
| run 4a (+ crop-3D, V2-S seeds, ConvNeXt, coronal B0, feature arm) | 0.2491 | 0.2176 / 0.2263 | 0.2667 | 39 |

The nested → public gap is a stable ≈ 0.018–0.021 (site shift: OOF AUROC 0.9655 vs public 0.9536), and the 20-scan smoke set
orders the submissions the same way as the public board (public ≈ 1.2–1.3 × smoke). Two facts from run 4a drive this design:

1. **Planes are complementary.** The coronal B0 arm kept weight 0.125–0.375 in every stack even though its own OOF (0.2773) is
   worse than axial V2-S (0.2708): the coronal view carries information the axial slabs do not. The tail of the putamen
   (posterior, the first region to lose uptake in PD) is best seen coronally; the caudate/putamen asymmetry is best seen
   axially; the anterior→posterior gradient along one striatum is a sagittal feature. No member so far sees all three.
2. **Site shift, not calibration, is the gap.** The strong "unrealistic" augmentation hurt on random and site folds, but
   intensity nonlinearity across cameras (quantisation, scatter) is exactly what the new reference-region histogram
   standardisation (`--norm hist`, landmarks in `template.npz`) removes; it has not yet been trained through.

## What the member is

`TriPlanarNet` (`datsc/models_b.py`): one ImageNet backbone (ConvNeXt-Tiny or EfficientNet-V2-S, torchvision commercial-use
weights, stem averaged to 3 slab channels) shared across **four views** of the registered 2 mm box — axial slabs at the template
striatal z, coronal slabs at the striatal y, and one sagittal slab stack per hemisphere centred on the left / right striatal
centroid (x 36.8 / 58.2 voxels from the template mask). Each view's pooled feature gets a learned view embedding; softmax
attention over the views fuses them (`fusion=attn`; `concat` is also implemented). The label-preserving mirror is exact:
axial/coronal images flip their X axis, the two sagittal stacks swap (the right stack is defined as the left window on the
mirrored volume), so training augmentation and test-time flip TTA are anatomically correct. `InputSpecB` (`datsc/train_b.py`)
produces the `[B, 4, 3, 160, 160]` tensor and the same (affine × z-shift × flip) variant set as the run-3 `InputSpec`, so the
existing trainer (EMA-final, mixup, base augmentation, slab-centre jitter, optional DANN/group-DRO) and the container
inference run unchanged through a spec factory (`make_spec`) and a `build_model("tri")` hook.

This follows the current multi-plane literature: cross-plane attention over axial/coronal/sagittal features of a shared 2D
backbone ([Axial-Centric Cross-Plane Attention, arXiv 2602.21636](https://arxiv.org/html/2602.21636)), 2.5D tri-view
fusion with cross-attention ([MOSAIC, 2025, via emergentmind](https://www.emergentmind.com/topics/2-5d-context-aware-model)),
and the finding that 2D foundation backbones over multi-plane slices scale better than 3D nets for volumetric classification
([Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification, arXiv 2512.12887](https://arxiv.org/pdf/2512.12887)).
Our twist is DaT-specific: the two sagittal views are hemisphere-aligned, which makes the mirror exact and lets the shared
backbone see both striata as the "same" image.

## Jobs (`JOBS_B`, ≈ 3.3 h on 2×T4 + 40 min re-OOF)

| Job | Folds | Purpose |
|---|---|---|
| `r2t_convnext_tiny` seed 0 | 5 random | the member (ref normalisation) |
| `r2t_convnext_tiny_hist` seed 0 | 5 random | same with `--norm hist` → does standardisation help a plane-fusion net? |
| `r2t_efficientnet_v2_s` seed 0 | 5 random | second backbone (V2-S was our best single 2.5D backbone) |
| `r2t_convnext_tiny` site folds | 5 site | leave-pseudo-site-out score → compare with B0 0.2818 and crop-3D site folds |
| `r2t_convnext_tiny` seed 1 | 5 random | seed for the stack |

Each job: 24 epochs, 160 px, batch 16, lr 5e-4 (backbone 1e-4), mixup 0.2, EMA 0.99, z-jitter ±1, TTA z ∈ {−1, 0, 1} × flip.
The re-OOF cell re-scores every checkpoint (run-3/4a arms included) with the shared TTA; the stack cell runs the nested
combiner over 9 arms (grid 8, `--typ` available), gated; the e2e cell prints every arm's smoke-set log loss next to its
stack weight and the stack's own smoke score, so the smoke-vs-public ladder can be read before spending a real slot.

## Expected gain (honest)

- Single tri-planar ConvNeXt-Tiny: 0.255–0.265 OOF (single-view ConvNeXt 0.2746, V2-S 3 seeds 0.2708; a fused view with
  4× the training images per backbone parameter usually buys 0.01–0.02).
- In the stack: nested 0.2491 → 0.240–0.245 if the member takes weight ≥ 0.2 (it is the only member seeing sagittal
  information, so its errors should be least correlated with the rest); public ≈ 0.255–0.26 (rank ≈ 12–20 on the Sept 7 board).
- `hist` is a coin flip: it removes camera intensity nonlinearity but compresses the background scale the ImageNet stem
  expects; the two variants are trained side by side and only the better one is kept.

## Inference budget

Per scan per checkpoint: 6 variants × 4 views = 24 backbone passes at 160 px. ConvNeXt-Tiny on an A100 (fp16) ≈ 1,500
img/s → 2,000 scans ≈ 32 s per checkpoint, 15 tri checkpoints ≈ 8 min; the run-4a arms add ≈ 7 min; preprocessing ≈ 3 min
with 20 workers. Well inside the 3 h limit. Zip: 15 × 56 MB (ConvNeXt fp16) + 5 × 42 MB (V2-S) + run-4a members ≈ 1.1 GB if
everything is kept — the assembly step drops zero-weight arms and, if needed, limits ConvNeXt to 10 checkpoints (< 850 MB).

## Risks and mitigations

- 160 px instead of 224 px for the ImageNet backbones: −0.005 expected on single views; offset by 4 views. If the OOF
  disappoints, rerun with `--hw 192 --batch-size 12` (+40 % time).
- fp16 divergence (2 of 15 V2-S folds in run 4a): the trainer now skips non-finite batches and sanitises logits.
- The site-fold run tells whether view fusion is site-robust; if its leave-site-out score is not better than 0.2818, the member
  is kept only for its stack weight, not as a shift fix.
- Nothing here touches the test set collectively; the smoke scans are labelled public data used as a read-out only.

## Files

`datsc/data_b.py` (sagittal slabs, hemisphere centres), `datsc/models_b.py` (TriPlanarNet), `datsc/train_b.py` (InputSpecB,
make_spec), `scripts/train_b.py` (CLI, tags `r2t_<backbone>[_hist]_f<f>_s<s>[_site]`), hooks in `datsc/models_v3.py`
(`build_model("tri")`), `datsc/infer_v3.py` (spec factory), `scripts/reoof_v3.py` (tri arm), `datsc/train_v3.py` (jitter for
tri), `tests/test_run6b.py`, `kaggle/run6b_triplanar/{gen_run6b.py, e2e_b.py, notebook.ipynb, kernel-metadata.json, HOWTO.md, HOWTO.pdf}`.
