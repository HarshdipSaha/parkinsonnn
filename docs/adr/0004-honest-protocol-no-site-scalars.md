---
status: accepted (2026-09-06)
---

# Fixed epochs with final averaged weights, and no acquisition scalars in the models

Run 2 kept the best epoch on each validation fold and fed eight per-scan scalars (four of which identify the acquisition site with 97 % accuracy) into the model heads. The out-of-fold score was flattered (0.300) and the public score was 0.339; the scalars produced a 0.13–0.20 probability bias when a site was held out. Run 3 trains for a fixed number of epochs, ships the final short-window exponential moving average, and models see only the image. Both choices make out-of-fold numbers slightly worse on paper and were taken deliberately so that the out-of-fold score predicts the leaderboard.

**Consequences**: do not add best-epoch selection or acquisition metadata back "to improve CV"; if a scalar is ever needed it must be image-derived and checked under site folds first; the raw model's validation metrics are logged next to the averaged model's so a lagging average is visible.
