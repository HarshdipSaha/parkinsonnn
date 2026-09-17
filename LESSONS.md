# What I learned running a machine-learning competition

A field guide for the next one. Written 2026-09-13, after eleven days and nine runs on a medical-imaging
code-submission challenge that went from rank 152 to rank ~16. The competition specifics do not matter; the
patterns do. Everything below is stated as a rule, with the evidence that earned it.

---

## The arc, in one table

| Run | What changed | Public score | Private score | Final Rank |
|---|---|---|---|---|
| 1 | First working pipeline (`id-321156`) | 0.3391 | 0.3924 | 152 |
| 2–3 | Proper spatial normalisation; a real ensemble (`id-321263`) | 0.2817 | 0.3217 | 66 |
| 4 | More members, nested cross-validation as the decision metric (`id-321298`) | 0.2667 | 0.3099 | 39 |
| 5–7 | Domain-invariance, new representations, a hemisphere-aware member | — | — | — |
| 8 | Ensemble-corrected training targets, all seeds, equal weights (`id-321967`) | 0.2557 | 0.3029 | 23 |
| 8b | Master v3 with tuned calibration ($\lambda=0.01$, `id-322091`) | **0.2473** | **0.3003** | **#50 (Best)** |
| 9 | Master v9 expanded seeds ensemble (91 ckpts, `id-322254`) | 0.2522 | 0.3019 | #51 |

Note the shape. **The first 0.06 came from fixing the pipeline. The next 0.02 came from ensembling. The last
0.01 came from arithmetic on cached predictions.** In the final private test evaluation, the entire leaderboard
shifted up by $+0.04$ to $+0.06$ log loss under unseen hospital camera systems, confirming that cross-scanner
robustness was the deciding factor.

---

## Part 1 — the ten rules

### 1. Every number needs a provenance, or it is a rumour

At one point a session wrote a score into the log that had never happened. A rule was added: *every score must
carry a submission id that exists on the platform*. Eleven days later, the two most important scores in the
project — the ones the entire endgame strategy was built on — had no ids. The rule had lapsed exactly when
time pressure made it matter most.

**Rule:** a number is data only if you can name where it came from. Record the id, the timestamp, and the
configuration in the same line as the value. If you cannot, mark it `[unverified]` and treat it as a guess.

### 2. Check that a claimed gain is *arithmetically possible*

A submission improved by 0.0084 after one constant was changed. That constant blends your prediction toward
the base rate; with the model fixed, its maximum possible effect was **0.0055**, and its realistic effect on
the actual data was **0.0020**. The observed gain was above the hard ceiling. It could not have come from
where everyone believed it came from — but the conclusion "this lever works, pull it harder" had already been
written into the plan.

**Rule:** before believing a result, compute the largest effect the proposed cause *could* have. Post-hoc
transforms, temperature, blending weights and priors all have bounds you can derive in five minutes. A gain
that exceeds its own ceiling is telling you the cause is something else.

### 3. Validation without shift cannot price robustness

Cross-validation on random folds has no distribution shift in it, so it will always prefer sharper, more
confident, less regularised predictions. It measures upside and is blind to tail risk. Every time we asked the
random-fold estimate where a regularisation constant should sit, it said "lower", monotonically, forever.

**Rule:** hold out something that resembles the *shift* you will face — by site, by scanner, by time, by
customer — and make your robustness decisions there. Where the two disagree, the shifted one is right about
regularisation and the random one is right about capacity.

### 4. Know which constraint is actually binding — and re-check it

We believed the constraint was GPU hours. It was not; it was the submission size ceiling. Then it turned out
size was not binding either — a file nearly twice the assumed limit had already been accepted months earlier —
nor was the inference time budget, which had six times the headroom we feared. The real constraint was
**diminishing returns**: ten extra model checkpoints bought 0.0001.

**Rule:** write down what you think is limiting you and test it before optimising against it. Three plans in a
row were shaped by a limit that was not real.

### 5. An average over heterogeneous things is not a unit

"847 MB across 71 checkpoints" became "11.9 MB per checkpoint" and then "we have room for 17 more". The real
sizes ranged from **3.8 MB to 51 MB** depending on the architecture. The budget for the thing we actually
wanted to add was three times larger than we thought.

**Rule:** before dividing, ask whether the numerator is made of one kind of thing. If not, the ratio is a
number with no referent.

### 6. "Wrong under every model" does not mean "irreducible" — if the models share a pipeline

A small set of samples was wrong under a 3D network, a 2.5D network, and a non-neural feature model. The
natural reading was irreducible label noise, and it was supported by the domain's known expert-disagreement
rate. But all three consumed the *same preprocessed input*. Consensus among models that share a pipeline stage
is evidence about the pipeline as much as about the labels.

**Rule:** when several models agree on being wrong, list what they share. Test the shared part before
concluding the data is at fault. And note which grouping you tested — we had shown the errors did not cluster
by *fold*, but folds are random by construction, so that test could never have found a *site* effect.

### 7. Look at the loss distribution, not the loss

6.5 % of samples carried 47 % of the total loss. Any average improvement was therefore mostly noise about
whether a handful of hard cases moved. A full new model generation improved the mean while making the hard
tail **worse by 2.2 nats** — two facts that the headline number hid completely.

**Rule:** plot per-sample loss, sorted, on day one. Decide explicitly whether you are working on the head or
the tail. They need different interventions and they respond to nothing in common.

### 8. Simple aggregation beats clever aggregation, almost always

Fitted ensemble weights scored *worse* than equal weights out of fold. Rank-averaging destroyed calibration.
Isotonic recalibration had 0.0002 of headroom. Dropping "weak" seeds cost 0.011. Nearly every sophisticated
combination step lost to the naive one.

**Rule:** default to equal weights and keep every seed. Spend your complexity budget on *members* — genuinely
different representations — not on the function that combines them. If a fitted combiner beats equal weights
by less than its own noise, it is overfitting and you cannot tell yet.

### 9. Put verification in the tooling, not in your discipline

Our build script could write an oversized submission with no error, and could silently trim the ensemble
*after* the gate had scored the untrimmed one. Our verification script could not see two of the model families
at all — they printed a different log line, so a dead component passed the check. None of this was discovered
by being careful; it was discovered by someone reading the code with the question "how could this lie to me?"

**Rule:** for each check you rely on, ask what it would fail to catch. Then make the failure loud. A gate that
cannot see a dead component will eventually ship one, and you will spend a submission finding out.

### 10. Understand the submission economics before the last week

Submissions were rationed on a rolling window. The final slot *opened* two days before the deadline — and the
plan of record spent it the moment it opened, on a configuration measurably worse than what was already
banked, forfeiting sixty hours of remaining time. The slot was never going to expire. Nobody had drawn the
timeline.

**Rule:** on day one, write down: how many submissions, over what window, what counts for the final standing,
whether free validation runs exist, and what the hard timings are. Free evaluation runs are worth more than
they look — ours were unlimited enough to test every packaging change at zero cost.

### 11. Beware the 3D convolution domain trap; project or sequence instead

In 3D medical imaging spanning heterogeneous hospital equipment, native slice thickness and reconstruction
filters vary widely. Isotropic 3D convolutions easily memorize z-axis interpolation textures and scanner noise
profiles (+0.035 to +0.043 transfer penalty on unseen cameras). Top solutions projected 3D volumes into 2D
multi-plane slabs (DenseNet-121) or processed sequential 2D slices with BiGRUs, collapsing out inter-slice
artifacts and achieving intrinsic camera invariance.

**Rule:** before investing your entire capacity budget into 3D CNNs on reconstructed medical volumes, test
whether a 2D multi-plane projection preserves the clinical signal. A 2D projection discards vendor-specific
z-axis interpolation artifacts that 3D kernels gladly overfit.

---


## Part 2 — what to do, in order, next time

**Week one — earn the right to tune.**
Build the dumbest end-to-end submission that scores, and submit it. You now have a pipeline and a baseline.
Fix the input representation before touching models — spatial normalisation was worth more than every
architecture change combined. Set up cross-validation you trust, including a shift-aware split. Read the rules
properly and write the submission economics into a file.

**Week two — build members, not machinery.**
Add genuinely different representations. Keep a results log where every run has a number, a configuration and
a date. Write down decisions you do not want to re-litigate; we used short architecture decision records and
they stopped at least three circular arguments. Before each run, write the decision rule — "this ships only if
it beats X by Y" — *in advance*, so you cannot rationalise afterwards.

**The endgame — verify more than you build.**
Freeze early. Reserve the last day entirely for packaging, verification and the submission itself. Compute the
hard drop-dead times backwards from the deadline, including queue and scoring. Check that the file you
validated is byte-identical to the one you upload. And know the gap between your local estimate and the real
score — ours was consistently +0.018 to +0.027, which made every local number interpretable.

---

## Part 3 — an honest assessment of how this project was run

**Done well, and worth repeating.** The cross-validation was honest — fixed epochs, no peeking at held-out
folds, decisions anchored to a nested estimate over 1,362 samples rather than to a 20-sample validation set
that was available and tempting. Decisions were written down as they were made. Pre-registered decision rules
meant results could not be rationalised after the fact. The free evaluation runs were used aggressively. When
external data was ruled out on licensing grounds, that was accepted rather than fudged — a real cost, taken
honestly.

**Done badly, and worth fixing.** Provenance discipline collapsed under deadline pressure, precisely when its
value peaked. A headline result was accepted without a sanity check that would have taken five minutes and
would have shown it was impossible. Three consecutive plans were shaped by a resource limit nobody had
measured. And an average was computed over things that were not alike, then used as a budget.

Notice that **every one of those four failures is a verification failure, not a modelling failure.** The
modelling was fine. What went wrong was believing things.

**Was this the right direction?** Yes, with one correction. The trajectory — fix the pipeline, then ensemble,
then calibrate — is the correct order and it produced a real result. The correction is that the effort split
was wrong at the end: in the last week, roughly all the energy went into producing more models and almost none
into checking that what shipped was what had been measured. By that point the marginal model was worth
0.0001 and the marginal verification was worth finding out that a headline number was wrong. **Late in a
competition, verification has a higher expected value than modelling** — and it is free.

---

## The one-page version

1. Every number carries its provenance, or it is a rumour.
2. Check a claimed gain against the maximum its cause could produce.
3. Random-fold validation cannot tell you how much to regularise.
4. Measure the constraint you think is binding.
5. Do not divide by a count of unlike things.
6. Models that share a pipeline agree about the pipeline, not only about the data.
7. Sort the per-sample loss on day one.
8. Equal weights, all seeds. Spend complexity on members.
9. Ask of every check: what would this fail to notice?
10. Draw the submission timeline before the last week, not during it.
11. Beware the 3D convolution domain trap; 2D multi-plane projections or slice sequences generalize better across scanner vendors.

And the meta-rule underneath all eleven: **the failure mode of a competent team is not bad modelling, it is
believing a number nobody checked.**
