# DaT-SPECT classification

Classifying dopamine-transporter SPECT scans as normal or abnormal for the DrivenData DaT Parkinson's Challenge, and shipping that classifier as a self-contained code submission scored by log loss.

## Language

### Scans and labels

**Scan**:
One DaT-SPECT examination: a 3D volume of tracer uptake in the head, identified by its uid.
_Avoid_: image, sample, case, volume (when the examination is meant)

**Uid**:
The eight-character identifier of a scan; the file name without extension.

**Label**:
Whether a scan is abnormal (reduced striatal uptake, value 1) or normal (value 0), as read by the organising radiologists.
_Avoid_: target, class, diagnosis

**Striatum**:
The paired deep-brain structure (caudate and putamen) whose tracer uptake decides the label. Left and right striata are mirror images about the mid-sagittal plane.

**Uptake**:
The tracer signal at a voxel. Specific binding is uptake above the non-specific background.

**Intensity regime**:
The raw-count scale a site stores its scans in: ultra-low (a handful of grey levels), low, or high. A property of the site, not of the patient.

**Site**:
The hospital-and-camera combination a scan came from. Unknown for every scan; approximated by the pseudo-site.

**Pseudo-site**:
An acquisition fingerprint standing in for the site: native voxel spacing, native grid shape and intensity regime. Small fingerprints are merged into "other".
_Avoid_: cluster, domain, scanner

**Public set / private set**:
The two undisclosed partitions of the hidden test scans. The public set feeds the leaderboard; the private set decides prizes.

### Frames

**Head box**:
The fixed 192 × 192 × 144 mm region at 2 mm isotropic spacing centred on a scan's head centre of mass. Every model input is derived from it.
_Avoid_: crop, patch, ROI (for the whole box)

**CoM frame**:
The head box as it comes from the scan: centred on the head centre of mass, orientation and scale as acquired.

**Study template**:
The label-blind mean scan built from training scans, left/right symmetric, that defines the registered frame and the striatal and reference regions.
_Avoid_: atlas, MNI template

**Registration**:
The rigid-plus-scale (and small affine) mapping of a scan's head box onto the study template.
_Avoid_: alignment, normalisation (spatial), warping

**Registered frame**:
A scan after registration: the striatum lies where the template's striatum lies.

**Registration fallback**:
The decision, taken per scan, that a registration is not trusted (implausible transform, agreement with the template below the study threshold, or clearly worse agreement than before registering) and that the CoM frame is used instead.

**Reference region**:
The brain tissue away from the striatum in the registered frame; its mean uptake defines the scan's reference level.

**Reference units**:
Uptake divided by the scan's reference level, so that a value of 1 is non-specific background and the striatum of a normal scan is roughly 2–4.
_Avoid_: normalised intensity, p75 units (that is the older head-mask scaling)

**Striatal plane**:
The template coordinate of the striatal centre along an axis; the axial striatal plane is where slabs are cut.

**Slab**:
A 2D image made by averaging an 8 mm stack of slices; a slab stack is three consecutive slabs around the striatal plane, treated as an image with three channels.
_Avoid_: slice (a slab is thicker than one slice), projection

### Models and validation

**Arm**:
An independent model family that produces one probability per scan from the head box: the 3D arm, the 2.5D arms, the ROI-LR arm.
_Avoid_: branch, head, model type, pipeline

**Checkpoint**:
One trained member of an arm, identified by fold and seed (or "all" for a full-data model). Members of an arm are averaged.
_Avoid_: model (ambiguous), weights file

**Full-data model**:
A checkpoint trained on every labelled scan, with no held-out fold; it contributes to inference but never to out-of-fold predictions.

**Folds**:
The canonical five random stratified partitions of the training scans, fixed since run 2. Site folds are the alternative partition that holds out whole pseudo-sites.

**Out-of-fold prediction**:
A checkpoint's prediction on the fold it never saw; the only basis for fitting the combiner and for reporting a score.
_Avoid_: validation prediction, CV score

**Honest protocol**:
Training for a fixed number of epochs and shipping the final averaged weights, so that out-of-fold predictions involve no choice made on the held-out fold.

**Test-time variants**:
The set of input views (slab-centre shifts and the left/right mirror) whose predictions are averaged for one scan. Defined once and used identically for out-of-fold predictions and for inference.
_Avoid_: TTA (fine in code, not in prose), augmentation (that is training-time)

**Combiner**:
The rule that turns the arms' probabilities into the shipped probability: weighted average of log-odds, one temperature, one prior mix.
_Avoid_: stacker, meta-learner, blender, ensemble (the ensemble is the set of arms; the combiner is the rule)

**Temperature**:
The divisor applied to the combined log-odds; never below 1, fitted on site-fold predictions when available.

**Prior mix**:
The small fraction of the shipped probability replaced by the training prevalence, as insurance against confident errors.

**Gate**:
A numeric threshold a component must pass before it is trusted (for example, the registration gate on ROI-feature discrimination).

### Submissions

**Submission zip**:
The self-contained bundle (entrypoint, code, template, checkpoints, manifest) that DrivenData executes on the hidden test scans.

**Manifest**:
The description of which arms, checkpoints and combiner parameters a submission zip uses.

**Smoke test**:
A free DrivenData run of a submission zip on twenty training scans, scored but not placed on the leaderboard; three per day.

**Real submission**:
A DrivenData run on the full hidden test set; three per rolling seven days, the leaderboard entries.
_Avoid_: normal submission (DrivenData's button label), scored submission

**Slot**:
One of the three real submissions available in a rolling seven-day window.

**Ladder**:
The wall-clock deadlines inside inference after which remaining stages are skipped, keeping the last complete predictions written so far.

**Run**:
One complete training campaign with a fixed design (run 1, run 2, run 3); each run has its own effort in the AI-DLC docs.
