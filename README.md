# Mouse Eye Keypoint Tracking

A reproducible DeepLabCut pipeline that tracks eight keypoints on a mouse eye
(pupil top/bottom/left/right, upper/lower eyelid, nasal/temporal corner),
derives pupil center/width and eye opening, and is built toward blink
detection and label-efficient generalization to new videos.

The repository is organised as **four sequential projects**, each a
self-contained unit with `scripts/`, `results/`, and (locally) the exact
`training-data/` its numbers rest on:

```
scaling-curve/           How does error scale with 20/50/100 labels?   [superseded]
batch-size-selection/    Which batch size? -> batch 2                  [complete]
model-selection/         Which backbone?   -> ResNet-50                [complete]
active-learning/         Which frame-selection algorithm? [running]
```

Shared infrastructure stays at the root: `dlc_projects/` (the DeepLabCut
workspace holding live labels and model weights), `local_data/` (the held-out
test set and caches), `environment/` (conda env and setup), `scripts/` (the
canonical tools each unit snapshots). Raw videos and papers live in their own
folders and never enter git.

## The data discipline behind every number

```
face.mp4 (5 min, 60 fps)
├── first 4 minutes ── training pool
│     ├── 80 frames   train        (chronologically first; weights learn here)
│     └── 20 frames   validation   (34-236 s; every SELECTION is made here)
└── last minute ────── 100 frames  test (REPORT ONLY - never selects,
                                    never tunes, never stops anything)
```

Temporal block splits prevent near-duplicate leakage (adjacent frames at
60 fps are near-identical); a verified 1.2 s gap separates train from
validation. Selection rules are declared before results are seen. Each unit
pins the exact label tables it used, because the label standard changed once
(2026-09-10: all 200 labels re-reviewed; `pupil_top` moved ~20 px) - numbers
on opposite sides of that boundary are never compared.

## Step 1 - Scaling curve (superseded, kept for the record)

ResNet-50 trained on 20 -> 50 -> 100 labels, judged on the fixed final-minute
set. Error *rose* with more labels (38 -> 42 -> 56 px), driven by a cluster of
catastrophic `pupil_left` failures - the first evidence that label quality and
frame diversity dominate label count, and the first recorded disagreement
between internal validation and the external test. Those findings triggered
everything that followed. Figures are reconstructions (originals predate git);
all numbers use the old label standard.

## Step 2 - Batch size selection -> batch 2

HRNet-W32 at batch 2/4/8/16, reviewed labels, 80/20 block split, 100 epochs,
LR milestones rescaled to [80, 95], snapshots every 10 epochs.

The pre-declared internal rule (lowest validation loss) picked **batch 2** -
and the untouched final-minute set independently ranked batch 2 lowest
(21.07 px). **Internal and external agreed for the first time**, the direct
evidence that the rebuilt 20-frame validation set resolves what the earlier
5-frame one could not. Confidence calibration recovered from 59/800 points
above likelihood 0.6 (old era) to 375/800.

See `batch-size-selection/results/`: overview, learning curves (stars = the
mAP-chosen snapshots), per-keypoint heatmaps, per-frame boxplots, and the
dual-ruler mAP figure.

## Step 3 - Model selection -> ResNet-50

Five bottom-up backbones (ResNet-50, HRNet-W18/W32/W48, CSPNeXt-S) at batch 2
on the identical split and labels. RTMPose-S excluded by design (top-down,
SSDLite detector - not controlled).

On accuracy the five are a statistical tie: validation losses span 1.4%, and a
paired frame-level bootstrap (10,000 resamples) puts zero in every pairwise
95% CI. The pre-declared tie-break - confidence on the validation frames -
separates decisively: **ResNet-50 73%** vs 61% (W32) vs ~13% (rest). The
report-only test set independently agrees (ResNet-50 lowest, 20.10 px), and it
is also the fastest practical trainer (1.7 h vs 3.7 h for W32), which matters
because active learning retrains every round.

A cautionary figure worth opening: `model-selection/results/03_internal_mAP_curves.png`
shows the loss rule and the mAP rule pointing at nearly the same epoch for
four models - but for HRNet-W18 the loss minimum sits at an epoch whose mAP is
64%, exactly why loss-only ranking had crowned the externally-worst model.

## Step 4 - Active learning (running): the convergence curves

Three branches compete under one frozen protocol
(`active-learning/FROZEN_PARAMETERS.md`): identical seed labels, identical
ResNet-50/batch-2/100-epoch training from scratch each round, k-means frame
extraction - the **only** difference is the outlier detector that nominates
candidate frames:

| branch | suspicion logic | knob |
|---|---|---|
| `uncertain` | the model's own low confidence | p_bound = 0.6 |
| `jump` | physically impossible frame-to-frame jumps | epsilon = 20 px |
| `fitting` | deviation from an ARIMA-fitted trajectory | epsilon = 20 px |

Each round: train on the branch's labels -> analyze the first four minutes ->
detector + k-means select 20 unreviewed frames -> human corrects all eight
keypoints -> retrain -> score ONCE on the final-minute set -> one point on
that branch's curve. Five rounds take each branch 80 -> 180 training frames.

The final deliverable is three convergence curves (x = cumulative reviewed
training frames, y = external RMSE). Interpretation rules fixed in advance:
a measured single-run noise band of ±2.4 px (two identical trainings differed
by that much) gates any claim of a lead; the plateau is judged retrospectively
after round 5, never used to stop early; RMSE is reported alongside the
per-frame median because a handful of catastrophic frames (typically blinks)
can dominate the mean - itself a finding that feeds the blink-detection goal.

Live progress: `active-learning/results/convergence.csv` and
`01_convergence_curves.png`.

## History and provenance

- `CHANGELOG.md` - what each tagged version established: `v0.1.0` (old-label
  batch sweep), `v0.2.0` (old-label architecture sweep and the
  checkpoint-matrix analysis that invalidated it), `v0.3.0` (reviewed-label
  batch sweep, first internal/external agreement).
- Superseded-era published packages remain at their tags; the old-label tables
  pinned in `scaling-curve/training-data/` are the only surviving copy of the
  pre-review labels.
- Setup: `environment/` (conda env, install guide, `check_setup.sh`).

Raw videos, labels, model weights, logs, and per-frame predictions stay local
by design; git carries code, documentation, and audited aggregate results.
