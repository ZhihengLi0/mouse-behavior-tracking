# Frozen protocol - new-video generalization scale experiment

Declared 2026-09-18, before any Pluto frame was labeled. Advisor-approved
goal: *the minimal labeling cost to apply the model to a brand-new video*
(kaiwen, 2026-09-18). These rules are fixed for THIS experiment and are the
template for every future video.

## Per-video split rule (applies to any new video)

- **Test set**: 100 frames evenly spaced over the FINAL 2 minutes. Labeled
  once, then frozen. REPORT-ONLY: never selects frames, never tunes, never
  stops anything, never enters training.
- **Validation set**: 20 frames evenly spaced over minute 15-16 (a segment
  excluded from the training pool). Labeled once, frozen. Used only to pick
  the best snapshot within each training run.
- **Training pool**: 0 to 15 min. All scale batches come from here, so a
  >=1 minute temporal gap separates pool from validation and ~3 minutes from
  test (same anti-leakage logic as the original 80/20/100 design).

## Fixed frame-selection algorithm (the "which frames" rule)

k-means on frame appearance (graysale, downsampled 32x24, every 5th frame of
the pool), `numpy` seed 42. Batch r uses k = 20*r clusters and takes the
frame nearest each centroid, drops frames already labeled in earlier batches
(collision-safe: enlarge k until 20 new unique frames), yielding exactly +20
per batch: cumulative 20, 40, 60, ...

## Training and evaluation per scale step

- Training set = all 676 previously reviewed old-video frames (the resource,
  per advisor) + cumulative Pluto batch frames. From scratch each step:
  ResNet-50, batch 2, 100 epochs, LR milestones [80, 95], MPS - the frozen
  recipe, unchanged.
- Best snapshot by Pluto validation mAP; test set scored ONCE per step.
- Curve: x = cumulative Pluto labels, y = external error (mean + median) on
  the frozen Pluto test set. Baseline point x=0 = production_v1 as-is.
- Stopping: judged retrospectively; plateau = two consecutive steps with
  median improvement < 0.5 px (the round-11 referee rule).

## Labeling standard

Same as rounds 6+: blink-frame eyelids are labeled whenever visible; pupil
points left empty only when fully occluded. Machine prelabels (production_v1
predictions) are shown for adjustment but never auto-accepted.

## Accounting

The one-time 120 frames (100 test + 20 val) are infrastructure, reported
separately from the scale curve's x axis, which counts training labels only
- identical bookkeeping to the original experiment.
