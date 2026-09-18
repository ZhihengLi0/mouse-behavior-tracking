# Frozen protocol - new-video generalization scale experiment

Declared 2026-09-18, before any Pluto frame was labeled. Advisor-approved
goal: *the minimal labeling cost to apply the model to a brand-new video*
(kaiwen, 2026-09-18). These rules are fixed for THIS experiment and are the
template for every future video.

## Per-video split rule (applies to any new video)

- **Test set**: 100 frames evenly spaced over the final 10% of the video
  (minimum 1 minute). Labeled once, then frozen. REPORT-ONLY: never selects
  frames, never tunes, never stops anything, never enters training.
- **Validation set**: 20 frames evenly spaced over the 80%-90% segment
  (minimum 1 minute). Labeled once, frozen. Used only to pick the best
  snapshot within each run; adjacent to the test segment so it reflects the
  same late-session conditions.
- **Training pool**: the first 80% of the video. All scale batches come from
  here. Validation and test stay CONTIGUOUS END BLOCKS, never interleaved
  with the pool: at 60 fps, frames 0.07 s apart are near-duplicates, so a
  spread-out validation set is effectively trained on (the era-1 leakage
  lesson). Coverage/diversity is the job of the k-means selection over the
  whole pool, not of the evaluation sets.
  (Amended twice on 2026-09-18, both BEFORE any frame was labeled: absolute
  minutes -> length-relative -> percentage-based. For this 19.9-min video
  all three yield the same frames.)

## Fixed frame-selection algorithm (the "which frames" rule)

Two-phase, pre-declared (amended 2026-09-18 before any labeling):

- **Batch 1 (bootstrap)**: pure k-means on frame appearance (grayscale,
  downsampled 32x24, per-frame zero-mean/unit-variance normalized so
  clusters form on shape rather than illumination, every 5th frame of the
  pool), `numpy` seed 42, k=20. Honest scope: k-means covers the ~20 most
  COMMON appearance modes; rare extremes are deliberately left to the
  error-guided batches, whose job is exactly the tail.
  With the unadapted model ~100% of frames are flagged, so error-guided
  selection has no discriminative power yet; diversity is the only signal.
- **Batch r >= 2 (error-guided)**: analyze the pool with the latest trained
  model, flag candidates with the `jump` outlier detector (epsilon = 20, the
  frozen AL setting; chosen as the preferred detector - best numbers,
  mildest failure modes, calibration-free - while noting the original
  experiment proved the three detectors equivalent only to EACH OTHER, not
  against pure k-means), then k-means (seed 42) among the flagged frames for
  20 new unique picks, collision-safe against all earlier labels.

This is a single-arm protocol: it measures minimal labeling cost under our
best selection practice, and does not re-test selection methods.

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
