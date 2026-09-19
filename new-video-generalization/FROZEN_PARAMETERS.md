# Frozen protocol - new-video generalization scale experiment

Declared 2026-09-18, before any Pluto frame was labeled. Advisor-approved
goal: *the minimal labeling cost to apply the model to a brand-new video*
(kaiwen, 2026-09-18). These rules are fixed for THIS experiment and are the
template for every future video.

## The one metric (declared first, because a mixed definition once forced a retraction)

Every number on the scale curve comes from ONE script and ONE definition:
per test frame, RMSE over the 8 keypoints = sqrt(mean of squared keypoint
errors); the curve's y value is the MEDIAN of that over the 100 test frames
("median frame RMSE"). Because the project goal concerns the error TAIL (finding
and fixing failing frames) and the median saturates early, the 90th
percentile of frame RMSE and the fraction of frames above 50 px are
reported in every row. The mean over frames and the per-frame mean-absolute
error are reported alongside in explicitly named columns, never substituted.
No headline number is ever computed by an ad-hoc inline snippet.

## Per-video split rule (applies to any new video; fps and length read from the file)

- **Test set**: 100 frames evenly spaced over the final 10% of the video
  (minimum 60 s) are EXTRACTED; labeled once, then frozen. REPORT-ONLY:
  never selects frames, never tunes, never enters training.
  **For Pluto spont_1 the frozen test set is 59 frames, not 100**: the
  annotator labeled the extracted frames in time order and stopped after
  slider positions 0-58 for labeling-cost reasons (2026-09-18, before any
  model had been scored on them). The 41 dropped frames are the LAST 49 s of
  the segment - a contiguous time block, not a difficulty-based exclusion;
  no frame was removed for being a blink or occluded. Within the 59 frames,
  439 of 472 keypoints are labeled: a point is left empty only when it is
  not visible (e.g. pupil points in a closed-eye frame), so errors on
  occluded points are by construction not measured - the per-frame RMSE is
  taken over the labeled points of that frame. Labels were made by
  correcting the unadapted model's predictions; untouched points therefore
  equal that model's output, a small bias in favour of x = 0.
- **Validation set**: 20 frames evenly spaced over the 10% before the test
  segment (minimum 60 s). Labeled once, frozen. Used only to pick the best
  snapshot within each run.
- **Training pool**: everything earlier.
- **Guard bands**: 2 s are removed between pool and validation and between
  validation and test, so no two sets contain near-duplicate frames (at
  60 fps, frames 0.07 s apart are effectively the same image - the era-1
  leakage lesson). Evaluation sets are contiguous end blocks, never
  interleaved with the pool; coverage is the selector's job.
- For the 19.9-min Pluto video: pool [0, 57158), val [57278, 64453),
  test [64573, 71748), guard 120 frames.

## Fixed frame-selection algorithm (implemented in scripts/select_scale_frames.py)

- **Batch 1 (bootstrap)**: k-means (scikit-learn, k = 20, n_init = 10,
  max_iter = 300, random_state = 42) over every 5th pool frame, each frame
  described by an edge fingerprint: difference of Gaussians (sigma 1 minus
  sigma 6) on a 128x96 downscale, reduced to 32x24 and normalized to zero
  mean / unit variance. The medoid of each cluster is picked.
  How this was settled (2026-09-18/19, all before batch 1 was labeled):
  1. Advisor-requested visualization (results/01, 02): appearance space is
     a continuum, so k-means acts as an even partition of it - 20 spread-out
     "ticks" - not a discovery of 20 discrete modes.
  2. Audit: stride, fingerprint resolution and seed change WHICH frames are
     picked but no measurable property of the pick set - fixed by reasoning.
  3. Label-free check on a video whose eye states are known (old video,
     10 k-means seeds, scripts/validate_selector.py): average coverage of
     pupil area / eye opening / pupil position is within noise of random
     picks (0.92 vs 0.88), but ALL FOUR rare extremes (lowest/highest
     opening and pupil area) are covered in 10/10 seeds with edge
     fingerprints, vs 7/10 with raw pixels and 27% of random draws.
  4. Three Pluto sessions (scripts/selector_check.py): edge fingerprints
     double cluster separation (silhouette 0.07 -> 0.15) and raise pick
     diversity on every video. A brightness criterion tried in two forms
     proved non-diagnostic (frame brightness co-varies with eye and face
     state) and was dropped.
  Conclusion: the selection method matters little for common states and
  matters for rare ones; k-means on edge fingerprints is the most reliable
  of the variants tested. Not yet shown on a different mouse.
- **Batch r >= 2 (error-guided)**: analyze the video with the latest trained
  model; flag pool frames where any keypoint moves more than 3% of the
  median eye width between consecutive frames (the `jump` detector, made
  scale-free: 3% is ~20 px on the original ~700-px eye, ~10 px on Pluto);
  k-means (same settings) among flagged frames; medoids picked.
- **All batches**: picks are at least 1 s apart from each other and from
  every previously labeled frame; a batch is exactly 20 frames or the script
  refuses. Re-running a populated set is refused unless --force.
- Single-arm protocol: it measures minimal labeling cost under this
  selection practice and does not re-test selection methods.

## Training and evaluation per scale step

- Training set = all 676 previously reviewed old-video frames (the resource,
  per advisor) + cumulative Pluto batch frames. From scratch each step:
  ResNet-50, batch 2, 100 epochs, LR milestones [80, 95], MPS - the frozen
  recipe, unchanged.
- Best snapshot by Pluto validation mAP; test set scored ONCE per step.
- Curve: x = cumulative Pluto labels, y = external error (mean + median) on
  the frozen Pluto test set. Baseline point x=0 = production_v1 as-is.
- Stopping (scale-free, amended after audit): let B be the running best
  median frame RMSE. The curve is declared flat when two consecutive steps
  each fail to improve B by more than 3% (3% is ~0.5 px on the original
  video; an absolute pixel threshold would not transfer to a smaller eye).
  Because one training run carries ~0.3-0.5 px of run-to-run noise, when the
  rule fires the two flat steps are retrained once with a second seed and
  the verdict stands only if both seeds agree. The x = 0 point (unadapted
  model) is plotted but excluded from the rule.

## Labeling standard

Same as rounds 6+: blink-frame eyelids are labeled whenever visible; pupil
points left empty only when fully occluded. Machine prelabels (production_v1
predictions) are shown for adjustment but never auto-accepted.

## Accounting

The one-time 120 frames (100 test + 20 val) are infrastructure, reported
separately from the scale curve's x axis, which counts training labels only
- identical bookkeeping to the original experiment.
