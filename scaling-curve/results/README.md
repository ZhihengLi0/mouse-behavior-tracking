# Scaling Study Reconstruction: 20 / 50 / 100 Training Frames

**Era-0 results (August 2026), reconstructed from the session handoff.** The
original figures (`eye_scaling_curve.png`, `eye_architecture_comparison_100train.png`,
the deliverables package) and the model weights (shuffles 1-4) were deleted in
the 2026-09-10 cleanup, and they cannot be regenerated: the weights are gone
and the labels have since been re-reviewed to a different standard. The figures
may still be downloadable from the Slack thread where the deliverables were
shared with the advisor around 2026-08-22 to 2026-08-28.

All numbers below are against the era-0/era-1 labels and the old 95/5 split.
**They are not comparable with anything produced after the 2026-09-10 label
review.** Their value is historical: they document how the project learned
what it learned.

## Design

- Model: ResNet-50 (DLC default), 95/5 split, snapshot-best by internal mAP.
- Training pools: 20, then 50, then 100 manually labelled frames from the
  first 4 minutes of `face.mp4` (expanded cumulatively; the 20-frame seed was
  k-means-extracted, the expansions used a sequential script that concentrated
  frames in the first 30 s).
- Fixed test set: the same 100 manually labelled final-minute frames.

## Scaling table (ResNet-50, fixed final-minute test)

| Train frames | Overall RMSE | Pupil-center RMSE | Pupil-width MAE |
|---:|---:|---:|---:|
| 20 | 38.27 px | 17.83 px | 36.36 px |
| 50 | 41.91 px | 18.44 px | 21.12 px |
| 100 | 55.94 px | 34.78 px | 25.57 px |

The curve was **not monotonic**: more labels made the external error worse.
Error sorting attributed the 100-frame degradation largely to a cluster of
severe `pupil_left` outliers (350-420 px) in specific test frames
(test_042/045/046/047/048/055, test_012, test_016). The 100-frame model had
*better* internal validation than the 50-frame model while being worse
externally - the first recorded instance of the internal/external
disagreement that later ended the first era.

## 100-frame two-model comparison (2026-08-28 audit)

Same 95/5 indices, same test frames, both at 200 epochs:

| Model | Overall RMSE | Pupil-center RMSE | Pupil-width MAE | Likelihood >= 0.6 |
|---|---:|---:|---:|---:|
| ResNet-50 (shuffle 3) | 56.82 px | 35.77 px | 32.48 px | 260/800 |
| HRNet-W32 (shuffle 4) | 25.68 px | 11.96 px | 26.31 px | 0/800 |

HRNet-W32 removed most of the `pupil_left` failure cluster
(`test_050_frame16232.png` remained the worst common frame), which motivated
the switch to HRNet-W32 and the subsequent batch-size sweep (tag v0.1.0).
Its confidence calibration was already known to be unusable
(`pcutoff=0.6` filtered out every point).

## What this study taught the project

1. More labels do not automatically help; label quality and frame diversity
   dominate at this scale.
2. Internal (5-frame) validation and the external test can disagree - later
   measured systematically in the era-1 checkpoint matrix (tag v0.2.0).
3. Structured failures (one keypoint, one frame cluster) matter more than
   average error, which motivated the outlier-montage tooling and eventually
   the active-learning design.

## Where the surviving artifacts are

- Full narrative and numbers: `../AI_HANDOFF.md` (sections around lines
  1590-1700, 2200-2330, 2560-2615).
- Era-1 published packages built on top of this study: main repo tags
  `v0.1.0` (batch sweep) and `v0.2.0` (architecture sweep, checkpoint matrix).
- Original figures: possibly in the advisor Slack thread (deliverables shared
  2026-08-22 to 2026-08-28); nowhere else.
