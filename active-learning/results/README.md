# Active Learning: Final Results (rounds 0-5 complete)

Three outlier detectors competed under one frozen protocol
(../FROZEN_PARAMETERS.md): identical ResNet-50 / batch 2 / 100-epoch training
from scratch each round, identical seed labels, k-means extraction; the only
variable was the detector nominating candidate frames (uncertain p_bound=0.6,
jump epsilon=20, fitting epsilon=20). Each branch grew 80 -> 180 reviewed
training frames over five rounds; every point below is one evaluation on the
report-only final-minute 100 frames.

## Headline result

**Adding 100 targeted labels did not improve external error for any branch,
and the three detectors are statistically indistinguishable.**

- Median frame error stayed flat for all branches across all rounds
  (16.3-17.5 px, entirely inside the ±2.4 px single-run noise band).
- No branch's mean ever finished below the 17.70 px baseline; the mean's
  excursions (up to 48 px) were driven by 3-6 catastrophic frames, not by
  typical-frame quality.
- The frozen plateau rule (two consecutive rounds < 5% improvement) fires at
  round 2 for every branch - improvement never materialised.

## What the flat curve means (the actual finding)

1. ~80 well-reviewed frames already saturate typical-frame performance on
   this test set. The scaling question the lab asked ("how many labels are
   enough?") has a concrete answer at this video's difficulty: about eighty,
   provided the labels are consistent.
2. The residual error budget is not label-starvation. It splits into
   (a) an irreducible-looking ~16-17 px floor on ordinary frames - of the
   same order as human relabeling shifts, i.e. close to label noise - and
   (b) a handful of blink/occlusion frames (led by the recurring
   160.5-160.7 s event and the historic frame16232) where "locate the pupil"
   is ill-posed because the pupil is not visible.
3. Every branch kept selecting those same pathological moments from opposite
   suspicion logics, and the human answer was to leave occluded points empty.
   The productive next step is therefore not more keypoint labels but
   likelihood-gated blink/occlusion handling - exactly the lab's next goal.

## Detector comparison

No winner. All three curves live inside the noise band of each other. jump
ended with the best median (16.26 px), fitting with the best final mean
(18.36 px), uncertain spent two rounds hostage to blink frames before
recovering - all differences are within ±2.4 px single-run noise. Under this
protocol and data scale, detector choice did not matter; frame budget spent
anywhere among the three bought the same (null) improvement.

## Files

- `convergence.csv` - all 18 points (branch, round, cumulative frames, mean,
  median, confidence, evaluated label).
- `01_convergence_curves.png` - mean + median curves with the noise band.
- `round1_selection_report.json` - round-1 selections and collision audit.

Caveats: single run per point (no repeated seeds); the report set, while
never used for selection, is one specific minute of one video; conclusions
are about this data scale and difficulty, not about active learning at large.
