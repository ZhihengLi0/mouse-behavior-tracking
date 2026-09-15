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

## Round 6 addendum (2026-09-13): the plateau-verification round

The lab meeting asked for extra rounds to confirm the plateau. Round 6
(+20 frames per branch, 200 training frames) produced the experiment's
first genuinely interesting movement: **all three branch medians dropped
together** - uncertain 15.18 px, jump 14.13 px, fitting 15.17 px - after six
rounds pinned at 16.3-17.5 px, and jump's point sits *below* the ±2.4 px
noise band (band floor 14.55 px). Means stayed unremarkable (17.5-19.9 px,
in/near band).

**Interpretation requires one honest caveat.** Round 6 is the first round
labeled under the amended eyelid rule (blink-frame eyelids are now labeled
rather than left empty), so two variables changed at once: +20 frames AND a
labeling-standard change. A synchronized drop across all three branches is
more consistent with the shared cause (the labeling amendment supplying
eyelid supervision on hard frames) than with three independent detectors
suddenly winning simultaneously. The rounds 0-5 conclusion (saturation under
the *old* labeling standard) stands; round 6 suggests the next gains come
from *what* is labeled, not *how many* - which is itself the meeting's
thesis.

**How to disambiguate** (round 7, if run): candidates are already selected;
labeling them under the same amended rule and watching whether medians keep
falling (labeling-standard effect saturates) or revert (noise) would settle
it. A control - relabeling only the round-6 frames' eyelids under the old
rule and retraining - would isolate the amendment's contribution exactly.

## Final chapter (2026-09-14): second saturation, experiment closed

The extension phase (rounds 6-11, all labeled under the amended blink-eyelid
rule) ended by the pre-declared referee rules:

- Rounds 0-5 (old labeling standard): medians flat at 16.3-17.5 px -
  **first saturation** at ~80 frames.
- Round 6: all three branches dropped together (the amendment supplying
  eyelid supervision on blink frames) - the floor moved to 14-15 px.
- Rounds 7-11: the three branches oscillate in a 14.0-15.4 px band with no
  branch beating its own r8-r10 minimum by more than 0.5 px at round 11
  (uncertain 14.80 -> 15.08, jump 13.97 -> 14.60, fitting 14.82 -> 15.14) -
  **second saturation**, declared by the pre-registered stopping rule at
  round 11 (cap was 12; round-12 labels exist but were never trained on).

Final story in one sentence: **label count saturates quickly (twice), and
the one intervention that moved the floor was changing what gets labeled,
not how much** - the amended eyelid rule bought ~2.3 px (17 -> 14.7) where
120 extra frames under the old standard bought nothing. Detector choice
never mattered at any stage. 12 rounds, 36 trainings, 300 human-reviewed
frames per branch, one held-out minute never touched by any decision except
the two pre-declared referee reads.

## Production model (post-experiment, 2026-09-15)

With the experiment closed, one model was trained on the union of every
human-reviewed label (80 seed + 596 unique frames across all branches and
the untrained round-12 sets; frozen recipe; shuffle 60). Scored once on the
report-only set as `al_production_v1`: **median 13.94 px, mean 14.77 px,
98% of points above likelihood 0.6** - the best of all three metrics in the
project, with the mean and confidence gains (previous best mean 15.07;
selection-time confidence 73%) showing what the merged pool's diversity
buys: fewer catastrophic frames and near-total calibration. This model
(shuffle 60, best snapshot) is the deployment model for new videos.
