# Active Learning: Final Results (rounds 0-5 complete)

> **CORRECTION (2026-09-18).** An internal audit found that the "median frame
> error" on this page mixed two definitions: rounds 0-5 used the per-frame
> RMSE over the 8 keypoints, rounds 6-11 (and production_v1) used the
> per-frame MEAN absolute error, which is ~1.7-2 px lower on the same data.
> The reported "round-6 drop" (17 -> 14-15 px), the "second saturation", and
> the claim that the amended eyelid-labeling rule moved the error floor were
> artifacts of that switch and are RETRACTED. Under one definition the curve
> is flat from round 0 to round 11 (median frame RMSE 15.7-17.5 px, baseline
> 16.95; the amendment's before/after difference is ~0.3 px, inside noise).
> What stands: ~80 labels saturate typical-frame accuracy and nothing up to
> 300 labels moves it; the three detectors are indistinguishable; the
> round-11 stop was correct. production_v1 under the same definition:
> median 16.01 px (baseline 16.95), mean-abs median 13.94 px, 98% of points
> above likelihood 0.6. `convergence.csv` and the figure are recomputed;
> both definitions are now stored in separate, explicitly named columns.
> Sections below are kept for the record; read them through this notice.


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

## Re-scored under the final-snapshot rule (2026-09-19)

`convergence_final_snapshot.csv`, `02_convergence_final_snapshot.png`,
`analysis-final-snapshot/scripts/rescore_final_snapshot.py`. Same 34 models,
same frozen test set; only the snapshot used for scoring changes, from
DeepLabCut's default (best validation mAP over all snapshots) to the final
snapshot (epoch 100). The recipe decays the learning rate at epochs 80 and
95; the default rule picked a pre-decay snapshot in 20 of 34 models (six
times epoch 30).

- **The early mean-error spikes were the snapshot rule.** uncertain rounds 2
  and 3 scored 48.3 and 44.4 px overall RMSE with their mAP-best snapshots
  (epochs 30 and 50) and score 18.6 and 18.1 px with their final snapshots.
  Across rounds 1-11 the overall RMSE has SD 6.89 px under the default rule
  and 1.04 px under the final-snapshot rule. When the default rule chose an
  epoch < 80 the two rules differ by 3.8 px on average; when it chose >= 80,
  by 0.1 px.
- **The saturation result stands and is cleaner**: median frame RMSE stays
  at 15.8-17.5 px from 80 to 300 labels (per-branch slopes +0.08, +0.05,
  -0.21 px per 100 frames); the 90th percentile stays at 21-25 px.
- **Detectors**: branch means over rounds 1-11 are jump 16.40, uncertain
  16.75, fitting 17.03 px (SD ~0.3 each). With the snapshot noise removed a
  small consistent ordering appears, but it is 0.3-0.6 px (under 4%), the
  rounds within a branch are not independent, and each point is one seed -
  practically negligible, not a basis for preferring a detector.
- The 120-frame spike analysis (`analysis-120frame-spike/`) attributed the
  spike to a "checkpoint lottery"; this re-scoring identifies the lottery's
  mechanism exactly.
