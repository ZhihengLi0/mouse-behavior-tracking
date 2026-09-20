# Time-series analysis - first results

Task 2 from the 2026-09-12 lab meeting: plot keypoint trajectories over time,
flag frames that deviate from a fitted trajectory, and attribute *when* the
model fails.

**Model**: jump round-5 (shuffle 45), one of the statistically tied round-5
trio; predictions are the existing `analyze_videos` output on the
first-4-minute clip (14,400 frames, 60 fps). No labels are used here - this
is an unsupervised error-mining pass, exactly what a large unlabeled video
would get.

## Method (pre-declared)

- Traces: pupil center x/y, eye opening (|eyelid_top - eyelid_bottom|), and
  **pupil area** (ellipse, pi/4 * width * height - the new derived statistic
  requested at the meeting).
- Fit: 1-second rolling median per trace. Anomaly = residual > 5 robust MADs.
- Confidence gate: min pupil likelihood < 0.6.
- Flagged frames merge into events with 0.25 s gap tolerance.
- `likely_blink` = low confidence AND eye opening below its 5th percentile.

## Result (`01_keypoint_timeseries.png`, `02_flagged_event_frames.png`,
`flagged_events.csv`)

124 events cover 2,059/14,400 frames (14%); 20 events classify as likely
blinks. Findings:

1. **Blinks are unmistakable in the derived traces**: eye opening and pupil
   area dip synchronously (~3-5 s, ~33 s, ~97 s, ~160 s, ~193 s, ~236 s), and
   the likelihood trace drops below 0.6 at the same moments - three
   independent signals agreeing without any labels.
2. **One catastrophic teleport** (pupil_center_y jumps 250 px at ~97 s) is
   exactly the failure mode seen in the round-2 autopsy overlays
   (`active-learning/results/r2_autopsy_*.png`): a spurious distant heatmap
   peak, flagged here automatically by the residual rule.
3. **pupil_left is the chronic low-confidence point** (red in most event
   snapshots), consistent with the autopsy finding that it gets attracted to
   the specular glint.
4. The 14% flagged fraction is deliberately permissive (5 MAD); tightening
   the threshold or requiring two of the three signals would shrink it - to
   be tuned when this becomes the production error-mining filter.

This pass is the prototype of the meeting's long-term goal: on a new
unlabeled video, these flagged events are where minimal human attention
should go first.

## Update 2026-09-14: rerun with the round-10 model

Same rule, stronger model (jump r10, shuffle 50): flagged frames shrink from
2,059 (14.3%) to 1,647 (11.4%) and events 124 -> 120 - model improvement
directly reduces the human-review workload. Likely-blink events drop 20 -> 9
for an instructive reason: with blink-frame eyelids now labeled (amended
rule), the model stays confident during blinks, so the low-confidence gate
rarely fires there. Blink detection should therefore lean on the eye-opening
/ pupil-area dips as the primary signal, with likelihood as a secondary
check - confirming the advisor's call to label eyelids through blinks.

## Update 2026-09-17: spike-by-spike behavioral attribution (advisor request)

`03_spike_behaviors.png`: for each major spike, zoomed traces (pupil x +
eye opening) with start/peak/end video frames. Human-verified verdicts:

- **~108.8 s pupil-x rise: a real SACCADE**, exactly as the advisor
  suspected - square-wave x excursion (622->645 px, held ~1 s, return),
  eye fully open in every frame, confidence high. The v1 rule had
  mislabeled it "blink" because opening grazed its 5th percentile.
- **191.9-193.1 s: NOT a full blink** - also as the advisor suspected.
  Opening dips 270->200 px (~25%, a squint/partial blink; a real closure
  at 233.3 s reaches 185 px within 3 frames). During the squint the pupil
  points teleport (conf 0.14) - the event is "partial blink + tracking
  loss", two phenomena stacked.
- Remaining spikes: 148-150 s = saccade + slow eyelid widening;
  124-126 s / 139.8-143.3 s / 225.5-227.7 s = clusters of small rapid
  x-oscillations with the eye open (micro eye movements / tracking jitter
  between nearby attractors - needs higher-zoom review to separate).

Classifier v2 requirements that follow: add a SACCADE class (center shift
with stable opening), grade blinks by absolute closure depth instead of a
video-relative percentile, and treat low confidence as a tracking-loss
flag rather than blink evidence - fully consistent with the advisor's
"confidence as secondary signal" position.

## Update 2026-09-19: pupil from three keypoints vs four (advisor request)

> **Superseded the same day.** The fixed-ratio variant (k = 0.68) described
> below has been dropped in favor of the ratio-free ENDPOINT method; the
> figure and script now show the endpoint method. The bullets below are kept
> as the record of what was tried. Current result: next section.

`06_pupil_3pt_vs_4pt.png`, `scripts/pupil_3pt_sensitivity.py`. pupil_top is
often hidden under the upper eyelid; how much do pupil size and location
change if only left, right and bottom are used?

- The pupil is modeled as an axis-aligned ELLIPSE throughout (advisor,
  2026-09-12). Such an ellipse has four unknowns and three points give three
  constraints, so one assumption is needed: a fixed aspect ratio (proposal)
  or width = height, i.e. a circle (counter-example).
- **Three points lose nothing** when the ellipse's aspect ratio is fixed at
  the video's own visible height/width ratio (k = 0.68; human labels: 0.71):
  versus the 4-point estimate, area differs by a median 2.3% and the center
  by 2.3 px over 14,400 frames; against the HUMAN 4-point pupil on the 100
  test frames, area error is 8.9% (4-point: 9.1%) and center error 12.3 px
  (12.2 px).
- **A circle through the three points does not work**: +50% area and a
  center 49 px too high, because the visible pupil is not circular in this
  view (height is ~70% of width).
- **The 4-point area is contaminated by eyelid position.** Pupil height
  correlates 0.89 with eye opening (width: 0.57): when the eye narrows the
  lid covers the pupil's upper edge and the visible height shrinks without
  any constriction. 4-point area drops 31% in partial blinks and 46% in deep
  blinks; the width-based area drops 16% and 21%. The "pupil area dips
  during blinks" reported earlier is therefore partly an occlusion artifact:
  fine as a blink cue, misleading as pupillometry. Width-based size is the
  more robust measure, though not fully immune under heavy occlusion.
- **Limitation of the 3-point estimate (noted 2026-09-19): vertical position
  during blinks.** A blink closes the eye from both sides: the visible pupil
  top moves down ~36 px and the bottom moves up ~44-49 px. The 4-point
  center averages the two and barely moves (4-6 px); the 3-point center
  depends on the bottom point alone and moves 29-37 px. So the two
  estimators fail in opposite places: size is steadier from three points,
  vertical position is steadier from four. Horizontal position is identical
  (both use the left-right midpoint), so saccade analysis is unaffected.
  Blink frames should be flagged from eye opening first and excluded from
  position traces under either estimator.
- **Ratio-free variant (2026-09-19): treat left/right as axis ENDPOINTS.**
  The fixed ratio k = 0.68 is specific to this mouse and camera. If the
  left and right labels are the endpoints of the horizontal axis, their mean
  y is the center height and the bottom point gives the half-height, so
  three points determine the ellipse with no constant to transfer
  (`endpoint3` in `pupil_3pt_sensitivity.py`). On the 14,050 frames where all
  four pupil points are confident it gives height/width 0.93, an area 35%
  above the 4-point ellipse and a center 36 px higher. The reason is a
  geometric inconsistency in the four labeled points themselves: the top
  point lies 66 px above the left-right midline but the bottom point 138 px
  below it (human labels: 75 vs 140 px, implied ratio 0.91). A complete
  ellipse would be symmetric, and a tilt of the ellipse cannot produce this
  asymmetry. Either the labeled "top" is the visible edge of a pupil whose
  upper part is covered (then the pupil is nearly round, 0.91-0.93, and the
  4-point ellipse underestimates it), or left/right are systematically
  labeled above the true widest point. The numbers cannot tell these apart;
  it has to be settled on images. Not yet measured: noise of this variant
  (left/right y is poorly defined on a near-vertical edge and its error is
  doubled in the height) and its behavior during blinks.

## Update 2026-09-19 (b): 3-point ENDPOINT ellipse, no fixed ratio - current method

`06_pupil_3pt_vs_4pt.png` (regenerated), `scripts/pupil_3pt_sensitivity.py`.

Definition: left/right are the endpoints of the horizontal axis, so center x
= their mid x, center y = their mean y, width = |R-L|; the bottom point
gives the half-height, height = 2 * (B.y - center y). pupil_top is not used.
No constant, so nothing has to be re-measured for a new mouse or camera.

What it measures: the WHOLE pupil. The upper half is reconstructed by
symmetry from the lower half, whereas the 4-point ellipse measures only the
VISIBLE pupil. Which one is wanted is a scientific choice (advisor).

Numbers (old video, production model; 14,050 frames with all four pupil
points confident; blink rows gated on L/R/B confidence only):

| | 4-pt ellipse | 3-pt endpoint | 3-pt circle | width only |
|---|---|---|---|---|
| height / width | 0.683 | 0.931 | 1 | - |
| median area (px^2) | 47,809 | 64,720 (+35%) | 70,764 (+48%) | - |
| median center y (px) | 486 | 449 (36 px higher) | 436 | - |
| frame-to-frame size jitter, eye open | 0.40% | 0.49% | 0.54% | 0.26% |
| center-y jitter, eye open | 0.29 px | 0.43 px | 0.53 px | - |
| corr(size, eye opening) | +0.90 | +0.75 | +0.44 | +0.54 |
| size change, partial blink (n=45) | -32% | -32% | -8% | -8% |
| size change, deep blink (n=4) | -47% | -44% | -7% | -12% |
| center-y shift, partial / deep blink | -8 / -6 px | 0 / +2 px | -27 / -41 px | - |
| model vs human, same estimator: area error median (p90) | 9.1% (22.3%) | 6.7% (16.5%) | 8.8% (22.0%) | 4.0% (10.9%) |
| model vs human, same estimator: center error median (p90) | 12.2 (20.0) px | 9.2 (16.1) px | 11.0 (22.2) px | - |

Reading:
- **Position: the endpoint method is the most blink-proof** (center y moves
  0-2 px in blinks, 4-point 6-8 px, circle 27-41 px) and the model reproduces
  the human value best (9.2 px vs 12.2 px). Cost: ~1.5x the jitter of the
  4-point center, still under half a pixel.
- **Size: the endpoint method does NOT fix blinks.** Its area falls as much
  as the 4-point area (-32% / -44%), because the lower lid pushes the bottom
  point up and that error is doubled in the height. Only width (and the
  circle, which is dominated by width) is nearly unaffected (-8% / -12%).
  Width is also the quantity the model reproduces best (4.0%) with the least
  jitter. For pupil SIZE the recommendation remains width; blink frames
  should be flagged from eye opening and excluded either way.
- Deep-blink rows rest on 4 frames and are anecdotal.
- Assumption to be confirmed on images: left/right are labeled at the
  pupil's true widest level. The top point sits 66 px above the L-R midline
  and the bottom 138 px below (human labels 75 / 140 px); if that asymmetry
  is occlusion of the upper pupil the endpoint method is right and the pupil
  is nearly round (0.91-0.93); if left/right are labeled systematically high
  the 4-point ellipse is right. L is also 32 px higher than R (a ~6 degree
  tilt that an axis-aligned ellipse ignores).
