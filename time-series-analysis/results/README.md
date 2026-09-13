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
