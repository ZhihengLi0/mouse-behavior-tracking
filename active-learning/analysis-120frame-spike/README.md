# Why did all three branches' mean error spike at 120 frames (round 2)?

Requested analysis (2026-09-13): find the frames behind the round-2 mean
spike, overlay predictions vs human labels, and explain the mechanism.

## Answer in one line

The "regression" is not a regression of the models - it is **3-5 individual
test frames per branch on which the round-2 checkpoint teleports one or two
keypoints hundreds of pixels into the fur**, while the same frames were fine
under the round-0 baseline; the other ~95 frames are unchanged.

## Evidence

**`01_spike_concentration.png`** - remove the k worst frames and recompute
the mean over the remaining 100-k test frames:

| branch | full-set mean | remove 3 | remove 5 | baseline mean |
|---|---|---|---|---|
| uncertain | 20.4 px | 16.8 px | 15.9 px | 15.3 px |
| jump | 15.5 px | 15.0 px | 14.9 px | 15.3 px |
| fitting | 16.3 px | 15.7 px | 15.4 px | 15.3 px |

jump's frame-level mean barely moved at all (its convergence-table spike is
amplified by the RMSE's squaring); uncertain's entire +5 px lives in ~5
frames; fitting's in ~4.

**`02_culprits_<branch>.png`** - the top culprit frames with all three
sources overlaid (green X = human label, blue square = round-0 baseline
prediction, red dot = round-2 prediction). The pattern repeats across
branches: on frames around t=270-292 s, the baseline model places every
point correctly (14-21 px), while the round-2 model throws `pupil_bottom` /
`eyelid_bottom` / `eyelid_top` 230-504 px into the fur - a spurious heatmap
peak far outside the eye, not a near-miss.

**`culprit_frames.csv`** - the ranked table (frame, time, baseline error,
round-2 error, delta).

## Mechanism

1. Each round retrains from scratch and picks its checkpoint by validation
   mAP - and the 20 validation frames contain no frame resembling these
   culprits (late-minute appearance, partial squint). Checkpoint behavior on
   them is a lottery ticket redrawn every round.
2. Round 2 was the round where each branch's newly added outlier frames were
   most ambiguous (half-blinks; two uncertain frames were even deleted as
   unlabelable under the old rule). Ambiguous or absent supervision on
   eyelid/pupil-bottom appearance makes distant fur textures competitive in
   the heatmap.
3. The round-6/7 fix confirms the diagnosis: once blink-frame eyelids were
   labeled (amended rule), the same statistic dropped below every earlier
   round - supervision, not frame count, was the missing ingredient.

Practical rule for the pipeline: catastrophic teleports are low-likelihood
and hundreds of px from the rolling-median trajectory, so the
`time-series-analysis` residual+likelihood filter catches them without any
labels.
