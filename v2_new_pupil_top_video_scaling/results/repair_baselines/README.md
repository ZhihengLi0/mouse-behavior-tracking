# P2 baseline: dropping the flagged frames (zero human cost), 2026-10-26

Script: `scripts/repair_baselines.py`. Inputs: the plateau model's whole-video prediction of every video, the cached
fit_pupil deviation of `results/unreliable_frames_sweep/cache/`, the 1330 step-8 truth frames (`results/error_detector/scores_lovo.csv`).
Nothing of earlier steps is modified.

## Rule

fit_pupil (one-step SARIMAX(3,0,1) deviation, mean of the 4 pupil points) above a share of the video's eye width (median
corner-to-corner distance of the prediction). The share is the quantile on the 1330 truth frames that flags 5 % of them
(`thresholds.csv`): 2.99 % of the eye width, i.e. 11.7–14.8 px depending on the video; the 9.2 % rule is 1.77 % (7–9 px).
Expressed relative to the eye width the rule gives the same detection as the fixed 12.8 px of 8.2 on these 19 videos
(eye widths 392–493 px) and follows the camera distance on a new video.

## Whole videos (`drop_per_video.csv`, 19 videos, 1 309 478 frames)

| rule | flagged frames | share | runs | runs with gaps ≤ 5 merged | single-frame runs | median run | longest run |
|---|---|---|---|---|---|---|---|
| 5 % | 69 907 | 0.2–14.9 % per video | 18 543 | 8 467 | 10 735 | 1–3 frames | 0.4–4.9 s |
| 9.2 % | 131 108 | | 30 169 | 12 464 | 15 092 | | |

- Videos 1, 8, 15, 18 carry most of the flags (10.7–14.9 % of their frames, 1 367–2 961 runs each); videos 0, 5, 7, 16
  below 1 %. A human cannot look at 18 543 runs: per-frame or per-run human repair of the flagged frames does not scale.
  Human cost can only be bounded by routes whose cost is "labels per video" (retraining) or by zero-human routes.
- **Area series after dropping the flagged frames**: the median area moves by 0.0–6.1 % (video 8 +6.1 %, video 6 +3.7 %, video
  18 −2.4 %, video 15 +1.0 %, all others < 1 %); the relative spread (1.4826 MAD / median) falls from 17.3 to 14.1 % in video 1 and from
  56.1 to 50.0 % in video 18, elsewhere by < 1.5 points. Timing impact: none (no value is changed, only removed).
- **Blink series**: the flag is about the pupil, so the recommended default keeps the eye-closure series untouched (it uses
  the eyelid points). For information: 1 014 of the 1 772 closure events of the step-10 rule contain a flagged frame and
  459 start on one; in videos 0, 5, 7, 13, 17 more than 70 % of the flagged frames lie inside closures (the pupil is
  undefined there anyway). Dropping flagged frames from the closure series as well would split events (video 8: 119 → 387)
  and move 459 onsets — not done.
- **Interpolation inside short runs (≤ 5 frames)** would change 0.15–6.6 % of the frames per video (`interp_short_runs_pct`).

## Truth frames (`truth_summary.csv`, 1330 frames)

| rule | flagged | > 20 px caught | > 50 px caught | area error > 20 % caught | left unflagged: > 20 px / > 50 px / area > 20 % | share of the remaining frames with area error > 20 % |
|---|---|---|---|---|---|---|
| none | 0 | – | – | – | 155 / 32 / 206 | 15.5 % |
| 5 % | 67 | 55 (35 %) | 25 (78 %) | 44 (21 %) | 100 / 7 / 162 | 12.8 % |
| 9.2 % | 123 | 77 (50 %) | 28 (88 %) | 64 (31 %) | 78 / 4 / 142 | 11.8 % |

- Dropping removes most of the very wrong frames (> 50 px: 32 → 7 → 4 left, 0.6 % / 0.3 % of the remaining frames) but
  hardly changes the share of frames with an area error above 20 % (15.5 → 12.8 → 11.8 %): the area error is dominated by
  frames the time-series signal does not see (median area error of the flagged frames 37 %, of the unflagged 4.8 %, but
  162 unflagged frames above 20 %).
- Linear interpolation of the flagged frames that lie in short runs (30 truth frames at 5 %): median area error 38.5 → 23.6 %,
  better on 18 frames, worse on 12. Not a repair, a coin toss with a timing cost; reported, not adopted.

## Reading

The flagged frames are the sporadic big errors; dropping them is free, alignment-safe and takes the > 50 px tier down to
0.3–0.6 % of the frames. The moderate, systematic errors (10–50 px on a pupil point, area 10–60 % off) remain after any
flag-and-drop rule at the same 12–13 % of frames; they need a change of the model or of the labels, not a detector.
