# The geometry check as a detector, 2026-10-27

Script: `scripts/geometry_detector.py`. Question (user, 10-27): the per-frame geometry check that gates the second-peak
repair (`results/second_peak_repair/`) is itself a "find the suspicious frame" rule — how does it compare with the
detectors already ranked at the 50-px tier (`results/tier50_reeval/`), and should the finder and the fixer not be one
system? Same 1330 truth frames, same truth (frame RMSE > 50 px, 32 frames), same leave-one-video-out thresholds.

Score: `geo_z` = largest |z| of the four pupil-point ratios (centre = the video's median, scale = the median MAD over the
19 videos, as in the repair script); a hard violation (pupil point outside the corners or the eyelid band, wrong
left/right or top/bottom ordering) adds 1000; closed frames score 0. The fixed rule of the repair script = `geo_z` ≥ 5 or a
hard violation. Files: `signals.csv`, `fixed_rules.csv`, `caught_per_video.csv`, `truth_frames.csv` (whole-video scores
local only).

| score | AUC 50 px (95 % interval) | AUC 20 px | caught of 32 at 1 / 2 / 5 % flagged | precision at 5 % |
|---|---|---|---|---|
| geo_z (geometry check) | 0.857 (0.744–0.966) | 0.824 | 15 / 25 / 27 | 34 % |
| fit_o200_max (chosen at the 50-px tier) | 0.986 (0.972–0.996) | 0.810 | 16 / 23 / 31 | 39 % |
| fit_pupil (8.2) | 0.973 (0.951–0.990) | 0.850 | 16 / 20 / 25 | 34 % |
| rank average (geo_z, fit_o200_max) | 0.930 (0.872–0.983) | 0.844 | 14 / 27 / 27 | 37 % |

| fixed rule | flagged share of the truth frames | caught of 32 | precision |
|---|---|---|---|
| geometry rule (k = 5 or hard violation) | 5.3 % | 27 | 39 % |
| fit_o200_max at 5 % flagged | 6.0 % | 31 | 39 % |
| OR of the two | 7.9 % | 31 | 30 % |
| AND of the two | 3.4 % | 27 | 60 % |

- **As a finder the geometry check is clearly weaker than the chosen time-series detector** (27 vs 31 of 32 at the same
  flagged share; AUC 0.86 vs 0.99) and adds nothing to it: the OR catches the same 31 and flags more, the rank average is
  worse than fit_o200_max alone. Its misses (`caught_per_video.csv`): the two closed-eye frames of video 11 (left alone by
  design), two throws of video 10, one of video 14 — all of them frames where the thrown point still lands inside the eye
  with a plausible pupil shape.
- The AND (both say so) is the high-precision rule: 60 % of its flags are > 50 px frames, 27 of 32 caught at 3.4 % flagged.
- Whole-video flagged share of the geometry rule runs from 0.2 % (videos 5, 16) to 18–21 % (videos 8, 18) and 12.5 %
  (video 15, 11 % hard violations: this video has 4.7 % closed frames and its eyelid band is tight), i.e. it follows
  the video's label/eye-state quality in the same way as the time-series flag share.

**Consequence for the pipeline** (user, 10-27: the finder and the fixer must be one system): the finder stays
fit_o200_max; the geometry check is only the acceptance test of the repair. That version is "system B" in
`results/second_peak_repair/README.md` (`scripts/triggered_repair.py`).
