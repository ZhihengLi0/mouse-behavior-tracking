# A parallel fitting rule for the eyelid and corner points (next to fit_pupil)

fit_pupil (page section 8.2) averages the fitting deviation of the 4 pupil points only; an error on an eyelid or
corner point enters its score only through the pupil. Question: does a second rule on the 4 eyelid / corner points
(fit_eye, same fitting deviation averaged over those points), used in parallel ("flag if either fires"), catch the
eyelid / corner errors, and what does it cost against spending the same frames on fit_pupil alone.

## Basic information

| item | value |
|---|---|
| frames and truth | the 1330 step-8 frames (19 videos × 50 test + 20 validation), plateau-point model of each video; wrong = frame RMSE vs human labels > 20 px (155 frames) / > 50 px (32); per-point errors from `results/error_detector/truth_pairs.csv` |
| eyelid / corner errors | of the 155 wrong frames, 12 have their largest error on an eyelid or corner point (nasal corner 5, upper eyelid 4, temporal corner 3); in 10 of the 12 the pupil is also wrong by > 20 px; list in `eye_wrong_frames.csv` |
| signals | `fit_pupil`, `fit_eye` from `results/unreliable_frames_sweep/signals_all_frames.csv` (forward SARIMAX(3,0,1) one-step deviation, mean over the 4 pupil / the 4 eyelid-corner points) |
| thresholds | leave-one-video-out: for a target flagged share, the threshold is the quantile on the other 18 videos, applied to the 19th; realised shares are therefore a little above the target |
| fair comparison | every combined rule is compared with fit_pupil alone at the same realised total flagged share |

Script: `scripts/eye_rule.py` → `eye_rule_table.csv`, `eye_wrong_frames.csv`.

## Result

| rule | flagged | recall > 20 px | precision | recall > 50 px | eyelid / corner wrong caught (of 12) |
|---|---|---|---|---|---|
| fit_pupil alone, target 5 % | 5.6 % | 38.1 % | 79.7 % | 78.1 % | 5 |
| fit_pupil 5 % OR fit_eye 1.5 % | 6.2 % | 38.7 % | 72.3 % | 81.2 % | 6 |
| fit_pupil 5 % OR fit_eye 2.5 % | 7.1 % | 40.6 % | 67.0 % | 81.2 % | 8 |
| fit_pupil alone at 7.1 % | 7.1 % | 46.5 % | 76.6 % | 87.5 % | 6 |
| fit_pupil alone, target 9.2 % | 9.4 % | 51.6 % | 64.0 % | 90.6 % | 6 |
| fit_pupil 9.2 % OR fit_eye 2.5 % | 10.7 % | 54.2 % | 59.2 % | 93.8 % | 9 |
| fit_pupil alone at 10.7 % | 11.1 % | 54.8 % | 57.8 % | 93.8 % | 7 |

Thresholds on all 19 videos: fit_pupil 12.8 px (5 %) and 7.8 px (9.2 %); fit_eye 14.4 px (1.5 %) and 9.3 px (2.5 %).

- The parallel eye rule does what it is meant to do: at +1.5 % frames it raises the eyelid / corner catches from 5–6 to
  8 of 12 (from 6–7 to 9 at the 9.2 % working point).
- It is not free: the same extra frames spent on a lower fit_pupil threshold give a higher total recall (46.5 % vs
  40.6 % at 7.1 % flagged) and a higher precision, because eyelid / corner errors are 8 % of the wrong frames and
  most of them are already caught through the pupil.
- Reading: use the parallel eye rule only if the eyelid / corner points matter downstream (eye width, blink); for
  the pupil, fit_pupil alone is the better use of the same human time. 12 frames is a small base; the exact counts
  are not precise.
