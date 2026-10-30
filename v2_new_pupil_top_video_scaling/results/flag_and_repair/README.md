# Flag by the fitting deviation, repair by the second peak, accept by the same criterion, 2026-10-30

Scripts: `scripts/fit_threshold.py` (the threshold), `scripts/flag_and_repair.py --video N --accept fit|geometry` (one
video), `scripts/flag_and_repair_all_videos.py` (the nineteen videos together, `all_videos/`). Inputs: the plateau
model's whole-video predictions, the sweep's cached fitting deviations (`results/unreliable_frames_sweep/cache/`), the
heatmap second peaks (`results/heatmap_signals/videoNN_shuffle*_peaks.csv`), the 1330 truth frames. Nothing existing
is modified; the earlier pipeline in `results/second_peak_repair/` is left as it was, with a note at its top.

## Three decisions (user, 2026-10-30)

1. **The threshold is read off the frames that must be found, not set as a budget.** The earlier 44 px was the 95 %
   quantile of the score on the truth frames, i.e. "flag 5 % of frames", a number with no basis in the errors.
2. **Nothing is deleted.** A flagged frame the repair cannot fix stays flagged (`flagged_not_repaired`). Whether such
   frames are left out later is a decision for whoever uses the data, not for this pipeline.
3. **The acceptance test of a repair should be the criterion that flagged the frame**, not a separate hand-made rule.
   The geometry check of `second_peak_repair.py` (5 MAD bands, 0.6 eye width, 0.1 lid margin) was designed on the same
   32 frames it was then judged on. It is kept here only as the comparison (`--accept geometry`).

## The finder and its threshold (`results/fit_threshold/`)

Finder, unchanged: DeepLabCut's fitting outlier score, SARIMAX (2,0,0) fitted on the whole video, one-step forward
prediction, distance between predicted and actual position, largest of the 8 key points (`fit_o200_max`). Points with
likelihood below 0.01 are treated as missing, as in the sweep.

Threshold rule: over the 1330 truth frames, for every integer threshold t, recall = share of the 32 frames with error
> 50 px whose score exceeds t, false share = share of the 1298 frames with error ≤ 50 px whose score exceeds t (user,
10-30: at the 50-px tier every frame at or below 50 px is a good frame, so flagging it is a false flag; the share among
the ≤ 20 px frames is given alongside). Chosen t maximises recall minus false share (the ROC optimum; no free constant).

| threshold | catches of the 32 > 50 px frames | flags among the 1298 ≤ 50 px frames | among the 1175 ≤ 20 px | flags among all truth frames |
|---|---|---|---|---|
| **36 px (chosen)** | **31** | **4.2 %** | 2.0 % | **6.4 %** |
| 44 px (previous, 95 % quantile) | 27 | 3.1 % | 1.4 % | 5.0 % |

Why 36: the 32 scores, sorted, are 11 px (video 14, the frame no detector finds), then 36.3, 36.4, 36.7, 39.8 (four
marginal video-8 frames of 50–55 px), then 45 px, then 26 frames from 56 px up to 580 px. Above 36 the four video-8
frames are lost for a gain of 0.3 % fewer false flags; below 36 no further > 50 px frame is gained until 10 px, where a
quarter of the good frames would be flagged. The choice is the same whether the good frames are counted as ≤ 50 px or
≤ 20 px.

Leave-one-video-out (`lovo.csv`): the same rule on the other 18 videos, applied to the held-out one. 18 of 19 folds
choose 36 px; the fold that holds out video 8 chooses 117 px, because the frames between 36 and 56 px are all video
8's. Over the 19 folds 25 of 32 frames are caught, 38 of 1298 good frames flagged. **The 36 px value therefore rests on
video 8**; without it the data would say "anything between 45 and 117 px". That is a property of the truth set (one
video holds all the marginal errors), to be said as such.

## The repair and the two acceptance tests

On a flagged open frame, every pupil point whose deviation exceeds the threshold is replaced by its own heatmap second
peak (cell × stride + the per-point median offset, as before). Then:

- `--accept fit`: the deviation of the new position from the same one-step prediction is computed; the frame is
  "repaired" if all 8 points are now within 36 px, otherwise it stays flagged and nothing changes. If the point over the
  threshold is an eyelid or corner point nothing is tried. Limitation: the prediction comes from the model's original
  trajectory, so a repaired frame does not change the prediction for the frames after it.
- `--accept geometry`: the earlier geometry rule, with the 36 px threshold. Its "a flagged frame that already passes the
  check" is reported as `flagged_passes_check`, its "drop" as `flagged_not_repaired`.

Both tests, same 19 videos, same threshold (`all_videos/totals.csv`, `comparison.csv`):

| | accept = fitting (same criterion) | accept = geometry (comparison) |
|---|---|---|
| > 50 px frames flagged, of 32 | 31 | 31 |
| … repaired to below 50 px (below 20 px) | 6 (2) | 16 (8) |
| … flagged, not repaired | 21 | 11 |
| … left alone: closed-eye 4, unflagged 1 | 5 | 5 |
| ≤ 50 px frames (good at this tier) flagged, of 1298 | 54 | 54 |
| … repaired / made worse by > 5 px / pushed above 50 px | 20 / 7 / 0 | 10 / 1 / 0 |
| … still flagged | 34 | 9 |
| the same among the ≤ 20 px frames: repaired / made worse / still flagged | 7 / 6 / 16 | 0 / 0 / 3 |
| whole videos: flagged / repaired / still flagged | 6.3 % / 2.2 % / 3.7 % | 6.3 % / 1.6 % / 1.1 % |
| touched share per video, min / median / max | 0.3 / 3.7 / 17.2 % | 0.1 / 1.4 / 11.8 % |

**The same criterion is the worse acceptance test, on both counts.** It repairs 6 of the 32 big errors where the
geometry test repairs 16, and of the 54 good frames (≤ 50 px) that are flagged it makes 7 worse and leaves 34 flagged, where the
geometry test makes 1 worse and leaves 9 flagged (for the rest it says "the frame passes, leave it"). The frame-by-frame comparison
(`all_videos/truth_summary_all.csv`, per-video `truth_frames.csv`) shows why, and it is the same reason both times:
the one-step prediction is built from the preceding frames. When the model is wrong for a stretch, the prediction
follows the wrong trajectory, so the correct second peak lies far from it and is refused (videos 1, 2, 11: six frames
the geometry test repairs to 7–17 px stay flagged). When a good frame is flagged because the eye really moved fast,
the prediction lags behind the true position, and a second peak that happens to sit near the lagging prediction is
accepted although it is farther from the truth (video 3 frame 70575: 9 → 24 px; video 18 frame 59921: 16 → 34 px).
Finding asks "does this frame disagree with the past"; accepting a replacement needs evidence from the frame itself,
which the time series cannot give. So the two steps cannot share one criterion, and the geometry test, hand-made as it
is, is doing the second job. Its constants remain the open point: they were set by hand and should be validated the
same way as the threshold (leave-one-video-out) before the rule is called final.

## Acceptance tests with a basis (`scripts/acceptance_thresholds.py`, `results/acceptance_thresholds/`, same day)

The geometry rule above was made by hand. Three acceptance tests whose quantity has a basis were added and run the same
way (same 36 px flagging, same candidate points and replacement as `--accept fit`, all 19 videos):

- **conf**: DeepLabCut's own confidence applied to the candidate point: the heatmap value of the second peak (the
  smallest over the replaced points). Accept if ≥ 0.21.
- **jump**: DeepLabCut's jump criterion applied to the candidate point: its displacement to the model's position in the
  previous and the next frame, the larger of the two, divided by the video's median eye width. Accept if ≤ 0.167.
- **labelgeo**: the geometry region spanned by the human labels. In the frame's own eye frame (origin = midpoint of the
  corners, unit = that frame's eye width), twelve scale-free quantities of the four pupil points (along coordinate,
  position inside the eyelid band as a fraction, the four ratios of the geometry rule) must each lie inside the min–max
  range of the 1330 human labels. No constant.

The two cuts were read off the truth frames by the rule used for the 36 px threshold: candidate repairs = the 75 truth
frames flagged at 36 px with a pupil point over the threshold; a repair is right if it lowers the frame error by 5 px
or more (40 of 75), wrong if it raises it by 5 px or more (31), neutral otherwise (4); the cut maximises the share of
right repairs accepted minus the share of wrong repairs accepted (`candidates.csv`, `chosen.csv`). Leave-one-video-out
(`lovo.csv`, `label_region_lovo.csv`): the conf cut is 0.21–0.215 in every fold; the jump cut moves between 0.15 and
0.25; the label region built from 18 videos leaves 58 of the held-out videos' 1318 fully labeled frames outside it
(4.4 %), i.e. the region is a little tight when a video is new.

All five acceptance tests on the 19 videos (`all_videos/totals.csv`):

| | fitting (same criterion) | geometry (hand-made) | conf | jump | labelgeo |
|---|---|---|---|---|---|
| > 50 px frames repaired to below 50 px (below 20 px), of 31 flagged | 6 (2) | **16 (8)** | 14 (7) | 5 (2) | 12 (8) |
| > 50 px frames still flagged | 21 | **11** | 13 | 22 | 15 |
| ≤ 50 px frames flagged, of 1298 | 54 | 54 | 54 | 54 | 54 |
| … repaired / made worse by > 5 px | 20 / 7 | **10 / 1** | 30 / 12 | 16 / 4 | 20 / 6 |
| … still flagged | 34 | **9** | 24 | 38 | 34 |
| whole videos: repaired / still flagged | 2.2 % / 3.7 % | 1.6 % / 1.1 % | 3.4 % / 2.4 % | 2.1 % / 3.7 % | 2.4 % / 3.4 % |

- **Of the tests with a basis, the second peak's confidence repairs the most big errors (14 of 31) but also damages
  the most good frames (12 of 54)**: a strong second peak is where the model saw a second plausible pupil edge, and on a
  good frame that was flagged because the eye moved fast the second peak is simply the wrong edge. Jump refuses almost
  everything, right and wrong alike: a point thrown for a stretch of frames has neighbours that are themselves wrong.
  The label region sits between the two (12 repaired, 6 damaged).
- **The hand-made geometry rule still does best on every line**, and the reason is visible in the table: it is the only
  test that first asks whether the flagged frame needs a repair at all. A flagged frame whose pupil points already pass
  the check is left alone (`flagged_passes_check`), which is what happens to most of the 54 flagged good frames; the
  three tests above try to repair every flagged frame with a point over the threshold.

### The label region used in the same two steps (`--accept labelregion`, same evening)

The fair version of the label-region test: a flagged frame whose twelve quantities already lie inside the label range
is left alone (`flagged_passes_check`); only otherwise are the points over the threshold replaced, and the repair is
accepted if the frame is then inside. Two widths of the range were run on all 19 videos: the range as the labels span
it (margin 0, `videoNN_labelregion0/`), and the range widened on both sides by 42 % of each quantity's width
(`videoNN_labelregion/`), 42 % being the smallest margin at which every held-out video's labels fall inside the range of
the other 18 (`label_region_margin.csv`).

| | geometry (hand-made) | label range, two steps, margin 0 | label range, two steps, margin 42 % |
|---|---|---|---|
| > 50 px frames repaired to below 50 px (below 20 px), of 31 flagged | **16 (8)** | 12 (8) | 12 (8) |
| > 50 px frames still flagged / left alone as "passes the check" | 11 / 4 | 13 / 6 | 7 / 12 |
| ≤ 50 px frames repaired / made worse by > 5 px, of 54 flagged | 10 / 1 | 5 / 1 | 2 / 2 |
| ≤ 50 px frames still flagged | **9** | 19 | 8 |
| whole videos: repaired / still flagged | 1.6 % / 1.1 % | 1.0 % / 2.3 % | 0.8 % / 1.6 % |

- **With the two steps the label range comes close to the hand-made rule without any constant**: the same 8 big errors
  brought below 20 px, one good frame made worse, but 4 fewer big errors repaired and twice as many good frames left
  flagged. On the 75 candidate truth frames the unwidened range accepts 20 right repairs and 1 wrong one, the same
  record as the hand-made rule; the difference on the whole set comes from the first step, where the hand-made bands
  (5 MAD around the video's own median) are tighter around a good frame than the pooled min–max of all labels.
- **The margin rule "every held-out label inside" is not a good basis**: the 42 % is set by two or three outlying human
  labels (the count of labels outside falls from 58 at margin 0 to 3 at 20 % and then needs 42 % for the last one), and
  the wider the range the more real errors are waved through as "passes the check" (12 of 31 at 42 %, 6 at margin 0;
  on the 75 candidates the right repairs accepted fall from 20 to 14, `README` table above). If a margin is wanted it
  should be read off the candidate repairs like the other cuts, not off the label coverage.
- Where this leaves the acceptance question: the hand-made geometry rule is still the best test on the truth frames;
  the label range in two steps is the version with a basis and is a little weaker. The honest statement for the
  meeting is that the geometry idea is validated by the label range, and the remaining gap is the width of the bands.

## Spot check by eye, same rule on every video

`all_videos/spotcheck_sample.csv`: per video and per acceptance test, up to 10 repaired and 10 still-flagged frames, one
per episode (touched frames at most 5 frames apart), episodes drawn at random (seed 0); sheets
`all_videos/spotcheck/videoNN_<test>.jpg` (local only; red = model, blue = accepted second peak; caption: frame, status,
points, episode length, fitting deviation). The `verdict` column is empty: real error / false alarm / cannot tell.

## Files

`videoNN_fit/`, `videoNN_geometry/`: `summary.csv` (truth frames), `whole_video.csv`, `truth_frames.csv`;
`frames_all.csv.gz` and `cache/videoNN_pred200.npy` local only. `all_videos/`: `whole_video_all.csv`,
`truth_summary_all.csv`, `totals.csv`, `comparison.csv`, `touched_structure.csv`, `share_per_video.png`,
`spotcheck_sample.csv`.
