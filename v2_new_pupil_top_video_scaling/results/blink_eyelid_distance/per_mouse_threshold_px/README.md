# One eye-opening threshold per mouse, in pixels - second look (2026-10-11), third look (2026-10-12)

> The files in this folder are now those of the **third look** (2026-10-12, `blink_per_mouse_threshold_px_v2.py`: all 19
> videos, truth = both blind samples, 327 verdicts); see the last section. The text of the second look below is kept as
> written (18 videos, 239 verdicts); its numbers are in git history (commit a13addb).


**Basic information.** Videos 0-17 (18 finished videos: mouse A 1, Pluto 10, Terra 7). Input: the whole-video prediction of
each video's plateau model (final snapshot), the same files as the first look in `../per_mouse_threshold/`. Truth for
choosing a threshold: the 239 blind single-frame verdicts of 2026-10-06 (videos 0-13 only; 211 of them on frames with
plausible eyelid points). Script: `../../../scripts/blink_per_mouse_threshold_px.py`. No model was trained and no existing
result was changed. This folder is for the discussion with Kaiwen; it is not part of the summary page.

**Why.** Kaiwen (Slack, 2026-10-11 22:41): use the upper and lower eyelid directly, i.e. the eyelid distance in pixels, not
divided by the eye width; the mouse is placed at the same position every day, so one pixel threshold per mouse. The first
look used r = opening / eye width; this folder repeats it with the opening in pixels and adds a check of the eye position.

**Quantity.** O = y(eyelid_bottom) - y(eyelid_top) in px per frame. A frame has a small opening at threshold T if O < T;
frames with implausible eyelid points (0.02-1.6% per video) fall back to the pupil confidence (< 0.527). The eye width is
used only for the plausibility test and for the crops of the example sheets.

## Is the mouse at the same place every day? (`eye_position_per_video.csv`, `eye_position_per_video.png`)

Eye centre = midpoint of the two eye corners, median over the video, image coordinates (px):

- Pluto: videos 1-7 (10-27 to 10-31) lie within x 520-529 and y 493-514, i.e. within 10 px of each other in x and 21 px
  in y. The three earliest days drift to the lower right: video 8 (10-24) x 540 / y 517, video 9 (10-23) 567 / 523,
  video 10 (10-22) 582 / 540, which is 60 px to the right and 45 px lower than the later days. The same three days have
  the smallest eye width (392-428 px against 417-468 px on the other days) and the smallest median opening (117-156 px).
- Terra: x 602-637, y 499-523 (35 px and 24 px between the extremes); video 14 (10-14) is the leftmost, video 12 the lowest.
- Inside one video the centre moves little: 5th-95th percentile about 10 px either way in x and 10-15 px in y (video 15:
  17 px).

So the position holds to about 10-20 px on most days, small against the openings compared here (120-180 px). Pluto's
first three days (10-22 to 10-24) are the exception: a shift of 45-60 px, a smaller eye width and a smaller opening at
the same time. The points alone cannot tell a different head position or camera distance from the mouse really keeping
its eye narrower on those days. It is the one block of days where a pixel threshold and a width-normalized threshold
disagree: in pixels these days have 45-79% of their frames below Pluto's threshold, in r (first look) 51-75%, so the
direction is the same, but the width normalization pulls video 8 (r 0.363, close to the threshold 0.365) up to the border.

## Thresholds (`threshold_per_mouse.csv`, `threshold_scan_per_mouse.png`)

Chosen on the trusted judged frames of each mouse by the criterion of the first round (largest recall minus false-alarm rate),
middle of the best range; the grid is 60-300 px in steps of 1 px.

| mouse | videos with verdicts | frames (judged not open) | threshold (px) | leave-one-video-out range | accuracy at the threshold | at 120 px | at 140 px | at 160 px | at 180 px |
|---|---|---|---|---|---|---|---|---|---|
| all pooled | 14 | 211 (98) | 149 | 147.5-156.5 | 91.9% | 80.1% | 88.6% | 90.5% | 77.7% |
| mouse A | 1 | 12 (0) | none (no not-open frame was judged) | - | - | 100% | 100% | 100% | 100% |
| Pluto | 10 | 151 (77) | 148 (best range 147-148) | 147.5-160.5 | 94.0% | 78.1% | 90.1% | 93.4% | 78.8% |
| Terra | 3 | 48 (21) | 132 (best range 130-134) | 131-159.5 | 89.6% | 81.2% | 81.2% | 79.2% | 68.8% |

- The two mice come out 16 px apart (148 against 132 px). Terra's value rests on 48 judged frames of 3 videos (11-13) and
  moves up to 160 px when one video is left out, so it is not settled; verdicts on videos 14-17 would be needed.
- Pluto's optimum is flat: accuracy 93-94% from 147 to 160 px. 148 px is r = 0.34 at Pluto's typical eye width (433 px);
  the first look chose r = 0.365 for Pluto, i.e. about 158 px, inside the same flat range.
- Mouse A has one video with the eye wide open throughout (median opening 289 px, never below 240 px); the pooled 149 px is
  used for it in the tables below.

## Share of frames with a small opening (`small_opening_share_per_video.csv`)

| video | mouse | median opening (px) | < 120 px | < 140 px | < 160 px | < 180 px | < mouse threshold |
|---|---|---|---|---|---|---|---|
| 0 | mouse A | 289 | 0.3% | 0.3% | 0.3% | 0.3% | 0.3% (149 px) |
| 1 | Pluto | 180 | 1.0% | 4.9% | 24.1% | 50.2% | 9.1% (148 px) |
| 2 | Pluto | 185 | 0.6% | 1.3% | 14.8% | 43.2% | 3.3% (148 px) |
| 3 | Pluto | 148 | 4.5% | 36.1% | 66.2% | 83.2% | 48.8% (148 px) |
| 4 | Pluto | 166 | 1.1% | 10.8% | 42.2% | 66.1% | 20.5% (148 px) |
| 5 | Pluto | 229 | 0.0% | 0.1% | 0.2% | 0.6% | 0.1% (148 px) |
| 6 | Pluto | 215 | 0.2% | 0.6% | 3.0% | 18.5% | 0.8% (148 px) |
| 7 | Pluto | 203 | 0.2% | 0.2% | 0.4% | 10.5% | 0.3% (148 px) |
| 8 | Pluto | 156 | 10.4% | 36.8% | 53.3% | 67.6% | 44.6% (148 px) |
| 9 | Pluto | 136 | 23.1% | 53.6% | 69.2% | 87.7% | 59.5% (148 px) |
| 10 | Pluto | 117 | 56.1% | 72.6% | 90.7% | 98.0% | 79.4% (148 px) |
| 11 | Terra | 144 | 3.8% | 39.6% | 81.6% | 93.0% | 20.8% (132 px) |
| 12 | Terra | 164 | 5.7% | 18.1% | 46.1% | 70.3% | 11.3% (132 px) |
| 13 | Terra | 203 | 0.3% | 0.3% | 0.4% | 8.3% | 0.3% (132 px) |
| 14 | Terra | 175 | 1.5% | 8.0% | 26.5% | 58.4% | 4.9% (132 px) |
| 15 | Terra | 153 | 30.4% | 41.6% | 54.5% | 63.6% | 36.8% (132 px) |
| 16 | Terra | 214 | 0.0% | 0.1% | 0.3% | 0.9% | 0.1% (132 px) |
| 17 | Terra | 196 | 0.5% | 0.8% | 1.7% | 7.4% | 0.6% (132 px) |

The share depends strongly on the threshold wherever the video's median is close to it: video 3 has 36% of its frames
below 140 px and 66% below 160 px, video 11 40% and 82%. On such days a single share is fragile; the continuous opening
(`opening_distribution_per_video.png`, `opening_timeseries_<mouse>.png`) describes them better, as Kaiwen suggested.
Days with the eye clearly open (videos 5-7, 13, 16, 17) stay below 1% at every candidate up to 160 px.

## Example frames (`examples_<mouse>.png`, `examples_<mouse>.csv`)

One row per candidate threshold (120, 140, 160, 180 px and the mouse's own; a candidate within 5 px of the mouse's own is
dropped), six frames whose opening is closest to it (within 3 px), taken round-robin from the mouse's videos, at least 10 s
apart inside a video, eyelid confidence >= 0.6. Yellow circles = eyelid points, blue = eye corners. Mouse A has no frame
near any candidate, so its sheet is empty. These sheets are what to look at together to settle the thresholds. One tile on
the Terra sheet (video 17, frame 3690, row 132 px) shows fur without the eye: a wrong prediction whose eyelid points still
pass the plausibility test, a reminder that a few of the "small opening" frames are prediction errors, not narrow eyes.

## Open points
- The threshold per mouse is to be chosen by looking at the example sheets together (Kaiwen's plan); 148 px (Pluto) and
  132 px (Terra) are the data's starting point, with a flat optimum of about 145-160 px for Pluto.
- Pluto's three earliest days (position shift, smaller width, smaller opening): whether one pixel threshold applies there
  unchanged is a decision, not a measurement.
- The verdicts cover videos 0-13; Terra's threshold needs more judged frames before it is used.

## Third look (2026-10-12): second blind sample added, all 19 videos

**Basic information.** Videos 0-18 (19 finished videos: mouse A 1, Pluto 10, Terra 8); input as before (plateau model's
whole-video prediction, final snapshot; video 18 added with its 40-label model 1912). Truth: the 239 verdicts of 2026-10-06
(videos 0-13) plus the 88 verdicts of 2026-10-12 on videos 14-18 (`../blink_notopen_frames_combined_all_videos.csv`; 289 of
the 327 on frames with plausible eyelid points). Script `../../../scripts/blink_per_mouse_threshold_px_v2.py` (the second-look
script with the combined truth file); same criterion, same grid, same candidates.

| mouse | videos with verdicts | frames (judged not open) | threshold (px) | best range | leave-one-video-out range | accuracy at the threshold | at 120 px | at 140 px | at 160 px | at 180 px |
|---|---|---|---|---|---|---|---|---|---|---|
| all pooled | 19 | 289 (141) | 156 | 156 | 149-156 | 91.7% | 77.5% | 88.6% | 90.7% | 78.9% |
| mouse A | 1 | 12 (0) | none (no not-open frame judged) | - | - | - | 100% | 100% | 100% | 100% |
| Pluto | 10 | 151 (77) | 148 (unchanged) | 147-148 | 147.5-160.5 | 94.0% | 78.1% | 90.1% | 93.4% | 78.8% |
| Terra | 8 | 126 (64) | **150.5** (was 132 on 48 frames) | 150-151 | 143.5-154.5 | 88.9% | 74.6% | 85.7% | 86.5% | 77.0% |

- With 126 instead of 48 judged frames Terra's threshold moves from 132 to 150.5 px and its leave-one-video-out range
  narrows from 131-160 to 143.5-154.5 px: it is settled now, and it is the same as Pluto's (148 px) within 3 px. One
  threshold of about 150 px would serve both mice; the pooled optimum is 156 px with accuracy 91.7% (90.7% at 160 px).
- Terra's accuracy at its threshold (88.9%) is lower than Pluto's (94.0%) because the two half-covered Terra days (11, 18)
  put many judged frames right at the border.
- Share of frames below the mouse threshold, Terra at 150.5 px (Pluto rows unchanged, see `small_opening_share_per_video.csv`):
  video 11 68.9%, 12 37.1%, 13 0.3%, 14 14.2%, 15 48.9%, 16 0.2%, 17 1.1%, **18 72.2%** (median opening 135 px, the
  narrowest Terra day; 55% of its frames below 140 px, 84% below 160 px). At 132 px these were 20.8 / 11.3 / 0.3 / 4.9 /
  36.8 / 0.1 / 0.6%, so for the borderline days (11, 12, 15, 18) the share depends strongly on the threshold.
- Example sheets `examples_Terra.png` / `examples_Pluto.png` redrawn with the new rows (120, 140, 160, 180 px and the
  mouse's own 150.5 / 147.5 px).

Open points: the threshold is now a data-based proposal for both mice (about 150 px); Kaiwen's look at the example sheets
remains the agreed step before it is used. Pluto's three earliest days (position shift) are unchanged.

