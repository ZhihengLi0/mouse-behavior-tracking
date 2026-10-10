# One eye-opening threshold per mouse - first look (2026-10-10)

**Basic information.** Videos 0-16 (17 finished videos: mouse A 1, Pluto 10, Terra 6). Input: the whole-video prediction of
each video's plateau model (final snapshot). Truth for choosing a threshold: the 239 blind single-frame verdicts of
2026-10-06 (videos 0-13 only; 211 of them on frames with plausible eyelid points). Script:
`../../../scripts/blink_per_mouse_threshold.py`. No model was trained and no existing result was changed. This folder is
a first look for discussion; it is not part of the summary page.

**Why.** Kaiwen (Slack, 2026-10-07/08): the aim is to know when the eye is opened little, not the blink movement itself; use
one threshold per mouse (the mouse was fixed at the same position and distance every day); for every threshold show a few
frames closest to it and look at them together; the opening may also simply be used as a continuous variable.

**Quantity.** r = eye opening / eye width per frame. Eye opening O = y(eyelid_bottom) - y(eyelid_top). Eye width W = the
median over the video of the corner-to-corner distance (one number per video, as in `blink_notopen_rule.py`). A frame has
a small opening at threshold T if r < T; frames with implausible eyelid points fall back to the pupil confidence (< 0.527).
The threshold is a fixed value of r, not a share of any video's or mouse's median.

## Thresholds (`threshold_per_mouse.csv`, `threshold_scan_per_mouse.png`)

Chosen on the trusted judged frames of each mouse by the criterion of the first round (largest recall minus false-alarm rate).

| mouse | videos with verdicts | frames (judged not open) | threshold | leave-one-video-out range | accuracy at the threshold | at 0.30 | at 0.347 | at 0.40 |
|---|---|---|---|---|---|---|---|---|
| all pooled | 14 | 211 (98) | 0.365 | 0.344-0.365 | 90.5% | 83.9% | 90.5% | 80.1% |
| mouse A | 1 | 12 (0) | none (no not-open frame was judged) | - | - | 100% | 100% | 100% |
| Pluto | 10 | 151 (77) | 0.365 | 0.338-0.365 | 92.7% | 80.8% | 91.4% | 82.1% |
| Terra | 3 | 48 (21) | 0.299 | 0.296-0.356 | 89.6% | 89.6% | 85.4% | 68.8% |

- The pooled optimum is a flat range: 0.347 (the first-round value) and 0.365 give the same accuracy (90.5%).
- Pluto and Terra come out different (0.365 against 0.299), but Terra's value rests on 48 frames of 3 videos and moves
  between 0.296 and 0.356 when one video is left out. It is not settled; verdicts on videos 14-16 would be needed.
- Mouse A has one video and no judged not-open frame; the pooled 0.347 is used for it in the tables below.

## Share of frames with a small opening (`small_opening_share_per_video.csv`)

| video | mouse | median r | r < 0.30 | r < 0.347 | r < 0.40 | r < mouse threshold |
|---|---|---|---|---|---|---|
| 0 | mouse A | 0.586 | 0.3% | 0.3% | 0.3% | 0.3% |
| 1 | Pluto | 0.415 | 1.5% | 11.8% | 41.7% | 22.1% |
| 2 | Pluto | 0.413 | 0.9% | 9.7% | 42.5% | 19.5% |
| 3 | Pluto | 0.356 | 9.4% | 43.6% | 72.2% | 56.3% |
| 4 | Pluto | 0.381 | 3.5% | 26.1% | 60.9% | 40.7% |
| 5 | Pluto | 0.489 | 0.1% | 0.2% | 0.7% | 0.3% |
| 6 | Pluto | 0.473 | 0.4% | 2.2% | 19.6% | 5.8% |
| 7 | Pluto | 0.456 | 0.2% | 0.3% | 9.1% | 0.5% |
| 8 | Pluto | 0.363 | 20.5% | 45.5% | 61.7% | 50.9% |
| 9 | Pluto | 0.330 | 28.9% | 56.6% | 75.5% | 61.8% |
| 10 | Pluto | 0.299 | 51.2% | 69.6% | 88.0% | 75.4% |
| 11 | Terra | 0.344 | 8.9% | 54.2% | 85.9% | 8.4% |
| 12 | Terra | 0.374 | 11.1% | 38.7% | 65.3% | 10.9% |
| 13 | Terra | 0.444 | 0.3% | 0.4% | 11.5% | 0.3% |
| 14 | Terra | 0.387 | 6.5% | 23.0% | 59.8% | 6.4% |
| 15 | Terra | 0.357 | 35.1% | 47.7% | 59.4% | 34.8% |
| 16 | Terra | 0.459 | 0.1% | 0.4% | 2.2% | 0.1% |

The share depends strongly on the threshold wherever the video's median r is close to it: video 11 has 8% of its frames
below 0.299 and 54% below 0.347. On such videos a threshold splits the bulk of the distribution, so the continuous r
(`r_distribution_per_video.png`, `r_timeseries_<mouse>.png`) describes them better than a share.

## Is the eye imaged at the same size every day? (`eye_width_per_video.csv`, `eye_width_per_video.png`)

Median eye width per video: Pluto 392-468 px (19.5% between the smallest and the largest), Terra 417-465 px (11.5%), mouse
A 493 px. This cannot be read as a change of camera distance: across the videos of a mouse the eye width goes with the
opening (correlation of the median width with the median opening in px 0.97 for Pluto and 0.96 for Terra), and inside one
video the width varies as much (5th-95th percentile, e.g. 386-470 px in video 1). The corner-to-corner distance shrinks
when the eye narrows, so it is not an independent scale reference. Consequences: (1) these data neither confirm nor
contradict a fixed imaging scale; (2) dividing by the eye width makes r differ less between days than the opening in px does
(median opening 117-229 px on Pluto, a factor 2.0; median r 0.299-0.489, a factor 1.6).

## Example frames (`examples_<mouse>.png`, `examples_<mouse>.csv`)

One row per candidate threshold (0.30, 0.347, 0.40 and the mouse's own; a candidate within 0.005 of the mouse's own is
dropped), six frames whose r is closest to it (within 0.01), taken round-robin from the mouse's videos, at least 10 s apart
inside a video, eyelid confidence >= 0.6. Yellow circles = eyelid points, blue = eye corners. Mouse A has no frame near
any candidate (its eye stays wide open; median r 0.586), so its sheet is empty.

## Open points
- Which threshold per mouse is to be decided by looking at the example sheets together; the numbers above are a starting point.
- r uses one eye width per video; whether to use the opening in px, r, or a width fixed per mouse is open (see the scale section).
- The verdicts cover videos 0-13; Terra's threshold needs more judged frames before it is used.
