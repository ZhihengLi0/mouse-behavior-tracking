# Mouse Eye Keypoint Tracking

**Project page (results, figures, every step): https://zhihengli0.github.io/mouse-behavior-tracking/**
(Chinese; English version: https://zhihengli0.github.io/mouse-behavior-tracking/index_en.html). The page is built from
`report/template.html` by `report/build_report.py` and redeploys on every push; this repository holds the code, the
documentation and the audited aggregate results behind it.

Eight keypoints on a mouse eye (pupil top / bottom / left / right = the endpoints of the pupil ellipse, upper / lower
eyelid, nasal / temporal corner) are tracked with DeepLabCut (3.0, PyTorch engine), an existing tool. What this
repository adds is the work around the tracker: a fixed protocol for labeling, training and evaluating every new video,
the scripts that run it (frame selection, model pre-labels corrected by hand, per-video training and scoring on frozen
test sets, back-tests, detectors for unreliable frames), and pupil size, eye opening and blinks derived frame by frame,
without temporal smoothing, so that the series can be aligned with brain recordings. The engineering question: **on a new video, how many human labels are needed, and how few frames must a person
correct, before the model is good enough?**

Zhiheng Li (University of Minnesota), with Kaiwen Sheng (Stanford). Raw videos, frames, labels, model weights and
per-frame predictions stay local; git carries code, documentation and audited aggregate results.

## Status (last updated 2026-10-28)

| what | state |
|---|---|
| Label standard | since 2026-09-20 the pupil is an **ellipse**: its four points are the ellipse endpoints, including the part hidden under the lid; on a double contour the left edge is labeled (2026-09-26); every point is labeled only when it can be seen clearly, otherwise left empty |
| Recipe (frozen) | ResNet-50, batch 2, trained from scratch, 120 epochs, LR drops at 96 / 114; headline = final snapshot, median per-frame RMSE over the 8 keypoints on 50 frozen test frames per video |
| Labels-per-video experiment | **finished 2026-10-12**: all 19 videos (mouse A: 1 video; mouse B "Pluto": 10 days; mouse C "Terra": 8 days) processed with the same protocol, each closed at its plateau (table below; tags `v0.9.x-videoNN-plateau`, last `v0.9.20-video18-plateau`) |
| Fewer labels | every video retrained with 5 and 10 labels instead of the first 20: 5 labels are enough on 5 of 19 videos, 10 labels on 7; 20 is the smallest batch that worked everywhere (`v2_new_pupil_top_video_scaling/results/README.md`) |
| Across-video analyses | back-test of every model on every test set (new videos do not hurt old ones), error and confidence per keypoint, pupil-area rules, blink rules, whole-video time series — sections 7–11 of the project page |
| Current phase (since 2026-10-24) | **unreliable frames**: find the frames whose tracking cannot be trusted with no extra labels, and repair them at minimum human cost; plan and log in `docs/next_phase_plan_2026-10-24.md`, results under `v2_new_pupil_top_video_scaling/results/` (`unreliable_frames_sweep`, `error_detector`, `tier50_reeval`, `heatmap_signals`, `second_peak_repair`); reviewed with Kaiwen Sheng on 2026-10-31 |

## Main result: labels needed per video

Plateau rule, the same for every video: running best of the final-snapshot error; the plateau is reached when two
consecutive 20-label steps improve the running best by <= 3 %, and the plateau point is the last step that still
improved it. Videos are in the order they were processed; the training set of each step = the labels carried from
the earlier videos (table further down) + this video's own labels so far.

| video | mouse, recording day | labels of earlier videos in the training set | 0 own labels | plateau | error at the plateau |
|---|---|---|---|---|---|
| 0 | mouse A, 5-minute recording | 0 | – | **60 labels** | 12.04 px |
| 1 | Pluto, 2025-10-31 | 100 | 208.23 px | **20 labels** | 10.37 px |
| 2 | Pluto, 2025-10-31 | 200 | 9.99 px | **20 labels** | 7.73 px |
| 3 | Pluto, 2025-10-31 | 260 | 4.43 px | **20 labels** | 3.40 px |
| 4 | Pluto, 2025-10-30 | 320 | 32.11 px | **40 labels** | 15.01 px (poor image quality) |
| 5 | Pluto, 2025-10-29 | 360 | 11.90 px | **20 labels** | 15.85 px |
| 6 | Pluto, 2025-10-28 | 380 | 9.36 px | **20 labels** | 6.75 px |
| 7 | Pluto, 2025-10-27 | 400 | 5.36 px | **0 labels** | 5.36 px (the previous video's model already fits this day) |
| 8 | Pluto, 2025-10-24 | 400 | 27.46 px | **20 labels** | 25.34 px (hard day: 12–16 % of test frames > 50 px at every step) |
| 9 | Pluto, 2025-10-23 | 420 | 23.41 px | **80 labels** | 8.20 px |
| 10 | Pluto, 2025-10-22 | 500 | 18.13 px | **40 labels** | 10.32 px |
| 11 | Terra, 2025-10-17 | 540 | 25.96 px | **40 labels** | 8.86 px |
| 12 | Terra, 2025-10-16 | 580 | 5.13 px | **0 labels** | 5.13 px (the previous video's model already fits this day) |
| 13 | Terra, 2025-10-15 | 580 | 69.13 px | **20 labels** | 9.43 px |
| 14 | Terra, 2025-10-14 | 600 | 13.33 px | **60 labels** | 10.30 px |
| 15 | Terra, 2025-10-13 | 660 | 10.57 px | **40 labels** | 9.73 px |
| 16 | Terra, 2025-10-10 | 700 | 11.35 px | **40 labels** | 7.15 px |
| 17 | Terra, 2025-10-09 | 740 | 6.38 px | **40 labels** | 4.47 px |
| 18 | Terra, 2025-10-08 | 780 | 38.24 px | **40 labels** | 12.38 px |

16 of the 19 videos reached their plateau with <= 40 of their own labels (9 with <= 20 or none); 820 labels in total over
19 videos. Full per-step series, p90 errors, confidence shares and the decisions taken: `<unit>/results/README.md`
of each video and `v2_new_pupil_top_video_scaling/results/README.md`.

What the numbers say:

- **A new mouse needs its own labels**: models trained only on mouse A score 170–350 px on mouse B; 20 mouse-B
  labels bring it to about 10 px. The first Terra video starts at 26 px and needs 40 labels.
- **Same day transfers, a new day may not**: video 3 was at 4.4 px before any of its own labels (thanks to the
  other 2025-10-31 videos); videos 7 and 12 needed no labels at all, while video 13 started at 69 px one day after
  video 12 and needed 20.
- **Adding videos does not hurt old ones**: every model is back-tested on every frozen test set
  (`v2_new_pupil_top_video_scaling/results/cross_video_curves.png`); earlier videos stay flat as labels are added.
- **Hard days stay hard**: videos 4, 5 and 8 (poor image quality, glints, a partly covered pupil) keep 15–25 px
  medians and a tail of frames > 50 px at every label count — these frames are the subject of the current phase.
- **Pupil area**: with the ellipse labels the four-point area has the lowest error (median 6.6 % on the test
  sets); eye closure is detected best by the mean confidence of the four pupil points.
- **Caveat — anchoring**: test sets are labeled by correcting a model's pre-labels, so the model that made the
  pre-labels looks a little better than it is on that test set.

## Current phase: unreliable frames (from 2026-10-24)

Truth = the 1330 human-labeled frames of the 19 videos scored against the plateau models; a frame counts as
unreliable when its RMSE is > 50 px (32 frames) — the tier that must be fixed before the time series can be used.
Finding: no label-free signal beats the time-series fitting deviation of the keypoints; the best version (largest
deviation over the 8 points, SARIMAX (2,0,0), threshold at 5 % of frames flagged) catches 31 of the 32 frames
(`results/tier50_reeval/`). Repair: on a flagged frame the deviating pupil points are replaced by the second peak of
the model's heatmap and the frame is kept only if the pupil geometry is back in the normal range, otherwise the
frame is dropped — no temporal smoothing, unflagged frames untouched (`results/second_peak_repair/README.md`, ten
videos). Numbers of this phase are interim until the 2026-10-31 review.

## Repository layout

```
v2_new_pupil_top_video_scaling/   the main experiment: one folder per video, indexed by how many earlier videos are in
  0_first5minvedio/            the training set; each has results/ (tracked) and training-data/ (local only)
  1_... 10_...                 videos 1-10 (mouse B, Pluto, 2025-10-31 -> 10-22); 11_... 18_... videos 11-18 (mouse C, Terra, 2025-10-17 -> 10-08)
    results/selection_sheets/  per video: the cluster + time-series sheet of every selected batch
    results/blink_area_analysis/ per video: eye time series and blink / pupil-area studies
  results/                     across videos: labels-to-plateau table and figure, cross-video back-test, pupil-area,
                               blink, confidence studies, and the unreliable-frames work of the current phase
  scripts/                     frame selection, pre-labels, training step, scoring, back-test, plots, detectors, repair
keypoint_definitions/          where each of the eight keypoints is placed, what changed, common mistakes (example figures)
report/                        the project page (template, build script, built pages)
docs/                          plans, meeting-driven notes, literature notes
v1_old_pupil_top/              eras 1-2 (5-minute video, earlier label standards) - kept for the record
dlc_projects/                  DeepLabCut workspaces (config tracked; labels and weights local)
environment/                   conda environment and setup check
CHANGELOG.md                   what each tagged version established (table of all tags)
```

## How one video is processed

1. **Split, model-free**: 50 test frames evenly spaced over the final 10 % of the video (frozen, report only),
   20 validation frames from the 10 % before it, 2-s guard bands, everything earlier is the selection pool.
2. **Batch 1**: 20 frames by k-means on image fingerprints (no model involved).
3. **Each step**: pre-label with the newest model, a person corrects, train from scratch on the carried labels of
   the earlier videos (table below) plus this video's labels so far, score on the frozen test set.
4. **Batch 2 onward**: predict the whole video, flag frames where any keypoint jumps more than 3 % of the eye width
   between consecutive frames, pick 20 of them by k-means (at least 1 s apart from each other and from every
   labeled frame).
5. **Stop** when the plateau rule fires; then back-test the new models on every earlier test set.
6. **Next video**: it starts from the plateau, and only this video's labels up to its plateau point are carried
   into the training sets of later videos.

### Labels carried into later videos (rule fixed 2026-10-01)

A video is finished when it reaches its plateau, the next video starts from there, and only the labels up to the
plateau point go into later training sets. This mirrors real use, where labeling stops once the plateau is reached.
Labels made beyond the plateau point stay on disk and are not used by later videos. Videos 0-3 carry the labels that
existed when the next video started (320 instead of 120); these 320 labels are the same in the training set of every
later model, so the comparison between later videos is not affected.

| video | labeled in total | plateau point | carried into later videos |
|---|---|---|---|
| 0 | 100 | 60 | 100 |
| 1 | 140 | 20 | 100 |
| 2 | 80 | 20 | 60 |
| 3 | 60 | 20 | 60 |
| 4 (run 2) | 80 | 40 | 40 |
| 5 | 40 | 20 | 20 |
| 6 | 60 | 20 | 20 |
| 7 | 40 | 0 | 0 |
| 8 | 60 | 20 | 20 |
| 9 | 120 | 80 | 80 |
| 10 | 80 | 40 | 40 |
| 11 | 80 | 40 | 40 |
| 12 | 40 | 0 | 0 |
| 13 | 80 | 20 | 20 |
| 14 | 120 | 60 | 60 |
| 15 | 80 | 40 | 40 |
| 16 | 100 | 40 | 40 |
| 17 | 100 | 40 | 40 |
| 18 | 80 | 40 | – (last video; 820 labels would be carried into a next one) |

Every number in a README comes from one script with one formula; if a definition changes, the whole series is
recomputed rather than mixed.

## History

- **Era 1-2 (2026-08 to 2026-09-19, `v1_old_pupil_top/`)**, 5-minute video, earlier label standards: batch size ->
  batch 2; backbone -> ResNet-50 (five backbones tied on accuracy, ResNet-50 won the pre-declared confidence
  tie-break and trains fastest); active learning (uncertainty vs jump vs trajectory-fit detectors) was a clean
  null - about 80 labels saturated that video and detector choice never mattered; a round-6 "improvement" was
  retracted after an audit found a mid-series metric switch (v0.5.1).
- **Era 3 (2026-09-20 to 2026-10-12)**: ellipse label standard, 120-epoch recipe, the per-video experiment above
  on all 19 videos; the project page went online on 2026-10-24.
- **Since 2026-10-24**: unreliable frames (detect, repair). Tags mark audited states; see the table in `CHANGELOG.md`.

## Next

- 2026-10-31: review the unreliable-frames results with Kaiwen Sheng (detector, repair rule, what each video loses).
- Keep one pipeline and run exactly the same experiments on every video; update the project page only with final results.
