# Mouse Eye Keypoint Tracking

A reproducible DeepLabCut pipeline that tracks eight keypoints on a mouse eye (pupil top / bottom / left / right,
upper / lower eyelid, nasal / temporal corner) and derives pupil size, eye opening and blinks, with the goal of
aligning eye behaviour with brain recordings. The engineering question behind the current work: **on a new video,
how many human labels are needed, and how few frames must a person correct, before the model is good enough?**

Zhiheng Li (University of Minnesota), with Kaiwen Sheng (Stanford). Raw videos, frames, labels, model weights and
per-frame predictions stay local; git carries code, documentation and audited aggregate results.

## Status (last updated 2026-09-29 23:30 CDT)

| what | state |
|---|---|
| Label standard | era 3 (since 2026-09-20): the pupil is an **ellipse**; its four points are the ellipse endpoints, including the part hidden under the lid |
| Recipe (frozen) | ResNet-50, batch 2, trained from scratch, 120 epochs, LR drops at 96 / 114; headline = final snapshot, median per-frame RMSE over the 8 keypoints on 50 frozen test frames |
| Videos done | video 0 (mouse A, 5 min), videos 1-3 (mouse B "Pluto", 2025-10-31) |
| Finished | **video 4 run 2** (Pluto, 2025-10-30, convention v2): plateau at **40 labels, 15.01 px** (0/20/40/60/80 = 32.11/24.64/15.01/16.82/17.54 px). Run 1 (v1): 140 labels, 14.74 px (archived, tag `v0.8.0-video4-convention1`). Open: the v2 labels mark a much smaller pupil (width 92 vs 215 px) |
| Finished | **video 5** (Pluto, 2025-10-29, convention v2): plateau at **20 labels, 15.85 px** (0/20/40 = 11.90/15.85/16.58 px; the 0-label model is > 50 px off on 12% of frames, 20 labels remove all of them) |
| Finished | **fewer-labels study** (videos 0-5, step 1 retrained with 5 or 10 of its 20 labels, training set otherwise as at that time): 5 labels were enough only on videos 3 and 5; on videos 0, 1, 2 and 4, 5 or 10 labels fall short of 20 (video 4: 10 labels worse than none). 20 labels is the smallest batch that worked on every video; table in `v2_new_pupil_top_video_scaling/results/README.md` |
| Finished | **video 6** (Pluto, 2025-10-28, v2 line continued): test set relabeled under v2 and frozen; 0 labels (video-5 model 631) = 9.36 px, 20 labels = 6.75 px (p90 11.2), 40 labels = 8.46 px, 60 labels = 10.68 px (p90 14.9): finished at **20 labels, 6.75 px** (rule fired at 60; user decision 2026-09-29). |
| Finished | **video 7** (Pluto, 2025-10-27, v2 line): test set relabeled under v2 and frozen; 0 labels (video-6 model 711) = 5.36 px, 20 labels = 5.86 px, 40 labels = 5.84 px: finished at **0 labels, 5.36 px** - the video-6 model already fits this day (user decision 2026-09-29). Next: video 8 (2025-10-24), test set being relabeled under v2 Cross-video curves now also show the 5 / 10-label subset models (open markers) |
| v3 (paused) | `v3_pupil_vedio_scaling/`: videos linked; labelling guide v0.2 records the 2026-09-29 decision (Kaiwen Sheng): the light band around the black core is pupil (fur / eyelash occlusion, IR-LED lightening), so the labeler's outer-edge labels are the reference and the guide's core-based procedure is withdrawn; v3 resumes after the procedure is re-derived from the human labels |
| Blink / area analysis | per-video time series in `<video>/results/blink_area_analysis/`; across videos (`v2_new_pupil_top_video_scaling/results/blink_area_consistency.png`): 4-point area better on 7 of 8 test sets; lowest pupil confidence < ~0.2 separates all 16 human closed-eye frames (the production 0.6 cut flags ~half of open frames); area correlates with eye opening in every video |
| Test-set only | videos 5-8 (2025-10-29 / 28 / 27 / 24): frozen test sets scored by every model so far; videos 9 (10-23) and 10 (10-22): test sets being labeled |

## Main results so far (era 3, `v2_new_pupil_top_video_scaling/`)

Labels needed per video (plateau = two consecutive 20-label steps that improve the best error by <= 3%):

| video | recording | earlier labels in the training set | error: 0 / 20 / 40 / 60 / 80 / 100 own labels | plateau |
|---|---|---|---|---|
| 0 | mouse A, 5 min | 0 | – / 12.93 / 13.00 / 12.04 px | 60 labels, 12.0 px |
| 1 | Pluto, 2025-10-31 | 100 | 208 / 10.37 / 12.17 / 11.35 px | 20 labels, 10.4 px |
| 2 | Pluto, 2025-10-31 | 200 | 9.99 / 7.73 / 7.57 / 7.83 px | 20 labels, 7.7 px |
| 3 | Pluto, 2025-10-31 | 260 | 4.43 / 3.40 / 3.87 / 3.63 px | 20 labels, 3.4 px |
| 4 | Pluto, 2025-10-30 (poor quality) | 320 | run 2 (v2): 32.11 / 24.64 / 15.01 / 16.82 / 17.54 px | 40 labels, 15.0 px (run 1 v1: 140 labels) |
| 5 | Pluto, 2025-10-29 | 360 | 11.90 / 15.85 / 16.58 px (v2) | 20 labels, 15.9 px (0 labels: 12% of frames > 50 px) |
| 6 | Pluto, 2025-10-28 | 380 | 9.36 / 6.75 / 8.46 / 10.68 px (v2) | 20 labels, 6.75 px (rule fired at 60) |
| 7 | Pluto, 2025-10-27 | 400 | 5.36 / 5.86 / 5.84 px (v2) | 0 labels, 5.36 px (rule fired at 40) |

What the numbers say:

- **A new mouse needs its own labels**: models trained only on mouse A score 170-350 px on mouse B; 20 mouse-B
  labels bring it to about 10 px.
- **Same day transfers, a new day does not**: video 3 was at 4.4 px before any of its own labels (thanks to the
  other 2025-10-31 videos), while video 4 (one day earlier, poor image quality) starts at 22.8 px and needs 140 of its own labels to plateau at 14.7 px.
- **Adding videos does not hurt old ones**: every model is back-tested on every frozen test set
  (`v2_new_pupil_top_video_scaling/results/cross_video_curves.png`); earlier videos stay flat as labels are added.
- **Pupil area**: with the new ellipse labels the 4-point area (width x top-to-bottom height) halves the error of
  the old 3-point rule (3.7% vs 6.9% on 196 test frames); per-frame point selection by model confidence does not
  help. The production script still uses the 3-point rule until the switch is approved.
- **Labeling convention v2 (2026-09-26)**: on double contours the LEFT edge is labeled; video 4 is being redone under v2 and the v1 test sets of videos 5-10 will be relabeled.
- **Caveat - anchoring**: test sets are labeled by correcting a model's pre-labels, so the model that made the
  pre-labels looks better than it is on that test set (clearest on videos 5 and 6).

## Repository layout

```
v3_pupil_vedio_scaling/        v3 (from 2026-09-28): same per-video layout, one Pluto video per recording day
  0_first5minvedio/            (spont_1 of each day, verified byte-for-byte against the lab Google Drive);
  1_20251031 ... 8_20251022    the videos were moved here from v2 (no second copy)
v2_new_pupil_top_video_scaling/   era 3 (current): one folder per video, indexed by how many earlier videos are in
  0_first5minvedio/            the training set; each has results/ (tracked) and training-data/ (local only)
  1_... 10_...                 videos 1-10 (mouse B)
    results/selection_sheets/  per video: the cluster + time-series sheet of every selected batch
    results/blink_area_analysis/ per video: eye time series and blink / pupil-area studies
  results/                     across videos: labels-to-plateau table, cross-video back-test, pupil-area study
  scripts/                     frame selection, pre-labels, training step, scoring, back-test, plots
v1_old_pupil_top/              eras 1-2 (5-minute video, earlier label standards) - kept for the record
dlc_projects/                  DeepLabCut workspaces (config tracked; labels and weights local)
environment/                   conda environment and setup check
docs/  report/  scripts/       shared notes, reports and older canonical tools
CHANGELOG.md                   what each tagged version established (table of all tags)
```

## How one video is processed (era 3 protocol)

1. **Split, model-free**: 50 test frames evenly spaced over the final 10% of the video (frozen, report only),
   20 validation frames from the 10% before it, 2-s guard bands, everything earlier is the selection pool.
2. **Batch 1**: 20 frames by k-means on image fingerprints (no model involved).
3. **Each step**: pre-label with the newest model, a person corrects, train from scratch on all labels of the
   earlier videos plus this video's labels so far, score on the frozen test set.
4. **Batch 2 onward**: predict the whole video, flag frames where any keypoint jumps more than 3% of the eye width
   between consecutive frames, pick 20 of them by k-means (at least 1 s apart from each other and from every
   labeled frame).
5. **Stop** when the plateau rule fires; then back-test the new models on every earlier test set.

Every number in a README comes from one script with one formula; if a definition changes, the whole series is
recomputed rather than mixed.

## History

- **Era 1-2 (2026-08 to 2026-09-19, `v1_old_pupil_top/`)**, 5-minute video, earlier label standards: batch size ->
  batch 2; backbone -> ResNet-50 (five backbones tied on accuracy, ResNet-50 won the pre-declared confidence
  tie-break and trains fastest); active learning (uncertainty vs jump vs trajectory-fit detectors) was a clean
  null - about 80 labels saturated that video and detector choice never mattered; a round-6 "improvement" was
  retracted after an audit found a mid-series metric switch (v0.5.1).
- **Era 3 (from 2026-09-20)**: new ellipse label standard, 120-epoch recipe, the per-video scaling experiment
  above. Tags `v0.1.0` ... `v0.9.0-convention2` mark audited states (every finished video, protocol or label
  convention change, and new analysis gets a tag); see the table in `CHANGELOG.md`.

## Next

- Finish video 4 (steps until the plateau rule fires), then continue the one-video-per-day plan across the
  remaining Pluto recordings, tracking the 0-label error of each new day and the error of every old video.
- Decide on switching the pupil area to the 4-point rule; improve blink handling (pupil-point confidence and
  axis-centre offset separate closed-eye frames much better than eyelid distance).
- Estimate how many frames per video still need human correction, and how well the jump rule finds them.
