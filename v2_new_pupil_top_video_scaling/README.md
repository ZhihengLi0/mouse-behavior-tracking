# New pupil standard: labels needed per video

Era 3 of the project (2026-09-20 to 2026-10-12: all 19 videos finished; since 2026-10-24 the same folder hosts the unreliable-frames work, see `results/`). The pupil is labeled as an ELLIPSE: `pupil_top / bottom / left / right`
are the four endpoints of the ellipse fitted to the pupil, including the part hidden under the upper lid.
Eyelids and eye corners keep the earlier strategy (lid points at the middle of each lid arc, corners at the
corners; they only serve blink detection). Numbers from the earlier eras (`../v1_old_pupil_top/`) are not comparable.

Fixed choices carried over from the earlier work: ResNet-50, batch 2, from scratch, final snapshot - now 120 epochs with LR milestones [96, 114] (2026-09-20) -,
k-means (DoG fingerprints) for the first batch, jump rule + k-means for later batches, median per-frame RMSE over
the 8 keypoints on a frozen test set.

Snapshot rules, fixed for every video (confirmed 2026-09-22): the headline number of a step is the FINAL snapshot
(epoch 120); the secondary column is the snapshot with the best validation mAP after the first learning-rate drop
(epochs >= 96, i.e. 100/110/120). The validation split never enters training; if its labels are corrected later,
only the secondary column is recomputed (`scripts/recompute_mAPlate.py`), nothing is retrained.
From the second video on, the curve starts at x = 0: the earlier videos' model applied unchanged
(`scripts/score_prior_model.py`). Each later video is trained with the carried labels of the earlier videos plus its own
(videos 0-3: the labels that existed when the next video started; from video 4 on: only the labels up to the plateau
point - rule fixed 2026-10-01, table in `results/README.md`).

```
scripts/                 shared tools (select_frames.py, make_prelabels.py, label_set.sh, ...)
results/                 ACROSS videos: x = number of videos, y = labels needed to reach the plateau
0_first5minvedio/        video 0 (the 5-minute recording, mouse A); the video copy sits in the unit folder under the same name (<unit>/<unit>.mp4, local only)
  training-data/         local only: frames and labels (test50, val20, batch01, ...)
  results/               WITHIN this video: x = labels on this video (20, 40, 60, ...), y = test error
    selection_sheets/    one sheet per selected batch: clusters + time series of the chosen frames
    blink_area_analysis/ per-video time series (pupil area 4-point / 3-point, eye opening, confidence) and blink / area studies
1_20251031_Pluto_spont_1/ video 1 (mouse B "Pluto", 2025-10-31, 19.9 min), same layout; finished at 20 labels, 10.37 px
2_20251031_Pluto_spont_3/ video 2 (Pluto, 2025-10-31, Drive session spont_3): 20 labels, 7.73 px
3_20251031_Pluto_spont_2/ video 3 (Pluto, 2025-10-31, Drive session spont_2): 20 labels, 3.40 px
4_20251030_Pluto_spont_1/ video 4 (Pluto, 2025-10-30; poor image quality; run 2 under convention v2): 40 labels, 15.01 px
5_20251029_Pluto_spont_1/ video 5 (Pluto, 2025-10-29): 20 labels, 15.85 px
6_20251028_Pluto_spont_1/ video 6 (Pluto, 2025-10-28): 20 labels, 6.75 px
7_20251027_Pluto_spont1/  video 7 (Pluto, 2025-10-27): 0 labels, 5.36 px (the video-6 model already fits this day)
8_20251024_Pluto_spont1/  video 8 (Pluto, 2025-10-24; hard day): 20 labels, 25.34 px
9_20251023_Pluto1/        video 9 (Pluto, 2025-10-23): 80 labels, 8.20 px
10_20251022_Pluto1/       video 10 (Pluto, 2025-10-22; last Pluto day): 40 labels, 10.32 px
11_20251017_Terra_spont_1/ video 11 (mouse C "Terra", 2025-10-17; first day of a new mouse): 40 labels, 8.86 px
12_20251016_Terra_spont_2/ video 12 (Terra, 2025-10-16): 0 labels, 5.13 px (the video-11 model already fits this day)
13_20251015_Terra_spont_1/ video 13 (Terra, 2025-10-15): 20 labels, 9.43 px
14_20251014_Terra_spont_1/ video 14 (Terra, 2025-10-14): 60 labels, 10.30 px
15_20251013_Terra_spont_1/ video 15 (Terra, 2025-10-13): 40 labels, 9.73 px
16_20251010_Terra_spont_1/ video 16 (Terra, 2025-10-10): 40 labels, 7.15 px
17_20251009_Terra_spont_1/ video 17 (Terra, 2025-10-09): 40 labels, 4.47 px
18_20251008_Terra_spont_1/ video 18 (Terra, 2025-10-08; the last video): 40 labels, 12.38 px
```

Per video: 50 test frames (final 10%, at least 60 s), 20 validation frames (the 10% before), 2-s guard bands,
training batches of 20 from the remaining pool.

## Protocol notes

- **2026-09-20, selector top-up (video 1, batch 3).** With a good model the jump rule flags few frames (389 of
  10,574, in 76 distinct seconds), and the cluster medoids alone gave only 11 frames that are 1 s apart from each
  other and from the 40 frames already labeled. The selector now tops a short batch up with further FLAGGED frames,
  largest jump first, under the same 1-s spacing (and only if that is still not enough, from the 2,000 largest
  jumps of the pool). Batch 3 = 11 medoids + 9 flagged frames; the threshold itself was not relaxed.

Units 11-18 (added 2026-09-30, all finished by 2026-10-12): mouse C "Terra", one recording day each, newest first - `11_20251017_Terra_spont_1`, `12_20251016_Terra_spont_2`, `13_20251015_Terra_spont_1`, `14_20251014_Terra_spont_1`, `15_20251013_Terra_spont_1`, `16_20251010_Terra_spont_1`, `17_20251009_Terra_spont_1`, `18_20251008_Terra_spont_1` (names = the lab Google Drive session names; 10-16 has no spont_1). Same layout as the Pluto units; they start after video 10.
