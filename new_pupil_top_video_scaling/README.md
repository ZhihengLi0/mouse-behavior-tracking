# New pupil standard: labels needed per video

Era 3 of the project (from 2026-09-20). The pupil is labeled as an ELLIPSE: `pupil_top / bottom / left / right`
are the four endpoints of the ellipse fitted to the pupil, including the part hidden under the upper lid.
Eyelids and eye corners keep the earlier strategy (lid points at the middle of each lid arc, corners at the
corners; they only serve blink detection). Numbers from the earlier eras (`../old_pupil_top/`) are not comparable.

Fixed choices carried over from the earlier work: ResNet-50, batch 2, from scratch, final snapshot - now 120 epochs with LR milestones [96, 114] (2026-09-20) -,
k-means (DoG fingerprints) for the first batch, jump rule + k-means for later batches, median per-frame RMSE over
the 8 keypoints on a frozen test set.

Snapshot rules, fixed for every video (confirmed 2026-09-22): the headline number of a step is the FINAL snapshot
(epoch 120); the secondary column is the snapshot with the best validation mAP after the first learning-rate drop
(epochs >= 96, i.e. 100/110/120). The validation split never enters training; if its labels are corrected later,
only the secondary column is recomputed (`scripts/recompute_mAPlate.py`), nothing is retrained.
From the second video on, the curve starts at x = 0: the earlier videos' model applied unchanged
(`scripts/score_prior_model.py`). Each later video is trained with ALL labels of the earlier videos plus its own.

```
scripts/                 shared tools (select_frames.py, make_prelabels.py, label_set.sh, ...)
results/                 ACROSS videos: x = number of videos, y = labels needed to reach the plateau
0_first5minvedio/        video 0 (the 5-minute recording, mouse A); the video copy sits in the unit folder under the same name (<unit>/<unit>.mp4, local only)
  training-data/         local only: frames and labels (test50, val20, batch01, ...)
  results/               WITHIN this video: x = labels on this video (20, 40, 60, ...), y = test error
    selection_sheets/    one sheet per selected batch: clusters + time series of the chosen frames
    blink_area_analysis/ per-video time series (pupil area 4-point / 3-point, eye opening, confidence) and blink / area studies
1_20251031_Pluto_spont_1/ video 1 (mouse B, 19.9 min), same layout
2_20251031_pluton2/      video 2 (mouse B, 2025-10-31, second recording), same layout
3_20251031_pluto3/       video 3 (mouse B, 2025-10-31, third recording), same layout
4_20251030_Pluto_spont_1/ video 4 (mouse B, 2025-10-30; poor image quality), same layout
5_20251029_Pluto_spont_1/ video 5 (mouse B, 2025-10-29): frozen test set only (scored by every model)
6_20251028_Pluto_spont_1/ video 6 (mouse B, 2025-10-28): frozen test set only (scored by every model)
```

Per video: 50 test frames (final 10%, at least 60 s), 20 validation frames (the 10% before), 2-s guard bands,
training batches of 20 from the remaining pool.

## Protocol notes

- **2026-09-20, selector top-up (video 1, batch 3).** With a good model the jump rule flags few frames (389 of
  10,574, in 76 distinct seconds), and the cluster medoids alone gave only 11 frames that are 1 s apart from each
  other and from the 40 frames already labeled. The selector now tops a short batch up with further FLAGGED frames,
  largest jump first, under the same 1-s spacing (and only if that is still not enough, from the 2,000 largest
  jumps of the pool). Batch 3 = 11 medoids + 9 flagged frames; the threshold itself was not relaxed.
