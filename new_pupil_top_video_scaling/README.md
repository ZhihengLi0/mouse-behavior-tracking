# New pupil standard: labels needed per video

Era 3 of the project (from 2026-09-20). The pupil is labeled as an ELLIPSE: `pupil_top / bottom / left / right`
are the four endpoints of the ellipse fitted to the pupil, including the part hidden under the upper lid.
Eyelids and eye corners keep the earlier strategy (lid points at the middle of each lid arc, corners at the
corners; they only serve blink detection). Numbers from the earlier eras (`../old_pupil_top/`) are not comparable.

Fixed choices carried over from the earlier work: ResNet-50, batch 2, from scratch, final snapshot - now 120 epochs with LR milestones [96, 114] (2026-09-20) -,
k-means (DoG fingerprints) for the first batch, jump rule + k-means for later batches, median per-frame RMSE over
the 8 keypoints on a frozen test set.

```
scripts/                 shared tools (select_frames.py, make_prelabels.py, label_set.sh, ...)
results/                 ACROSS videos: x = number of videos, y = labels needed to reach the plateau
first5minvedio/          video 1 (the 5-minute recording)
  training-data/         local only: frames and labels (test50, val20, batch01, ...)
  results/               WITHIN this video: x = labels on this video (20, 40, 60, ...), y = test error
<next video name>/       same layout, one folder per video
```

Per video: 50 test frames (final 10%, at least 60 s), 20 validation frames (the 10% before), 2-s guard bands,
training batches of 20 from the remaining pool.

## Protocol notes

- **2026-09-20, selector top-up (video 1, batch 3).** With a good model the jump rule flags few frames (389 of
  10,574, in 76 distinct seconds), and the cluster medoids alone gave only 11 frames that are 1 s apart from each
  other and from the 40 frames already labeled. The selector now tops a short batch up with further FLAGGED frames,
  largest jump first, under the same 1-s spacing (and only if that is still not enough, from the 2,000 largest
  jumps of the pool). Batch 3 = 11 medoids + 9 flagged frames; the threshold itself was not relaxed.
