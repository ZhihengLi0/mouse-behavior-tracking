# Video 0: the 5-minute recording (index 0 = no earlier video in the training set)

Scale curve within this video: test error on its 50 frozen test frames (y) against the number of labels
from this video (x = 20, 40, 60, ...). New pupil standard (ellipse endpoints).

## Old-era model vs new-standard model on the same frames (2026-09-23)

`scripts/pupil_area_variants.py` part B -> `pupil_old_vs_new_model.csv/.png`. The old-era model (Aug 17 project,
100 labels, snapshot best-100) predicted the first 4 minutes of this video (14,400 frames); the new-standard model of
step 5 (100 labels) predicted the whole video. The 3-point area rule ignores pupil_top, so on these frames only the
models differ, not the label standard. Corr(old, new 3-point area) 0.761 on the 12,446 frames both rules trust,
median new/old ratio 1.025, median frame-wise disagreement 3.6%. Untrusted frames under the 2026-09-19 rule: old
model 1,287 (8.9%, 86 runs), new model 1,522 (10.6%, 73 runs), 855 in common. The new model's 3-point area is 6.2%
above its own 4-point area (bias of the old rule; corr 0.934). No labeled frame of this video has an empty pupil, so
the blink trigger cannot be checked against closed-eye labels here; of the 120 labeled frames 22 (old) / 25 (new)
are flagged untrusted (these frames were chosen by the jump rule, so they over-represent blinks).
