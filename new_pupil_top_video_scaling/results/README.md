# Across videos

The final deliverable of this unit: labels needed to reach the plateau (y) against the number of videos
already in the training set (x). One point per video, taken from `<video>/results/`.

Plateau rule (same for every video): running best of the final-snapshot series (median frame RMSE on that
video's 50 frozen test frames); the plateau is reached when two consecutive 20-label steps each improve the
running best by <= 3%. The plateau point is the last step that still improved it.

| video | mouse / length | labels of earlier videos in the training set | final-snapshot series (px, 20/40/60/80/100 labels) | plateau | error at plateau |
|---|---|---|---|---|---|
| 1 `first5minvedio` | mouse A, 5.0 min | 0 | 12.93 / 13.00 / 12.04 / 13.07 / 12.29 | **60 labels** | 12.04 px |
| 2 `20251031_Pluto_spont_1` | mouse B, 19.9 min | 100 (all of video 1) | 10.37 / 12.17 / 11.35 / 11.59 / – | **20 labels** | 10.37 px |

Notes
- With 50 test frames the median moves by about 1 px between neighbouring steps from sampling alone, so the
  3% rule is at the noise floor; differences of 1 px between steps of the same video are not meaningful.
- Video 2, label-free signals kept improving after the median flattened: pool frames flagged by the jump rule
  21% -> 11% -> 6% -> 6% (steps 1-4), test keypoints with confidence >= 0.6: 59% -> 75% -> 74% -> 82%.
- Video 2, step 5 (100 labels) was selected (`selection_100_frames.png`) but not trained; whether to run it for
  a full 5-point curve like video 1 is an open decision.
