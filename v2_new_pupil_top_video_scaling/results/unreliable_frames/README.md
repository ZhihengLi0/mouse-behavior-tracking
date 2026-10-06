# Can unreliable frames be found without labels? (2026-10-04; fitting detector added and rerun with video 13 on 2026-10-06)

Question (Kaiwen, 2026-09-23 and 2026-10-04): are the frames flagged by the jump rule really wrong, how many wrong frames
does it miss, and is there a robust label-free signal for "this prediction is not reliable"?

**Basic information.** Videos 0-13 (finished videos). Frames: the 50 test frames and the 20 validation frames of each
video, 980 in total; they are evenly spaced in time (not picked by any model), human-labeled under the keypoint
definitions made explicit on 2026-09-20, and never used for training. Model of a video = the model at its plateau point
(plateau at 0 labels: the 20-label model), ResNet-50, final snapshot, whole-video prediction, no confidence cut-off.
Truth: frame RMSE against the human labels (formula of `scale_step.py`); "wrong" = frame RMSE > 50 px (30 frames, 3.1%)
or > 20 px (123 frames, 12.6%). Script: `../../scripts/unreliable_frames.py` (formulas in its header).

| file | content |
|---|---|
| `unreliable_frames_all_videos.csv` | one row per frame: frame RMSE, jump, lowest and mean confidence, fitting deviation, rule flags |
| `unreliable_rules_all_videos.csv` | flagged share, precision and recall of each rule |
| `unreliable_signals_all_videos.csv` | AUC and Spearman correlation of each signal, per video and pooled |
| `unreliable_frames_all_videos.png` | jump, lowest confidence and fitting deviation against the frame error; recall and precision of the rules |
| `fitting_dev_videoNN.npy` (not in git) | cached per-frame fitting deviation of each whole video |

| rule | frames flagged | of the 30 frames > 50 px | flagged frames that are > 50 px | of the 123 frames > 20 px | flagged frames that are > 20 px |
|---|---|---|---|---|---|
| jump > 3% of the eye width (rule of `select_frames.py`) | 121 (12.3%) | 20 found (67%), 10 missed | 17% | 57 found (46%) | 47% |
| lowest confidence < 0.6 | 548 (55.9%) | 30 found (100%) | 5.5% | 109 found (89%) | 20% |
| mean confidence < t (leave one video out; 0.688) | 247 (25.2%) | 24 found (80%) | 10% | 68 found (55%) | 28% |
| jump rule OR lowest confidence < 0.6 | 562 (57.3%) | 30 found (100%) | 5% | 110 found (89%) | 20% |
| fitting: mean SARIMAX deviation > 20 px (DLC's `fitting` detector, step 3) | 26 (2.7%) | 22 found (73%), 8 missed | 85% | 23 found (19%) | 88% |

AUC pooled (frames > 50 px against the rest): fitting 0.98, lowest confidence 0.92, mean confidence 0.91, jump 0.84; for
frames > 20 px: 0.84 / 0.79 / 0.78 / 0.74.

The `fitting` signal is DLC's third outlier detector re-implemented (the three were compared as *selectors of training
frames* in step 3 of the summary page; here they are measured as *detectors of wrong frames*): per keypoint coordinate a
SARIMAX(3,0,1) model is fitted to the whole video (points with confidence < 0.01 set to missing, series centred because
SARIMAX has no intercept; without centring the optimizer failed on videos 1, 5 and 12 and produced a constant deviation),
deviation of a keypoint = distance between the one-step prediction and the predicted position, fitting = mean over the 8
keypoints, flagged above DLC's default epsilon of 20 px.

1. The jump rule is selective but incomplete: it flags 12% of the frames, about half of the flagged frames have an error
   above 20 px, and it finds two thirds of the gross errors. It misses 10 of the 30 frames above 50 px: frames in which
   the prediction is wrong but does not move from the previous frame.
2. The confidence finds every gross error (lowest confidence < 0.6: 30 of 30) but flags 56% of all frames, so 94% of
   what it flags is not a gross error.
3. The fitting detector is the most selective: it flags 2.7% of the frames and 85% of them are gross errors; it finds 22
   of the 30 (73%), more than the jump rule with a quarter of the flags. It finds gross errors only (19% of the frames
   above 20 px); the 8 gross errors it misses are frames where the wrong prediction is smooth in time.
4. No rule here is both complete and selective. Per video the signals differ a lot (AUC for > 20 px between 0.3 and 1.0
   with 1-38 wrong frames per video), so the pooled numbers should not be read as valid for every video.
5. Limits: 30 gross-error frames, 17 of them in videos 1 and 8; one model per video; the frames are single frames, so
   runs of consecutive wrong frames are not evaluated as events.
