# Can unreliable frames be found without labels? (2026-10-04; fitting detector added and rerun with video 13 on 2026-10-06; rerun with video 14 on 2026-10-07; rerun with video 15 on 2026-10-08; rerun with video 16 on 2026-10-10)

Question (Kaiwen, 2026-09-23 and 2026-10-04): are the frames flagged by the jump rule really wrong, how many wrong frames
does it miss, and is there a robust label-free signal for "this prediction is not reliable"?

**Basic information.** Videos 0-16 (finished videos). Frames: the 50 test frames and the 20 validation frames of each
video, 1190 in total; they are evenly spaced in time (not picked by any model), human-labeled under the keypoint
definitions made explicit on 2026-09-20, and never used for training. Model of a video = the model at its plateau point
(plateau at 0 labels: the 20-label model), ResNet-50, final snapshot, whole-video prediction, no confidence cut-off.
Truth: frame RMSE against the human labels (formula of `scale_step.py`); "wrong" = frame RMSE > 50 px (32 frames, 2.7%)
or > 20 px (143 frames, 12.0%). Script: `../../scripts/unreliable_frames.py` (formulas in its header).

| file | content |
|---|---|
| `unreliable_frames_all_videos.csv` | one row per frame: frame RMSE, jump, lowest and mean confidence, fitting deviation, rule flags |
| `unreliable_rules_all_videos.csv` | flagged share, precision and recall of each rule |
| `unreliable_signals_all_videos.csv` | AUC and Spearman correlation of each signal, per video and pooled |
| `unreliable_frames_all_videos.png` | jump, lowest confidence and fitting deviation against the frame error; recall and precision of the rules |
| `fitting_dev_videoNN.npy` (not in git) | cached per-frame fitting deviation of each whole video |

| rule | frames flagged | of the 32 frames > 50 px | flagged frames that are > 50 px | of the 143 frames > 20 px | flagged frames that are > 20 px |
|---|---|---|---|---|---|
| jump > 3% of the eye width (rule of `select_frames.py`) | 145 (12.2%) | 21 found (66%), 11 missed | 14.5% | 65 found (45%) | 45% |
| lowest confidence < 0.6 | 615 (51.7%) | 32 found (100%) | 5.2% | 124 found (87%) | 20% |
| mean confidence < t (leave one video out; 0.790) | 344 (28.9%) | 29 found (91%) | 8.4% | 86 found (60%) | 25% |
| jump rule OR lowest confidence < 0.6 | 635 (53.4%) | 32 found (100%) | 5.0% | 126 found (88%) | 20% |
| fitting: mean SARIMAX deviation > 20 px (DLC's `fitting` detector, step 3) | 27 (2.3%) | 23 found (72%), 9 missed | 85% | 24 found (17%) | 89% |

AUC pooled (frames > 50 px against the rest): fitting 0.97, lowest confidence 0.94, mean confidence 0.91, jump 0.85; for
frames > 20 px: 0.85 / 0.79 / 0.78 / 0.73.

The `fitting` signal is DLC's third outlier detector re-implemented (the three were compared as *selectors of training
frames* in step 3 of the summary page; here they are measured as *detectors of wrong frames*): per keypoint coordinate a
SARIMAX(3,0,1) model is fitted to the whole video (points with confidence < 0.01 set to missing, series centred because
SARIMAX has no intercept; without centring the optimizer failed on videos 1, 5 and 12 and produced a constant deviation),
deviation of a keypoint = distance between the one-step prediction and the predicted position, fitting = mean over the 8
keypoints, flagged above DLC's default epsilon of 20 px.

1. The jump rule is selective but incomplete: it flags 12% of the frames, about half of the flagged frames have an error
   above 20 px, and it finds two thirds of the gross errors. It misses 11 of the 32 frames above 50 px: frames in which
   the prediction is wrong but does not move from the previous frame.
2. The confidence finds every gross error (lowest confidence < 0.6: 32 of 32) but flags 52% of all frames, so 95% of
   what it flags is not a gross error.
3. The fitting detector is the most selective: it flags 2.3% of the frames and 85% of them are gross errors; it finds 23
   of the 32 (72%), more than the jump rule with a quarter of the flags. It finds gross errors only (17% of the frames
   above 20 px); the 9 gross errors it misses are frames where the wrong prediction is smooth in time.
4. No rule here is both complete and selective. Per video the signals differ a lot (AUC for > 20 px between 0.3 and 1.0
   with 1-38 wrong frames per video), so the pooled numbers should not be read as valid for every video.
5. Limits: 32 gross-error frames, 17 of them in videos 1 and 8; one model per video; the frames are single frames, so
   runs of consecutive wrong frames are not evaluated as events.
