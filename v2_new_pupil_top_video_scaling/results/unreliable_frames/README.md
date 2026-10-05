# Can unreliable frames be found without labels? (2026-10-04)

Question (Kaiwen, 2026-09-23 and 2026-10-04): are the frames flagged by the jump rule really wrong, how many wrong frames
does it miss, and is there a robust label-free signal for "this prediction is not reliable"?

**Basic information.** Videos 0-12 (finished videos). Frames: the 50 test frames and the 20 validation frames of each
video, 910 in total; they are evenly spaced in time (not picked by any model), human-labeled under the keypoint
definitions made explicit on 2026-09-20, and never used for training. Model of a video = the model at its plateau point
(plateau at 0 labels: the 20-label model), ResNet-50, final snapshot, whole-video prediction, no confidence cut-off.
Truth: frame RMSE against the human labels (formula of `scale_step.py`); "wrong" = frame RMSE > 50 px (30 frames, 3.3%)
or > 20 px (118 frames, 13.0%). Script: `../../scripts/unreliable_frames.py` (formulas in its header).

| file | content |
|---|---|
| `unreliable_frames_all_videos.csv` | one row per frame: frame RMSE, jump, lowest and mean confidence, rule flags |
| `unreliable_rules_all_videos.csv` | flagged share, precision and recall of each rule |
| `unreliable_signals_all_videos.csv` | AUC and Spearman correlation of each signal, per video and pooled |
| `unreliable_frames_all_videos.png` | jump and lowest confidence against the frame error; recall and precision of the rules |

| rule | frames flagged | of the 30 frames > 50 px | flagged frames that are > 50 px | of the 118 frames > 20 px | flagged frames that are > 20 px |
|---|---|---|---|---|---|
| jump > 3% of the eye width (rule of `select_frames.py`) | 115 (12.6%) | 20 found (67%), 10 missed | 17% | 55 found (47%) | 48% |
| lowest confidence < 0.6 | 481 (52.9%) | 30 found (100%) | 6% | 105 found (89%) | 22% |
| mean confidence < t (leave one video out; 0.688) | 207 (22.7%) | 24 found (80%) | 12% | 66 found (56%) | 32% |
| jump rule OR lowest confidence < 0.6 | 495 (54.4%) | 30 found (100%) | 6% | 106 found (90%) | 21% |

AUC pooled (frames > 50 px against the rest): lowest confidence 0.92, mean confidence 0.90, jump 0.84; for frames > 20 px:
0.81 / 0.79 / 0.75.

1. The jump rule is selective but incomplete: it flags 13% of the frames, about half of the flagged frames have an error
   above 20 px, and it finds two thirds of the gross errors. It misses 10 of the 30 frames above 50 px: frames in which
   the prediction is wrong but does not move from the previous frame.
2. The confidence finds every gross error (lowest confidence < 0.6: 30 of 30) but flags half of all frames, so 94% of
   what it flags is not a gross error.
3. No rule here is both complete and selective. Per video the signals differ a lot (AUC for > 20 px between 0.3 and 1.0
   with 1-38 wrong frames per video), so the pooled numbers should not be read as valid for every video.
4. Limits: 30 gross-error frames, 17 of them in videos 1 and 8; one model per video; the frames are single frames, so
   runs of consecutive wrong frames are not evaluated as events.
