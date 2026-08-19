# 02 Eye Project

Goal: use `face.mp4` to track pupil position, pupil size, and blink events.

## 1. Create A New DLC Project

The current project already exists at:

```text
dlc_projects/EyePupilBlink-Zhiheng-2026-08-17/config.yaml
```

It was created with `copy_videos=False`, so the project uses a symlink to `face.mp4` instead of copying the large raw video.

If recreating the project from scratch, use:

```python
import deeplabcut

deeplabcut.create_new_project(
    "EyePupilBlink",
    "Zhiheng",
    ["/Users/lizhiheng/Desktop/research/生物/face.mp4"],
    working_directory="/Users/lizhiheng/Desktop/research/生物/dlc_projects",
    copy_videos=False,
)
```

## 2. Bodyparts

The active bodyparts are:

```text
pupil_top
pupil_bottom
pupil_left
pupil_right
eyelid_top
eyelid_bottom
eye_nasal_corner
eye_temporal_corner
```

In `config.yaml`:

```yaml
bodyparts:
- pupil_top
- pupil_bottom
- pupil_left
- pupil_right
- eyelid_top
- eyelid_bottom
- eye_nasal_corner
- eye_temporal_corner
```

Definitions:

- `pupil_top`: top edge of the visible pupil.
- `pupil_bottom`: bottom edge of the visible pupil.
- `pupil_left`: left edge of the visible pupil.
- `pupil_right`: right edge of the visible pupil.
- `eyelid_top`: upper eyelid point at the eye opening.
- `eyelid_bottom`: lower eyelid point at the eye opening.
- `eye_nasal_corner`: inner/nasal eye corner.
- `eye_temporal_corner`: outer/temporal eye corner.

## 3. Extract Frames

Training frames must come from the first 4 minutes of the 5-minute `face.mp4`.

Current first-pass training extraction:

```text
start: 0.0
stop: 0.8
numframes2pick: 20
algo: kmeans
```

This means:

```text
extract 20 representative training frames from the first 80% of the video
```

For the final training-frame-count curve, repeat this with larger training sets from the same first-4-minute pool, for example 50, 100, and 150 frames.

The held-out test set is separate:

```text
local_data/test_sets/eye_last_minute_100/
```

It contains 100 manually labeled frames from the last minute and must not be used for training.

## 4. Label Frames

Open training-frame labeling with:

```bash
bash scripts/label_eye_frames.sh
```

Label carefully. Do not label reflections as pupil edges.
If the pupil boundary is hard to see, use a consistent ellipse approximation.

For the held-out test set, use:

```bash
bash scripts/label_eye_test_frames.sh
```

## 5. Check Labels

Run:

```python
import deeplabcut
config_path = "/Users/lizhiheng/Desktop/research/生物/dlc_projects/EyePupilBlink-Zhiheng-YYYY-MM-DD/config.yaml"
deeplabcut.check_labels(config_path)
```

Replace `YYYY-MM-DD` with the actual generated folder date.

Inspect the label check images manually before training.

## 6. Train First Model

```python
import deeplabcut
config_path = "/Users/lizhiheng/Desktop/research/生物/dlc_projects/EyePupilBlink-Zhiheng-2026-08-17/config.yaml"

deeplabcut.create_training_dataset(config_path)
deeplabcut.train_network(
    config_path,
    shuffle=1,
    maxiters=1000,
    displayiters=100,
    saveiters=500,
)
```

The first 20-frame model is only a baseline and is not expected to be final quality.

## 7. Evaluate On Fixed Test Set

Run:

```bash
/Users/lizhiheng/miniforge3/envs/DEEPLABCUT/bin/python scripts/evaluate_eye_test_set.py
/Users/lizhiheng/miniforge3/envs/DEEPLABCUT/bin/python scripts/plot_eye_test_results.py
```

Current 20-frame baseline:

```text
overall keypoint RMSE: 38.27 px
pupil center RMSE:    17.83 px
pupil width MAE:      36.36 px
```

The final deliverable should compare test error across training-frame counts:

```text
20, 50, 100, 150, ...
```
