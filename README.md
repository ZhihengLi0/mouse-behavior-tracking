# Mouse Behavior Tracking

Local DeepLabCut workflow for mouse eye, pupil, blink, face, ear, and forepaw tracking.

This repository tracks code, configuration templates, and notes only.
Raw videos, PDFs, trained models, extracted labels, and generated results stay local.

## Repository Logic

GitHub stores the reproducible pipeline:

- setup notes and project documentation.
- DeepLabCut config files.
- scripts for extraction, labeling launchers, evaluation, and plotting.

Local-only files store the private or heavy research data:

- `face.mp4`: eye close-up video for pupil and blink tracking.
- `body.mp4`: face/body video for facial features, ears, and forepaws.
- `*.pdf`: local reading materials and unpublished/review materials.
- `local_data/`: extracted frames, labels, predictions, result plots, caches, and AI handoff notes.

## Current Eye Project

Active DeepLabCut project:

```text
dlc_projects/EyePupilBlink-Zhiheng-2026-08-17/config.yaml
```

Tracked eye keypoints:

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

Current train/test split:

```text
training pool: first 4 minutes of face.mp4
held-out test pool: last 1 minute of face.mp4
```

## Current Status

Completed:

1. DeepLabCut environment setup.
2. Eye project creation.
3. 20 training frames extracted from the first 4 minutes.
4. 20 training frames manually labeled.
5. First model trained from the 20-frame training set.
6. 100-frame held-out test set extracted from the last minute.
7. 100 test frames manually labeled.
8. The 20-frame model evaluated on the 100-frame test set.
9. A summary plot generated locally.

Current 20-frame baseline:

```text
overall keypoint RMSE: 38.27 px
pupil center RMSE:    17.83 px
pupil width MAE:      36.36 px
```

Local result plot:

```text
local_data/test_sets/eye_last_minute_100/predictions_20train/eye_test_20train_summary.png
```

The 20-frame model is only a baseline. The next scientific step is to repeat training with larger training sets from the first 4 minutes, such as 50, 100, and 150 frames, and evaluate each model on the same fixed 100-frame test set.

## Useful Commands

Check the environment:

```bash
bash scripts/check_setup.sh
```

Open training-frame labeling:

```bash
bash scripts/label_eye_frames.sh
```

Extract the fixed 100-frame eye test set:

```bash
/Users/lizhiheng/miniforge3/envs/DEEPLABCUT/bin/python scripts/extract_eye_test_frames.py
```

Open test-frame labeling:

```bash
bash scripts/label_eye_test_frames.sh
```

Evaluate the current model on the test set:

```bash
/Users/lizhiheng/miniforge3/envs/DEEPLABCUT/bin/python scripts/evaluate_eye_test_set.py
```

Regenerate the current result plot:

```bash
/Users/lizhiheng/miniforge3/envs/DEEPLABCUT/bin/python scripts/plot_eye_test_results.py
```

## Documentation

Start with:

```text
docs/01_local_setup.md
docs/02_eye_project.md
docs/04_metrics_and_outputs.md
```

The body video workflow is documented but deferred until the eye pipeline is more mature.
