# Mouse Behavior Tracking

Local DeepLabCut workflow for mouse eye, pupil, blink, face, ear, and forepaw tracking.

This repository tracks code, configuration templates, and notes only.
Raw videos, PDFs, trained models, extracted labels, and generated results stay local.

## Repository Logic

There are two layers:

1. GitHub/repository layer: code, configuration, documentation, and reproducible commands.
2. Local research-data layer: raw videos, PDFs, extracted frames, labels, model weights, predictions, and plots.

The local research-data layer is intentionally ignored by Git. This keeps private or large files out of GitHub.

Main local inputs:

- `face.mp4`: eye video for pupil and blink tracking.
- `body.mp4`: deferred face/body video for facial features, ears, and forepaws.
- `*.pdf`: local reading material.
- `local_data/`: temporary files and local results.

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

The first 20/50/100-frame scaling study and the 100-frame ResNet-50 versus
HRNet-W32 comparison are complete. The controlled HRNet-W32 batch-size
sweep is also complete for batch sizes `1, 2, 4, 8, 16`.

A controlled batch-size `32` extension uses the same HRNet-W32 model,
95/5 split, 200 epochs, and locked external test (`shuffle9`). Batch 64
is intentionally excluded because this machine has 16 GB of memory.

Every batch-size run used the same 100-frame label pool from the first
four minutes, the same 95/5 internal split, 200 epochs, and the same
locked 100-frame final-minute test set.

| Batch | DLC best epoch | Minimum internal valid loss | External overall RMSE | Pupil center RMSE | Pupil width MAE |
|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 0.00836 | 51.31 px | 15.76 px | 32.95 px |
| 2 | 90 | 0.00824 | 20.06 px | 9.53 px | 20.11 px |
| 4 | 20 | 0.00820 | 24.31 px | 10.53 px | 22.86 px |
| 8 | 10 | 0.00806 | 25.68 px | 11.96 px | 26.31 px |
| 16 | 10 | 0.00790 | 38.61 px | 16.68 px | 37.99 px |

The pre-specified internal-validation rule selects batch size **16**.
Batch size **2** has the lowest external-test errors, but that final-minute
result is report-only and must not be used retroactively for tuning. The
disagreement matters because internal validation contains only five
images and all five runs were performed once.

Audited, publication-safe outputs are in
[`results/01_batch_size_sweep/`](results/01_batch_size_sweep/).
The detailed interpretation is in the
[result report](results/01_batch_size_sweep/README.md).

## Where To Look

- `docs/`: learning workflow and metric definitions.
- `scripts/`: extraction, labeling, training, evaluation, and publishing.
- `results/`: versioned aggregate tables, figures, and reports.
- `local_data/`: ignored local predictions, outlier montages, and test assets.
- `dlc_projects/`: DeepLabCut project; generated models and labels are ignored.

Next scientific work:

1. Use the frozen internally selected batch size for a controlled
   architecture comparison.
2. Improve or explicitly account for the small internal validation set.
3. Calibrate DLC likelihood before applying a confidence cutoff.
4. Run the active-learning comparison using K-means frame selection and
   the `uncertain`, `jump`, and `fitting` outlier methods.

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

Regenerate outlier tables and montages:

```bash
/Users/lizhiheng/miniforge3/envs/DEEPLABCUT/bin/python scripts/plot_eye_outliers.py --train-frames 100
/Users/lizhiheng/miniforge3/envs/DEEPLABCUT/bin/python scripts/plot_eye_outliers.py --train-frames 100 --model-label hrnet_w32
```

## Documentation

Start with:

```text
docs/01_local_setup.md
docs/02_eye_project.md
docs/04_metrics_and_outputs.md
```

The body video workflow is documented but deferred until the eye pipeline is more mature.

## HRNet-W32 Batch-Size Sweep

The sweep completed on 2026-09-08. Existing `shuffle4` is batch size 8;
`shuffle5-8` are batch sizes 1, 2, 4, and 16. All five model runs,
external evaluations, and plots completed successfully.

Regenerate and audit the versioned result package:

```bash
/Users/lizhiheng/miniforge3/envs/DEEPLABCUT/bin/python scripts/publish_eye_batch_sweep.py
```

The command validates completion, architecture, configured batch sizes,
tested snapshot epochs, and identical external frame order before writing
the result package. Full local artifacts remain in:

```text
local_data/experiments/01_batch_size_sweep/
local_data/test_sets/eye_last_minute_100/predictions_100train_hrnet_w32_batch*/
```

These local directories contain predictions and logs and remain ignored
by Git. Only aggregate result tables, plots, and reports are published.

Batch-32 extension status:

```bash
bash scripts/status_eye_batch32_extension.sh
```
