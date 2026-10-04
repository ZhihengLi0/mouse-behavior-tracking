# Model selection: results

Five backbones trained on the same data as the batch-size sweep (5-minute video of the first mouse; 80 training and
20 validation frames from the first 4 minutes; batch 2, 100 epochs, CPU, one run each) and scored on the 100
final-minute test frames. `model_selection_summary.csv`, `01_model_selection_overview.png`.

## Both metrics (added 2026-10-04)

`two_metrics.csv`, `overall_vs_median.png` (`../../batch-size-selection/scripts/plot_overall_vs_median.py`; from the
per-frame errors already on disk, no retraining).

| Model | overall RMSE | median frame RMSE | p90 frame RMSE | frames > 50 px | worst frame |
|---|---:|---:|---:|---:|---:|
| ResNet-50 | 20.10 px | 18.39 px | 26.2 px | 0% | 41.2 px |
| HRNet-W32 | 21.07 px | 19.71 px | 27.3 px | 0% | 35.4 px |
| HRNet-W48 | 24.73 px | 19.25 px | 27.5 px | 1% | 135.7 px |
| CSPNeXt-S | 35.45 px | 19.63 px | 26.1 px | 2% | 214.4 px |
| HRNet-W18 | 47.88 px | 20.47 px | 111.8 px | 15% | 118.6 px |

The overall RMSE spreads from 20 to 48 px, the median frame RMSE only from 18.4 to 20.5 px: the overall RMSE is pulled
up by a few far-off frames (15% of the frames for HRNet-W18, single frames for HRNet-W48 and CSPNeXt-S). This is why the
median frame RMSE is the main metric from the active-learning step on, with the share of frames above 50 px reported
next to it. ResNet-50 is the lowest under both metrics.
