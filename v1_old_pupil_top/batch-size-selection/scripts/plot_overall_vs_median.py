#!/usr/bin/env python3
"""Score the batch-size runs and the model-selection runs with BOTH metrics, from the per-frame errors already on disk.

    python plot_overall_vs_median.py

overall RMSE       = sqrt(mean of e^2 over all keypoints of all 100 test frames)          (used when the two steps were run)
median frame RMSE  = median over the test frames of sqrt(mean of e^2 over the 8 keypoints)  (used from the next step on)
e = distance in px between the predicted and the human position of a keypoint. No new training and no new inference:
the inputs are <unit>/predictions/*/eye_test_per_frame_errors.csv written by evaluate_eye_test_set.py.
Outputs: <unit>/results/two_metrics.csv and <unit>/results/overall_vs_median.png for both units."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

V1 = Path(__file__).resolve().parents[2]
BP = ["pupil_top", "pupil_bottom", "pupil_left", "pupil_right", "eyelid_top", "eyelid_bottom", "eye_nasal_corner", "eye_temporal_corner"]
B = V1 / "batch-size-selection/predictions"
M = V1 / "model-selection/predictions"
RUNS = {
    "batch-size-selection": ("HRNet-W32, batch size", [("2", B / "predictions_100train_b8020_batch2"), ("4", B / "predictions_100train_b8020_batch4"),
                                                       ("8", B / "predictions_100train_b8020_batch8"), ("16", B / "predictions_100train_b8020_batch16")]),
    "model-selection": ("model (batch 2)", [("ResNet-50", M / "predictions_100train_arch2_resnet_50"), ("HRNet-W32", B / "predictions_100train_b8020_batch2"),
                                            ("HRNet-W48", M / "predictions_100train_arch2_hrnet_w48"), ("CSPNeXt-S", M / "predictions_100train_arch2_cspnext_s"),
                                            ("HRNet-W18", M / "predictions_100train_arch2_hrnet_w18")]),
}

for unit, (xlabel, runs) in RUNS.items():
    rows = []
    for name, d in runs:
        e = pd.read_csv(d / "eye_test_per_frame_errors.csv")[[f"{b}_error_px" for b in BP]]
        frame = np.sqrt((e ** 2).mean(axis=1))
        rows.append({"run": name, "n_test_frames": len(e), "overall_rmse_px": round(float(np.sqrt(np.nanmean(e.to_numpy() ** 2))), 2),
                     "median_frame_rmse_px": round(float(frame.median()), 2), "p90_frame_rmse_px": round(float(frame.quantile(0.9)), 2),
                     "frames_rmse_gt_50px_pct": round(float((frame > 50).mean() * 100), 1), "max_frame_rmse_px": round(float(frame.max()), 1)})
    t = pd.DataFrame(rows)
    out = V1 / unit / "results"
    t.to_csv(out / "two_metrics.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    top = max(t["overall_rmse_px"].max(), t["median_frame_rmse_px"].max()) * 1.15
    for ax, col, title, unit_txt in ((axes[0], "overall_rmse_px", "RMSE over all keypoints (metric used at the time)", "px"),
                                     (axes[1], "median_frame_rmse_px", "median frame RMSE (metric used from the next step on)", "px"),
                                     (axes[2], "frames_rmse_gt_50px_pct", "test frames with frame RMSE > 50 px", "%")):
        bars = ax.bar(t["run"], t[col], color="#2c6a9b", width=0.6)
        ax.bar_label(bars, fmt="%.1f", fontsize=10, padding=2)
        ax.set_title(title, fontsize=11); ax.set_xlabel(xlabel); ax.set_ylabel(unit_txt)
        ax.set_ylim(0, top if unit_txt == "px" else max(5, t[col].max() * 1.25))
        ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    fig.suptitle(f"{unit}: the same runs scored with both metrics on the 100 final-minute test frames (left and middle share one y scale)", fontsize=11)
    fig.savefig(out / "overall_vs_median.png", dpi=130); plt.close(fig)
    print(unit); print(t.to_string(index=False))
