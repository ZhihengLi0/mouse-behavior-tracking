#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
TEST_ROOT = PROJECT_DIR / "local_data" / "test_sets" / "eye_last_minute_100"
OUT_CSV = TEST_ROOT / "eye_architecture_comparison_100train.csv"
OUT_PNG = TEST_ROOT / "eye_architecture_comparison_100train.png"
MODELS = {
    "ResNet-50": TEST_ROOT / "predictions_100train" / "eye_test_summary.csv",
    "HRNet-W32": TEST_ROOT / "predictions_100train_hrnet_w32" / "eye_test_summary.csv",
}
METRICS = {
    "overall_keypoint_rmse_px": "Overall keypoint RMSE",
    "pupil_center_rmse_px": "Pupil center RMSE",
    "pupil_width_mae_px": "Pupil width MAE",
}


def main() -> None:
    rows = []
    for model_name, summary_path in MODELS.items():
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing model summary: {summary_path}")

        summary = pd.read_csv(summary_path).set_index("metric")
        row = {"model": model_name, "training_frames": 100}
        for metric in METRICS:
            row[metric] = float(summary.loc[metric, "value"])
        rows.append(row)

    data = pd.DataFrame(rows)
    data.to_csv(OUT_CSV, index=False)

    plot_data = data.set_index("model")[list(METRICS)].rename(columns=METRICS)
    ax = plot_data.plot.bar(figsize=(9, 5), rot=0, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_title("Eye model architecture comparison (100 training frames)")
    ax.set_xlabel("model architecture")
    ax.set_ylabel("held-out test error (px; lower is better)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="metric")
    ax.figure.tight_layout()
    ax.figure.savefig(OUT_PNG, dpi=200)
    plt.close(ax.figure)

    print(f"saved_csv: {OUT_CSV}")
    print(f"saved_plot: {OUT_PNG}")
    print(data.to_string(index=False))


if __name__ == "__main__":
    main()
