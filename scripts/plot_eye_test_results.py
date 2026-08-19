#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
TEST_ROOT = PROJECT_DIR / "local_data" / "test_sets" / "eye_last_minute_100"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot eye held-out test metrics for a training-frame count.")
    parser.add_argument("--train-frames", type=int, default=20, help="Training frame count label for the result folder.")
    args = parser.parse_args()

    result_dir = TEST_ROOT / f"predictions_{args.train_frames}train"
    summary_csv = result_dir / "eye_test_summary.csv"
    per_frame_csv = result_dir / "eye_test_per_frame_errors.csv"
    out_png = result_dir / f"eye_test_{args.train_frames}train_summary.png"

    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_csv}")
    if not per_frame_csv.exists():
        raise FileNotFoundError(f"Missing per-frame file: {per_frame_csv}")

    summary = pd.read_csv(summary_csv)
    per_frame = pd.read_csv(per_frame_csv, index_col=0)

    keypoint_rmse = summary[
        summary["metric"].str.endswith("_rmse_px")
        & ~summary["metric"].str.startswith("overall")
        & ~summary["metric"].str.contains("pcutoff")
    ].copy()
    keypoint_rmse["bodypart"] = keypoint_rmse["metric"].str.replace("_rmse_px", "", regex=False)

    overall = summary.set_index("metric").loc["overall_keypoint_rmse_px", "value"]
    center = summary.set_index("metric").loc["pupil_center_rmse_px", "value"]
    width = summary.set_index("metric").loc["pupil_width_mae_px", "value"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)

    axes[0].bar(keypoint_rmse["bodypart"], keypoint_rmse["value"], color="#4C78A8")
    axes[0].axhline(overall, color="#D62728", linestyle="--", linewidth=1.5, label=f"overall RMSE = {overall:.2f}px")
    axes[0].set_title(f"{args.train_frames} training frames: keypoint test RMSE")
    axes[0].set_ylabel("RMSE (px)")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].legend()

    frame_numbers = range(len(per_frame))
    axes[1].plot(frame_numbers, per_frame["pupil_center_error_px"], color="#2CA02C", linewidth=1.8)
    axes[1].axhline(center, color="#D62728", linestyle="--", linewidth=1.5, label=f"pupil center RMSE = {center:.2f}px")
    axes[1].set_title(f"Pupil center error across 100 held-out frames; width MAE = {width:.2f}px")
    axes[1].set_xlabel("test frame index")
    axes[1].set_ylabel("center error (px)")
    axes[1].set_xlim(0, max(0, len(per_frame) - 1))
    axes[1].set_xticks(range(0, len(per_frame), 10))
    axes[1].legend()

    fig.savefig(out_png, dpi=200)
    print(f"saved_plot: {out_png}")


if __name__ == "__main__":
    main()
