#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_DIR / "local_data" / "test_sets" / "eye_last_minute_100" / "predictions_20train"
SUMMARY_CSV = RESULT_DIR / "eye_test_summary.csv"
PER_FRAME_CSV = RESULT_DIR / "eye_test_per_frame_errors.csv"
OUT_PNG = RESULT_DIR / "eye_test_20train_summary.png"


def main() -> None:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing summary file: {SUMMARY_CSV}")
    if not PER_FRAME_CSV.exists():
        raise FileNotFoundError(f"Missing per-frame file: {PER_FRAME_CSV}")

    summary = pd.read_csv(SUMMARY_CSV)
    per_frame = pd.read_csv(PER_FRAME_CSV, index_col=0)

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
    axes[0].set_title("20 training frames: keypoint test RMSE")
    axes[0].set_ylabel("RMSE (px)")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].legend()

    axes[1].plot(per_frame.index, per_frame["pupil_center_error_px"], color="#2CA02C", linewidth=1.8)
    axes[1].axhline(center, color="#D62728", linestyle="--", linewidth=1.5, label=f"pupil center RMSE = {center:.2f}px")
    axes[1].set_title(f"Pupil center error across 100 held-out frames; width MAE = {width:.2f}px")
    axes[1].set_xlabel("test frame")
    axes[1].set_ylabel("center error (px)")
    axes[1].legend()

    fig.savefig(OUT_PNG, dpi=200)
    print(f"saved_plot: {OUT_PNG}")


if __name__ == "__main__":
    main()
