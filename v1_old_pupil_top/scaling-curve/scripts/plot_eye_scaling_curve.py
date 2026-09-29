#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
TEST_ROOT = PROJECT_DIR / "local_data" / "test_sets" / "eye_last_minute_100"
OUT_CSV = TEST_ROOT / "eye_scaling_summary.csv"
OUT_PNG = TEST_ROOT / "eye_scaling_curve.png"
METRICS = [
    "overall_keypoint_rmse_px",
    "pupil_center_rmse_px",
    "pupil_width_mae_px",
]


def train_frame_count(path: Path) -> int | None:
    name = path.name
    if not name.startswith("predictions_") or not name.endswith("train"):
        return None
    value = name.removeprefix("predictions_").removesuffix("train")
    return int(value) if value.isdigit() else None


def main() -> None:
    rows = []
    for result_dir in sorted(TEST_ROOT.glob("predictions_*train")):
        n_frames = train_frame_count(result_dir)
        summary_csv = result_dir / "eye_test_summary.csv"
        if n_frames is None or not summary_csv.exists():
            continue

        summary = pd.read_csv(summary_csv).set_index("metric")
        row = {"train_frames": n_frames}
        for metric in METRICS:
            row[metric] = float(summary.loc[metric, "value"])
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No prediction summaries found under {TEST_ROOT}")

    data = pd.DataFrame(rows).sort_values("train_frames")
    data.to_csv(OUT_CSV, index=False)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(data["train_frames"], data["overall_keypoint_rmse_px"], marker="o", label="overall keypoint RMSE")
    ax.plot(data["train_frames"], data["pupil_center_rmse_px"], marker="o", label="pupil center RMSE")
    ax.plot(data["train_frames"], data["pupil_width_mae_px"], marker="o", label="pupil width MAE")
    ax.set_title("Eye tracking scaling curve")
    ax.set_xlabel("training frames from first 4 minutes")
    ax.set_ylabel("held-out test error (px)")
    ax.set_xticks(data["train_frames"])
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(OUT_PNG, dpi=200)

    print(f"saved_csv: {OUT_CSV}")
    print(f"saved_plot: {OUT_PNG}")
    print(data.to_string(index=False))


if __name__ == "__main__":
    main()
