#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
TEST_ROOT = PROJECT_DIR / "local_data" / "test_sets" / "eye_last_minute_100"
TEST_LABEL_DIR = TEST_ROOT / "dlc_label_project" / "labeled-data" / "eye_last_minute_100"
TEST_LABELS = TEST_LABEL_DIR / "CollectedData_Zhiheng.h5"


def frame_name(index_value: object) -> str:
    if isinstance(index_value, tuple):
        return Path(str(index_value[-1])).name
    return Path(str(index_value)).name


def drop_scorer_level(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    if df.columns.nlevels == 4:
        return df.droplevel([0, 1], axis=1)
    if df.columns.nlevels == 3:
        return df.droplevel(0, axis=1)
    return df


def xy(df: pd.DataFrame, bodypart: str) -> np.ndarray:
    return df.loc[:, [(bodypart, "x"), (bodypart, "y")]].to_numpy(dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the worst eye-tracking frames with manual and predicted points.")
    parser.add_argument("--train-frames", type=int, required=True)
    parser.add_argument("--model-label", default="")
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()

    if args.model_label and not re.fullmatch(r"[A-Za-z0-9_-]+", args.model_label):
        raise ValueError("--model-label may contain only letters, numbers, underscores, and hyphens")
    if args.count < 1:
        raise ValueError("--count must be positive")

    output_name = f"predictions_{args.train_frames}train"
    if args.model_label:
        output_name += f"_{args.model_label}"
    result_dir = TEST_ROOT / output_name

    prediction_files = sorted(result_dir.glob("image_predictions_*.h5"))
    if not prediction_files:
        raise FileNotFoundError(f"No prediction H5 found in {result_dir}")

    manual = pd.read_hdf(TEST_LABELS)
    predicted = pd.read_hdf(prediction_files[-1])
    manual.index = [frame_name(value) for value in manual.index]
    predicted.index = [frame_name(value) for value in predicted.index]
    manual = drop_scorer_level(manual).sort_index()
    predicted = drop_scorer_level(predicted).sort_index()

    common = manual.index.intersection(predicted.index)
    manual = manual.loc[common]
    predicted = predicted.loc[common]
    bodyparts = sorted({column[0] for column in manual.columns})

    errors = {}
    for bodypart in bodyparts:
        errors[bodypart] = np.linalg.norm(xy(predicted, bodypart) - xy(manual, bodypart), axis=1)
    error_data = pd.DataFrame(errors, index=common)

    pupil_parts = ["pupil_top", "pupil_bottom", "pupil_left", "pupil_right"]
    manual_center = sum(xy(manual, part) for part in pupil_parts) / len(pupil_parts)
    predicted_center = sum(xy(predicted, part) for part in pupil_parts) / len(pupil_parts)
    center_error = np.linalg.norm(predicted_center - manual_center, axis=1)
    manual_width = np.linalg.norm(xy(manual, "pupil_left") - xy(manual, "pupil_right"), axis=1)
    predicted_width = np.linalg.norm(xy(predicted, "pupil_left") - xy(predicted, "pupil_right"), axis=1)

    audit = pd.DataFrame(index=common)
    audit["max_keypoint"] = error_data.idxmax(axis=1)
    audit["max_keypoint_error_px"] = error_data.max(axis=1)
    audit["pupil_center_error_px"] = center_error
    audit["pupil_width_error_px"] = np.abs(predicted_width - manual_width)
    audit = audit.sort_values("max_keypoint_error_px", ascending=False).head(args.count)
    audit.to_csv(result_dir / f"outlier_top{args.count}.csv")

    columns = 4
    rows = int(np.ceil(len(audit) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(20, 4.6 * rows), squeeze=False, constrained_layout=True)
    for axis, (name, row) in zip(axes.flat, audit.iterrows(), strict=False):
        image_path = TEST_LABEL_DIR / name
        axis.imshow(plt.imread(image_path))
        for bodypart in bodyparts:
            manual_point = manual.loc[name, [(bodypart, "x"), (bodypart, "y")]].to_numpy(dtype=float)
            predicted_point = predicted.loc[name, [(bodypart, "x"), (bodypart, "y")]].to_numpy(dtype=float)
            axis.scatter(*manual_point, s=34, facecolors="none", edgecolors="#00B050", linewidths=1.8)
            axis.scatter(*predicted_point, s=34, marker="x", c="#E31A1C", linewidths=1.8)
        axis.set_title(f"{name}\nworst: {row['max_keypoint']} {row['max_keypoint_error_px']:.1f}px", fontsize=10)
        axis.set_axis_off()

    for axis in axes.flat[len(audit) :]:
        axis.set_visible(False)

    label = args.model_label or "default"
    fig.suptitle(
        f"Top {len(audit)} outliers: {args.train_frames} training frames, {label}\n"
        "green circles = manual labels; red crosses = predictions",
        fontsize=16,
    )
    out_png = result_dir / f"outlier_top{args.count}_full_width_green_manual_red_prediction.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    print(f"saved_csv: {result_dir / f'outlier_top{args.count}.csv'}")
    print(f"saved_plot: {out_png}")
    print(audit.to_string())


if __name__ == "__main__":
    main()
