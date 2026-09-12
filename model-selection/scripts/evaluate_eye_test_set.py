#!/usr/bin/env python3
from __future__ import annotations

import json
import argparse
import re
from pathlib import Path

import deeplabcut
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
MAIN_CONFIG = PROJECT_DIR / "dlc_projects" / "EyePupilBlink-Zhiheng-2026-08-17" / "config.yaml"
TEST_ROOT = PROJECT_DIR / "local_data" / "test_sets" / "eye_last_minute_100"
TEST_LABEL_DIR = TEST_ROOT / "dlc_label_project" / "labeled-data" / "eye_last_minute_100"
TEST_LABELS = TEST_LABEL_DIR / "CollectedData_Zhiheng.h5"
PCUTOFF = 0.6
PUPIL_PARTS = ["pupil_top", "pupil_bottom", "pupil_left", "pupil_right"]


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


def xy_array(df: pd.DataFrame, bodypart: str) -> np.ndarray:
    return df.loc[:, [(bodypart, "x"), (bodypart, "y")]].to_numpy(dtype=float)


def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=1)


def pupil_metrics(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    top = xy_array(df, "pupil_top")
    bottom = xy_array(df, "pupil_bottom")
    left = xy_array(df, "pupil_left")
    right = xy_array(df, "pupil_right")

    center = (top + bottom + left + right) / 4.0
    width = distance(left, right)
    height = distance(top, bottom)
    area = np.pi * (width / 2.0) * (height / 2.0)
    return center, width, area


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an eye DLC model on the fixed 100-frame held-out test set.")
    parser.add_argument("--train-frames", type=int, default=20, help="Training frame count label for the output folder.")
    parser.add_argument("--shuffle", type=int, default=1, help="DLC shuffle/model number to evaluate.")
    parser.add_argument(
        "--model-label",
        default="",
        help="Optional architecture label appended to the output folder, such as hrnet_w32.",
    )
    parser.add_argument(
        "--trainingsetindex",
        type=int,
        default=0,
        help=(
            "Index into the project config's TrainingFraction list; must match "
            "the fraction the shuffle was built with (0 = 0.95 era, 1 = 0.8 era)."
        ),
    )
    parser.add_argument(
        "--snapshot-index",
        type=int,
        default=None,
        help=(
            "Index into the sorted snapshot list (the best snapshot sorts last, "
            "so -1 picks it when one exists). Omit to use the project config. "
            "Use this only to evaluate a specific numbered snapshot, such as "
            "when a resumed run destroyed the best checkpoint."
        ),
    )
    args = parser.parse_args()

    if args.model_label and not re.fullmatch(r"[A-Za-z0-9_-]+", args.model_label):
        raise ValueError("--model-label may contain only letters, numbers, underscores, and hyphens")

    output_name = f"predictions_{args.train_frames}train"
    if args.model_label:
        output_name += f"_{args.model_label}"
    prediction_dir = TEST_ROOT / output_name

    if not MAIN_CONFIG.exists():
        raise FileNotFoundError(f"Missing config: {MAIN_CONFIG}")
    if not TEST_LABELS.exists():
        raise FileNotFoundError(f"Missing manual test labels: {TEST_LABELS}")

    image_files = sorted(TEST_LABEL_DIR.glob("test_*.png"))
    if len(image_files) != 100:
        raise RuntimeError(f"Expected 100 test images in {TEST_LABEL_DIR}, found {len(image_files)}")

    prediction_dir.mkdir(parents=True, exist_ok=True)

    deeplabcut.analyze_images(
        str(MAIN_CONFIG),
        [str(TEST_LABEL_DIR)],
        frame_type=".png",
        destfolder=str(prediction_dir),
        shuffle=args.shuffle,
        trainingsetindex=args.trainingsetindex,
        save_as_csv=True,
        plotting=False,
        pcutoff=0.0,
        device="cpu",
        **({} if args.snapshot_index is None else {"snapshot_index": args.snapshot_index}),
    )

    prediction_files = sorted(prediction_dir.glob("image_predictions_*.h5"))
    if not prediction_files:
        raise FileNotFoundError(f"No prediction H5 files found in {prediction_dir}")

    predictions_h5 = prediction_files[-1]
    manual = pd.read_hdf(TEST_LABELS)
    pred = pd.read_hdf(predictions_h5)

    manual.index = [frame_name(idx) for idx in manual.index]
    pred.index = [frame_name(idx) for idx in pred.index]

    manual = drop_scorer_level(manual).sort_index()
    pred = drop_scorer_level(pred).sort_index()

    common_frames = manual.index.intersection(pred.index)
    if len(common_frames) != 100:
        raise RuntimeError(f"Expected 100 matched frames, found {len(common_frames)}")

    manual = manual.loc[common_frames]
    pred = pred.loc[common_frames]

    bodyparts = sorted({col[0] for col in manual.columns})
    per_frame = pd.DataFrame(index=common_frames)
    summary_rows = []

    all_errors = []
    all_errors_pcutoff = []

    for bodypart in bodyparts:
        gt_xy = xy_array(manual, bodypart)
        pred_xy = xy_array(pred, bodypart)
        err = distance(pred_xy, gt_xy)
        likelihood = pred.loc[:, (bodypart, "likelihood")].to_numpy(dtype=float)
        keep = likelihood >= PCUTOFF

        per_frame[f"{bodypart}_error_px"] = err
        per_frame[f"{bodypart}_likelihood"] = likelihood
        all_errors.append(err)
        all_errors_pcutoff.append(err[keep])

        summary_rows.append(
            {
                "metric": f"{bodypart}_rmse_px",
                "value": float(np.sqrt(np.mean(err**2))),
                "n": int(len(err)),
            }
        )
        summary_rows.append(
            {
                "metric": f"{bodypart}_rmse_px_pcutoff_{PCUTOFF}",
                "value": float(np.sqrt(np.mean(err[keep] ** 2))) if np.any(keep) else np.nan,
                "n": int(np.sum(keep)),
            }
        )
        summary_rows.append(
            {
                "metric": f"{bodypart}_mean_likelihood",
                "value": float(np.mean(likelihood)),
                "n": int(len(likelihood)),
            }
        )

    all_errors_arr = np.concatenate(all_errors)
    pcutoff_error_groups = [x for x in all_errors_pcutoff if len(x) > 0]
    pcutoff_errors_arr = np.concatenate(pcutoff_error_groups) if pcutoff_error_groups else np.array([])

    manual_center, manual_width, manual_area = pupil_metrics(manual)
    pred_center, pred_width, pred_area = pupil_metrics(pred)
    center_error = distance(pred_center, manual_center)
    width_error = np.abs(pred_width - manual_width)
    area_error = np.abs(pred_area - manual_area)

    per_frame["pupil_center_error_px"] = center_error
    per_frame["pupil_width_error_px"] = width_error
    per_frame["pupil_area_error_px2"] = area_error

    summary_rows.extend(
        [
            {"metric": "overall_keypoint_rmse_px", "value": float(np.sqrt(np.mean(all_errors_arr**2))), "n": int(len(all_errors_arr))},
            {
                "metric": f"overall_keypoint_rmse_px_pcutoff_{PCUTOFF}",
                "value": float(np.sqrt(np.mean(pcutoff_errors_arr**2))) if len(pcutoff_errors_arr) else np.nan,
                "n": int(len(pcutoff_errors_arr)),
            },
            {"metric": "pupil_center_rmse_px", "value": float(np.sqrt(np.mean(center_error**2))), "n": int(len(center_error))},
            {"metric": "pupil_width_mae_px", "value": float(np.mean(width_error)), "n": int(len(width_error))},
            {"metric": "pupil_area_mae_px2", "value": float(np.mean(area_error)), "n": int(len(area_error))},
        ]
    )

    summary = pd.DataFrame(summary_rows)
    per_frame_path = prediction_dir / "eye_test_per_frame_errors.csv"
    summary_path = prediction_dir / "eye_test_summary.csv"
    json_path = prediction_dir / "eye_test_summary.json"

    per_frame.to_csv(per_frame_path)
    summary.to_csv(summary_path, index=False)
    json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    print(f"manual_labels: {TEST_LABELS}")
    print(f"predictions: {predictions_h5}")
    print(f"train_frames: {args.train_frames}")
    print(f"shuffle: {args.shuffle}")
    print(f"model_label: {args.model_label or 'default'}")
    print(f"matched_frames: {len(common_frames)}")
    print(f"per_frame_errors: {per_frame_path}")
    print(f"summary: {summary_path}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
