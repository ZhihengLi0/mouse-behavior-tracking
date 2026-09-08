#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "dlc_projects" / "EyePupilBlink-Zhiheng-2026-08-17"
MODEL_ROOT = PROJECT / "dlc-models-pytorch" / "iteration-0"
TEST_ROOT = ROOT / "local_data" / "test_sets" / "eye_last_minute_100"
EXPERIMENT_ROOT = ROOT / "local_data" / "experiments" / "01_batch_size_sweep"
OUTPUT_ROOT = ROOT / "results" / "01_batch_size_sweep"
RUNS = [(1, 5), (2, 6), (4, 7), (8, 4), (16, 8)]
BODY_PARTS = [
    "pupil_top",
    "pupil_bottom",
    "pupil_left",
    "pupil_right",
    "eyelid_top",
    "eyelid_bottom",
    "eye_nasal_corner",
    "eye_temporal_corner",
]
DISPLAY_NAMES = {
    "pupil_top": "pupil top",
    "pupil_bottom": "pupil bottom",
    "pupil_left": "pupil left",
    "pupil_right": "pupil right",
    "eyelid_top": "eyelid top",
    "eyelid_bottom": "eyelid bottom",
    "eye_nasal_corner": "nasal corner",
    "eye_temporal_corner": "temporal corner",
}


def metric(table: pd.DataFrame, name: str, field: str = "value") -> float:
    rows = table.loc[table["metric"] == name, field]
    if rows.empty:
        raise RuntimeError(f"Missing metric: {name}")
    return float(rows.iloc[0])


def best_snapshot_epoch(train_dir: Path) -> int:
    snapshots = list(train_dir.glob("snapshot-best-*.pt"))
    if len(snapshots) != 1:
        raise RuntimeError(
            f"Expected one best snapshot in {train_dir}, found {len(snapshots)}"
        )
    match = re.fullmatch(r"snapshot-best-(\d+)\.pt", snapshots[0].name)
    if not match:
        raise RuntimeError(f"Unexpected snapshot name: {snapshots[0].name}")
    return int(match.group(1))


def frame_statistics(per_frame: pd.DataFrame) -> dict[str, float]:
    columns = [f"{part}_error_px" for part in BODY_PARTS]
    frame_rmse = np.sqrt(np.mean(np.square(per_frame[columns]), axis=1))
    center = per_frame["pupil_center_error_px"]
    return {
        "external_frame_rmse_median_px": float(np.median(frame_rmse)),
        "external_frame_rmse_p90_px": float(np.quantile(frame_rmse, 0.90)),
        "external_pupil_center_error_median_px": float(np.median(center)),
        "external_pupil_center_error_p90_px": float(np.quantile(center, 0.90)),
    }


def audit_datasets() -> dict[str, object]:
    train_labels = (
        PROJECT / "labeled-data" / "face" / "CollectedData_Zhiheng.h5"
    )
    test_labels = (
        TEST_ROOT
        / "dlc_label_project"
        / "labeled-data"
        / "eye_last_minute_100"
        / "CollectedData_Zhiheng.h5"
    )

    def inspect(path: Path, prefix: str) -> tuple[pd.DataFrame, list[int]]:
        data = pd.read_hdf(path)
        if len(data) != 100:
            raise RuntimeError(f"{prefix} labels contain {len(data)} rows")
        missing = int(data.isna().sum().sum())
        if missing:
            raise RuntimeError(f"{prefix} labels contain {missing} missing values")
        names = [
            str(value[-1] if isinstance(value, tuple) else value)
            for value in data.index
        ]
        frames = []
        for name in names:
            match = re.search(r"(?:img|frame)(\d+)", name)
            if not match:
                raise RuntimeError(f"Cannot parse frame number from {name}")
            frames.append(int(match.group(1)))
        return data, frames

    _, train_frames = inspect(train_labels, "training pool")
    _, test_frames = inspect(test_labels, "external test")
    if set(train_frames) & set(test_frames):
        raise RuntimeError("Training and external-test frame numbers overlap")
    if max(train_frames) >= min(test_frames):
        raise RuntimeError("Training frames are not earlier than test frames")
    return {
        "training_label_rows": len(train_frames),
        "training_frame_min": min(train_frames),
        "training_frame_max": max(train_frames),
        "external_test_label_rows": len(test_frames),
        "external_test_frame_min": min(test_frames),
        "external_test_frame_max": max(test_frames),
        "missing_coordinate_values": 0,
        "train_test_frame_overlap": False,
    }


def load_results():
    if not (EXPERIMENT_ROOT / "runner_finished_at.txt").exists():
        raise RuntimeError("Batch sweep has no completion marker")
    metadata = pd.read_csv(EXPERIMENT_ROOT / "run_metadata.csv")
    if set(metadata["batch_size"]) != {batch for batch, _ in RUNS}:
        raise RuntimeError("Metadata does not contain all five batch sizes")
    if set(metadata["status"]) != {"complete"}:
        raise RuntimeError("At least one run is incomplete")

    summary_rows = []
    keypoint_rows = []
    curve_rows = []
    frame_data = {}
    expected_frames = None

    for batch_size, shuffle in RUNS:
        train_dir = (
            MODEL_ROOT
            / f"EyePupilBlinkAug17-trainset95shuffle{shuffle}"
            / "train"
        )
        stats = pd.read_csv(train_dir / "learning_stats.csv")
        if int(stats["step"].max()) != 200:
            raise RuntimeError(f"Batch {batch_size} did not reach 200 epochs")

        config = yaml.safe_load(
            (train_dir / "pytorch_config.yaml").read_text(encoding="utf-8")
        )
        if int(config["train_settings"]["batch_size"]) != batch_size:
            raise RuntimeError(f"Batch {batch_size} config mismatch")
        if config["model"]["backbone"]["model_name"] != "hrnet_w32":
            raise RuntimeError(f"Batch {batch_size} is not HRNet-W32")
        if config["runner"]["key_metric"] != "test.mAP":
            raise RuntimeError(f"Batch {batch_size} checkpoint rule mismatch")

        evaluated = stats.dropna(subset=["losses/eval.total_loss"])
        min_valid = evaluated.loc[evaluated["losses/eval.total_loss"].idxmin()]
        snapshot_epoch = best_snapshot_epoch(train_dir)
        tested_rows = stats.loc[stats["step"] == snapshot_epoch]
        if len(tested_rows) != 1:
            raise RuntimeError(
                f"No unique stats row for batch {batch_size}, epoch {snapshot_epoch}"
            )
        tested = tested_rows.iloc[0]

        prediction_dir = (
            TEST_ROOT / f"predictions_100train_hrnet_w32_batch{batch_size}"
        )
        test_summary = pd.read_csv(prediction_dir / "eye_test_summary.csv")
        per_frame = pd.read_csv(
            prediction_dir / "eye_test_per_frame_errors.csv", index_col=0
        )
        if len(per_frame) != 100:
            raise RuntimeError(
                f"Batch {batch_size} test has {len(per_frame)} frames"
            )
        frame_names = per_frame.index.tolist()
        if expected_frames is None:
            expected_frames = frame_names
        elif frame_names != expected_frames:
            raise RuntimeError(f"Batch {batch_size} used a different test set")
        frame_data[batch_size] = per_frame

        likelihoods = [
            metric(test_summary, f"{part}_mean_likelihood")
            for part in BODY_PARTS
        ]
        confident = int(
            metric(
                test_summary,
                "overall_keypoint_rmse_px_pcutoff_0.6",
                "n",
            )
        )
        run_meta = metadata.loc[
            metadata["batch_size"] == batch_size
        ].iloc[-1]
        row = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "architecture": "HRNet-W32",
            "epochs_completed": 200,
            "dlc_checkpoint_metric": "internal test.mAP",
            "dlc_best_snapshot_epoch": snapshot_epoch,
            "internal_mAP_at_tested_snapshot_percent": float(
                tested["metrics/test.mAP"]
            ),
            "internal_valid_loss_at_tested_snapshot": float(
                tested["losses/eval.total_loss"]
            ),
            "minimum_internal_valid_loss": float(
                min_valid["losses/eval.total_loss"]
            ),
            "minimum_internal_valid_loss_epoch": int(min_valid["step"]),
            "external_overall_keypoint_rmse_px": metric(
                test_summary, "overall_keypoint_rmse_px"
            ),
            "external_pupil_center_rmse_px": metric(
                test_summary, "pupil_center_rmse_px"
            ),
            "external_pupil_width_mae_px": metric(
                test_summary, "pupil_width_mae_px"
            ),
            "external_mean_likelihood": float(np.mean(likelihoods)),
            "external_points_likelihood_ge_0_6": confident,
            "external_points_total": 800,
            "external_fraction_likelihood_ge_0_6": confident / 800,
            "pipeline_elapsed_hours": (
                np.nan
                if batch_size == 8
                else float(run_meta["training_duration_hours"])
            ),
            "reused_existing_model": batch_size == 8,
        }
        row.update(frame_statistics(per_frame))
        summary_rows.append(row)

        for part in BODY_PARTS:
            keypoint_rows.append(
                {
                    "batch_size": batch_size,
                    "bodypart": part,
                    "rmse_px": metric(test_summary, f"{part}_rmse_px"),
                    "mean_likelihood": metric(
                        test_summary, f"{part}_mean_likelihood"
                    ),
                }
            )

        curve = stats[
            [
                "step",
                "losses/train.total_loss",
                "losses/eval.total_loss",
                "metrics/test.mAP",
                "metrics/test.rmse",
            ]
        ].copy()
        curve.insert(0, "batch_size", batch_size)
        curve_rows.append(curve)

    summary = pd.DataFrame(summary_rows).sort_values("batch_size")
    return (
        summary.reset_index(drop=True),
        pd.DataFrame(keypoint_rows),
        pd.concat(curve_rows, ignore_index=True),
        frame_data,
    )


def save_overview(data: pd.DataFrame) -> None:
    x = data["batch_size"].to_numpy()
    selected = data.loc[data["minimum_internal_valid_loss"].idxmin()]
    external_best = data.loc[
        data["external_overall_keypoint_rmse_px"].idxmin()
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    ax = axes[0, 0]
    series = [
        ("external_overall_keypoint_rmse_px", "overall keypoint RMSE", "#2F6B9A"),
        ("external_pupil_center_rmse_px", "pupil center RMSE", "#D1495B"),
        ("external_pupil_width_mae_px", "pupil width MAE", "#2A9D8F"),
    ]
    for column, label, color in series:
        ax.plot(x, data[column], "o-", color=color, label=label)
    ax.set(
        title="Locked final-minute test (report only)",
        ylabel="error (px; lower is better)",
        xticks=x,
    )
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(x, data["minimum_internal_valid_loss"], "o-", color="#6A4C93")
    ax.scatter(
        selected["batch_size"],
        selected["minimum_internal_valid_loss"],
        marker="*",
        s=180,
        color="#E9C46A",
        edgecolor="black",
        zorder=3,
    )
    ax.annotate(
        f"protocol selection: batch {int(selected['batch_size'])}",
        (selected["batch_size"], selected["minimum_internal_valid_loss"]),
        xytext=(-105, 18),
        textcoords="offset points",
        fontsize=8,
    )
    ax.set(
        title="Pre-specified selection metric",
        ylabel="minimum internal validation loss",
        xticks=x,
    )

    ax = axes[1, 0]
    positions = np.arange(len(data))
    width = 0.34
    ax.bar(
        positions - width / 2,
        data["external_mean_likelihood"],
        width,
        color="#457B9D",
        label="mean likelihood",
    )
    ax.bar(
        positions + width / 2,
        data["external_fraction_likelihood_ge_0_6"],
        width,
        color="#E76F51",
        label="fraction >= 0.6",
    )
    ax.set(
        title="External-test confidence diagnostic",
        ylabel="fraction / mean likelihood",
        xticks=positions,
        xticklabels=x,
        ylim=(0, 1),
    )
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.scatter(
        data["minimum_internal_valid_loss"],
        data["external_overall_keypoint_rmse_px"],
        s=70,
        color="#264653",
    )
    for row in data.itertuples():
        ax.annotate(
            f"batch {row.batch_size}",
            (
                row.minimum_internal_valid_loss,
                row.external_overall_keypoint_rmse_px,
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.scatter(
        external_best["minimum_internal_valid_loss"],
        external_best["external_overall_keypoint_rmse_px"],
        marker="s",
        facecolors="none",
        edgecolors="#E63946",
        s=150,
        linewidths=2,
    )
    ax.set(
        title="Internal selection vs external report",
        xlabel="minimum internal validation loss",
        ylabel="external overall RMSE (px)",
    )

    for ax in axes.flat:
        ax.grid(alpha=0.22)
    fig.suptitle(
        "HRNet-W32 batch-size sweep: 100-frame training pool",
        fontsize=14,
    )
    fig.savefig(OUTPUT_ROOT / "01_batch_size_overview.png", dpi=220)
    plt.close(fig)


def save_learning_curves(
    curves: pd.DataFrame, summary: pd.DataFrame
) -> None:
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.8, len(RUNS)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for color, (batch_size, _) in zip(colors, RUNS, strict=True):
        subset = curves.loc[curves["batch_size"] == batch_size]
        axes[0].plot(
            subset["step"],
            subset["losses/train.total_loss"],
            color=color,
            label=f"batch {batch_size}",
            linewidth=1.4,
        )
        evaluated = subset.dropna(subset=["losses/eval.total_loss"])
        axes[1].plot(
            evaluated["step"],
            evaluated["losses/eval.total_loss"],
            "o-",
            color=color,
            label=f"batch {batch_size}",
            markersize=3,
        )
        tested_epoch = int(
            summary.loc[
                summary["batch_size"] == batch_size,
                "dlc_best_snapshot_epoch",
            ].iloc[0]
        )
        tested_row = evaluated.loc[evaluated["step"] == tested_epoch]
        axes[1].scatter(
            tested_row["step"],
            tested_row["losses/eval.total_loss"],
            marker="*",
            s=80,
            color=color,
            edgecolor="black",
            zorder=3,
        )
    axes[0].set(title="Training loss", xlabel="epoch", ylabel="total loss")
    axes[1].set(
        title="Validation loss (stars = tested DLC snapshots)",
        xlabel="epoch",
        ylabel="total loss",
    )
    for ax in axes:
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8)
    fig.suptitle("Learning curves; identical 95/5 split", fontsize=13)
    fig.savefig(OUTPUT_ROOT / "02_learning_curves.png", dpi=220)
    plt.close(fig)


def save_keypoint_heatmaps(keypoints: pd.DataFrame) -> None:
    rmse = keypoints.pivot(
        index="bodypart", columns="batch_size", values="rmse_px"
    ).loc[BODY_PARTS]
    likelihood = keypoints.pivot(
        index="bodypart", columns="batch_size", values="mean_likelihood"
    ).loc[BODY_PARTS]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    settings = [
        (axes[0], rmse, "Keypoint RMSE (px; lower is better)", "YlOrRd", ".1f"),
        (
            axes[1],
            likelihood,
            "Mean DLC likelihood (higher is better)",
            "YlGnBu",
            ".2f",
        ),
    ]
    for ax, matrix, title, cmap, fmt in settings:
        image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap=cmap)
        ax.set_xticks(
            range(len(matrix.columns)),
            labels=[str(value) for value in matrix.columns],
        )
        ax.set_yticks(
            range(len(matrix.index)),
            labels=[DISPLAY_NAMES[value] for value in matrix.index],
        )
        ax.set(xlabel="batch size", title=title)
        midpoint = (
            float(np.nanmin(matrix.to_numpy()))
            + float(np.nanmax(matrix.to_numpy()))
        ) / 2
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iloc[i, j]
                ax.text(
                    j,
                    i,
                    format(value, fmt),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value > midpoint else "black",
                )
        fig.colorbar(image, ax=ax, shrink=0.78)
    fig.suptitle("Locked final-minute test by keypoint", fontsize=13)
    fig.savefig(OUTPUT_ROOT / "03_keypoint_diagnostics.png", dpi=220)
    plt.close(fig)


def save_error_distributions(frame_data) -> None:
    frame_rmse = []
    center_error = []
    labels = []
    for batch_size, _ in RUNS:
        per_frame = frame_data[batch_size]
        columns = [f"{part}_error_px" for part in BODY_PARTS]
        frame_rmse.append(
            np.sqrt(np.mean(np.square(per_frame[columns]), axis=1))
        )
        center_error.append(per_frame["pupil_center_error_px"].to_numpy())
        labels.append(str(batch_size))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    settings = [
        (
            axes[0],
            frame_rmse,
            "Per-frame keypoint RMSE distribution",
            "frame RMSE (px)",
        ),
        (
            axes[1],
            center_error,
            "Per-frame pupil-center error distribution",
            "center error (px)",
        ),
    ]
    for ax, values, title, ylabel in settings:
        plot = ax.boxplot(
            values,
            labels=labels,
            showfliers=True,
            patch_artist=True,
        )
        for patch, color in zip(
            plot["boxes"], plt.get_cmap("Set2").colors, strict=False
        ):
            patch.set_facecolor(color)
        ax.set(title=title, xlabel="batch size", ylabel=ylabel)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Same 100 held-out frames in every run", fontsize=13)
    fig.savefig(OUTPUT_ROOT / "04_error_distributions.png", dpi=220)
    plt.close(fig)


def markdown_table(data: pd.DataFrame) -> str:
    lines = [
        "| Batch | DLC best epoch | Min valid loss (epoch) | "
        "Overall RMSE | Center RMSE | Width MAE | Likelihood >= 0.6 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in data.itertuples():
        lines.append(
            f"| {row.batch_size} | {row.dlc_best_snapshot_epoch} | "
            f"{row.minimum_internal_valid_loss:.5f} "
            f"({row.minimum_internal_valid_loss_epoch}) | "
            f"{row.external_overall_keypoint_rmse_px:.2f} px | "
            f"{row.external_pupil_center_rmse_px:.2f} px | "
            f"{row.external_pupil_width_mae_px:.2f} px | "
            f"{row.external_points_likelihood_ge_0_6}/800 |"
        )
    return "\n".join(lines)


def save_report(data: pd.DataFrame) -> None:
    selected = data.loc[data["minimum_internal_valid_loss"].idxmin()]
    external_best = data.loc[
        data["external_overall_keypoint_rmse_px"].idxmin()
    ]
    loss_span = 100 * (
        data["minimum_internal_valid_loss"].max()
        / data["minimum_internal_valid_loss"].min()
        - 1
    )
    tested_epochs = ", ".join(
        str(int(value)) for value in data["dlc_best_snapshot_epoch"]
    )
    report = f"""# HRNet-W32 Batch-Size Sweep

## Question

This experiment tests batch sizes `1, 2, 4, 8, 16` before fixing a
batch size for the architecture comparison. Every run uses HRNet-W32.

## Locked Design

- Training pool: 100 frames from the first four minutes.
- Internal split: identical 95 training / 5 validation frames.
- Schedule: 200 epochs on CPU.
- External test: identical 100 manually labeled final-minute frames
  containing 800 keypoints.
- Final-minute labels were never added to training.
- Within-run checkpoint: DeepLabCut `snapshot-best-*`, selected by
  maximum internal `test.mAP`.
- Across-batch rule: lowest observed internal validation total loss.
  External-test error is report-only.

## Results

{markdown_table(data)}

The pre-specified internal criterion selects **batch {int(selected['batch_size'])}** (minimum validation loss {selected['minimum_internal_valid_loss']:.5f} at epoch {int(selected['minimum_internal_valid_loss_epoch'])}). The locked external test has its lowest overall RMSE at **batch {int(external_best['batch_size'])}** ({external_best['external_overall_keypoint_rmse_px']:.2f} px), but that result must not be used retroactively to select hyperparameters.

## Figures

![Batch-size overview](01_batch_size_overview.png)

![Learning curves](02_learning_curves.png)

![Keypoint diagnostics](03_keypoint_diagnostics.png)

![Per-frame error distributions](04_error_distributions.png)

## 中文摘要

按实验开始前固定的内部验证损失规则，应选择 **batch {int(selected['batch_size'])}** 进入下一步模型架构比较。最后一分钟测试中 batch {int(external_best['batch_size'])} 的观测误差最低，但该测试集只能用于最终报告，不能反过来改变超参数。内部验证集只有 5 张图片，而且每个 batch size 只运行了一次，因此当前 batch 选择应视为暂定结果。所有模型的置信度校准都较弱，在校准前不应直接使用 `pcutoff=0.6` 过滤预测点。

## Interpretation

1. Internal validation and final-minute test rankings disagree. The
   internal validation set has only five images, and minimum validation
   losses span only {loss_span:.1f}% across all runs.
2. Batch 2 is the strongest external-test observation: overall RMSE {external_best['external_overall_keypoint_rmse_px']:.2f} px, pupil-center RMSE {external_best['external_pupil_center_rmse_px']:.2f} px, and pupil-width MAE {external_best['external_pupil_width_mae_px']:.2f} px. This is report-only, not a valid reason to change the frozen rule.
3. Confidence is poorly calibrated. Only batch 2 produces more than 2/800 predictions with likelihood >= 0.6, and it has only {int(external_best['external_points_likelihood_ge_0_6'])}/800. Do not use `pcutoff=0.6` as a hard filter before calibration.
4. Tested snapshots are not the epochs with minimum validation loss.
   DeepLabCut saves its best checkpoint by maximum internal mAP. Tested
   epochs are {tested_epochs} for batches 1, 2, 4, 8, and 16.
5. Fixed epochs do not mean equal optimizer updates: larger batches
   make fewer updates per epoch. This is a fixed-200-epoch sweep, not a
   compute- or update-matched experiment.
6. Each batch size was run once. Small differences need repeated runs
   before they can be interpreted as precise batch-size effects.

## Decision

To preserve test-set isolation, use **batch {int(selected['batch_size'])}** for the next architecture comparison because the pre-specified internal rule selected it. Treat this as provisional. Before a definitive production-model decision, use a larger internal validation set or repeated runs and freeze one internally consistent checkpoint/selection rule.

## Files

- `batch_size_summary.csv`: one audited row per run.
- `keypoint_metrics.csv`: per-keypoint RMSE and likelihood.
- `learning_curves.csv`: epoch-level train/validation records.
- `selection.json`: protocol and report-only selections.
- `audit.json`: completion and train/test isolation checks.
- `01_batch_size_overview.png`: metrics and ranking mismatch.
- `02_learning_curves.png`: train and validation trajectories.
- `03_keypoint_diagnostics.png`: keypoint heatmaps.
- `04_error_distributions.png`: per-frame errors and outliers.

Raw videos, frames, labels, predictions, outlier montages, and model
weights stay in ignored local directories and are not on GitHub.
"""
    (OUTPUT_ROOT / "README.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    dataset_audit = audit_datasets()
    summary, keypoints, curves, frame_data = load_results()
    selected_idx = summary["minimum_internal_valid_loss"].idxmin()
    external_idx = summary[
        "external_overall_keypoint_rmse_px"
    ].idxmin()
    summary["selected_by_prespecified_internal_rule"] = False
    summary.loc[selected_idx, "selected_by_prespecified_internal_rule"] = True
    summary["lowest_external_rmse_report_only"] = False
    summary.loc[external_idx, "lowest_external_rmse_report_only"] = True

    summary.to_csv(OUTPUT_ROOT / "batch_size_summary.csv", index=False)
    keypoints.to_csv(OUTPUT_ROOT / "keypoint_metrics.csv", index=False)
    curves.to_csv(OUTPUT_ROOT / "learning_curves.csv", index=False)
    selection = {
        "selection_rule": "lowest minimum internal validation total loss",
        "selected_batch_size": int(summary.loc[selected_idx, "batch_size"]),
        "selected_minimum_internal_validation_loss": float(
            summary.loc[selected_idx, "minimum_internal_valid_loss"]
        ),
        "external_test_policy": (
            "report only; not used for hyperparameter selection"
        ),
        "lowest_external_overall_rmse_batch_size": int(
            summary.loc[external_idx, "batch_size"]
        ),
        "checkpoint_rule_within_each_run": (
            "DeepLabCut snapshot-best selected by maximum internal test.mAP"
        ),
    }
    (OUTPUT_ROOT / "selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    audit = {
        **dataset_audit,
        "five_batch_runs_complete": True,
        "all_runs_reached_200_epochs": True,
        "all_models_hrnet_w32": True,
        "configured_batch_sizes_verified": True,
        "identical_external_test_frame_order": True,
        "external_test_used_for_reporting_only": True,
    }
    (OUTPUT_ROOT / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    save_overview(summary)
    save_learning_curves(curves, summary)
    save_keypoint_heatmaps(keypoints)
    save_error_distributions(frame_data)
    save_report(summary)
    print(summary.to_string(index=False))
    print(f"published: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
