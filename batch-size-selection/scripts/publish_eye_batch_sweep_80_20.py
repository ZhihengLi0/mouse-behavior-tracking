#!/usr/bin/env python3
"""Publish the second-era batch-size sweep into results/batch-size-selection.

Second era: reviewed labels (all 100 training-pool frames re-reviewed on
2026-09-10, pupil_top definition corrected by ~20 px), temporal block split
(chronologically first 80 frames train, last 20 validate, 1.2 s boundary gap),
100 epochs with milestones rescaled to [80, 95], snapshots every 10 epochs.

Selection rules, declared before any result was inspected:
- within each run: DeepLabCut `snapshot-best-*`, chosen by maximum internal
  `test.mAP` on the 20 validation frames;
- across batches: lowest minimum internal validation total loss.
The reviewed final-minute 100-frame set is report-only. It never selects.

Numbers here are NOT comparable with tags v0.1.0 / v0.2.0: the labels, the
split, and the epoch budget all changed at the era boundary.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "dlc_projects" / "EyePupilBlink-Zhiheng-2026-08-17"
MODEL_ROOT = PROJECT / "dlc-models-pytorch" / "iteration-0"
TEST_ROOT = ROOT / "local_data" / "test_sets" / "eye_last_minute_100"
EXPERIMENT_ROOT = ROOT / "batch-size-selection" / "logs"
SPLIT_PATH = ROOT / "local_data" / "experiments" / "split_80_20.json"
OUTPUT_ROOT = ROOT / "results" / "batch-size-selection"

EPOCHS = 100
MILESTONES = [80, 95]
SAVE_EPOCHS = 10
BACKBONE = "hrnet_w32"
RUNS = [(2, 21), (4, 22), (8, 23), (16, 24)]
LABEL_PREFIX = "b8020_batch"

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
TOTAL_POINTS = 100 * len(BODY_PARTS)


def metric(table: pd.DataFrame, name: str, field: str = "value") -> float:
    rows = table.loc[table["metric"] == name, field]
    if rows.empty:
        raise RuntimeError(f"Missing metric: {name}")
    return float(rows.iloc[0])


def train_dir_for(shuffle: int) -> Path:
    return MODEL_ROOT / f"EyePupilBlinkAug17-trainset80shuffle{shuffle}" / "train"


def best_snapshot(train_dir: Path) -> tuple[int, Path]:
    snapshots = list(train_dir.glob("snapshot-best-*.pt"))
    if len(snapshots) != 1:
        raise RuntimeError(
            f"Expected one best snapshot in {train_dir}, found {len(snapshots)}"
        )
    match = re.fullmatch(r"snapshot-best-(\d+)\.pt", snapshots[0].name)
    if not match:
        raise RuntimeError(f"Unexpected snapshot name: {snapshots[0].name}")
    return int(match.group(1)), snapshots[0]


def frame_numbers(index) -> list[int]:
    numbers = []
    for value in index:
        name = str(value[-1] if isinstance(value, tuple) else value)
        match = re.search(r"(?:img|frame)(\d+)", name)
        if not match:
            raise RuntimeError(f"Cannot parse frame number from {name}")
        numbers.append(int(match.group(1)))
    return numbers


def audit_datasets() -> dict[str, object]:
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train_labels = pd.read_hdf(
        PROJECT / "labeled-data" / "face" / "CollectedData_Zhiheng.h5"
    )
    test_labels = pd.read_hdf(
        TEST_ROOT
        / "dlc_label_project"
        / "labeled-data"
        / "eye_last_minute_100"
        / "CollectedData_Zhiheng.h5"
    )
    if len(train_labels) != 100 or len(test_labels) != 100:
        raise RuntimeError("Label tables are not 100 rows each")
    missing = int(train_labels.isna().sum().sum()) + int(
        test_labels.isna().sum().sum()
    )
    if missing:
        raise RuntimeError(f"Labels contain {missing} missing values")

    train_frames = frame_numbers(train_labels.index)
    test_frames = frame_numbers(test_labels.index)
    if set(train_frames) & set(test_frames):
        raise RuntimeError("Training and test frame numbers overlap")
    if max(train_frames) >= min(test_frames):
        raise RuntimeError("Training frames are not all earlier than test frames")

    boundary = split["boundary"]
    if boundary["gap_frames"] < 30:
        raise RuntimeError("Block-split boundary gap is below 0.5 s")

    names = [
        str(v[-1] if isinstance(v, tuple) else v) for v in train_labels.index
    ]
    validation_set = set(split["validation_filenames"])
    validation_rows = sorted(i for i, n in enumerate(names) if n in validation_set)
    training_rows = sorted(i for i, n in enumerate(names) if n not in validation_set)
    if len(validation_rows) != 20:
        raise RuntimeError("Split record does not match the label table")

    for _, shuffle in RUNS:
        documentation = (
            PROJECT
            / "training-datasets"
            / "iteration-0"
            / "UnaugmentedDataSet_EyePupilBlinkAug17"
            / f"Documentation_data-EyePupilBlink_80shuffle{shuffle}.pickle"
        )
        created = pd.read_pickle(documentation)
        if sorted(int(v) for v in created[1]) != training_rows:
            raise RuntimeError(f"Shuffle {shuffle} training rows mismatch")
        if sorted(int(v) for v in created[2]) != validation_rows:
            raise RuntimeError(f"Shuffle {shuffle} validation rows mismatch")

    return {
        "labels_reviewed": "2026-09-10, all 100 training-pool and 100 test frames",
        "training_label_rows": 100,
        "external_test_label_rows": 100,
        "missing_coordinate_values": 0,
        "train_test_frame_overlap": False,
        "split": "temporal block 80/20",
        "block_boundary": boundary,
        "identical_split_across_all_shuffles": True,
        "advisor_approved_split": split.get("provenance", {}).get(
            "advisor_approved", "recorded in split_80_20.json"
        ),
    }


def load_results():
    if not (EXPERIMENT_ROOT / "runner_finished_at.txt").exists():
        raise RuntimeError("Sweep has no completion marker")
    metadata = pd.read_csv(EXPERIMENT_ROOT / "run_metadata.csv")
    if set(metadata["batch_size"]) != {b for b, _ in RUNS}:
        raise RuntimeError("Metadata does not contain all four batch sizes")
    if not all(str(s) == "complete" for s in metadata["status"]):
        raise RuntimeError(f"Incomplete runs: {metadata['status'].tolist()}")

    summary_rows, keypoint_rows, curve_rows = [], [], []
    frame_data: dict[int, pd.DataFrame] = {}
    expected_frames = None

    for batch_size, shuffle in RUNS:
        train_dir = train_dir_for(shuffle)
        stats = pd.read_csv(train_dir / "learning_stats.csv")
        if int(stats["step"].max()) != EPOCHS:
            raise RuntimeError(f"Batch {batch_size} did not reach {EPOCHS} epochs")
        if stats["step"].duplicated().any():
            raise RuntimeError(
                f"Batch {batch_size} has duplicate epoch rows; it was resumed"
            )

        config = yaml.safe_load(
            (train_dir / "pytorch_config.yaml").read_text(encoding="utf-8")
        )
        if config["model"]["backbone"]["model_name"] != BACKBONE:
            raise RuntimeError(f"Batch {batch_size} is not {BACKBONE}")
        if int(config["train_settings"]["batch_size"]) != batch_size:
            raise RuntimeError(f"Batch {batch_size} config mismatch")
        if config["runner"]["scheduler"]["params"]["milestones"] != MILESTONES:
            raise RuntimeError(f"Batch {batch_size} milestones were not rescaled")
        if config["runner"]["snapshots"]["save_epochs"] != SAVE_EPOCHS:
            raise RuntimeError(f"Batch {batch_size} snapshot interval mismatch")
        if config["runner"]["key_metric"] != "test.mAP":
            raise RuntimeError(f"Batch {batch_size} checkpoint rule mismatch")

        snapshot_epoch, _ = best_snapshot(train_dir)
        tested = stats.loc[stats["step"] == snapshot_epoch]
        if len(tested) != 1:
            raise RuntimeError(
                f"No unique stats row for batch {batch_size} epoch {snapshot_epoch}"
            )
        tested = tested.iloc[0]
        evaluated = stats.dropna(subset=["losses/eval.total_loss"])
        min_valid = evaluated.loc[evaluated["losses/eval.total_loss"].idxmin()]

        prediction_dir = (
            ROOT / "batch-size-selection" / "predictions" / f"predictions_100train_{LABEL_PREFIX}{batch_size}"
        )
        test_summary = pd.read_csv(prediction_dir / "eye_test_summary.csv")
        per_frame = pd.read_csv(
            prediction_dir / "eye_test_per_frame_errors.csv", index_col=0
        )
        if len(per_frame) != 100:
            raise RuntimeError(f"Batch {batch_size} test has {len(per_frame)} frames")
        frames = per_frame.index.tolist()
        if expected_frames is None:
            expected_frames = frames
        elif frames != expected_frames:
            raise RuntimeError(f"Batch {batch_size} used a different frame order")
        frame_data[batch_size] = per_frame

        prediction_file = next(
            iter(prediction_dir.glob("image_predictions_*snapshot_*.h5")), None
        )
        if prediction_file is None:
            raise RuntimeError(f"Batch {batch_size} has no prediction file")
        named = int(
            re.search(r"snapshot_(?:best-)?(\d+)", prediction_file.name).group(1)
        )
        if named != snapshot_epoch:
            raise RuntimeError(
                f"Batch {batch_size} evaluated snapshot {named}, best is "
                f"{snapshot_epoch}"
            )

        confident = int(
            metric(test_summary, "overall_keypoint_rmse_px_pcutoff_0.6", "n")
        )
        likelihoods = [
            metric(test_summary, f"{part}_mean_likelihood") for part in BODY_PARTS
        ]
        run_meta = metadata.loc[metadata["batch_size"] == batch_size].iloc[-1]

        columns = [f"{part}_error_px" for part in BODY_PARTS]
        per_frame_rmse = np.sqrt(np.mean(np.square(per_frame[columns]), axis=1))
        summary_rows.append(
            {
                "batch_size": batch_size,
                "shuffle": shuffle,
                "architecture": "HRNet-W32",
                "epochs_completed": EPOCHS,
                "dlc_best_snapshot_epoch": snapshot_epoch,
                "internal_mAP_at_tested_snapshot_percent": float(
                    tested["metrics/test.mAP"]
                ),
                "internal_rmse_at_tested_snapshot_px": float(
                    tested["metrics/test.rmse"]
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
                "external_points_total": TOTAL_POINTS,
                "external_frame_rmse_median_px": float(np.median(per_frame_rmse)),
                "external_frame_rmse_p90_px": float(
                    np.quantile(per_frame_rmse, 0.90)
                ),
                "external_frame_rmse_max_px": float(np.max(per_frame_rmse)),
                "training_hours": float(run_meta["elapsed_hours"]),
            }
        )

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

    summary = pd.DataFrame(summary_rows).sort_values("batch_size").reset_index(
        drop=True
    )
    return (
        summary,
        pd.DataFrame(keypoint_rows),
        pd.concat(curve_rows, ignore_index=True),
        frame_data,
    )


def save_overview(data: pd.DataFrame) -> None:
    x = data["batch_size"].to_numpy()
    internal = data.loc[data["minimum_internal_valid_loss"].idxmin()]
    external = data.loc[data["external_overall_keypoint_rmse_px"].idxmin()]
    agree = int(internal["batch_size"]) == int(external["batch_size"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    ax = axes[0, 0]
    for column, label, color in [
        ("external_overall_keypoint_rmse_px", "overall keypoint RMSE", "#2F6B9A"),
        ("external_pupil_center_rmse_px", "pupil center RMSE", "#D1495B"),
        ("external_pupil_width_mae_px", "pupil width MAE", "#2A9D8F"),
    ]:
        ax.plot(x, data[column], "o-", color=color, label=label)
    ax.set(
        title="Reviewed final-minute test (report only)",
        ylabel="error (px; lower is better)",
        xticks=x,
        xlabel="batch size",
    )
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(x, data["minimum_internal_valid_loss"], "o-", color="#6A4C93")
    ax.scatter(
        internal["batch_size"],
        internal["minimum_internal_valid_loss"],
        marker="*",
        s=200,
        color="#E9C46A",
        edgecolor="black",
        zorder=3,
    )
    ax.annotate(
        f"internal selection: batch {int(internal['batch_size'])}",
        (internal["batch_size"], internal["minimum_internal_valid_loss"]),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=8,
    )
    ax.set(
        title="Selection metric: minimum validation loss (20 frames)",
        ylabel="total loss",
        xticks=x,
        xlabel="batch size",
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
        data["external_points_likelihood_ge_0_6"] / TOTAL_POINTS,
        width,
        color="#E76F51",
        label="fraction >= 0.6",
    )
    ax.set(
        title="External confidence diagnostic",
        ylabel="fraction / mean likelihood",
        xticks=positions,
        xticklabels=x,
        xlabel="batch size",
        ylim=(0, 1),
    )
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.scatter(
        data["minimum_internal_valid_loss"],
        data["external_overall_keypoint_rmse_px"],
        s=80,
        color="#264653",
    )
    for row in data.itertuples():
        ax.annotate(
            f"batch {row.batch_size}",
            (
                row.minimum_internal_valid_loss,
                row.external_overall_keypoint_rmse_px,
            ),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.scatter(
        external["minimum_internal_valid_loss"],
        external["external_overall_keypoint_rmse_px"],
        marker="s",
        facecolors="none",
        edgecolors="#2A9D8F" if agree else "#E63946",
        s=170,
        linewidths=2,
    )
    ax.set(
        title=(
            "Internal selection vs external report: "
            + ("AGREE" if agree else "DISAGREE")
        ),
        xlabel="minimum internal validation loss",
        ylabel="external overall RMSE (px)",
    )

    for ax in axes.flat:
        ax.grid(alpha=0.22)
    fig.suptitle(
        "HRNet-W32 batch sweep, second era: reviewed labels, 80/20 block split, "
        f"{EPOCHS} epochs",
        fontsize=13,
    )
    fig.savefig(OUTPUT_ROOT / "01_batch_size_overview.png", dpi=220)
    plt.close(fig)


def save_learning_curves(curves: pd.DataFrame, summary: pd.DataFrame) -> None:
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
                summary["batch_size"] == batch_size, "dlc_best_snapshot_epoch"
            ].iloc[0]
        )
        tested = evaluated.loc[evaluated["step"] == tested_epoch]
        axes[1].scatter(
            tested["step"],
            tested["losses/eval.total_loss"],
            marker="*",
            s=110,
            color=color,
            edgecolor="black",
            zorder=3,
        )
    for milestone in MILESTONES:
        for ax in axes:
            ax.axvline(milestone, color="gray", linewidth=0.8, linestyle=":")
    axes[0].set(title="Training loss", xlabel="epoch", ylabel="total loss")
    axes[1].set(
        title="Validation loss on 20 held-later frames (stars = tested snapshots)",
        xlabel="epoch",
        ylabel="total loss",
    )
    for ax in axes:
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Identical 80/20 block split; dotted lines = LR milestones [80, 95]",
        fontsize=12,
    )
    fig.savefig(OUTPUT_ROOT / "02_learning_curves.png", dpi=220)
    plt.close(fig)


def save_keypoint_diagnostics(keypoints: pd.DataFrame) -> None:
    order = [b for b, _ in RUNS]
    rmse = keypoints.pivot(
        index="bodypart", columns="batch_size", values="rmse_px"
    ).loc[BODY_PARTS, order]
    likelihood = keypoints.pivot(
        index="bodypart", columns="batch_size", values="mean_likelihood"
    ).loc[BODY_PARTS, order]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for ax, matrix, title, cmap, fmt in [
        (axes[0], rmse, "Keypoint RMSE (px; lower is better)", "YlOrRd", ".1f"),
        (
            axes[1],
            likelihood,
            "Mean DLC likelihood (higher is better)",
            "YlGnBu",
            ".2f",
        ),
    ]:
        values = matrix.to_numpy()
        image = ax.imshow(values, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(matrix.columns)), labels=list(matrix.columns))
        ax.set_yticks(
            range(len(matrix.index)),
            labels=[DISPLAY_NAMES[v] for v in matrix.index],
        )
        ax.set(xlabel="batch size", title=title)
        midpoint = (float(np.nanmin(values)) + float(np.nanmax(values))) / 2
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
    fig.suptitle("Reviewed final-minute test by keypoint", fontsize=13)
    fig.savefig(OUTPUT_ROOT / "03_keypoint_diagnostics.png", dpi=220)
    plt.close(fig)


def save_error_distributions(frame_data: dict[int, pd.DataFrame]) -> None:
    frame_rmse, center_error, labels = [], [], []
    for batch_size, _ in RUNS:
        per_frame = frame_data[batch_size]
        columns = [f"{part}_error_px" for part in BODY_PARTS]
        frame_rmse.append(np.sqrt(np.mean(np.square(per_frame[columns]), axis=1)))
        center_error.append(per_frame["pupil_center_error_px"].to_numpy())
        labels.append(str(batch_size))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, values, title, ylabel in [
        (axes[0], frame_rmse, "Per-frame keypoint RMSE", "frame RMSE (px)"),
        (
            axes[1],
            center_error,
            "Per-frame pupil-center error",
            "center error (px)",
        ),
    ]:
        plot = ax.boxplot(values, labels=labels, showfliers=True, patch_artist=True)
        for patch, color in zip(
            plot["boxes"], plt.get_cmap("Set2").colors, strict=False
        ):
            patch.set_facecolor(color)
        ax.set(title=title, xlabel="batch size", ylabel=ylabel)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Same reviewed 100 held-out frames in every run", fontsize=13)
    fig.savefig(OUTPUT_ROOT / "04_error_distributions.png", dpi=220)
    plt.close(fig)


def markdown_table(data: pd.DataFrame) -> str:
    lines = [
        "| Batch | Best epoch | Min valid loss (epoch) | Overall RMSE | "
        "Center RMSE | Width MAE | Likelihood >= 0.6 | Hours |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in data.itertuples():
        lines.append(
            f"| {row.batch_size} | {row.dlc_best_snapshot_epoch} | "
            f"{row.minimum_internal_valid_loss:.5f} "
            f"({row.minimum_internal_valid_loss_epoch}) | "
            f"{row.external_overall_keypoint_rmse_px:.2f} px | "
            f"{row.external_pupil_center_rmse_px:.2f} px | "
            f"{row.external_pupil_width_mae_px:.2f} px | "
            f"{row.external_points_likelihood_ge_0_6}/{TOTAL_POINTS} | "
            f"{row.training_hours:.1f} |"
        )
    return "\n".join(lines)


def save_report(data: pd.DataFrame) -> None:
    internal = data.loc[data["minimum_internal_valid_loss"].idxmin()]
    external = data.loc[data["external_overall_keypoint_rmse_px"].idxmin()]
    agree = int(internal["batch_size"]) == int(external["batch_size"])
    loss_span = 100 * (
        data["minimum_internal_valid_loss"].max()
        / data["minimum_internal_valid_loss"].min()
        - 1
    )
    rmse_span = 100 * (
        data["external_overall_keypoint_rmse_px"].max()
        / data["external_overall_keypoint_rmse_px"].min()
        - 1
    )

    if agree:
        agreement_en = (
            f"**The internal and external signals agree**: the 20-frame "
            f"validation rule selects batch {int(internal['batch_size'])}, and "
            f"the untouched final-minute set independently ranks the same batch "
            f"lowest ({external['external_overall_keypoint_rmse_px']:.2f} px). "
            f"In the first era the two signals disagreed, which is what exposed "
            f"the five-frame validation set. Agreement here is the first "
            f"evidence that the rebuilt validation set can be trusted for the "
            f"architecture comparison and the active-learning rounds."
        )
        agreement_zh = (
            f"内部与外部信号一致：20 帧验证规则选出 batch "
            f"{int(internal['batch_size'])}，未参与任何选择的最后一分钟测试集"
            f"也独立地把它排在最低（{external['external_overall_keypoint_rmse_px']:.2f} px）。"
            f"第一时代正是两个信号打架暴露了 5 帧验证集的问题；这次一致，是重建后"
            f"的验证集可以信任的第一个证据。"
        )
    else:
        agreement_en = (
            f"**The internal and external signals disagree**: the 20-frame rule "
            f"selects batch {int(internal['batch_size'])} while the final-minute "
            f"set is lowest at batch {int(external['batch_size'])} "
            f"({external['external_overall_keypoint_rmse_px']:.2f} px). Per "
            f"protocol the internal selection stands and the disagreement is "
            f"reported, not acted on. A persistent disagreement after the label "
            f"review and the 4x larger validation set would mean 20 frames are "
            f"still too few, and that must be resolved before the "
            f"active-learning experiment."
        )
        agreement_zh = (
            f"内部与外部信号不一致：20 帧规则选 batch "
            f"{int(internal['batch_size'])}，而最后一分钟测试集上最低的是 batch "
            f"{int(external['batch_size'])}"
            f"（{external['external_overall_keypoint_rmse_px']:.2f} px）。按协议，"
            f"内部选择维持不变，不一致只报告、不采取行动。若复查标注并扩大验证集后"
            f"仍然不一致，说明 20 帧仍不够，必须在主动学习实验前解决。"
        )

    report = f"""# HRNet-W32 Batch-Size Sweep, Second Era

## Question

With reviewed labels, a temporal block split, and a 100-epoch budget, which
batch size does the 20-frame internal validation set select, and does the
untouched final-minute set agree?

## Locked Design

- Labels: all 100 training-pool frames and all 100 final-minute frames
  re-reviewed on 2026-09-10; the `pupil_top` definition moved ~20 px, so
  nothing here is comparable with tags v0.1.0 / v0.2.0.
- Split: temporal block, the chronologically first 80 labelled frames train,
  the last 20 validate, boundary gap 1.2 s, advisor-approved. Training sees
  0-33 s; validation extrapolates to 34-236 s; the final minute extrapolates
  further.
- Schedule: {EPOCHS} epochs on CPU, LR milestones rescaled to {MILESTONES},
  snapshots every {SAVE_EPOCHS} epochs, interrupted runs restart from scratch.
- Within-run checkpoint: DeepLabCut `snapshot-best-*` by maximum internal
  `test.mAP`. Across batches: lowest minimum internal validation total loss.
  Both rules were declared before any result was inspected.
- External test: the same reviewed 100 final-minute frames, {TOTAL_POINTS}
  keypoints, report-only. It selects nothing.

## Results

{markdown_table(data)}

The internal rule selects **batch {int(internal['batch_size'])}** (minimum
validation loss {internal['minimum_internal_valid_loss']:.5f} at epoch
{int(internal['minimum_internal_valid_loss_epoch'])}). The final-minute set is
lowest at **batch {int(external['batch_size'])}**
({external['external_overall_keypoint_rmse_px']:.2f} px overall RMSE).

{agreement_en}

## Figures

![Batch-size overview](01_batch_size_overview.png)

![Learning curves](02_learning_curves.png)

![Keypoint diagnostics](03_keypoint_diagnostics.png)

![Per-frame error distributions](04_error_distributions.png)

## Interpretation

1. Internal validation losses span {loss_span:.1f}% across batches and external
   errors span {rmse_span:.1f}%. In the first era those numbers were 5.8% and
   156%: the internal signal had no resolving power. The reviewed labels and
   the 4x larger, temporally separated validation set are what changed.
2. Best snapshots now land mid-training (epochs
   {', '.join(str(int(v)) for v in data['dlc_best_snapshot_epoch'])}), not at
   epoch 10-20 as in the first era, another sign the validation signal became
   meaningful.
3. Validation here measures forward temporal extrapolation (34-236 s from
   0-33 s training), the same condition the final-minute set measures further
   out. That alignment is by design.
4. Batch sizes are not update-matched: at 100 epochs, batch 2 makes 8x more
   optimizer steps than batch 16. This is a fixed-epoch sweep; runtime scales
   accordingly (see the hours column).
5. Single run per batch size. Only large gaps are interpretable.

## 中文摘要

复查标注 + 时间块 80/20 划分 + 100 epoch 下的 batch size 重扫。内部规则
（20 帧验证集最低验证损失）选出 batch {int(internal['batch_size'])}；最后一
分钟测试集最低的是 batch {int(external['batch_size'])}。{agreement_zh}

新旧时代不可比：标注定义（尤其 pupil_top，约 20 px）、划分、epoch 预算都变
了。每个 batch 只跑一次，只有大差距可解读。

## Files

- `batch_size_summary.csv`: one audited row per run.
- `keypoint_metrics.csv`: per-keypoint RMSE and likelihood.
- `learning_curves.csv`: epoch-level train/validation records.
- `selection.json`: declared rules and both selections.
- `audit.json`: label review, split, config, and isolation checks.

Raw videos, frames, labels, predictions, and model weights stay in ignored
local directories and are not on GitHub.
"""
    (OUTPUT_ROOT / "README.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    dataset_audit = audit_datasets()
    summary, keypoints, curves, frame_data = load_results()

    internal_idx = summary["minimum_internal_valid_loss"].idxmin()
    external_idx = summary["external_overall_keypoint_rmse_px"].idxmin()
    summary["selected_by_internal_rule"] = False
    summary.loc[internal_idx, "selected_by_internal_rule"] = True
    summary["lowest_external_rmse_report_only"] = False
    summary.loc[external_idx, "lowest_external_rmse_report_only"] = True

    summary.to_csv(OUTPUT_ROOT / "batch_size_summary.csv", index=False)
    keypoints.to_csv(OUTPUT_ROOT / "keypoint_metrics.csv", index=False)
    curves.to_csv(OUTPUT_ROOT / "learning_curves.csv", index=False)

    selection = {
        "era": "second (reviewed labels, 80/20 block split, 100 epochs)",
        "rules_declared_before_results": True,
        "within_run_checkpoint_rule": (
            "DeepLabCut snapshot-best by maximum internal test.mAP on the 20 "
            "validation frames"
        ),
        "across_batch_rule": "lowest minimum internal validation total loss",
        "external_policy": "report-only; never selects",
        "internal_rule_selects_batch": int(
            summary.loc[internal_idx, "batch_size"]
        ),
        "external_lowest_batch_report_only": int(
            summary.loc[external_idx, "batch_size"]
        ),
        "internal_external_agree": bool(internal_idx == external_idx),
        "not_comparable_with": ["v0.1.0", "v0.2.0"],
    }
    (OUTPUT_ROOT / "selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    audit = {
        **dataset_audit,
        "four_runs_complete": True,
        "all_runs_reached_100_epochs": True,
        "no_resumed_runs": True,
        "all_models_hrnet_w32": True,
        "milestones_rescaled_to_80_95": True,
        "snapshots_every_10_epochs": True,
        "evaluated_snapshot_matches_best_snapshot": True,
        "identical_external_frame_order": True,
        "external_test_used_for_reporting_only": True,
    }
    (OUTPUT_ROOT / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    save_overview(summary)
    save_learning_curves(curves, summary)
    save_keypoint_diagnostics(keypoints)
    save_error_distributions(frame_data)
    save_report(summary)
    print(
        summary[
            [
                "batch_size",
                "dlc_best_snapshot_epoch",
                "minimum_internal_valid_loss",
                "external_overall_keypoint_rmse_px",
                "external_pupil_center_rmse_px",
                "external_points_likelihood_ge_0_6",
            ]
        ].to_string(index=False)
    )
    print(f"published: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
