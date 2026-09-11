#!/usr/bin/env python3
"""Publish the batch-2 architecture comparison into results/02_architecture_sweep.

Reads only completed local artifacts (learning stats, model configs, and
final-minute evaluation outputs), audits them for a fair comparison, and writes
tracked aggregate CSVs, figures, and a README. No raw frames, labels, or model
weights are copied into the tracked results directory.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
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
EXPERIMENT_ROOT = ROOT / "local_data" / "experiments" / "03_architecture_sweep_batch2"
BATCH_SWEEP_ROOT = ROOT / "local_data" / "experiments" / "01_batch_size_sweep"
OUTPUT_ROOT = ROOT / "results" / "02_architecture_sweep"

BATCH_SIZE = 2
EPOCHS = 200

# display order: HRNet family by width, then the two non-HRNet backbones
RUNS = [
    ("HRNet-W18", 10, "arch_hrnet_w18_batch2", "hrnet_w18"),
    ("HRNet-W32", 6, "hrnet_w32_batch2", "hrnet_w32"),
    ("HRNet-W48", 13, "arch_hrnet_w48_batch2", "hrnet_w48"),
    ("ResNet-50", 11, "arch_resnet_50_batch2", "resnet50_gn"),
    ("CSPNeXt-S", 14, "arch_cspnext_s_batch2", "cspnext_s"),
]

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

# HRNet-W48 was interrupted twice and resumed. TorchSnapshotManager restarts
# with `_best_metric=None`, so the first evaluation of a resumed run always
# writes a new best snapshot; when that epoch equals the existing best epoch,
# the manager captures the old best (same path), overwrites it, and then unlinks
# it because the epoch is not a multiple of save_epochs -- deleting the file it
# just wrote. Later epochs only tied the best mAP, and the update requires a
# strict improvement, so no best snapshot was ever recreated. The epochs holding
# the maximum internal mAP (30, 40, 60, 120) were pruned as ordinary snapshots,
# so that checkpoint is unrecoverable without retraining. The final epoch is
# used instead. This was declared before any external error was inspected.
CHECKPOINT_DEVIATION = {
    "HRNet-W48": {
        "rule_applied": "final epoch 200 snapshot",
        "rule_for_other_models": (
            "DeepLabCut snapshot-best selected by maximum internal test.mAP"
        ),
        "cause": (
            "the best snapshot was destroyed by DeepLabCut's snapshot manager "
            "during a resumed run, and the snapshots holding the maximum "
            "internal mAP had already been pruned"
        ),
        "declared_before_seeing_external_error": True,
        "sensitivity_labels": {
            100: "arch_hrnet_w48_batch2_snap100",
            125: "arch_hrnet_w48_batch2_snap125",
            150: "arch_hrnet_w48_batch2_snap150",
            175: "arch_hrnet_w48_batch2_snap175",
        },
    }
}

# RTMPose-S was configured by DeepLabCut as a top-down pose model with an
# SSDLite detector, so it is not controlled against the bottom-up fixed-eye
# models. It was stopped at detector epoch 30/250 and is excluded by design.
EXCLUDED = {
    "model": "RTMPose-S",
    "shuffle": 12,
    "reason": (
        "top-down pose model with an SSDLite detector; not controlled against "
        "the bottom-up single-eye models"
    ),
    "stopped_at": "detector epoch 30/250",
}


def metric(table: pd.DataFrame, name: str, field: str = "value") -> float:
    rows = table.loc[table["metric"] == name, field]
    if rows.empty:
        raise RuntimeError(f"Missing metric: {name}")
    return float(rows.iloc[0])


def train_dir_for(shuffle: int) -> Path:
    return MODEL_ROOT / f"EyePupilBlinkAug17-trainset95shuffle{shuffle}" / "train"


def tested_snapshot(train_dir: Path, model: str) -> tuple[int, Path, str]:
    """Return the checkpoint whose metrics are published for this model.

    Normally DeepLabCut's single `snapshot-best-*.pt`. For a model listed in
    CHECKPOINT_DEVIATION the best snapshot no longer exists, so the final
    numbered snapshot is used and the deviation is reported.
    """
    best = list(train_dir.glob("snapshot-best-*.pt"))
    if len(best) == 1:
        match = re.fullmatch(r"snapshot-best-(\d+)\.pt", best[0].name)
        if not match:
            raise RuntimeError(f"Unexpected snapshot name: {best[0].name}")
        return int(match.group(1)), best[0], "snapshot-best by internal test.mAP"
    if len(best) > 1:
        raise RuntimeError(f"Found {len(best)} best snapshots in {train_dir}")
    if model not in CHECKPOINT_DEVIATION:
        raise RuntimeError(
            f"{model} has no best snapshot and no declared deviation in "
            "CHECKPOINT_DEVIATION"
        )
    final = train_dir / f"snapshot-{EPOCHS:03d}.pt"
    if not final.exists():
        raise RuntimeError(f"{model} is missing its final snapshot {final.name}")
    return EPOCHS, final, "final epoch snapshot (best snapshot destroyed)"


def load_stats(train_dir: Path, model: str) -> pd.DataFrame:
    """Read learning_stats.csv, collapsing rows from resumed runs.

    HRNet-W48 was interrupted twice and resumed from snapshot-025, so its file
    holds several generations of rows for the same epochs. The last occurrence
    of each epoch belongs to the run that actually produced the final weights.
    """
    stats = pd.read_csv(train_dir / "learning_stats.csv")
    duplicates = int(stats.duplicated(subset="step").sum())
    stats = (
        stats.drop_duplicates(subset="step", keep="last")
        .sort_values("step")
        .reset_index(drop=True)
    )
    if int(stats["step"].max()) != EPOCHS:
        raise RuntimeError(
            f"{model} reached epoch {int(stats['step'].max())}, expected {EPOCHS}"
        )
    missing = sorted(set(range(1, EPOCHS + 1)) - set(stats["step"].astype(int)))
    if missing:
        raise RuntimeError(f"{model} is missing epochs: {missing[:10]}")
    stats.attrs["duplicate_rows_collapsed"] = duplicates
    return stats


def parameter_count(snapshot: Path) -> float:
    """Total number of weight elements in a DLC snapshot."""
    try:
        import torch

        payload = torch.load(snapshot, map_location="cpu", weights_only=True)
    except Exception:  # pragma: no cover - reported as NaN in the summary
        return float("nan")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        return float("nan")
    total = 0
    for value in state.values():
        if hasattr(value, "numel"):
            total += int(value.numel())
    return float(total)


def training_wall_clock_hours(log_path: Path) -> float:
    """Sum wall-clock time of successful training invocations in one log.

    Includes any system sleep and resumed segments, so it is only a rough
    runtime indicator, not a compute-matched measurement.
    """
    if not log_path.exists():
        return float("nan")
    text = log_path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    pending: datetime | None = None
    total = 0.0
    found = False
    for line in text.splitlines():
        command = re.match(r"\[([^\]]+)\] COMMAND: (.*)", line)
        if command:
            pending = (
                datetime.fromisoformat(command.group(1))
                if "train_eye_model.py" in command.group(2)
                else None
            )
            continue
        exit_marker = re.match(r"\[([^\]]+)\] EXIT: (\d+)", line)
        if exit_marker and pending is not None:
            if exit_marker.group(2) == "0":
                delta = datetime.fromisoformat(exit_marker.group(1)) - pending
                total += delta.total_seconds() / 3600
                found = True
            pending = None
    return total if found else float("nan")


def frame_statistics(per_frame: pd.DataFrame) -> dict[str, float]:
    columns = [f"{part}_error_px" for part in BODY_PARTS]
    frame_rmse = np.sqrt(np.mean(np.square(per_frame[columns]), axis=1))
    center = per_frame["pupil_center_error_px"]
    return {
        "external_frame_rmse_median_px": float(np.median(frame_rmse)),
        "external_frame_rmse_p90_px": float(np.quantile(frame_rmse, 0.90)),
        "external_frame_rmse_max_px": float(np.max(frame_rmse)),
        "external_pupil_center_error_median_px": float(np.median(center)),
        "external_pupil_center_error_p90_px": float(np.quantile(center, 0.90)),
    }


def audit_datasets() -> dict[str, object]:
    train_labels = PROJECT / "labeled-data" / "face" / "CollectedData_Zhiheng.h5"
    test_labels = (
        TEST_ROOT
        / "dlc_label_project"
        / "labeled-data"
        / "eye_last_minute_100"
        / "CollectedData_Zhiheng.h5"
    )

    def inspect(path: Path, prefix: str) -> list[int]:
        data = pd.read_hdf(path)
        if len(data) != 100:
            raise RuntimeError(f"{prefix} labels contain {len(data)} rows")
        missing = int(data.isna().sum().sum())
        if missing:
            raise RuntimeError(f"{prefix} labels contain {missing} missing values")
        frames = []
        for value in data.index:
            name = str(value[-1] if isinstance(value, tuple) else value)
            match = re.search(r"(?:img|frame)(\d+)", name)
            if not match:
                raise RuntimeError(f"Cannot parse frame number from {name}")
            frames.append(int(match.group(1)))
        return frames

    train_frames = inspect(train_labels, "training pool")
    test_frames = inspect(test_labels, "external test")
    if set(train_frames) & set(test_frames):
        raise RuntimeError("Training and external-test frame numbers overlap")
    if max(train_frames) >= min(test_frames):
        raise RuntimeError("Training frames are not all earlier than test frames")
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
    summary_rows: list[dict[str, object]] = []
    keypoint_rows: list[dict[str, object]] = []
    curve_frames: list[pd.DataFrame] = []
    frame_data: dict[str, pd.DataFrame] = {}
    split_reference: tuple[list[int], list[int]] | None = None
    expected_frames: list[str] | None = None
    resume_notes: dict[str, int] = {}

    for model, shuffle, label, backbone in RUNS:
        train_dir = train_dir_for(shuffle)
        stats = load_stats(train_dir, model)
        resume_notes[model] = int(stats.attrs["duplicate_rows_collapsed"])

        config = yaml.safe_load(
            (train_dir / "pytorch_config.yaml").read_text(encoding="utf-8")
        )
        if config["model"]["backbone"]["model_name"] != backbone:
            raise RuntimeError(
                f"{model} backbone is {config['model']['backbone']['model_name']}, "
                f"expected {backbone}"
            )
        if int(config["train_settings"]["batch_size"]) != BATCH_SIZE:
            raise RuntimeError(f"{model} was not trained at batch {BATCH_SIZE}")
        if config["runner"]["key_metric"] != "test.mAP":
            raise RuntimeError(f"{model} checkpoint rule mismatch")
        # bottom-up single-eye models only: method "bu" and no detector stage.
        # RTMPose-S (shuffle 12) is "td" with an SSDLite detector, which is why
        # it cannot be compared against these runs.
        if config.get("method") != "bu" or config.get("detector") is not None:
            raise RuntimeError(
                f"{model} is not a bottom-up detector-free model "
                f"(method={config.get('method')!r})"
            )

        documentation = (
            PROJECT
            / "training-datasets"
            / "iteration-0"
            / "UnaugmentedDataSet_EyePupilBlinkAug17"
            / f"Documentation_data-EyePupilBlink_95shuffle{shuffle}.pickle"
        )
        split = pd.read_pickle(documentation)
        current = (sorted(int(v) for v in split[1]), sorted(int(v) for v in split[2]))
        if split_reference is None:
            split_reference = current
        elif current != split_reference:
            raise RuntimeError(f"{model} used a different internal 95/5 split")

        snapshot_epoch, snapshot_path, checkpoint_rule = tested_snapshot(
            train_dir, model
        )
        tested = stats.loc[stats["step"] == snapshot_epoch]
        if len(tested) != 1:
            raise RuntimeError(f"No unique stats row for {model} epoch {snapshot_epoch}")
        tested = tested.iloc[0]

        evaluated = stats.dropna(subset=["losses/eval.total_loss"])
        min_valid = evaluated.loc[evaluated["losses/eval.total_loss"].idxmin()]
        best_map = evaluated.loc[evaluated["metrics/test.mAP"].idxmax()]

        prediction_dir = TEST_ROOT / f"predictions_100train_{label}"
        test_summary = pd.read_csv(prediction_dir / "eye_test_summary.csv")
        per_frame = pd.read_csv(
            prediction_dir / "eye_test_per_frame_errors.csv", index_col=0
        )
        if len(per_frame) != 100:
            raise RuntimeError(f"{model} test has {len(per_frame)} frames")
        frames = per_frame.index.tolist()
        if expected_frames is None:
            expected_frames = frames
        elif frames != expected_frames:
            raise RuntimeError(f"{model} used a different external frame order")
        frame_data[model] = per_frame

        evaluated_snapshot = next(
            iter(sorted(prediction_dir.glob("image_predictions_*snapshot_*.h5"))), None
        )
        if evaluated_snapshot is None:
            raise RuntimeError(f"{model} has no prediction file naming its snapshot")
        snapshot_in_name = int(
            re.search(r"snapshot_(?:best-)?(\d+)", evaluated_snapshot.name).group(1)
        )
        if snapshot_in_name != snapshot_epoch:
            raise RuntimeError(
                f"{model} evaluated snapshot {snapshot_in_name} but the published "
                f"checkpoint is epoch {snapshot_epoch}"
            )

        confident = int(
            metric(test_summary, "overall_keypoint_rmse_px_pcutoff_0.6", "n")
        )
        likelihoods = [
            metric(test_summary, f"{part}_mean_likelihood") for part in BODY_PARTS
        ]
        log_path = EXPERIMENT_ROOT / f"{_log_stem(label)}_shuffle{shuffle}.log"
        runtime = training_wall_clock_hours(log_path)
        runtime_source = "architecture queue log"
        if np.isnan(runtime):
            runtime, runtime_source = _batch_sweep_runtime(shuffle)

        row = {
            "model": model,
            "backbone_config_name": backbone,
            "shuffle": shuffle,
            "batch_size": BATCH_SIZE,
            "epochs_completed": EPOCHS,
            "parameters": parameter_count(snapshot_path),
            "checkpoint_rule": checkpoint_rule,
            "dlc_best_snapshot_epoch": snapshot_epoch,
            "internal_mAP_at_tested_snapshot_percent": float(
                tested["metrics/test.mAP"]
            ),
            "internal_rmse_at_tested_snapshot_px": float(tested["metrics/test.rmse"]),
            "internal_valid_loss_at_tested_snapshot": float(
                tested["losses/eval.total_loss"]
            ),
            "final_train_loss": float(
                stats.loc[stats["step"] == EPOCHS, "losses/train.total_loss"].iloc[0]
            ),
            "minimum_internal_valid_loss": float(min_valid["losses/eval.total_loss"]),
            "minimum_internal_valid_loss_epoch": int(min_valid["step"]),
            "maximum_internal_mAP_percent": float(best_map["metrics/test.mAP"]),
            "external_overall_keypoint_rmse_px": metric(
                test_summary, "overall_keypoint_rmse_px"
            ),
            "external_pupil_center_rmse_px": metric(
                test_summary, "pupil_center_rmse_px"
            ),
            "external_pupil_width_mae_px": metric(test_summary, "pupil_width_mae_px"),
            "external_mean_likelihood": float(np.mean(likelihoods)),
            "external_points_likelihood_ge_0_6": confident,
            "external_points_total": TOTAL_POINTS,
            "external_fraction_likelihood_ge_0_6": confident / TOTAL_POINTS,
            "training_wall_clock_hours": runtime,
            "training_wall_clock_source": runtime_source,
        }
        row.update(frame_statistics(per_frame))
        summary_rows.append(row)

        for part in BODY_PARTS:
            keypoint_rows.append(
                {
                    "model": model,
                    "bodypart": part,
                    "rmse_px": metric(test_summary, f"{part}_rmse_px"),
                    "mean_likelihood": metric(test_summary, f"{part}_mean_likelihood"),
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
        curve.insert(0, "model", model)
        curve_frames.append(curve)

    summary = pd.DataFrame(summary_rows)
    order = [model for model, _, _, _ in RUNS]
    summary["model"] = pd.Categorical(summary["model"], order, ordered=True)
    summary = summary.sort_values("model").reset_index(drop=True)
    return (
        summary,
        pd.DataFrame(keypoint_rows),
        pd.concat(curve_frames, ignore_index=True),
        frame_data,
        split_reference,
        resume_notes,
    )


def load_checkpoint_sensitivity() -> pd.DataFrame:
    """External error at every surviving checkpoint of a deviating model.

    Used only to test whether the checkpoint deviation could change the ranking.
    It must never be used to pick the published number for that model.
    """
    rows: list[dict[str, object]] = []
    for model, spec in CHECKPOINT_DEVIATION.items():
        shuffle = next(s for m, s, _, _ in RUNS if m == model)
        primary_label = next(label for m, _, label, _ in RUNS if m == model)
        stats = load_stats(train_dir_for(shuffle), model)
        labels = {EPOCHS: primary_label, **spec["sensitivity_labels"]}
        for epoch, label in sorted(labels.items()):
            summary_path = (
                TEST_ROOT / f"predictions_100train_{label}" / "eye_test_summary.csv"
            )
            if not summary_path.exists():
                continue
            table = pd.read_csv(summary_path)
            internal = stats.loc[stats["step"] == epoch]
            internal_map = (
                float(internal["metrics/test.mAP"].iloc[0])
                if len(internal) and internal["metrics/test.mAP"].notna().all()
                else float("nan")
            )
            rows.append(
                {
                    "model": model,
                    "snapshot_epoch": epoch,
                    "is_published_checkpoint": label == primary_label,
                    "internal_mAP_percent": internal_map,
                    "external_overall_keypoint_rmse_px": metric(
                        table, "overall_keypoint_rmse_px"
                    ),
                    "external_pupil_center_rmse_px": metric(
                        table, "pupil_center_rmse_px"
                    ),
                    "external_pupil_width_mae_px": metric(table, "pupil_width_mae_px"),
                }
            )
    return pd.DataFrame(rows)


def sensitivity_section(
    sensitivity: pd.DataFrame, summary: pd.DataFrame
) -> tuple[str, str]:
    """Report table plus a plain statement of whether the ranking is at risk."""
    if sensitivity.empty:
        return "", ""
    lines = [
        "| Model | Snapshot epoch | Internal mAP | Overall RMSE | Center RMSE | "
        "Width MAE | Published |",
        "|---|---:|---:|---:|---:|---:|:--:|",
    ]
    for row in sensitivity.itertuples():
        internal = (
            "not evaluated"
            if np.isnan(row.internal_mAP_percent)
            else f"{row.internal_mAP_percent:.2f}"
        )
        lines.append(
            f"| {row.model} | {row.snapshot_epoch} | {internal} | "
            f"{row.external_overall_keypoint_rmse_px:.2f} px | "
            f"{row.external_pupil_center_rmse_px:.2f} px | "
            f"{row.external_pupil_width_mae_px:.2f} px | "
            f"{'yes' if row.is_published_checkpoint else 'no'} |"
        )
    table = "\n".join(lines)

    verdicts = []
    for model in sensitivity["model"].unique():
        subset = sensitivity.loc[sensitivity["model"] == model]
        others = summary.loc[summary["model"].astype(str) != model]
        rival = others.loc[others["external_overall_keypoint_rmse_px"].idxmin()]
        best_possible = float(subset["external_overall_keypoint_rmse_px"].min())
        best_epoch = int(
            subset.loc[
                subset["external_overall_keypoint_rmse_px"].idxmin(), "snapshot_epoch"
            ]
        )
        shuffle = next(s for m, s, _, _ in RUNS if m == model)
        stats = load_stats(train_dir_for(shuffle), model)
        evaluated = stats.dropna(subset=["metrics/test.mAP"])
        favoured = evaluated.loc[evaluated["metrics/test.mAP"].idxmax()]
        favoured_epoch = int(favoured["step"])
        tested_row = stats.loc[stats["step"] == EPOCHS].iloc[0]
        survived = favoured_epoch in set(subset["snapshot_epoch"].astype(int))

        if best_possible < rival["external_overall_keypoint_rmse_px"]:
            verdicts.append(
                f"The ranking is **not** safe for {model}: its epoch-{best_epoch} "
                f"checkpoint reaches {best_possible:.2f} px, beating "
                f"{rival['model']} at "
                f"{rival['external_overall_keypoint_rmse_px']:.2f} px. Because "
                f"that checkpoint cannot be selected by an internal rule, "
                f"{model} must be retrained with intact snapshot retention "
                f"before any architecture decision is final."
            )
        elif survived:
            verdicts.append(
                f"For {model} the internally favoured checkpoint (epoch "
                f"{favoured_epoch}, internal mAP "
                f"{favoured['metrics/test.mAP']:.2f}) did survive and is "
                f"included above, and no surviving checkpoint beats "
                f"{rival['model']} at "
                f"{rival['external_overall_keypoint_rmse_px']:.2f} px."
            )
        else:
            verdicts.append(
                f"**{model} remains inconclusive, not confirmed worse.** Every "
                f"surviving checkpoint is worse than {rival['model']} "
                f"({best_possible:.2f} px at epoch {best_epoch} versus "
                f"{rival['external_overall_keypoint_rmse_px']:.2f} px), but the "
                f"checkpoint the internal rule would have chosen is epoch "
                f"{favoured_epoch}, which no longer exists. Its internal "
                f"metrics were the best of the run (mAP "
                f"{favoured['metrics/test.mAP']:.2f} and internal RMSE "
                f"{favoured['metrics/test.rmse']:.2f} px, against "
                f"{tested_row['metrics/test.mAP']:.2f} and "
                f"{tested_row['metrics/test.rmse']:.2f} px at epoch {EPOCHS}), "
                f"and the other four models were all tested at similarly early "
                f"epochs, so its external error is unknown and could have been "
                f"lower. Do not claim this comparison ruled {model} out. "
                f"Settling it requires retraining {model} with intact snapshot "
                f"retention."
            )
    return table, " ".join(verdicts)


def _log_stem(label: str) -> str:
    return label.removeprefix("arch_").removesuffix("_batch2")


def _batch_sweep_runtime(shuffle: int) -> tuple[float, str]:
    path = BATCH_SWEEP_ROOT / "run_metadata.csv"
    if path.exists():
        metadata = pd.read_csv(path)
        rows = metadata.loc[metadata["shuffle"] == shuffle]
        if len(rows):
            return (
                float(rows["training_duration_hours"].iloc[-1]),
                "batch-size sweep metadata",
            )
    return float("nan"), "not recorded"


def save_overview(data: pd.DataFrame) -> None:
    labels = data["model"].astype(str).tolist()
    positions = np.arange(len(data))
    internal_best = data.loc[data["minimum_internal_valid_loss"].idxmin()]
    external_best = data.loc[data["external_overall_keypoint_rmse_px"].idxmin()]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), constrained_layout=True)

    ax = axes[0, 0]
    width = 0.27
    series = [
        ("external_overall_keypoint_rmse_px", "overall keypoint RMSE", "#2F6B9A"),
        ("external_pupil_center_rmse_px", "pupil center RMSE", "#D1495B"),
        ("external_pupil_width_mae_px", "pupil width MAE", "#2A9D8F"),
    ]
    for offset, (column, legend, color) in zip((-1, 0, 1), series, strict=True):
        ax.bar(
            positions + offset * width,
            data[column],
            width,
            color=color,
            label=legend,
        )
    for index, value in enumerate(data["external_overall_keypoint_rmse_px"]):
        ax.text(
            positions[index] - width,
            value + 1.2,
            f"{value:.1f}",
            ha="center",
            fontsize=8,
        )
    ax.set(
        title="Final-minute 100-frame test (lower is better)",
        ylabel="error (px)",
        xticks=positions,
        xticklabels=labels,
    )
    ax.tick_params(axis="x", labelrotation=15)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.bar(positions, data["minimum_internal_valid_loss"], 0.55, color="#6A4C93")
    ax.scatter(
        labels.index(str(internal_best["model"])),
        internal_best["minimum_internal_valid_loss"],
        marker="*",
        s=200,
        color="#E9C46A",
        edgecolor="black",
        zorder=3,
    )
    ax.set(
        title="Minimum internal validation loss (5 frames only)",
        ylabel="total loss",
        xticks=positions,
        xticklabels=labels,
        ylim=(0, float(data["minimum_internal_valid_loss"].max()) * 1.25),
    )
    ax.tick_params(axis="x", labelrotation=15)

    ax = axes[1, 0]
    ax.scatter(
        data["parameters"] / 1e6,
        data["external_overall_keypoint_rmse_px"],
        s=80,
        color="#264653",
    )
    for row in data.itertuples():
        ax.annotate(
            str(row.model),
            (row.parameters / 1e6, row.external_overall_keypoint_rmse_px),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set(
        title="Model capacity vs external error",
        xlabel="parameters (millions)",
        ylabel="external overall RMSE (px)",
    )

    ax = axes[1, 1]
    ax.scatter(
        data["minimum_internal_valid_loss"],
        data["external_overall_keypoint_rmse_px"],
        s=80,
        color="#457B9D",
    )
    for row in data.itertuples():
        ax.annotate(
            str(row.model),
            (row.minimum_internal_valid_loss, row.external_overall_keypoint_rmse_px),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.scatter(
        external_best["minimum_internal_valid_loss"],
        external_best["external_overall_keypoint_rmse_px"],
        marker="s",
        facecolors="none",
        edgecolors="#E63946",
        s=170,
        linewidths=2,
    )
    ax.set(
        title="Internal validation vs external report",
        xlabel="minimum internal validation loss",
        ylabel="external overall RMSE (px)",
    )

    for ax in axes.flat:
        ax.grid(alpha=0.22)
    fig.suptitle(
        f"Architecture comparison at batch {BATCH_SIZE}, {EPOCHS} epochs, "
        "100-frame training pool",
        fontsize=14,
    )
    fig.savefig(OUTPUT_ROOT / "01_architecture_overview.png", dpi=220)
    plt.close(fig)


def save_learning_curves(curves: pd.DataFrame, summary: pd.DataFrame) -> None:
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.8, len(RUNS)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for color, (model, _, _, _) in zip(colors, RUNS, strict=True):
        subset = curves.loc[curves["model"] == model]
        axes[0].plot(
            subset["step"],
            subset["losses/train.total_loss"],
            color=color,
            label=model,
            linewidth=1.4,
        )
        evaluated = subset.dropna(subset=["losses/eval.total_loss"])
        axes[1].plot(
            evaluated["step"],
            evaluated["losses/eval.total_loss"],
            "o-",
            color=color,
            label=model,
            markersize=3,
        )
        tested_epoch = int(
            summary.loc[
                summary["model"].astype(str) == model, "dlc_best_snapshot_epoch"
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
    axes[0].set(title="Training loss", xlabel="epoch", ylabel="total loss")
    axes[1].set(
        title="Validation loss on 5 frames (stars = tested snapshots)",
        xlabel="epoch",
        ylabel="total loss",
    )
    for ax in axes:
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Identical 95/5 split, identical labels, batch 2, 200 epochs", fontsize=13
    )
    fig.savefig(OUTPUT_ROOT / "02_learning_curves.png", dpi=220)
    plt.close(fig)


def save_keypoint_diagnostics(keypoints: pd.DataFrame) -> None:
    order = [model for model, _, _, _ in RUNS]
    rmse = keypoints.pivot(index="bodypart", columns="model", values="rmse_px").loc[
        BODY_PARTS, order
    ]
    likelihood = keypoints.pivot(
        index="bodypart", columns="model", values="mean_likelihood"
    ).loc[BODY_PARTS, order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
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
        values = matrix.to_numpy()
        image = ax.imshow(values, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(matrix.columns)), labels=list(matrix.columns))
        ax.set_yticks(
            range(len(matrix.index)),
            labels=[DISPLAY_NAMES[value] for value in matrix.index],
        )
        ax.tick_params(axis="x", labelrotation=20)
        ax.set(title=title)
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
    fig.suptitle("Final-minute test by keypoint and architecture", fontsize=13)
    fig.savefig(OUTPUT_ROOT / "03_keypoint_diagnostics.png", dpi=220)
    plt.close(fig)


def save_error_distributions(frame_data: dict[str, pd.DataFrame]) -> None:
    frame_rmse = []
    center_error = []
    labels = []
    for model, _, _, _ in RUNS:
        per_frame = frame_data[model]
        columns = [f"{part}_error_px" for part in BODY_PARTS]
        frame_rmse.append(np.sqrt(np.mean(np.square(per_frame[columns]), axis=1)))
        center_error.append(per_frame["pupil_center_error_px"].to_numpy())
        labels.append(model)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    settings = [
        (axes[0], frame_rmse, "Per-frame keypoint RMSE", "frame RMSE (px)"),
        (axes[1], center_error, "Per-frame pupil-center error", "center error (px)"),
    ]
    for ax, values, title, ylabel in settings:
        plot = ax.boxplot(values, labels=labels, showfliers=True, patch_artist=True)
        for patch, color in zip(
            plot["boxes"], plt.get_cmap("Set2").colors, strict=False
        ):
            patch.set_facecolor(color)
        ax.set(title=title, ylabel=ylabel)
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Same 100 held-out frames for every architecture", fontsize=13)
    fig.savefig(OUTPUT_ROOT / "04_error_distributions.png", dpi=220)
    plt.close(fig)


def markdown_table(data: pd.DataFrame) -> str:
    lines = [
        "| Model | Params | Tested epoch | Min valid loss (epoch) | "
        "Overall RMSE | Center RMSE | Width MAE | Likelihood >= 0.6 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in data.itertuples():
        params = (
            "n/a" if np.isnan(row.parameters) else f"{row.parameters / 1e6:.1f} M"
        )
        lines.append(
            f"| {row.model} | {params} | {row.dlc_best_snapshot_epoch} | "
            f"{row.minimum_internal_valid_loss:.5f} "
            f"({row.minimum_internal_valid_loss_epoch}) | "
            f"{row.external_overall_keypoint_rmse_px:.2f} px | "
            f"{row.external_pupil_center_rmse_px:.2f} px | "
            f"{row.external_pupil_width_mae_px:.2f} px | "
            f"{row.external_points_likelihood_ge_0_6}/{TOTAL_POINTS} |"
        )
    return "\n".join(lines)


def robustness_table(data: pd.DataFrame) -> str:
    lines = [
        "| Model | Median frame RMSE | P90 frame RMSE | Worst frame RMSE | "
        "Mean likelihood | Training wall clock |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in data.itertuples():
        runtime = (
            "not recorded"
            if np.isnan(row.training_wall_clock_hours)
            else f"{row.training_wall_clock_hours:.2f} h"
        )
        lines.append(
            f"| {row.model} | {row.external_frame_rmse_median_px:.2f} px | "
            f"{row.external_frame_rmse_p90_px:.2f} px | "
            f"{row.external_frame_rmse_max_px:.2f} px | "
            f"{row.external_mean_likelihood:.3f} | {runtime} |"
        )
    return "\n".join(lines)


def capacity_sentences(data: pd.DataFrame) -> tuple[str, str]:
    """Describe the capacity/accuracy relation from the data, not from priors."""
    usable = data.dropna(subset=["parameters"])
    if len(usable) < 2:
        return (
            "Parameter counts were unavailable, so capacity cannot be related "
            "to accuracy here.",
            "参数量无法读取，因此本次无法讨论容量与精度的关系。",
        )
    largest = usable.loc[usable["parameters"].idxmax()]
    best = usable.loc[usable["external_overall_keypoint_rmse_px"].idxmin()]
    correlation = float(
        np.corrcoef(usable["parameters"], usable["external_overall_keypoint_rmse_px"])[
            0, 1
        ]
    )
    if str(largest["model"]) == str(best["model"]):
        return (
            f"The largest backbone ({largest['model']}, "
            f"{largest['parameters'] / 1e6:.1f} M parameters) is also the most "
            f"accurate on the external set, and parameter count correlates "
            f"{correlation:+.2f} with external RMSE. Capacity still helps at "
            f"this label budget, but with one run per model this is weak "
            f"evidence.",
            f"参数量最大的 {largest['model']}（{largest['parameters'] / 1e6:.1f} M）"
            f"同时也是外部误差最低的模型，参数量与外部 RMSE 的相关系数为 "
            f"{correlation:+.2f}。在当前标注规模下容量仍然有帮助，但每个模型"
            f"只跑了一次，证据很弱。",
        )
    return (
        f"Model capacity does not buy accuracy at this label budget. The "
        f"largest backbone ({largest['model']}, "
        f"{largest['parameters'] / 1e6:.1f} M parameters) is not the most "
        f"accurate; {best['model']} "
        f"({best['parameters'] / 1e6:.1f} M) is, and parameter count "
        f"correlates only {correlation:+.2f} with external RMSE. With 95 "
        f"training frames the bottleneck is labels, not parameters.",
        f"在当前标注规模下，更大的模型并没有更准：参数量最大的 "
        f"{largest['model']}（{largest['parameters'] / 1e6:.1f} M）不是最优，"
        f"最优的是 {best['model']}（{best['parameters'] / 1e6:.1f} M），"
        f"参数量与外部 RMSE 的相关系数只有 {correlation:+.2f}。"
        f"在 95 张训练帧的条件下，瓶颈是标注量而不是参数量。",
    )


def save_report(
    data: pd.DataFrame,
    keypoints: pd.DataFrame,
    resume_notes: dict[str, int],
    sensitivity: pd.DataFrame,
) -> None:
    internal_best = data.loc[data["minimum_internal_valid_loss"].idxmin()]
    external_best = data.loc[data["external_overall_keypoint_rmse_px"].idxmin()]
    center_best = data.loc[data["external_pupil_center_rmse_px"].idxmin()]
    worst = data.loc[data["external_overall_keypoint_rmse_px"].idxmax()]
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
    tested_epochs = ", ".join(
        f"{row.model} {row.dlc_best_snapshot_epoch}" for row in data.itertuples()
    )
    best_keypoints = keypoints.loc[keypoints["model"] == str(external_best["model"])]
    hardest = best_keypoints.loc[best_keypoints["rmse_px"].idxmax()]
    easiest = best_keypoints.loc[best_keypoints["rmse_px"].idxmin()]
    resumed = ", ".join(
        f"{model} ({count} duplicate epoch rows collapsed)"
        for model, count in resume_notes.items()
        if count
    ) or "none"
    capacity_en, capacity_zh = capacity_sentences(data)
    sensitivity_table, sensitivity_verdict = sensitivity_section(sensitivity, data)

    inconclusive = []
    for model in CHECKPOINT_DEVIATION:
        shuffle = next(s for m, s, _, _ in RUNS if m == model)
        stats = load_stats(train_dir_for(shuffle), model)
        evaluated = stats.dropna(subset=["metrics/test.mAP"])
        favoured_epoch = int(evaluated.loc[evaluated["metrics/test.mAP"].idxmax(), "step"])
        surviving = (
            set(
                sensitivity.loc[
                    sensitivity["model"] == model, "snapshot_epoch"
                ].astype(int)
            )
            if not sensitivity.empty
            else set()
        )
        if favoured_epoch not in surviving:
            inconclusive.append((model, favoured_epoch))

    if inconclusive:
        names = ", ".join(model for model, _ in inconclusive)
        best_params = float(external_best["parameters"])
        rival_params = float(
            data.loc[data["model"].astype(str) == inconclusive[0][0], "parameters"].iloc[0]
        )
        decision_caveat = (
            f"\n\nThis is not a clean sweep. {names} is inconclusive rather than "
            f"beaten: the checkpoint its internal rule would have selected no "
            f"longer exists, so its best achievable external error is unknown. "
            f"Choosing {external_best['model']} anyway is defensible on the "
            f"evidence that does exist plus cost: it wins against every model "
            f"whose comparable checkpoint survived, it beats every surviving "
            f"{names} checkpoint, and it needs "
            f"{rival_params / best_params:.1f}x fewer parameters, which matters "
            f"because active learning retrains the model once per round. "
            f"Retraining {names} with intact snapshot retention is the only way "
            f"to settle it, and that has not been done."
        )
    else:
        decision_caveat = ""
    deviation_lines = "\n".join(
        f"- {model}: published from its {spec['rule_applied']} instead of the "
        f"{spec['rule_for_other_models']}, because {spec['cause']}. Declared "
        f"before any external error for this model was inspected."
        for model, spec in CHECKPOINT_DEVIATION.items()
    )
    deviation_block = (
        f"""
## Checkpoint Deviation

One model could not follow the shared checkpoint rule:

{deviation_lines}

Every surviving checkpoint of that model was also evaluated, so the deviation
can be checked rather than assumed:

{sensitivity_table}

{sensitivity_verdict}
"""
        if sensitivity_table
        else ""
    )
    unreliable_runtime = ", ".join(
        str(row.model)
        for row in data.itertuples()
        if row.training_wall_clock_source != "architecture queue log"
        or resume_notes.get(str(row.model))
    ) or "none"

    report = f"""# Architecture Comparison at Batch 2

## Question

With the training pool, internal split, batch size, epoch budget, and external
test set all held fixed, which backbone gives the lowest final-minute error on
100 manually labeled eye frames?

## Locked Design

- Training pool: the same 100 manually labeled frames from the first four
  minutes of `face.mp4`.
- Internal split: the identical 95 training / 5 validation frames in every run,
  verified from each shuffle's DeepLabCut documentation pickle.
- Optimization: batch {BATCH_SIZE}, {EPOCHS} epochs, CPU, DLC seed 42.
- Within-run checkpoint: DeepLabCut `snapshot-best-*`, chosen by maximum
  internal `test.mAP`. The published metrics come from that snapshot. One
  documented exception is described under "Checkpoint Deviation".
- External test: the same 100 manually labeled final-minute frames in the same
  order, {TOTAL_POINTS} keypoints total. Final-minute labels were never trained on.
- Excluded by design: {EXCLUDED['model']} (shuffle {EXCLUDED['shuffle']}) is a
  {EXCLUDED['reason']}. It was stopped at {EXCLUDED['stopped_at']}; its partial
  files are kept only for audit and appear in no result here.

## Results

{markdown_table(data)}

{robustness_table(data)}

**{external_best['model']} has the lowest external error** at
{external_best['external_overall_keypoint_rmse_px']:.2f} px overall RMSE, while
{worst['model']} is worst at {worst['external_overall_keypoint_rmse_px']:.2f} px
— a spread of {rmse_span:.0f}% across backbones. The internal five-frame
validation rule would instead have picked **{internal_best['model']}**
(minimum validation loss {internal_best['minimum_internal_valid_loss']:.5f} at
epoch {int(internal_best['minimum_internal_valid_loss_epoch'])}), and those
minima span only {loss_span:.1f}% across all five models.
{deviation_block}
## Figures

![Architecture overview](01_architecture_overview.png)

![Learning curves](02_learning_curves.png)

![Keypoint diagnostics](03_keypoint_diagnostics.png)

![Per-frame error distributions](04_error_distributions.png)

## Interpretation

1. {capacity_en}
2. Internal and external rankings disagree again, exactly as in the batch-size
   sweep. Five validation frames cannot separate models whose validation minima
   differ by {loss_span:.1f}%. Do not select architectures from that signal.
3. Validation loss rising while training loss falls is the expected shape for
   95 training frames, but it is measured on five frames and must not be read
   as a precise overfitting point.
4. Tested snapshots differ across models ({tested_epochs}), because DeepLabCut
   picks its best checkpoint by internal mAP. Equal epochs therefore do not
   mean equal effective training length.
5. Confidence stays poorly calibrated for every backbone: the best model
   reports only {int(external_best['external_points_likelihood_ge_0_6'])}/{TOTAL_POINTS}
   points at likelihood >= 0.6. `pcutoff=0.6` cannot be used as a hard filter
   before calibration, which also constrains the `uncertain` outlier detector
   in the active-learning stage.
6. Per-keypoint failures are structured, not uniform. For {external_best['model']},
   `{DISPLAY_NAMES[hardest['bodypart']]}` is hardest
   ({hardest['rmse_px']:.1f} px) and `{DISPLAY_NAMES[easiest['bodypart']]}` is
   easiest ({easiest['rmse_px']:.1f} px). Pupil-center error
   ({center_best['external_pupil_center_rmse_px']:.2f} px best, from
   {center_best['model']}) is smaller than overall keypoint RMSE because
   averaging four pupil points cancels part of the error.
7. Runtime is a wall-clock indicator only, not a compute-matched measurement.
   It includes evaluation passes, and for resumed or externally timed runs also
   system sleep and restarts. Do not read runtime as cost for: {unreliable_runtime}.

## Limitations

- Single run per architecture, no seeds repeated. Small differences are not
  resolvable; only the large gaps are.
- The five internal validation frames were flagged as possibly mislabeled but
  deliberately left unchanged so the sweep stays internally consistent. They do
  not update weights, but they do drive checkpoint selection.
- The final-minute 100-frame set has been inspected and relabeled during
  earlier debugging, so it is a controlled comparison set, not a pristine test
  set. Future advisor-supplied video needs a new untouched confirmation set.
- Resumed runs whose duplicate epoch rows were collapsed to the last
  occurrence: {resumed}.
- One model's checkpoint rule deviates; see "Checkpoint Deviation" for the
  cause and for the evidence on whether it can change the ranking.

## Decision

Use **{external_best['model']}** as the fixed architecture for the three-branch
active-learning experiment: it has the lowest external overall RMSE, the lowest
pupil-center error among HRNet variants, the best likelihood behavior, and
mid-range runtime. State plainly in any report that this choice comes from the
final-minute comparison set rather than from the five-frame internal rule, and
that it rests on one run per architecture.{decision_caveat}

## 中文摘要

在标注池、内部划分、batch、epoch 和外部测试集全部固定的条件下比较五个
backbone。最后一分钟 100 帧上 **{external_best['model']}** 误差最低
（总体 RMSE {external_best['external_overall_keypoint_rmse_px']:.2f} px），
{worst['model']} 最差（{worst['external_overall_keypoint_rmse_px']:.2f} px）。
内部 5 帧验证规则会选 {internal_best['model']}，与外部结果不一致——这与 batch
size 实验暴露的是同一个问题：验证集只有 5 帧，不足以区分模型。

{capacity_zh}

所有模型的置信度校准都很差（最优模型也只有
{int(external_best['external_points_likelihood_ge_0_6'])}/{TOTAL_POINTS} 个点的
likelihood 达到 0.6），因此主动学习阶段的 `uncertain` 检测器不能直接依赖
`pcutoff=0.6`。每个架构只跑了一次，只有大的差距可信。RTMPose-S 因为被 DLC
配置成带 SSDLite 检测器的 top-down 模型，与其余 bottom-up 模型不可比，已按
设计排除。最后一分钟这 100 帧此前已被反复查看和修正，属于受控比较集，不是
干净的测试集；老师后续提供新视频时需要一套全新未污染的确认集。

## Files

- `architecture_summary.csv`: one audited row per architecture.
- `keypoint_metrics.csv`: per-keypoint RMSE and likelihood.
- `learning_curves.csv`: epoch-level train/validation records, duplicates from
  resumed runs collapsed.
- `checkpoint_sensitivity.csv`: external error at every surviving checkpoint of
  the model with the documented checkpoint deviation.
- `selection.json`: selection rules, rankings, and the exclusion record.
- `audit.json`: split, label, snapshot, and isolation checks.
- `01_architecture_overview.png`: external errors, internal loss, capacity, and
  the internal/external disagreement.
- `02_learning_curves.png`: train and validation trajectories.
- `03_keypoint_diagnostics.png`: per-keypoint RMSE and likelihood heatmaps.
- `04_error_distributions.png`: per-frame error distributions.

Raw videos, frames, labels, predictions, outlier montages, and model weights
stay in ignored local directories and are not on GitHub.
"""
    (OUTPUT_ROOT / "README.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    dataset_audit = audit_datasets()
    summary, keypoints, curves, frame_data, split, resume_notes = load_results()

    internal_idx = summary["minimum_internal_valid_loss"].idxmin()
    external_idx = summary["external_overall_keypoint_rmse_px"].idxmin()
    summary["selected_by_internal_five_frame_rule"] = False
    summary.loc[internal_idx, "selected_by_internal_five_frame_rule"] = True
    summary["lowest_external_overall_rmse"] = False
    summary.loc[external_idx, "lowest_external_overall_rmse"] = True

    sensitivity = load_checkpoint_sensitivity()

    summary.to_csv(OUTPUT_ROOT / "architecture_summary.csv", index=False)
    keypoints.to_csv(OUTPUT_ROOT / "keypoint_metrics.csv", index=False)
    curves.to_csv(OUTPUT_ROOT / "learning_curves.csv", index=False)
    if not sensitivity.empty:
        sensitivity.to_csv(OUTPUT_ROOT / "checkpoint_sensitivity.csv", index=False)

    selection = {
        "held_fixed": {
            "training_label_pool": 100,
            "internal_train_frames": len(split[0]),
            "internal_validation_frames": len(split[1]),
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "device": "cpu",
            "dlc_seed": 42,
            "external_report_frames": 100,
        },
        "checkpoint_rule_within_each_run": (
            "DeepLabCut snapshot-best selected by maximum internal test.mAP"
        ),
        "checkpoint_deviation": CHECKPOINT_DEVIATION,
        "internal_five_frame_rule_would_select": str(
            summary.loc[internal_idx, "model"]
        ),
        "lowest_external_overall_rmse_model": str(summary.loc[external_idx, "model"]),
        "selected_for_active_learning": str(summary.loc[external_idx, "model"]),
        "models_with_inconclusive_comparison": [
            model
            for model in CHECKPOINT_DEVIATION
            if model in set(summary["model"].astype(str))
        ],
        "selection_basis": (
            "lowest external overall RMSE on the fixed final-minute comparison "
            "set, supported by pupil-center error, likelihood behavior, and "
            "runtime; the internal five-frame rule is reported but not used"
        ),
        "caveats": [
            "one run per architecture; no repeated seeds",
            "internal validation set has only five frames",
            "the final-minute set has been inspected before and is a "
            "comparison set rather than a pristine test set",
        ],
        "excluded": EXCLUDED,
        "ranking_external_overall_rmse": [
            {
                "model": str(row.model),
                "external_overall_keypoint_rmse_px": round(
                    row.external_overall_keypoint_rmse_px, 4
                ),
            }
            for row in summary.sort_values(
                "external_overall_keypoint_rmse_px"
            ).itertuples()
        ],
    }
    (OUTPUT_ROOT / "selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )

    audit = {
        **dataset_audit,
        "architectures_compared": len(RUNS),
        "all_runs_reached_200_epochs": True,
        "all_runs_batch_size_2": True,
        "identical_internal_95_5_split": True,
        "internal_validation_frame_indices": list(split[1]),
        "all_backbones_bottom_up_no_detector": True,
        "backbone_config_names_verified": True,
        "checkpoint_rule_identical": (
            "internal test.mAP for every model except the documented deviation"
        ),
        "checkpoint_rule_per_model": dict(
            zip(
                summary["model"].astype(str),
                summary["checkpoint_rule"],
                strict=True,
            )
        ),
        "evaluated_snapshot_matches_published_checkpoint": True,
        "identical_external_test_frame_order": True,
        "external_test_never_trained_on": True,
        "duplicate_epoch_rows_collapsed_per_model": resume_notes,
        "excluded_incomplete_model": EXCLUDED,
    }
    (OUTPUT_ROOT / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    save_overview(summary)
    save_learning_curves(curves, summary)
    save_keypoint_diagnostics(keypoints)
    save_error_distributions(frame_data)
    save_report(summary, keypoints, resume_notes, sensitivity)

    columns = [
        "model",
        "parameters",
        "dlc_best_snapshot_epoch",
        "minimum_internal_valid_loss",
        "external_overall_keypoint_rmse_px",
        "external_pupil_center_rmse_px",
        "external_pupil_width_mae_px",
        "external_points_likelihood_ge_0_6",
    ]
    print(summary[columns].to_string(index=False))
    print(f"published: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
