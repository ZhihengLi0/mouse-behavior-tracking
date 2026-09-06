#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DLC_PROJECT = ROOT / "dlc_projects" / "EyePupilBlink-Zhiheng-2026-08-17"
TEST_ROOT = ROOT / "local_data" / "test_sets" / "eye_last_minute_100"
OUT = ROOT / "local_data" / "experiments" / "01_batch_size_sweep"
RUNS = [(1, 5), (2, 6), (4, 7), (8, 4), (16, 8)]


def summary_value(summary: pd.DataFrame, metric: str) -> float:
    rows = summary.loc[summary["metric"] == metric, "value"]
    return float(rows.iloc[0]) if not rows.empty else float("nan")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metadata_path = OUT / "run_metadata.csv"
    metadata = pd.read_csv(metadata_path) if metadata_path.exists() else pd.DataFrame()

    rows = []
    for batch_size, shuffle in RUNS:
        train_dir = DLC_PROJECT / "dlc-models-pytorch" / "iteration-0" / f"EyePupilBlinkAug17-trainset95shuffle{shuffle}" / "train"
        stats_path = train_dir / "learning_stats.csv"
        summary_path = TEST_ROOT / f"predictions_100train_hrnet_w32_batch{batch_size}" / "eye_test_summary.csv"
        if not stats_path.exists() or not summary_path.exists():
            continue

        stats = pd.read_csv(stats_path)
        evaluated = stats.dropna(subset=["losses/eval.total_loss"])
        if evaluated.empty:
            continue
        best_idx = evaluated["losses/eval.total_loss"].idxmin()
        summary = pd.read_csv(summary_path)
        cutoff = summary.loc[summary["metric"] == "overall_keypoint_rmse_px_pcutoff_0.6"]

        row = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "epochs_completed": int(stats["step"].max()),
            "min_internal_valid_loss": float(stats.loc[best_idx, "losses/eval.total_loss"]),
            "min_valid_loss_epoch": int(stats.loc[best_idx, "step"]),
            "internal_rmse_at_min_valid_loss_px": float(stats.loc[best_idx, "metrics/test.rmse"]),
            "external_overall_rmse_px": summary_value(summary, "overall_keypoint_rmse_px"),
            "external_pupil_center_rmse_px": summary_value(summary, "pupil_center_rmse_px"),
            "external_pupil_width_mae_px": summary_value(summary, "pupil_width_mae_px"),
            "external_points_likelihood_ge_0_6": int(cutoff["n"].iloc[0]) if not cutoff.empty else 0,
        }
        if not metadata.empty and batch_size in set(metadata["batch_size"]):
            meta = metadata.loc[metadata["batch_size"] == batch_size].iloc[-1]
            row["training_duration_hours"] = float(meta.get("training_duration_hours", np.nan))
            row["status"] = str(meta.get("status", "complete"))
        rows.append(row)

    if not rows:
        print("No completed evaluated runs yet.")
        return

    data = pd.DataFrame(rows).sort_values("batch_size")
    winner_idx = data["min_internal_valid_loss"].idxmin()
    data["selected_by_internal_validation"] = False
    data.loc[winner_idx, "selected_by_internal_validation"] = True
    data.to_csv(OUT / "batch_size_comparison.csv", index=False)

    winner = data.loc[winner_idx].to_dict()
    (OUT / "selection.json").write_text(json.dumps(winner, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    x = data["batch_size"]
    axes[0].plot(x, data["external_overall_rmse_px"], marker="o", label="overall keypoint RMSE")
    axes[0].plot(x, data["external_pupil_center_rmse_px"], marker="o", label="pupil center RMSE")
    axes[0].plot(x, data["external_pupil_width_mae_px"], marker="o", label="pupil width MAE")
    axes[0].set_title("HRNet-W32 batch-size sweep: locked final-minute evaluation")
    axes[0].set_ylabel("external test error (px; lower is better)")
    axes[0].set_xticks(x)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(x, data["min_internal_valid_loss"], marker="o", color="#D62728")
    chosen_batch = int(data.loc[winner_idx, "batch_size"])
    chosen_loss = float(data.loc[winner_idx, "min_internal_valid_loss"])
    axes[1].scatter([chosen_batch], [chosen_loss], marker="*", s=180, color="#2CA02C", zorder=3, label="selected")
    axes[1].set_title("Selection metric: minimum internal validation loss")
    axes[1].set_xlabel("batch size")
    axes[1].set_ylabel("minimum validation loss")
    axes[1].set_xticks(x)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.savefig(OUT / "batch_size_comparison.png", dpi=200)
    plt.close(fig)
    print(data.to_string(index=False))
    print(f"selected_batch_size: {chosen_batch}")


if __name__ == "__main__":
    main()
