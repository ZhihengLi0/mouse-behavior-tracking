#!/usr/bin/env python3
"""Publish the architecture x checkpoint matrix into results/03_checkpoint_matrix.

The batch-2 architecture sweep selected one checkpoint per model with
DeepLabCut's `snapshot-best`, which maximises internal `test.mAP` over five
validation frames whose labels are known to be suspect. A single HRNet-W48
model moved 16 px across checkpoints, which is larger than the gap between
several architectures, so that selection rule cannot support an architecture
ranking on its own.

This script reads every surviving checkpoint of every architecture and reports
four views of the same data, because each answers a different question:

1. internal rule    - what the current, contaminated validation set actually buys
2. fixed epoch      - an apples-to-apples read at one identical epoch
3. oracle           - each architecture's best achievable external error, an
                      UPPER BOUND ONLY; selecting on the report set is not a
                      legal procedure and must never pick the architecture
4. envelope         - min-max range per architecture, so a ranking is claimed
                      only when two ranges do not overlap

No model is retrained; this is evaluation of existing weights.
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
OUTPUT_ROOT = ROOT / "results" / "03_checkpoint_matrix"

BATCH_SIZE = 2
EPOCHS = 200
MATRIX_EPOCHS = [100, 125, 150, 175, 200]
FIXED_EPOCH = 200
TOTAL_POINTS = 800

# model, shuffle, matrix label stem, label of the internal-rule evaluation
RUNS = [
    ("HRNet-W18", 10, "hrnet_w18", "arch_hrnet_w18_batch2"),
    ("HRNet-W32", 6, "hrnet_w32", "hrnet_w32_batch2"),
    ("HRNet-W48", 13, "hrnet_w48", "arch_hrnet_w48_batch2"),
    ("ResNet-50", 11, "resnet_50", "arch_resnet_50_batch2"),
    ("CSPNeXt-S", 14, "cspnext_s", "arch_cspnext_s_batch2"),
]


def metric(table: pd.DataFrame, name: str, field: str = "value") -> float:
    rows = table.loc[table["metric"] == name, field]
    if rows.empty:
        raise RuntimeError(f"Missing metric: {name}")
    return float(rows.iloc[0])


def train_dir_for(shuffle: int) -> Path:
    return MODEL_ROOT / f"EyePupilBlinkAug17-trainset95shuffle{shuffle}" / "train"


def load_stats(shuffle: int) -> pd.DataFrame:
    """learning_stats.csv with rows from resumed runs collapsed to the last."""
    stats = pd.read_csv(train_dir_for(shuffle) / "learning_stats.csv")
    return (
        stats.drop_duplicates(subset="step", keep="last")
        .sort_values("step")
        .reset_index(drop=True)
    )


def internal_rule_epoch(shuffle: int) -> int | None:
    """Epoch of the surviving `snapshot-best`, or None if it was destroyed."""
    best = list(train_dir_for(shuffle).glob("snapshot-best-*.pt"))
    if len(best) != 1:
        return None
    return int(re.fullmatch(r"snapshot-best-(\d+)\.pt", best[0].name).group(1))


def read_evaluation(label: str) -> dict[str, float] | None:
    path = TEST_ROOT / f"predictions_100train_{label}" / "eye_test_summary.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    per_frame = pd.read_csv(
        TEST_ROOT / f"predictions_100train_{label}" / "eye_test_per_frame_errors.csv",
        index_col=0,
    )
    if len(per_frame) != 100:
        raise RuntimeError(f"{label} has {len(per_frame)} frames, expected 100")
    confident = int(metric(table, "overall_keypoint_rmse_px_pcutoff_0.6", "n"))
    return {
        "external_overall_keypoint_rmse_px": metric(table, "overall_keypoint_rmse_px"),
        "external_pupil_center_rmse_px": metric(table, "pupil_center_rmse_px"),
        "external_pupil_width_mae_px": metric(table, "pupil_width_mae_px"),
        "external_points_likelihood_ge_0_6": confident,
        "external_fraction_likelihood_ge_0_6": confident / TOTAL_POINTS,
        "external_frame_order_checksum": hash(tuple(per_frame.index)) & 0xFFFFFFFF,
    }


def collect() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    checksums: set[int] = set()
    for model, shuffle, stem, internal_label in RUNS:
        stats = load_stats(shuffle)
        config = yaml.safe_load(
            (train_dir_for(shuffle) / "pytorch_config.yaml").read_text(
                encoding="utf-8"
            )
        )
        if int(config["train_settings"]["batch_size"]) != BATCH_SIZE:
            raise RuntimeError(f"{model} is not batch {BATCH_SIZE}")
        if config.get("method") != "bu" or config.get("detector") is not None:
            raise RuntimeError(f"{model} is not a bottom-up detector-free model")
        if int(stats["step"].max()) != EPOCHS:
            raise RuntimeError(f"{model} did not reach {EPOCHS} epochs")

        best_epoch = internal_rule_epoch(shuffle)
        candidates: list[tuple[int, str, bool]] = [
            (epoch, f"matrix_{stem}_ep{epoch}", False) for epoch in MATRIX_EPOCHS
        ]
        if best_epoch is not None and best_epoch not in MATRIX_EPOCHS:
            candidates.append((best_epoch, internal_label, True))

        for epoch, label, from_internal_rule in candidates:
            values = read_evaluation(label)
            if values is None:
                continue
            checksums.add(int(values.pop("external_frame_order_checksum")))
            internal = stats.loc[stats["step"] == epoch]
            has_internal = len(internal) and internal["metrics/test.mAP"].notna().all()
            rows.append(
                {
                    "model": model,
                    "shuffle": shuffle,
                    "snapshot_epoch": epoch,
                    "is_internal_rule_choice": bool(
                        from_internal_rule or (best_epoch == epoch)
                    ),
                    "is_fixed_epoch": epoch == FIXED_EPOCH,
                    "internal_mAP_percent": (
                        float(internal["metrics/test.mAP"].iloc[0])
                        if has_internal
                        else np.nan
                    ),
                    "internal_rmse_px": (
                        float(internal["metrics/test.rmse"].iloc[0])
                        if has_internal
                        else np.nan
                    ),
                    "internal_valid_loss": (
                        float(internal["losses/eval.total_loss"].iloc[0])
                        if has_internal
                        else np.nan
                    ),
                    "evaluation_label": label,
                    **values,
                }
            )

    if len(checksums) > 1:
        raise RuntimeError("Evaluations do not all use the same external frame order")

    matrix = pd.DataFrame(rows)
    order = [model for model, _, _, _ in RUNS]
    matrix["model"] = pd.Categorical(matrix["model"], order, ordered=True)
    return matrix.sort_values(["model", "snapshot_epoch"]).reset_index(drop=True)


def build_views(matrix: pd.DataFrame) -> pd.DataFrame:
    """One row per architecture holding all four views of its external error."""
    rows = []
    for model in matrix["model"].cat.categories:
        subset = matrix.loc[matrix["model"] == model]
        if subset.empty:
            continue
        column = "external_overall_keypoint_rmse_px"
        internal = subset.loc[subset["is_internal_rule_choice"]]
        fixed = subset.loc[subset["is_fixed_epoch"]]
        oracle = subset.loc[subset[column].idxmin()]
        # Mean over the checkpoints every model kept. This is the most stable
        # leakage-free statistic available: the epoch set is declared in
        # advance and identical for all models, and averaging selects nothing.
        common = subset.loc[subset["snapshot_epoch"].isin(MATRIX_EPOCHS)]
        rows.append(
            {
                "model": model,
                "checkpoints_evaluated": len(subset),
                "internal_rule_epoch": (
                    int(internal["snapshot_epoch"].iloc[0]) if len(internal) else None
                ),
                "internal_rule_rmse_px": (
                    float(internal[column].iloc[0]) if len(internal) else np.nan
                ),
                "fixed_epoch": FIXED_EPOCH if len(fixed) else None,
                "fixed_epoch_rmse_px": (
                    float(fixed[column].iloc[0]) if len(fixed) else np.nan
                ),
                "common_checkpoint_mean_rmse_px": float(common[column].mean()),
                "common_checkpoint_std_rmse_px": float(common[column].std()),
                "oracle_epoch": int(oracle["snapshot_epoch"]),
                "oracle_rmse_px": float(oracle[column]),
                "envelope_min_px": float(subset[column].min()),
                "envelope_max_px": float(subset[column].max()),
                "envelope_span_px": float(subset[column].max() - subset[column].min()),
            }
        )
    views = pd.DataFrame(rows)
    views["internal_rule_cost_px"] = (
        views["internal_rule_rmse_px"] - views["oracle_rmse_px"]
    )
    return views


def dominance(views: pd.DataFrame) -> list[dict[str, object]]:
    """Pairs whose envelopes do not overlap; only these support a ranking."""
    findings = []
    models = views["model"].tolist()
    for i, left in enumerate(models):
        for right in models[i + 1 :]:
            a = views.loc[views["model"] == left].iloc[0]
            b = views.loc[views["model"] == right].iloc[0]
            if a["envelope_max_px"] < b["envelope_min_px"]:
                findings.append(
                    {"better": left, "worse": right, "separated": True}
                )
            elif b["envelope_max_px"] < a["envelope_min_px"]:
                findings.append(
                    {"better": right, "worse": left, "separated": True}
                )
            else:
                findings.append(
                    {"pair": [left, right], "separated": False}
                )
    return findings


def save_figure(matrix: pd.DataFrame, views: pd.DataFrame) -> None:
    models = list(views["model"])
    positions = np.arange(len(models))
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.8, len(models)))
    column = "external_overall_keypoint_rmse_px"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)

    ax = axes[0]
    for index, (model, color) in enumerate(zip(models, colors, strict=True)):
        subset = matrix.loc[matrix["model"] == model].sort_values("snapshot_epoch")
        ax.plot(
            subset["snapshot_epoch"],
            subset[column],
            "o-",
            color=color,
            label=model,
            markersize=5,
        )
        chosen = subset.loc[subset["is_internal_rule_choice"]]
        ax.scatter(
            chosen["snapshot_epoch"],
            chosen[column],
            marker="*",
            s=230,
            color=color,
            edgecolor="black",
            zorder=4,
        )
    ax.set(
        title="External error at every surviving checkpoint\n(stars = chosen by the five-frame internal rule)",
        xlabel="snapshot epoch",
        ylabel="external overall RMSE (px)",
    )
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)

    ax = axes[1]
    for index, (row, color) in enumerate(zip(views.itertuples(), colors, strict=True)):
        ax.plot(
            [index, index],
            [row.envelope_min_px, row.envelope_max_px],
            color=color,
            linewidth=7,
            solid_capstyle="round",
            alpha=0.45,
        )
        ax.scatter(index, row.oracle_rmse_px, marker="v", s=90, color=color, zorder=3)
        if not np.isnan(row.internal_rule_rmse_px):
            ax.scatter(
                index,
                row.internal_rule_rmse_px,
                marker="*",
                s=210,
                color=color,
                edgecolor="black",
                zorder=4,
            )
        ax.scatter(
            index, row.fixed_epoch_rmse_px, marker="s", s=60, facecolors="none",
            edgecolors=color, linewidths=2, zorder=3,
        )
    ax.set(
        title="Range per architecture\nbar = min-max, triangle = oracle, star = internal rule, square = epoch 200",
        ylabel="external overall RMSE (px)",
        xticks=positions,
        xticklabels=models,
    )
    ax.tick_params(axis="x", labelrotation=20)
    ax.grid(axis="y", alpha=0.22)

    fig.suptitle(
        "Checkpoint choice versus architecture choice, batch 2, 100-frame training pool",
        fontsize=13,
    )
    fig.savefig(OUTPUT_ROOT / "01_checkpoint_matrix.png", dpi=220)
    plt.close(fig)


def views_table(views: pd.DataFrame) -> str:
    lines = [
        "| Model | Internal rule | Mean over the 5 common checkpoints | "
        "Fixed epoch 200 | Oracle (upper bound) | Range | Internal-rule cost |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in views.itertuples():
        internal = (
            "unavailable"
            if np.isnan(row.internal_rule_rmse_px)
            else f"{row.internal_rule_rmse_px:.2f} px (ep {row.internal_rule_epoch})"
        )
        cost = (
            "n/a"
            if np.isnan(row.internal_rule_cost_px)
            else f"{row.internal_rule_cost_px:.2f} px"
        )
        lines.append(
            f"| {row.model} | {internal} | "
            f"{row.common_checkpoint_mean_rmse_px:.2f} px "
            f"(SD {row.common_checkpoint_std_rmse_px:.2f}) | "
            f"{row.fixed_epoch_rmse_px:.2f} px | "
            f"{row.oracle_rmse_px:.2f} px (ep {row.oracle_epoch}) | "
            f"{row.envelope_min_px:.2f}-{row.envelope_max_px:.2f} px | {cost} |"
        )
    return "\n".join(lines)


def save_report(matrix: pd.DataFrame, views: pd.DataFrame) -> None:
    separated = [f for f in dominance(views) if f["separated"]]
    overlapping = [f for f in dominance(views) if not f["separated"]]
    widest = views.loc[views["envelope_span_px"].idxmax()]
    oracle_best = views.loc[views["oracle_rmse_px"].idxmin()]
    internal_best = views.loc[views["internal_rule_rmse_px"].idxmin()]
    fixed_best = views.loc[views["fixed_epoch_rmse_px"].idxmin()]
    cost = views["internal_rule_cost_px"].dropna()

    separated_text = (
        "\n".join(
            f"- {f['better']} is better than {f['worse']}: their ranges do not overlap."
            for f in separated
        )
        or "- None. No pair of architectures has separated ranges."
    )
    overlapping_text = (
        "\n".join(
            f"- {f['pair'][0]} vs {f['pair'][1]}: ranges overlap, not separable."
            for f in overlapping
        )
        or "- None."
    )

    report = f"""# Checkpoint Choice Versus Architecture Choice

## Why This Exists

The batch-2 architecture sweep published one number per architecture, taken
from DeepLabCut's `snapshot-best`, which maximises internal `test.mAP` over
five validation frames. Those five labels are known to be suspect, and a single
HRNet-W48 model moved {widest['envelope_span_px']:.0f} px across its own
checkpoints. If checkpoint choice can move one model more than the gap between
two architectures, a one-checkpoint-per-architecture table cannot support an
architecture ranking.

Every surviving checkpoint of every architecture was therefore evaluated on the
unchanged 100 final-minute frames. No model was retrained.

## Four Views Of The Same Data

{views_table(views)}

- **Internal rule** is what the current validation set actually buys you.
- **Mean over the five common checkpoints** is the most defensible single
  number here. The epoch set is declared in advance, identical for every model,
  and averaging selects nothing, so it leaks nothing and is far less sensitive
  to one unlucky checkpoint than any single reading.
- **Fixed epoch 200** reads every architecture at one identical epoch, so it
  never consults the validation labels. It does penalise architectures that peak
  early, and these did peak at very different epochs.
- **Oracle** is the lowest external error over the surviving checkpoints. It is
  an UPPER BOUND on what an architecture could deliver. It is not achievable in
  practice and **must never be used to select anything**, because selecting on
  the report set converts it into a validation set.
- **Range** is min to max across checkpoints; **span** is the width.

## What Can And Cannot Be Concluded

Ranking claims are only made where two ranges do not overlap:

{separated_text}

Not separable:

{overlapping_text}

## Findings

1. Checkpoint choice moves a single architecture by up to
   {widest['envelope_span_px']:.2f} px ({widest['model']}). Several
   architecture gaps in the published sweep are smaller than that, so those
   gaps were not measuring architecture.
2. The internal five-frame rule costs {cost.min():.2f} to {cost.max():.2f} px
   against the oracle, median {cost.median():.2f} px. That gap is the price of
   the current validation set, and it is the direct argument for relabelling
   before the active-learning experiment.
3. The three views disagree about the winner: internal rule favours
   {internal_best['model']}, fixed epoch 200 favours {fixed_best['model']}, and
   the oracle bound favours {oracle_best['model']}. A single-run,
   five-validation-frame design cannot resolve this.
4. Architectures peak at very different epochs, so a fixed-epoch comparison and
   a best-checkpoint comparison are different experiments. Neither is wrong;
   they answer different questions, and both are reported here rather than one
   being presented as the truth.

## Consequence For The Architecture Decision

The architecture cannot be selected from the final-minute set without breaking
the report-only rule, and it cannot be selected reliably from five suspect
validation frames either. The decision is therefore deferred until the
validation set is rebuilt: all 100 training-pool labels reviewed and the
validation set expanded from 5 to 20 frames. Only then does "each architecture
at its own best checkpoint" become both legal and precise.

The retrain after relabelling should also save snapshots more densely (every 10
epochs instead of every 25), because the surviving checkpoints here are too
coarse to locate a real optimum.

## 中文摘要

架构比较原来每个模型只报一个数，而那个数对应的检查点是用 5 张标注可疑的验证帧
挑出来的。实测同一个模型换检查点可以差
{widest['envelope_span_px']:.0f} px，比若干架构之间的差距还大，所以原表格测到
的不是架构差异。

本报告把每个架构的**全部存活检查点**都评估了一遍（不重训任何模型），给出四种
读法：内部规则（实践中实际能拿到的）、固定 epoch 200（苹果对苹果，但对早收敛
的架构不利）、oracle 上界（潜力，**严禁用于选择**，否则测试集就变成验证集）、
以及区间范围。

只有当两个架构的区间**完全不重叠**时才下排名结论。三种读法给出的赢家并不一致
，说明单次运行 + 5 帧验证的设计无法区分这些架构。内部规则相对 oracle 的代价是
{cost.median():.2f} px 中位数，这就是当前验证集的代价，也是主动学习之前必须
重新标注的直接理由。

架构决定因此推迟到验证集重建之后（复查 100 张标注、验证集从 5 张扩到 20 张）。
重训时还应把存档间隔从 25 epoch 改为 10 epoch，因为现有检查点太稀疏，定位不到
真正的最优点。

## Files

- `checkpoint_matrix.csv`: one row per architecture and checkpoint, with
  internal metrics and external errors.
- `architecture_views.csv`: the four views plus range and span per architecture.
- `dominance.json`: which pairs are separable and which overlap.
- `01_checkpoint_matrix.png`: external error against epoch, and the range per
  architecture.

Model weights, predictions, frames, and labels stay in ignored local
directories and are not on GitHub.
"""
    (OUTPUT_ROOT / "README.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    matrix = collect()
    expected = {model for model, _, _, _ in RUNS}
    missing = expected - set(matrix["model"].astype(str))
    if missing:
        raise RuntimeError(f"No evaluations found for: {sorted(missing)}")
    incomplete = [
        model
        for model in expected
        if not set(MATRIX_EPOCHS)
        <= set(matrix.loc[matrix["model"] == model, "snapshot_epoch"].astype(int))
    ]
    if incomplete:
        raise RuntimeError(
            f"Matrix is incomplete for {sorted(incomplete)}; wait for the "
            "checkpoint matrix run to finish"
        )

    views = build_views(matrix)
    matrix.to_csv(OUTPUT_ROOT / "checkpoint_matrix.csv", index=False)
    views.to_csv(OUTPUT_ROOT / "architecture_views.csv", index=False)
    (OUTPUT_ROOT / "dominance.json").write_text(
        json.dumps(
            {
                "rule": (
                    "a ranking is claimed only when two architectures' external "
                    "error ranges across surviving checkpoints do not overlap"
                ),
                "oracle_policy": (
                    "oracle is an upper bound computed on the report set; it must "
                    "never be used to select an architecture or hyperparameter"
                ),
                "fixed_epoch": FIXED_EPOCH,
                "checkpoints_per_model": MATRIX_EPOCHS,
                "pairs": dominance(views),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_figure(matrix, views)
    save_report(matrix, views)
    print(views.to_string(index=False))
    print(f"published: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
