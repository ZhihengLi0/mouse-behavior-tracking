#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import deeplabcut
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "dlc_projects" / "EyePupilBlink-Zhiheng-2026-08-17" / "config.yaml"
MODEL_ROOT = ROOT / "dlc_projects" / "EyePupilBlink-Zhiheng-2026-08-17" / "dlc-models-pytorch" / "iteration-0"


def latest_numbered_snapshot(train_dir: Path) -> Path | None:
    snapshots = []
    for path in train_dir.glob("snapshot-*.pt"):
        suffix = path.stem.removeprefix("snapshot-")
        if suffix.isdigit():
            snapshots.append((int(suffix), path))
    return max(snapshots, default=(0, None))[1]


def reached_epochs(train_dir: Path, epochs: int) -> bool:
    stats_path = train_dir / "learning_stats.csv"
    if not stats_path.exists():
        return False
    stats = pd.read_csv(stats_path)
    return bool(len(stats) and int(stats["step"].max()) >= epochs)


def has_best_snapshot(train_dir: Path) -> bool:
    """Whether DeepLabCut's best-metric checkpoint survived training.

    Resuming can destroy it. TorchSnapshotManager restarts with
    `_best_metric=None`, so the first evaluation of a resumed run always writes
    a new best. If that epoch equals the epoch of the existing best, the manager
    captures the old best (same path), overwrites it, and then unlinks it when
    the epoch is not a multiple of `save_epochs` -- deleting the file it just
    wrote. No later epoch that merely ties the best metric will recreate it,
    because the update requires a strict improvement.
    """
    return bool(list(train_dir.glob("snapshot-best-*.pt")))


def completed(train_dir: Path, epochs: int) -> bool:
    return reached_epochs(train_dir, epochs) and has_best_snapshot(train_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuffle", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--trainset-fraction", type=int, default=95,
        help="Training-set percentage in the DLC model folder name.",
    )
    parser.add_argument(
        "--trainingsetindex", type=int, default=0,
        help=(
            "Index into the project config's TrainingFraction list. It must "
            "point at the fraction this shuffle was built with, or "
            "train_network cannot find the shuffle."
        ),
    )
    parser.add_argument(
        "--save-epochs", type=int, default=25,
        help="Snapshot interval. train_network's argument overrides the model config.",
    )
    parser.add_argument(
        "--max-snapshots", type=int, default=5,
        help="Numbered snapshots to keep. Raise it to stop pruning useful checkpoints.",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help=(
            "Never resume from a numbered snapshot; restart from scratch "
            "instead. Resuming resets DeepLabCut's memory of the best metric, "
            "which can destroy the best snapshot (see has_best_snapshot)."
        ),
    )
    args = parser.parse_args()

    train_dir = (
        MODEL_ROOT
        / f"EyePupilBlinkAug17-trainset{args.trainset_fraction}shuffle{args.shuffle}"
        / "train"
    )
    if completed(train_dir, args.epochs):
        print(f"already_complete: shuffle={args.shuffle} batch_size={args.batch_size}")
        return
    if reached_epochs(train_dir, args.epochs) and not has_best_snapshot(train_dir):
        # Retraining would silently discard finished weights. Stop instead and
        # let a human decide between evaluating a numbered snapshot and a
        # controlled retrain.
        raise RuntimeError(
            f"Shuffle {args.shuffle} already reached {args.epochs} epochs but has "
            "no snapshot-best-*.pt (destroyed by a resumed run). Refusing to "
            "retrain over finished weights; audit the numbered snapshots first."
        )

    resume = None if args.no_resume else latest_numbered_snapshot(train_dir)
    epochs_to_train = args.epochs
    if args.no_resume:
        print("no_resume: training from scratch", flush=True)
    if resume:
        # DLC treats `epochs` as additional epochs beyond the resumed snapshot
        # (observed: epochs=200 resumed from snapshot-025 targets 225). Subtract
        # the snapshot epoch so every run ends at exactly args.epochs total.
        resume_epoch = int(resume.stem.removeprefix("snapshot-"))
        epochs_to_train = args.epochs - resume_epoch
        print(
            f"resuming_from: {resume} "
            f"(epoch {resume_epoch}, {epochs_to_train} more to reach {args.epochs})",
            flush=True,
        )

    deeplabcut.train_network(
        str(CONFIG),
        shuffle=args.shuffle,
        trainingsetindex=args.trainingsetindex,
        epochs=epochs_to_train,
        save_epochs=args.save_epochs,
        max_snapshots_to_keep=args.max_snapshots,
        batch_size=args.batch_size,
        device=args.device,
        snapshot_path=str(resume) if resume else None,
    )

    if not completed(train_dir, args.epochs):
        raise RuntimeError(f"Training did not complete {args.epochs} epochs for shuffle {args.shuffle}")


if __name__ == "__main__":
    main()
