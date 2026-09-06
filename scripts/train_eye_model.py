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


def completed(train_dir: Path, epochs: int) -> bool:
    stats_path = train_dir / "learning_stats.csv"
    if not stats_path.exists() or not list(train_dir.glob("snapshot-best-*.pt")):
        return False
    stats = pd.read_csv(stats_path)
    return bool(len(stats) and int(stats["step"].max()) >= epochs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuffle", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train_dir = MODEL_ROOT / f"EyePupilBlinkAug17-trainset95shuffle{args.shuffle}" / "train"
    if completed(train_dir, args.epochs):
        print(f"already_complete: shuffle={args.shuffle} batch_size={args.batch_size}")
        return

    resume = latest_numbered_snapshot(train_dir)
    if resume:
        print(f"resuming_from: {resume}")

    deeplabcut.train_network(
        str(CONFIG),
        shuffle=args.shuffle,
        epochs=args.epochs,
        save_epochs=25,
        max_snapshots_to_keep=5,
        batch_size=args.batch_size,
        device=args.device,
        snapshot_path=str(resume) if resume else None,
    )

    if not completed(train_dir, args.epochs):
        raise RuntimeError(f"Training did not complete {args.epochs} epochs for shuffle {args.shuffle}")


if __name__ == "__main__":
    main()
