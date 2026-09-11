#!/usr/bin/env python3
"""Create the 80/20 era: one fixed split, new shuffles, corrected schedules.

The first experimental era (tag v0.2.0) used a 95 training / 5 internal
validation split over the same 100 labelled frames. Five validation frames
could not separate architectures, and they drove checkpoint selection, so the
architecture ranking measured the selection rule rather than the architectures.

This script starts the second era, agreed with the advisor:

- internal split becomes 80 training / 20 validation, so internal validation
  can actually separate models
- epoch budget drops from 200 to 100, because validation loss bottoms out near
  epoch 50-75

Two configuration changes are mandatory for the shorter schedule, and both are
silent failures if forgotten:

- learning-rate milestones must scale from [160, 190] to [80, 95]. Left at the
  200-epoch values they never fire in a 100-epoch run, so the model trains at
  the initial rate throughout and never reaches its fine-tuning phase.
- snapshots must be saved every 10 epochs with retention high enough that none
  are pruned, because the v0.2.0 snapshots were too coarse to locate an optimum
  and its best snapshot was lost to a resumed run.

The validation frames are chosen by uniform temporal spacing over the sorted
label table, not at random and not by any measured quantity, so the choice
cannot have been influenced by any result.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import deeplabcut
import numpy as np
import pandas as pd
from deeplabcut.utils import auxiliaryfunctions


ROOT = Path(__file__).resolve().parents[1]
DLC_PROJECT = ROOT / "dlc_projects" / "EyePupilBlink-Zhiheng-2026-08-17"
CONFIG = DLC_PROJECT / "config.yaml"
TRAINING_DIR = (
    DLC_PROJECT
    / "training-datasets"
    / "iteration-0"
    / "UnaugmentedDataSet_EyePupilBlinkAug17"
)
MODEL_ROOT = DLC_PROJECT / "dlc-models-pytorch" / "iteration-0"
TRAIN_LABELS = DLC_PROJECT / "labeled-data" / "face" / "CollectedData_Zhiheng.h5"
SPLIT_PATH = ROOT / "local_data" / "experiments" / "split_80_20.json"

EPOCHS = 100
VALIDATION_FRAMES = 20
VALIDATION_OFFSET = 2  # skip frame 0, which is the very first video frame
SAVE_EPOCHS = 10
MAX_SNAPSHOTS = 12  # 10 numbered saves plus headroom, so none are pruned
MILESTONES = [80, 95]

# batch-size sweep first, all HRNet-W32; architectures are added later at the
# batch this sweep selects
RUNS = [
    ("hrnet_w32", 21, 2),
    ("hrnet_w32", 22, 4),
    ("hrnet_w32", 23, 8),
    ("hrnet_w32", 24, 16),
]


def label_frame_names() -> list[str]:
    labels = pd.read_hdf(TRAIN_LABELS)
    if len(labels) != 100:
        raise RuntimeError(f"Expected 100 labelled frames, found {len(labels)}")
    missing = int(labels.isna().sum().sum())
    if missing:
        raise RuntimeError(f"Labels contain {missing} missing coordinates")
    return [str(v[-1] if isinstance(v, tuple) else v) for v in labels.index]


def build_split(names: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Uniformly spaced validation rows over the label table's own order."""
    total = len(names)
    step = total // VALIDATION_FRAMES
    validation = np.arange(VALIDATION_OFFSET, total, step)[:VALIDATION_FRAMES]
    if len(validation) != VALIDATION_FRAMES:
        raise RuntimeError(f"Selected {len(validation)} validation frames")
    training = np.array([i for i in range(total) if i not in set(validation)])
    if len(training) != total - VALIDATION_FRAMES:
        raise RuntimeError(f"Selected {len(training)} training frames")
    record = {
        "era": "80/20",
        "supersedes": "95/5 split used up to tag v0.2.0",
        "selection_method": (
            f"every {step}th row of the label table starting at row "
            f"{VALIDATION_OFFSET}; no measured quantity was consulted"
        ),
        "epochs": EPOCHS,
        "save_epochs": SAVE_EPOCHS,
        "max_snapshots": MAX_SNAPSHOTS,
        "lr_milestones": MILESTONES,
        "training_rows": training.tolist(),
        "validation_rows": validation.tolist(),
        "training_filenames": [names[i] for i in training],
        "validation_filenames": [names[i] for i in validation],
    }
    return training, validation, record


def patch_config(shuffle: int, batch_size: int) -> None:
    path = (
        MODEL_ROOT
        / f"EyePupilBlinkAug17-trainset80shuffle{shuffle}"
        / "train"
        / "pytorch_config.yaml"
    )
    if not path.exists():
        raise RuntimeError(f"Missing model config: {path}")
    config = auxiliaryfunctions.read_plainconfig(path)
    config["train_settings"]["batch_size"] = batch_size
    config["train_settings"]["epochs"] = EPOCHS
    scheduler = config["runner"]["scheduler"]
    if scheduler["type"] != "LRListScheduler":
        raise RuntimeError(f"Unexpected scheduler {scheduler['type']} in {path}")
    if len(scheduler["params"]["milestones"]) != len(MILESTONES):
        raise RuntimeError(f"Unexpected milestone count in {path}")
    scheduler["params"]["milestones"] = MILESTONES
    config["runner"]["snapshots"]["save_epochs"] = SAVE_EPOCHS
    config["runner"]["snapshots"]["max_snapshots"] = MAX_SNAPSHOTS
    auxiliaryfunctions.write_plainconfig(path, config)

    written = auxiliaryfunctions.read_plainconfig(path)
    checks = {
        "batch_size": written["train_settings"]["batch_size"] == batch_size,
        "epochs": written["train_settings"]["epochs"] == EPOCHS,
        "milestones": written["runner"]["scheduler"]["params"]["milestones"]
        == MILESTONES,
        "save_epochs": written["runner"]["snapshots"]["save_epochs"] == SAVE_EPOCHS,
        "max_snapshots": written["runner"]["snapshots"]["max_snapshots"]
        == MAX_SNAPSHOTS,
        "key_metric": written["runner"]["key_metric"] == "test.mAP",
        "bottom_up": written.get("method") == "bu"
        and written.get("detector") is None,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"Config verification failed for shuffle {shuffle}: {failed}")


def create_shuffle(
    model: str, shuffle: int, batch_size: int, training: np.ndarray, validation: np.ndarray
) -> None:
    documentation = (
        TRAINING_DIR / f"Documentation_data-EyePupilBlink_80shuffle{shuffle}.pickle"
    )
    train_dir = MODEL_ROOT / f"EyePupilBlinkAug17-trainset80shuffle{shuffle}" / "train"
    model_config = train_dir / "pytorch_config.yaml"

    if documentation.exists() != model_config.exists():
        raise RuntimeError(f"Shuffle {shuffle} is partially present; clean it first")
    if not documentation.exists():
        deeplabcut.create_training_dataset(
            str(CONFIG),
            Shuffles=[shuffle],
            trainIndices=[training],
            testIndices=[validation],
            net_type=model,
            userfeedback=False,
            engine=deeplabcut.Engine.PYTORCH,
        )

    created = pickle.load(documentation.open("rb"))
    if not np.array_equal(np.sort(created[1]), np.sort(training)):
        raise RuntimeError(f"Shuffle {shuffle} training rows do not match the split")
    if not np.array_equal(np.sort(created[2]), np.sort(validation)):
        raise RuntimeError(f"Shuffle {shuffle} validation rows do not match the split")
    patch_config(shuffle, batch_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the split; create no shuffles.",
    )
    args = parser.parse_args()

    names = label_frame_names()
    training, validation, record = build_split(names)

    print(f"label table rows: {len(names)}")
    print(f"training frames: {len(training)}")
    print(f"validation frames: {len(validation)}")
    print("validation filenames:")
    for name in record["validation_filenames"]:
        print(f"  {name}")

    if args.dry_run:
        print("\ndry run: no shuffles created")
        return

    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nsplit recorded: {SPLIT_PATH}")

    for model, shuffle, batch_size in RUNS:
        create_shuffle(model, shuffle, batch_size, training, validation)
        print(f"ready: shuffle {shuffle}  {model}  batch {batch_size}  {EPOCHS} epochs")

    print(
        f"\nall shuffles verified: batch size, {EPOCHS} epochs, milestones "
        f"{MILESTONES}, save_epochs {SAVE_EPOCHS}, max_snapshots {MAX_SNAPSHOTS}"
    )


if __name__ == "__main__":
    main()
