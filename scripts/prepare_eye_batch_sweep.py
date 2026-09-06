#!/usr/bin/env python3
from __future__ import annotations

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
RAW_VIDEO = ROOT / "face.mp4"
PROJECT_VIDEO = DLC_PROJECT / "videos" / "face.mp4"
TRAINING_DIR = DLC_PROJECT / "training-datasets" / "iteration-0" / "UnaugmentedDataSet_EyePupilBlinkAug17"
REFERENCE_DOC = TRAINING_DIR / "Documentation_data-EyePupilBlink_95shuffle4.pickle"
TRAIN_LABELS = DLC_PROJECT / "labeled-data" / "face" / "CollectedData_Zhiheng.h5"
TEST_LABELS = ROOT / "local_data" / "test_sets" / "eye_last_minute_100" / "dlc_label_project" / "labeled-data" / "eye_last_minute_100" / "CollectedData_Zhiheng.h5"
EXPERIMENT_DIR = ROOT / "local_data" / "experiments" / "01_batch_size_sweep"
RUNS = [
    {"batch_size": 8, "shuffle": 4, "existing_baseline": True},
    {"batch_size": 1, "shuffle": 5, "existing_baseline": False},
    {"batch_size": 2, "shuffle": 6, "existing_baseline": False},
    {"batch_size": 4, "shuffle": 7, "existing_baseline": False},
    {"batch_size": 16, "shuffle": 8, "existing_baseline": False},
]


def repair_project_paths() -> None:
    cfg = auxiliaryfunctions.read_config(CONFIG)
    old_video_sets = cfg.get("video_sets", {})
    crop = "0, 928, 0, 736"
    if old_video_sets:
        crop = next(iter(old_video_sets.values())).get("crop", crop)
    cfg["project_path"] = str(DLC_PROJECT)
    cfg["video_sets"] = {str(PROJECT_VIDEO): {"crop": crop}}
    auxiliaryfunctions.write_config(CONFIG, cfg)

    PROJECT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    if PROJECT_VIDEO.is_symlink():
        if PROJECT_VIDEO.resolve(strict=False) != RAW_VIDEO:
            PROJECT_VIDEO.unlink()
            PROJECT_VIDEO.symlink_to(RAW_VIDEO)
    elif not PROJECT_VIDEO.exists():
        PROJECT_VIDEO.symlink_to(RAW_VIDEO)


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        metadata = pickle.load(handle)
    return np.asarray(metadata[1]), np.asarray(metadata[2])


def validate_labels() -> None:
    for label, path, expected_rows in [
        ("training", TRAIN_LABELS, 100),
        ("held-out test", TEST_LABELS, 100),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label} labels: {path}")
        data = pd.read_hdf(path)
        if len(data) != expected_rows:
            raise RuntimeError(f"Expected {expected_rows} {label} rows, found {len(data)}")
        missing = int(data.isna().sum().sum())
        if missing:
            raise RuntimeError(f"{label} labels contain {missing} missing coordinate values")


def main() -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG.exists() or not RAW_VIDEO.exists():
        raise FileNotFoundError("Project config or face.mp4 is missing")

    repair_project_paths()
    validate_labels()

    if not REFERENCE_DOC.exists():
        raise FileNotFoundError(f"Missing reference split: {REFERENCE_DOC}")
    reference_train, reference_test = load_split(REFERENCE_DOC)
    if len(reference_train) != 95 or len(reference_test) != 5:
        raise RuntimeError("Reference shuffle4 is not the expected 95/5 split")

    for run in RUNS:
        shuffle = run["shuffle"]
        documentation = TRAINING_DIR / f"Documentation_data-EyePupilBlink_95shuffle{shuffle}.pickle"
        model_config = DLC_PROJECT / "dlc-models-pytorch" / "iteration-0" / f"EyePupilBlinkAug17-trainset95shuffle{shuffle}" / "train" / "pytorch_config.yaml"

        if not documentation.exists() or not model_config.exists():
            if documentation.exists() or model_config.exists():
                raise RuntimeError(f"Shuffle {shuffle} is partially present; refusing automatic overwrite")
            deeplabcut.create_training_dataset(
                str(CONFIG),
                Shuffles=[shuffle],
                trainIndices=[reference_train],
                testIndices=[reference_test],
                net_type="hrnet_w32",
                userfeedback=False,
                engine=deeplabcut.Engine.PYTORCH,
            )

        train_indices, test_indices = load_split(documentation)
        if not np.array_equal(train_indices, reference_train):
            raise RuntimeError(f"Shuffle {shuffle} training indices differ from shuffle4")
        if not np.array_equal(test_indices, reference_test):
            raise RuntimeError(f"Shuffle {shuffle} validation indices differ from shuffle4")
        if "net_type: hrnet_w32" not in model_config.read_text(encoding="utf-8"):
            raise RuntimeError(f"Shuffle {shuffle} is not HRNet-W32")

        model_cfg = auxiliaryfunctions.read_plainconfig(model_config)
        model_cfg["train_settings"]["batch_size"] = run["batch_size"]
        auxiliaryfunctions.write_plainconfig(model_config, model_cfg)

    design = {
        "experiment": "HRNet-W32 batch-size sweep",
        "selection_rule": "lowest minimum internal validation total loss; external test is report-only",
        "epochs": 200,
        "device": "cpu",
        "training_label_pool": 100,
        "internal_training_rows": 95,
        "internal_validation_rows": 5,
        "external_test_rows": 100,
        "reference_shuffle": 4,
        "train_indices": reference_train.tolist(),
        "validation_indices": reference_test.tolist(),
        "runs": RUNS,
    }
    (EXPERIMENT_DIR / "design.json").write_text(json.dumps(design, indent=2), encoding="utf-8")
    print(json.dumps(design, indent=2))


if __name__ == "__main__":
    main()
