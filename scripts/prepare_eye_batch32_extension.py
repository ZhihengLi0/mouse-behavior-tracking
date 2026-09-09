#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import deeplabcut
import numpy as np
from deeplabcut.utils import auxiliaryfunctions

from prepare_eye_batch_sweep import (
    CONFIG,
    DLC_PROJECT,
    REFERENCE_DOC,
    TRAINING_DIR,
    load_split,
    repair_project_paths,
    validate_labels,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "local_data" / "experiments" / "02_batch32_extension"
BATCH_SIZE = 32
SHUFFLE = 9


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    repair_project_paths()
    validate_labels()

    reference_train, reference_test = load_split(REFERENCE_DOC)
    if len(reference_train) != 95 or len(reference_test) != 5:
        raise RuntimeError("Reference shuffle4 is not the expected 95/5 split")

    documentation = (
        TRAINING_DIR
        / f"Documentation_data-EyePupilBlink_95shuffle{SHUFFLE}.pickle"
    )
    train_dir = (
        DLC_PROJECT
        / "dlc-models-pytorch"
        / "iteration-0"
        / f"EyePupilBlinkAug17-trainset95shuffle{SHUFFLE}"
        / "train"
    )
    model_config = train_dir / "pytorch_config.yaml"

    if not documentation.exists() or not model_config.exists():
        if documentation.exists() or model_config.exists():
            raise RuntimeError(
                f"Shuffle {SHUFFLE} is partially present; refusing overwrite"
            )
        deeplabcut.create_training_dataset(
            str(CONFIG),
            Shuffles=[SHUFFLE],
            trainIndices=[reference_train],
            testIndices=[reference_test],
            net_type="hrnet_w32",
            userfeedback=False,
            engine=deeplabcut.Engine.PYTORCH,
        )

    train_indices, test_indices = load_split(documentation)
    if not np.array_equal(train_indices, reference_train):
        raise RuntimeError("Shuffle 9 training indices differ from shuffle4")
    if not np.array_equal(test_indices, reference_test):
        raise RuntimeError("Shuffle 9 validation indices differ from shuffle4")

    model_cfg = auxiliaryfunctions.read_plainconfig(model_config)
    if model_cfg["model"]["backbone"]["model_name"] != "hrnet_w32":
        raise RuntimeError("Shuffle 9 is not HRNet-W32")
    model_cfg["train_settings"]["batch_size"] = BATCH_SIZE
    auxiliaryfunctions.write_plainconfig(model_config, model_cfg)

    design = {
        "experiment": "HRNet-W32 batch-size 32 extension",
        "relationship": "controlled extension of 01_batch_size_sweep",
        "selection_rule": (
            "lowest minimum internal validation total loss; "
            "external test is report-only"
        ),
        "epochs": 200,
        "device": "cpu",
        "training_label_pool": 100,
        "internal_training_rows": 95,
        "internal_validation_rows": 5,
        "external_test_rows": 100,
        "reference_shuffle": 4,
        "batch_size": BATCH_SIZE,
        "shuffle": SHUFFLE,
        "train_indices": reference_train.tolist(),
        "validation_indices": reference_test.tolist(),
    }
    (OUT / "design.json").write_text(
        json.dumps(design, indent=2), encoding="utf-8"
    )
    print(json.dumps(design, indent=2))


if __name__ == "__main__":
    main()
