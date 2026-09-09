#!/usr/bin/env python3
from __future__ import annotations

import csv
import fcntl
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import deeplabcut
import numpy as np
import pandas as pd
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
PYTHON = Path(sys.executable)
OUT = ROOT / "local_data" / "experiments" / "03_architecture_sweep_batch2"
TEST_ROOT = ROOT / "local_data" / "test_sets" / "eye_last_minute_100"
RUNS = [
    ("hrnet_w18", 10),
    ("resnet_50", 11),
    ("rtmpose_s", 12),
    ("hrnet_w48", 13),
]


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def run_logged(command: list[str], log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{now()}] COMMAND: {' '.join(command)}\n")
        log.flush()
        process = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "MPLCONFIGDIR": str(ROOT / "local_data" / "matplotlib"),
                "NUMBA_CACHE_DIR": str(ROOT / "local_data" / "numba"),
            },
            check=False,
        )
        log.write(f"[{now()}] EXIT: {process.returncode}\n")
        return process.returncode


def prepare(model: str, shuffle: int) -> None:
    reference_train, reference_test = load_split(REFERENCE_DOC)
    documentation = (
        TRAINING_DIR
        / f"Documentation_data-EyePupilBlink_95shuffle{shuffle}.pickle"
    )
    train_dir = (
        DLC_PROJECT
        / "dlc-models-pytorch"
        / "iteration-0"
        / f"EyePupilBlinkAug17-trainset95shuffle{shuffle}"
        / "train"
    )
    model_config = train_dir / "pytorch_config.yaml"
    if not documentation.exists() or not model_config.exists():
        if documentation.exists() or model_config.exists():
            raise RuntimeError(f"Shuffle {shuffle} is partially present")
        deeplabcut.create_training_dataset(
            str(CONFIG),
            Shuffles=[shuffle],
            trainIndices=[reference_train],
            testIndices=[reference_test],
            net_type=model,
            userfeedback=False,
            engine=deeplabcut.Engine.PYTORCH,
        )
    created = pickle.load(documentation.open("rb"))
    if not np.array_equal(created[1], reference_train):
        raise RuntimeError(f"Shuffle {shuffle} training split mismatch")
    if not np.array_equal(created[2], reference_test):
        raise RuntimeError(f"Shuffle {shuffle} validation split mismatch")
    model_cfg = auxiliaryfunctions.read_plainconfig(model_config)
    model_cfg["train_settings"]["batch_size"] = 2
    auxiliaryfunctions.write_plainconfig(model_config, model_cfg)


def training_complete(shuffle: int) -> bool:
    train_dir = (
        DLC_PROJECT
        / "dlc-models-pytorch"
        / "iteration-0"
        / f"EyePupilBlinkAug17-trainset95shuffle{shuffle}"
        / "train"
    )
    stats_path = train_dir / "learning_stats.csv"
    if not stats_path.exists() or not list(train_dir.glob("snapshot-best-*.pt")):
        return False
    stats = pd.read_csv(stats_path)
    return bool(len(stats) and int(stats["step"].max()) >= 200)


def write_rows(rows: list[dict[str, object]]) -> None:
    columns = [
        "model", "shuffle", "batch_size", "status",
        "started_at", "finished_at", "elapsed_hours", "log",
    ]
    with (OUT / "run_metadata.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "runner.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another architecture queue is active") from None
    lock.write(f"{os.getpid()}\n")
    lock.flush()
    (OUT / "background.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    (OUT / "runner_started_at.txt").write_text(now() + "\n", encoding="utf-8")
    repair_project_paths()
    validate_labels()

    rows = []
    for model, shuffle in RUNS:
        log_path = OUT / f"{model}_shuffle{shuffle}.log"
        started_at = now()
        started = time.monotonic()
        status = "complete"
        try:
            prepare(model, shuffle)
            if not training_complete(shuffle):
                code = run_logged(
                    [
                        str(PYTHON), str(ROOT / "scripts/train_eye_model.py"),
                        "--shuffle", str(shuffle), "--batch-size", "2",
                        "--epochs", "200", "--device", "cpu",
                    ],
                    log_path,
                )
                if code:
                    status = "training_failed"
            label = f"arch_{model}_batch2"
            summary = (
                TEST_ROOT / f"predictions_100train_{label}"
                / "eye_test_summary.csv"
            )
            if status == "complete" and not summary.exists():
                code = run_logged(
                    [
                        str(PYTHON),
                        str(ROOT / "scripts/evaluate_eye_test_set.py"),
                        "--train-frames", "100", "--shuffle", str(shuffle),
                        "--model-label", label,
                    ],
                    log_path,
                )
                if code:
                    status = "evaluation_failed"
            if status == "complete":
                for script in ["plot_eye_test_results.py", "plot_eye_outliers.py"]:
                    code = run_logged(
                        [
                            str(PYTHON), str(ROOT / f"scripts/{script}"),
                            "--train-frames", "100", "--model-label", label,
                        ],
                        log_path,
                    )
                    if code:
                        status = "plot_failed"
                        break
        except Exception as error:
            status = f"exception:{type(error).__name__}"
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[{now()}] {status}: {error}\n")

        rows.append(
            {
                "model": model,
                "shuffle": shuffle,
                "batch_size": 2,
                "status": status,
                "started_at": started_at,
                "finished_at": now(),
                "elapsed_hours": f"{(time.monotonic() - started) / 3600:.4f}",
                "log": str(log_path.relative_to(ROOT)),
            }
        )
        write_rows(rows)

    (OUT / "runner_finished_at.txt").write_text(now() + "\n", encoding="utf-8")
    if any(row["status"] != "complete" for row in rows):
        raise SystemExit("Architecture queue completed with failures")


if __name__ == "__main__":
    main()
