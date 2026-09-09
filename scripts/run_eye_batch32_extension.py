#!/usr/bin/env python3
from __future__ import annotations

import csv
import argparse
import fcntl
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
SCRIPTS = ROOT / "scripts"
PROJECT = ROOT / "dlc_projects" / "EyePupilBlink-Zhiheng-2026-08-17"
OUT = ROOT / "local_data" / "experiments" / "02_batch32_extension"
BATCH_SIZE = 32
SHUFFLE = 9
EPOCHS = 200


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


def training_complete() -> bool:
    train_dir = (
        PROJECT
        / "dlc-models-pytorch"
        / "iteration-0"
        / f"EyePupilBlinkAug17-trainset95shuffle{SHUFFLE}"
        / "train"
    )
    stats_path = train_dir / "learning_stats.csv"
    if not stats_path.exists() or not list(train_dir.glob("snapshot-best-*.pt")):
        return False
    stats = pd.read_csv(stats_path)
    return bool(len(stats) and int(stats["step"].max()) >= EPOCHS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the memory-intensive HRNet-W32 batch-32 extension."
    )
    parser.add_argument(
        "--force-memory-risk",
        action="store_true",
        help="Required after this run exhausted swap on a 16 GiB Mac.",
    )
    args = parser.parse_args()
    if not args.force_memory_risk:
        raise SystemExit(
            "Refusing batch 32 by default: the 2026-09-08 attempt expanded "
            "swap to 15 GiB before epoch 1. Use --force-memory-risk only "
            "on a machine with substantially more RAM."
        )

    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "runner.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another batch-32 runner is already active") from None
    lock.write(f"{os.getpid()}\n")
    lock.flush()
    (OUT / "background.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    (OUT / "runner_started_at.txt").write_text(now() + "\n", encoding="utf-8")
    for folder in ["matplotlib", "numba"]:
        (ROOT / "local_data" / folder).mkdir(parents=True, exist_ok=True)

    if run_logged(
        [str(PYTHON), str(SCRIPTS / "prepare_eye_batch32_extension.py")],
        OUT / "prepare.log",
    ):
        raise SystemExit("Preparation failed; see prepare.log")

    log_path = OUT / "batch32_shuffle9.log"
    started_at = now()
    started = time.monotonic()
    status = "complete"
    if not training_complete():
        code = run_logged(
            [
                str(PYTHON), str(SCRIPTS / "train_eye_model.py"),
                "--shuffle", str(SHUFFLE),
                "--batch-size", str(BATCH_SIZE),
                "--epochs", str(EPOCHS),
                "--device", "cpu",
            ],
            log_path,
        )
        if code:
            status = "training_failed"

    model_label = "hrnet_w32_batch32"
    prediction_dir = (
        ROOT / "local_data" / "test_sets" / "eye_last_minute_100"
        / f"predictions_100train_{model_label}"
    )
    if (
        status == "complete"
        and not (prediction_dir / "eye_test_summary.csv").exists()
    ):
        code = run_logged(
            [
                str(PYTHON), str(SCRIPTS / "evaluate_eye_test_set.py"),
                "--train-frames", "100",
                "--shuffle", str(SHUFFLE),
                "--model-label", model_label,
            ],
            log_path,
        )
        if code:
            status = "evaluation_failed"

    if status == "complete":
        for script in ["plot_eye_test_results.py", "plot_eye_outliers.py"]:
            code = run_logged(
                [
                    str(PYTHON), str(SCRIPTS / script),
                    "--train-frames", "100",
                    "--model-label", model_label,
                ],
                log_path,
            )
            if code:
                status = "plot_failed"
                break

    row = {
        "batch_size": BATCH_SIZE,
        "shuffle": SHUFFLE,
        "status": status,
        "started_at": started_at,
        "finished_at": now(),
        "training_duration_hours": (
            f"{(time.monotonic() - started) / 3600:.4f}"
        ),
        "log": str(log_path.relative_to(ROOT)),
    }
    with (OUT / "run_metadata.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    if status != "complete":
        raise SystemExit(f"Batch-32 extension failed: {status}")
    (OUT / "runner_finished_at.txt").write_text(now() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
