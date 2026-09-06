#!/usr/bin/env python3
from __future__ import annotations

import csv
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
DLC_PROJECT = ROOT / "dlc_projects" / "EyePupilBlink-Zhiheng-2026-08-17"
OUT = ROOT / "local_data" / "experiments" / "01_batch_size_sweep"
RUNS = [
    {"batch_size": 8, "shuffle": 4},
    {"batch_size": 1, "shuffle": 5},
    {"batch_size": 2, "shuffle": 6},
    {"batch_size": 4, "shuffle": 7},
    {"batch_size": 16, "shuffle": 8},
]
EPOCHS = 200


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def evaluation_complete(model_label: str) -> bool:
    summary = (
        ROOT
        / "local_data"
        / "test_sets"
        / "eye_last_minute_100"
        / f"predictions_100train_{model_label}"
        / "eye_test_summary.csv"
    )
    return summary.exists()


def training_complete(shuffle: int) -> bool:
    train_dir = DLC_PROJECT / "dlc-models-pytorch" / "iteration-0" / f"EyePupilBlinkAug17-trainset95shuffle{shuffle}" / "train"
    stats_path = train_dir / "learning_stats.csv"
    if not stats_path.exists() or not list(train_dir.glob("snapshot-best-*.pt")):
        return False
    stats = pd.read_csv(stats_path)
    return bool(len(stats) and int(stats["step"].max()) >= EPOCHS)


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


def write_metadata(rows: list[dict[str, object]]) -> None:
    columns = ["batch_size", "shuffle", "status", "started_at", "finished_at", "training_duration_hours", "log"]
    with (OUT / "run_metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lock_handle = (OUT / "runner.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another batch sweep runner is already active") from None
    lock_handle.write(f"{os.getpid()}\n")
    lock_handle.flush()
    (OUT / "background.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    (ROOT / "local_data" / "matplotlib").mkdir(parents=True, exist_ok=True)
    (ROOT / "local_data" / "numba").mkdir(parents=True, exist_ok=True)
    (OUT / "runner_started_at.txt").write_text(now() + "\n", encoding="utf-8")

    if run_logged([str(PYTHON), str(SCRIPTS / "prepare_eye_batch_sweep.py")], OUT / "prepare.log"):
        raise SystemExit("Preparation failed; see prepare.log")

    rows: list[dict[str, object]] = []
    for run in RUNS:
        batch_size = run["batch_size"]
        shuffle = run["shuffle"]
        log_path = OUT / f"batch{batch_size}_shuffle{shuffle}.log"
        started_at = now()
        started = time.monotonic()
        status = "complete"

        if not training_complete(shuffle):
            code = run_logged(
                [
                    str(PYTHON),
                    str(SCRIPTS / "train_eye_model.py"),
                    "--shuffle", str(shuffle),
                    "--batch-size", str(batch_size),
                    "--epochs", str(EPOCHS),
                    "--device", "cpu",
                ],
                log_path,
            )
            if code:
                status = "training_failed"

        if status == "complete":
            model_label = f"hrnet_w32_batch{batch_size}"
            if evaluation_complete(model_label):
                code = 0
            else:
                code = run_logged(
                    [
                        str(PYTHON),
                        str(SCRIPTS / "evaluate_eye_test_set.py"),
                        "--train-frames", "100",
                        "--shuffle", str(shuffle),
                        "--model-label", model_label,
                    ],
                    log_path,
                )

            if code:
                status = "evaluation_failed"
            else:
                for script in ["plot_eye_test_results.py", "plot_eye_outliers.py"]:
                    code = run_logged(
                        [
                            str(PYTHON),
                            str(SCRIPTS / script),
                            "--train-frames", "100",
                            "--model-label", model_label,
                        ],
                        log_path,
                    )
                    if code:
                        status = "plot_failed"
                        break

        duration_hours = (time.monotonic() - started) / 3600
        rows.append({
            "batch_size": batch_size,
            "shuffle": shuffle,
            "status": status,
            "started_at": started_at,
            "finished_at": now(),
            "training_duration_hours": f"{duration_hours:.4f}",
            "log": str(log_path.relative_to(ROOT)),
        })
        write_metadata(rows)
        run_logged([str(PYTHON), str(SCRIPTS / "plot_eye_batch_sweep.py")], OUT / "summary.log")

    (OUT / "runner_finished_at.txt").write_text(now() + "\n", encoding="utf-8")
    failed = [row for row in rows if row["status"] != "complete"]
    if failed:
        raise SystemExit(f"Batch sweep finished with failures: {failed}")


if __name__ == "__main__":
    main()
