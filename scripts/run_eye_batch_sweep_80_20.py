#!/usr/bin/env python3
"""Train the 80/20-era batch-size sweep. Training only, no evaluation.

Evaluation is deliberately left out. Relabelling the training pool changed the
definition of `pupil_top` by about 20 px, so the final-minute test labels are
now on a different standard and must be reviewed before any external error
number means anything. Training does not depend on those labels, so it runs
while the review happens.

Interrupted runs restart from scratch rather than resume: resuming resets
DeepLabCut's memory of the best metric, which is how the previous era lost
HRNet-W48's best snapshot. At 100 epochs a restart is cheaper than the risk.
"""
from __future__ import annotations

import csv
import fcntl
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
OUT = ROOT / "local_data" / "experiments" / "batch-size-selection"
SPLIT = ROOT / "local_data" / "experiments" / "split_80_20.json"

EPOCHS = 100
SAVE_EPOCHS = 10
MAX_SNAPSHOTS = 12
TRAINSET_FRACTION = 80
# index of 0.8 in the project config's TrainingFraction list, which is
# [0.95, 0.8]. 0.8 was appended rather than inserted so the 95/5-era shuffles
# keep resolving at index 0.
TRAININGSETINDEX = 1

# batch size, shuffle; cheapest first so a failure shows up early
RUNS = [(16, 24), (8, 23), (4, 22), (2, 21)]


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_rows(rows: list[dict[str, object]]) -> None:
    columns = [
        "batch_size", "shuffle", "status", "started_at", "finished_at",
        "elapsed_hours", "log",
    ]
    with (OUT / "run_metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


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


def main() -> None:
    if not SPLIT.exists():
        raise SystemExit(f"Missing split record: {SPLIT}. Run prepare_eye_split_80_20.py")
    OUT.mkdir(parents=True, exist_ok=True)

    lock = (OUT / "runner.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another 80/20 batch sweep is active") from None
    lock.write(f"{os.getpid()}\n")
    lock.flush()
    (OUT / "background.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    (OUT / "runner_started_at.txt").write_text(now() + "\n", encoding="utf-8")

    rows: list[dict[str, object]] = []
    for batch_size, shuffle in RUNS:
        log_path = OUT / f"batch{batch_size}_shuffle{shuffle}.log"
        started_at = now()
        started = time.monotonic()
        code = run_logged(
            [
                str(PYTHON), str(ROOT / "scripts/train_eye_model.py"),
                "--shuffle", str(shuffle),
                "--batch-size", str(batch_size),
                "--epochs", str(EPOCHS),
                "--device", "cpu",
                "--trainset-fraction", str(TRAINSET_FRACTION),
                "--trainingsetindex", str(TRAININGSETINDEX),
                "--save-epochs", str(SAVE_EPOCHS),
                "--max-snapshots", str(MAX_SNAPSHOTS),
                "--no-resume",
            ],
            log_path,
        )
        rows.append(
            {
                "batch_size": batch_size,
                "shuffle": shuffle,
                "status": "complete" if code == 0 else f"training_failed:{code}",
                "started_at": started_at,
                "finished_at": now(),
                "elapsed_hours": f"{(time.monotonic() - started) / 3600:.4f}",
                "log": str(log_path.relative_to(ROOT)),
            }
        )
        write_rows(rows)

    (OUT / "runner_finished_at.txt").write_text(now() + "\n", encoding="utf-8")
    if any(row["status"] != "complete" for row in rows):
        raise SystemExit("80/20 batch sweep completed with failures")


if __name__ == "__main__":
    main()
