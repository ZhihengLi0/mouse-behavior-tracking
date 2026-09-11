#!/usr/bin/env python3
"""Train the second-era architecture comparison at batch 2. Training only.

Batch 2 was selected by the pre-declared internal rule on the 20-frame
block-split validation set. The HRNet-W32 entry is shuffle 21 from the batch
sweep (identical batch, epochs, split, labels) and is reused, not retrained.

External evaluation is a separate, later step so that it can never influence
what gets trained. Interrupted runs restart from scratch (--no-resume);
resuming destroyed a best snapshot in the first era. Lid-close sleep merely
pauses training and is safe.
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
OUT = ROOT / "local_data" / "experiments" / "05_arch_sweep_80_20"
SPLIT = ROOT / "local_data" / "experiments" / "split_80_20.json"

BATCH_SIZE = 2
EPOCHS = 100
SAVE_EPOCHS = 10
MAX_SNAPSHOTS = 12
TRAINSET_FRACTION = 80
TRAININGSETINDEX = 1  # index of 0.8 in the project TrainingFraction list

# cheapest first so a failure surfaces early
RUNS = [
    ("cspnext_s", 25),
    ("hrnet_w18", 26),
    ("resnet_50", 27),
    ("hrnet_w48", 28),
]


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_rows(rows: list[dict[str, object]]) -> None:
    columns = [
        "model", "shuffle", "batch_size", "status", "started_at",
        "finished_at", "elapsed_hours", "log",
    ]
    with (OUT / "run_metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
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
        raise SystemExit(f"Missing split record: {SPLIT}")
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "runner.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another architecture sweep is active") from None
    lock.write(f"{os.getpid()}\n")
    lock.flush()
    (OUT / "background.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    (OUT / "runner_started_at.txt").write_text(now() + "\n", encoding="utf-8")

    rows: list[dict[str, object]] = []
    for model, shuffle in RUNS:
        log_path = OUT / f"{model}_shuffle{shuffle}.log"
        started_at = now()
        started = time.monotonic()
        code = run_logged(
            [
                str(PYTHON), str(ROOT / "scripts/train_eye_model.py"),
                "--shuffle", str(shuffle),
                "--batch-size", str(BATCH_SIZE),
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
                "model": model,
                "shuffle": shuffle,
                "batch_size": BATCH_SIZE,
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
        raise SystemExit("Architecture sweep completed with failures")


if __name__ == "__main__":
    main()
