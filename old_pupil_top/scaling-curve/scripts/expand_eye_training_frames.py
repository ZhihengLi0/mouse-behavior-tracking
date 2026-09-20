#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
VIDEO_PATH = PROJECT_DIR / "face.mp4"
LABEL_DIR = (
    PROJECT_DIR
    / "dlc_projects"
    / "EyePupilBlink-Zhiheng-2026-08-17"
    / "labeled-data"
    / "face"
)
TRAIN_SECONDS = 240
FRAME_RE = re.compile(r"img(\d+)\.png$")


def existing_frame_indices() -> set[int]:
    indices: set[int] = set()
    for path in LABEL_DIR.glob("img*.png"):
        match = FRAME_RE.match(path.name)
        if match:
            indices.add(int(match.group(1)))
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add training frames from the first 4 minutes without deleting existing labels."
    )
    parser.add_argument(
        "--target",
        type=int,
        required=True,
        help="Total number of training frame images wanted in labeled-data/face.",
    )
    args = parser.parse_args()

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Missing video: {VIDEO_PATH}")
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    existing = existing_frame_indices()
    if len(existing) >= args.target:
        print(f"existing_frames: {len(existing)}")
        print(f"target_frames: {args.target}")
        print("No new frames needed.")
        return

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    max_train_frame = min(total - 1, int(round(TRAIN_SECONDS * fps)) - 1)
    if max_train_frame <= 0:
        raise RuntimeError("Could not determine a valid first-4-minute training range.")

    needed = args.target - len(existing)
    candidates = np.linspace(0, max_train_frame, args.target * 4).round().astype(int)
    new_indices: list[int] = []
    for frame_idx in candidates:
        idx = int(frame_idx)
        if idx not in existing and idx not in new_indices:
            new_indices.append(idx)
        if len(new_indices) == needed:
            break

    if len(new_indices) != needed:
        raise RuntimeError(f"Needed {needed} new frames, but found {len(new_indices)} candidates.")

    saved = 0
    wanted = set(new_indices)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx in wanted:
            out_path = LABEL_DIR / f"img{frame_idx:05d}.png"
            cv2.imwrite(str(out_path), frame)
            saved += 1
            if saved == needed:
                break

        frame_idx += 1

    cap.release()

    final_count = len(existing_frame_indices())
    print(f"video: {VIDEO_PATH}")
    print(f"label_dir: {LABEL_DIR}")
    print(f"fps: {fps:.3f}")
    print(f"training_range_seconds: 0-{TRAIN_SECONDS}")
    print(f"training_range_frames: 0-{max_train_frame}")
    print(f"existing_frames_before: {len(existing)}")
    print(f"new_frames_saved: {saved}")
    print(f"target_frames: {args.target}")
    print(f"final_frame_count: {final_count}")

    if final_count != args.target:
        raise RuntimeError(f"Expected {args.target} frames after extraction, found {final_count}")


if __name__ == "__main__":
    main()
