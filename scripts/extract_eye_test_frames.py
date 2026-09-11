#!/usr/bin/env python3
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
TEST_DIR = PROJECT_DIR / "local_data" / "test_sets" / "eye_last_minute_100"
VIDEO_PATH = PROJECT_DIR / "最原始的视频body1h，eye5min" / "face.mp4"
OUT_DIR = TEST_DIR / "frames"
N_FRAMES = 100
TEST_SECONDS = 60


def main() -> None:
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Missing video: {VIDEO_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for old_frame in OUT_DIR.glob("test_*.png"):
        old_frame.unlink()

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total <= 0:
        raise RuntimeError(f"Video has no readable frames: {VIDEO_PATH}")

    start_frame = max(0, int(round(total - TEST_SECONDS * fps)))
    indices = np.linspace(start_frame, total - 1, N_FRAMES).round().astype(int)
    frame_index_to_output_index = {int(frame_idx): i for i, frame_idx in enumerate(indices)}

    saved = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx in frame_index_to_output_index:
            i = frame_index_to_output_index[frame_idx]
            out_path = OUT_DIR / f"test_{i:03d}_frame{frame_idx:05d}.png"
            cv2.imwrite(str(out_path), frame)
            saved += 1

        frame_idx += 1

    cap.release()

    if saved != N_FRAMES:
        raise RuntimeError(f"Expected {N_FRAMES} frames, but saved {saved}")

    print(f"video: {VIDEO_PATH}")
    print(f"total_frames: {total}")
    print(f"fps: {fps:.3f}")
    print(f"saved_frames: {saved}")
    print(f"output_dir: {OUT_DIR}")
    print(f"test_start_frame: {start_frame}")
    print(f"first_frame: {indices[0]}")
    print(f"last_frame: {indices[-1]}")


if __name__ == "__main__":
    main()
