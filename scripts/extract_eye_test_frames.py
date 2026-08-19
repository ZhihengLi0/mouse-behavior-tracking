#!/usr/bin/env python3
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
TEST_DIR = PROJECT_DIR / "local_data" / "test_sets" / "eye_last_minute_100"
VIDEO_PATH = TEST_DIR / "face_last60s.mp4"
OUT_DIR = TEST_DIR / "frames"
N_FRAMES = 100


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

    indices = np.linspace(0, total - 1, N_FRAMES).round().astype(int)

    saved = 0
    for i, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_idx}")

        out_path = OUT_DIR / f"test_{i:03d}_frame{frame_idx:05d}.png"
        cv2.imwrite(str(out_path), frame)
        saved += 1

    cap.release()

    print(f"video: {VIDEO_PATH}")
    print(f"total_frames: {total}")
    print(f"fps: {fps:.3f}")
    print(f"saved_frames: {saved}")
    print(f"output_dir: {OUT_DIR}")
    print(f"first_frame: {indices[0]}")
    print(f"last_frame: {indices[-1]}")


if __name__ == "__main__":
    main()
