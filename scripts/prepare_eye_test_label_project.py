#!/usr/bin/env python3
from pathlib import Path
import shutil

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
MAIN_CONFIG = PROJECT_DIR / "dlc_projects" / "EyePupilBlink-Zhiheng-2026-08-17" / "config.yaml"
TEST_ROOT = PROJECT_DIR / "local_data" / "test_sets" / "eye_last_minute_100"
SOURCE_FRAMES = TEST_ROOT / "frames"
TEST_PROJECT = TEST_ROOT / "dlc_label_project"
TEST_CONFIG = TEST_PROJECT / "config.yaml"
TEST_LABEL_DIR = TEST_PROJECT / "labeled-data" / "eye_last_minute_100"


def main() -> None:
    if not MAIN_CONFIG.exists():
        raise FileNotFoundError(f"Missing main config: {MAIN_CONFIG}")
    if not SOURCE_FRAMES.exists():
        raise FileNotFoundError(
            f"Missing test frames: {SOURCE_FRAMES}. Run scripts/extract_eye_test_frames.py first."
        )

    frames = sorted(SOURCE_FRAMES.glob("test_*.png"))
    if len(frames) != 100:
        raise RuntimeError(f"Expected 100 test frames in {SOURCE_FRAMES}, found {len(frames)}")

    TEST_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    (TEST_PROJECT / "videos").mkdir(parents=True, exist_ok=True)
    (TEST_PROJECT / "training-datasets").mkdir(parents=True, exist_ok=True)
    (TEST_PROJECT / "dlc-models-pytorch").mkdir(parents=True, exist_ok=True)

    for frame in frames:
        target = TEST_LABEL_DIR / frame.name
        if not target.exists():
            shutil.copy2(frame, target)

    with MAIN_CONFIG.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["Task"] = "EyeLastMinuteTestSet"
    cfg["project_path"] = str(TEST_PROJECT)
    cfg["video_sets"] = {}
    cfg["start"] = 0.0
    cfg["stop"] = 1.0
    cfg["numframes2pick"] = 100

    with TEST_CONFIG.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    print(f"test_project: {TEST_PROJECT}")
    print(f"test_config: {TEST_CONFIG}")
    print(f"label_folder: {TEST_LABEL_DIR}")
    print(f"frames_ready: {len(list(TEST_LABEL_DIR.glob('test_*.png')))}")


if __name__ == "__main__":
    main()
