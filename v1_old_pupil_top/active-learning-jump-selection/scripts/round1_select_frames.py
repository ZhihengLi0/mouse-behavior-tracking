#!/usr/bin/env python3
"""Active learning round 1: machine half, runs unattended overnight.

Produces, for each of the three frozen branches (uncertain / jump / fitting),
a k-means-selected set of 20 candidate frames from the first four minutes,
ready for human review in the morning. No training happens here: round 1's
model is the already-trained ResNet-50 (shuffle 27).

Steps
1. Cut a frame-accurate first-4-minutes clip (frames 0-14399) so the outlier
   detectors can never see the report-only final minute.
2. Register the clip in the DLC project and analyze it with ResNet-50.
3. For each branch: numpy seed 42, DLC extract_outlier_frames with the frozen
   detector settings, harvest the picked frames into
   active-learning/frames/<branch>/, write a manifest, and clean the shared
   staging folder so branches stay independent.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "dlc_projects" / "EyePupilBlink-Zhiheng-2026-08-17"
CONFIG = PROJECT / "config.yaml"
UNIT = ROOT / "active-learning"
CLIP = UNIT / "training-data" / "face_first4min.mp4"
SOURCE = ROOT / "最原始的视频body1h，eye5min" / "face.mp4"
STAGING = PROJECT / "labeled-data" / "face_first4min"
SHUFFLE = 27  # ResNet-50, the frozen architecture
TRAININGSETINDEX = 1
FIRST_4MIN_FRAMES = 14400
SEED = 42

BRANCHES = [
    ("uncertain", {"outlieralgorithm": "uncertain", "p_bound": 0.6}),
    ("jump", {"outlieralgorithm": "jump", "epsilon": 20}),
    ("fitting", {"outlieralgorithm": "fitting", "epsilon": 20}),
]


def log(msg: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def make_clip() -> None:
    if CLIP.exists():
        log(f"clip exists: {CLIP}")
        return
    import cv2

    log("cutting frame-accurate first-4-minutes clip ...")
    cap = cv2.VideoCapture(str(SOURCE))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    CLIP.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(
        str(CLIP), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    n = 0
    while n < FIRST_4MIN_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        out.write(frame)
        n += 1
    cap.release()
    out.release()
    if n != FIRST_4MIN_FRAMES:
        raise RuntimeError(f"clip has {n} frames, expected {FIRST_4MIN_FRAMES}")
    log(f"clip written: {n} frames at {fps:.0f} fps")


def labeled_frame_numbers() -> set[int]:
    labels = pd.read_hdf(PROJECT / "labeled-data" / "face" / "CollectedData_Zhiheng.h5")
    numbers = set()
    for v in labels.index:
        name = str(v[-1] if isinstance(v, tuple) else v)
        numbers.add(int(name.replace("img", "").replace(".png", "")))
    return numbers


def main() -> None:
    import deeplabcut

    make_clip()

    import yaml

    cfg = yaml.safe_load(CONFIG.read_text())
    if not any("face_first4min" in k for k in cfg["video_sets"]):
        log("registering clip in the DLC project ...")
        deeplabcut.add_new_videos(str(CONFIG), [str(CLIP)], copy_videos=False)

    log("analyzing 14400 frames with ResNet-50 (this is the long step) ...")
    deeplabcut.analyze_videos(
        str(CONFIG),
        [str(CLIP)],
        shuffle=SHUFFLE,
        trainingsetindex=TRAININGSETINDEX,
        snapshot_index=-1,
        device="cpu",
        save_as_csv=False,
    )
    log("analysis done")

    already = labeled_frame_numbers()
    summary = {}
    for branch, params in BRANCHES:
        log(f"branch {branch}: extracting outliers ({params}) ...")
        if STAGING.exists():
            shutil.rmtree(STAGING)
        np.random.seed(SEED)
        deeplabcut.extract_outlier_frames(
            str(CONFIG),
            [str(CLIP)],
            shuffle=SHUFFLE,
            trainingsetindex=TRAININGSETINDEX,
            extractionalgorithm="kmeans",
            automatic=True,
            savelabeled=False,
            **params,
        )
        if not STAGING.exists():
            raise RuntimeError(f"branch {branch}: no frames were extracted")
        dest = UNIT / "frames" / branch
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(STAGING), str(dest))
        picked = sorted(
            int(p.stem.replace("img", "")) for p in dest.glob("img*.png")
        )
        collisions = sorted(set(picked) & already)
        manifest = pd.DataFrame(
            {
                "frame": picked,
                "time_s": [round(f / 60.0, 2) for f in picked],
                "already_labeled_collision": [f in collisions for f in picked],
            }
        )
        manifest.to_csv(dest / "manifest.csv", index=False)
        summary[branch] = {
            "n_selected": len(picked),
            "frames": picked,
            "collisions_with_seed_labels": collisions,
        }
        log(
            f"branch {branch}: {len(picked)} frames -> {dest} "
            f"(collisions: {len(collisions)})"
        )

    report = UNIT / "results" / "round1_selection_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"summary written: {report}")
    (UNIT / "logs" / "round1_selection_DONE.txt").write_text(
        datetime.now().isoformat() + "\n", encoding="utf-8"
    )
    log("ROUND 1 SELECTION COMPLETE")


if __name__ == "__main__":
    main()
