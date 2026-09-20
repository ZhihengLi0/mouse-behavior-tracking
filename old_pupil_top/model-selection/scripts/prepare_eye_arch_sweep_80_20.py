#!/usr/bin/env python3
"""Create the second-era architecture-comparison shuffles at batch 2.

Batch 2 was selected minutes earlier by the pre-declared internal rule (lowest
minimum validation loss on the 20-frame block-split validation set). The
HRNet-W32 entry of this comparison is shuffle 21 from the batch sweep itself
(batch 2, 100 epochs, same split) and is reused, not retrained. This script
builds the other four backbones on the identical split and schedule.

RTMPose-S stays excluded by design: DeepLabCut configures it top-down with an
SSDLite detector, so it is not controlled against these bottom-up models.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_eye_split_80_20 import (  # noqa: E402
    build_split,
    create_shuffle,
    label_frame_names,
    SPLIT_PATH,
)

BATCH_SIZE = 2
# model name (DLC net_type), shuffle; cheapest first so failures surface early
RUNS = [
    ("cspnext_s", 25, BATCH_SIZE),
    ("hrnet_w18", 26, BATCH_SIZE),
    ("resnet_50", 27, BATCH_SIZE),
    ("hrnet_w48", 28, BATCH_SIZE),
]


def main() -> None:
    if not SPLIT_PATH.exists():
        raise SystemExit(f"Missing split record: {SPLIT_PATH}")
    names = label_frame_names()
    training, validation, record = build_split(names)
    import json

    stored = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    if stored["validation_filenames"] != record["validation_filenames"]:
        raise SystemExit(
            "Label table changed since split_80_20.json was written; refusing "
            "to build architecture shuffles on a different split"
        )
    for model, shuffle, batch_size in RUNS:
        create_shuffle(model, shuffle, batch_size, training, validation)
        print(f"ready: shuffle {shuffle}  {model}  batch {batch_size}  100 epochs")
    print("hrnet_w32 entry: shuffle 21 (batch sweep), reused without retraining")


if __name__ == "__main__":
    main()
