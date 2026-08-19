#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/Users/lizhiheng/miniforge3/envs/DEEPLABCUT/bin/python"
CONFIG="$PROJECT_DIR/dlc_projects/EyePupilBlink-Zhiheng-2026-08-17/config.yaml"
IMAGE_FOLDER="$PROJECT_DIR/local_data/test_sets/eye_last_minute_100/frames"

if [ ! -d "$IMAGE_FOLDER" ]; then
  echo "Missing test frame folder: $IMAGE_FOLDER" >&2
  echo "Run scripts/extract_eye_test_frames.py first." >&2
  exit 1
fi

mkdir -p \
  "$PROJECT_DIR/local_data/matplotlib" \
  "$PROJECT_DIR/local_data/numba" \
  "$PROJECT_DIR/local_data/xdg_cache" \
  "$PROJECT_DIR/local_data/napari_config"

export MPLCONFIGDIR="$PROJECT_DIR/local_data/matplotlib"
export NUMBA_CACHE_DIR="$PROJECT_DIR/local_data/numba"
export XDG_CACHE_HOME="$PROJECT_DIR/local_data/xdg_cache"
export NAPARI_CONFIG="$PROJECT_DIR/local_data/napari_config/settings.yaml"

"$PYTHON" - <<PY
from deeplabcut.gui.widgets import launch_napari
import napari

config = "$CONFIG"
image_folder = "$IMAGE_FOLDER"

viewer = launch_napari(
    files=[image_folder, config],
    plugin="napari-deeplabcut",
    stack=False,
)

napari.run()
PY
